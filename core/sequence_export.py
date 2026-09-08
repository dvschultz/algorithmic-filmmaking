"""Export timeline sequences to video files."""

import logging
import json
import os
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING
from dataclasses import dataclass, replace
from fractions import Fraction

from models.sequence import Sequence, SequenceClip
from models.clip import Source
from models.media_time import VideoRange, frame_boundary, frame_rate
from core.render_plan import RenderSegment
from core.binary_resolver import find_binary, get_subprocess_kwargs

if TYPE_CHECKING:
    from models.frame import Frame

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for sequence export."""

    output_path: Path
    fps: float = 30.0
    width: Optional[int] = None  # None = use source resolution
    height: Optional[int] = None
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    preset: str = "fast"
    crf: int = 18  # Quality (lower = better)
    show_chromatic_color_bar: bool = False
    chromatic_color_bar_height_ratio: float = 0.04
    chromatic_color_bar_min_height: int = 12
    music_path: Optional[Path] = None  # Music file to mux onto exported video
    cancel_check: Callable[[], bool] | None = None


class SequenceExporter:
    """Exports timeline sequences to video files using FFmpeg."""

    def __init__(self, ffmpeg_path: str | None = None):
        binary = ffmpeg_path or find_binary("ffmpeg")
        if not binary:
            raise RuntimeError("FFmpeg not found")
        self.ffmpeg_path: str = binary

    def export(
        self,
        sequence: Sequence,
        sources: dict[str, Source],
        clips: dict[str, tuple],  # clip_id -> (Clip, Source)
        config: ExportConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        frames: Optional[dict[str, "Frame"]] = None,
    ) -> bool:
        """
        Export a sequence to a video file.

        Args:
            sequence: The Sequence to export
            sources: Dict of source_id -> Source
            clips: Dict of clip_id -> (Clip, Source)
            config: Export configuration
            progress_callback: Optional callback (progress 0-1, message)
            frames: Optional dict of frame_id -> Frame for frame-based entries

        Returns:
            True if export succeeded
        """
        from core.render_plan import compile_render_plan
        from models.media_time import VideoRange

        def cancelled():
            return config.cancel_check is not None and config.cancel_check()

        try:
            logger.info("Starting sequence export: %s", config.output_path)
            if cancelled():
                return False
            plan = compile_render_plan(
                sequence, sources, clips, frames=frames, output_fps=config.fps,
                music_path=config.music_path,
            )
            from core.media_timing import probe_video_timing
            verified = {}
            audio_streams = {}
            for segment in plan.segments:
                if segment.kind == "video":
                    assert segment.path is not None
                    if segment.path not in verified:
                        if progress_callback:
                            progress_callback(0, f"Verifying source timing: {segment.path.name}")
                        verified[segment.path] = probe_video_timing(segment.path, config.cancel_check)
                        audio_streams[segment.path] = self._has_audio(segment.path)
                    timing = verified[segment.path]
                    assert isinstance(segment.media, VideoRange)
                    if segment.media.end_frame > timing.frame_count:
                        raise ValueError(f"Entry {segment.entry_id} extends beyond the last decoded source frame")
                    if timing.variable or segment.source_boundaries is not None:
                        selected = timing.boundaries[segment.media.start_frame:segment.media.end_frame]
                        if segment.source_boundaries != selected or segment.media.end != timing.boundaries[segment.media.end_frame]:
                            raise ValueError(f"Variable-rate source timing is missing or changed for {segment.entry_id}; reimport the source and rebuild this entry")
                    elif segment.media.rate != timing.rate:
                        raise ValueError(f"Source frame rate changed for {segment.entry_id}; reimport the source and rebuild this entry")
                if segment.reverse and segment.timeline.duration > self._REVERSE_MAX_DURATION:
                    raise ValueError(
                        f"Reverse entry {segment.entry_id} exceeds the {self._REVERSE_MAX_DURATION:g}-second limit; split it before rendering"
                    )
                if segment.path is not None and segment.path.resolve() == config.output_path.resolve():
                    raise ValueError("Export output must not overwrite source media")
            if plan.music_path is not None and plan.music_path.resolve() == config.output_path.resolve():
                raise ValueError("Export output must not overwrite the music source")
            widths = [source.width for source in sources.values() if source.width]
            heights = [source.height for source in sources.values() if source.height]
            widths.extend(frame.width for frame in (frames or {}).values() if frame.width)
            heights.extend(frame.height for frame in (frames or {}).values() if frame.height)
            config = replace(
                config, width=config.width or max(widths, default=1280),
                height=config.height or max(heights, default=720),
            )
            assert config.width is not None and config.height is not None
            if config.width < 2 or config.height < 2 or config.width % 2 or config.height % 2:
                raise ValueError("Export dimensions must be positive even integers")
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            # All intermediates share the destination filesystem. A failed or
            # cancelled replacement leaves an existing successful export intact.
            with tempfile.TemporaryDirectory(prefix=".sequence_render_", dir=config.output_path.parent) as directory:
                temporary = Path(directory)
                intermediate_config = replace(config, video_codec="libx264", crf=0, audio_codec="pcm_s16le")
                segment_paths = []
                visible = [segment for segment in plan.segments if segment.frame_count]
                for index, segment in enumerate(visible):
                    if cancelled():
                        return False
                    if progress_callback:
                        progress_callback(index / len(visible) * 0.8, f"Processing segment {index + 1}/{len(visible)}")
                    segment_path = temporary / f"segment_{index:06d}.mov"
                    duration = float(segment.frame_count / plan.output_rate)
                    samples = (
                        frame_boundary(segment.end_frame / plan.output_rate, 48000)
                        - frame_boundary(segment.start_frame / plan.output_rate, 48000)
                    )
                    color = (segment.color or (0, 0, 0)) if config.show_chromatic_color_bar else None
                    if segment.kind == "gap":
                        success = self._export_gap_segment(segment_path, duration, intermediate_config, audio_samples=samples)
                    elif segment.kind == "still":
                        assert segment.path is not None
                        success = self._export_frame_segment(
                            segment.path, segment_path, duration, config.fps, intermediate_config,
                            bar_color=color, seq_clip=segment, audio_samples=samples,
                        )
                    else:
                        assert segment.path is not None
                        assert isinstance(segment.media, VideoRange)
                        success = self._export_segment(
                            source_path=segment.path, output_path=segment_path,
                            start_frame=segment.media.start_frame, end_frame=segment.media.end_frame,
                            source_fps=float(segment.media.rate), config=intermediate_config,
                            bar_color=color, seq_clip=segment,
                            media_range=segment.media, frame_count=segment.frame_count, audio_samples=samples,
                            source_origin=verified[segment.path].origin,
                            has_audio=audio_streams[segment.path],
                        )
                    if not success:
                        return False
                    segment_paths.append(segment_path)
                if cancelled():
                    return False
                if progress_callback:
                    progress_callback(0.8, "Concatenating clips...")
                staged = temporary / ("complete" + config.output_path.suffix)
                concat_output = temporary / "without_music.mp4" if plan.music_path else staged
                if not self._concat_segments(segment_paths=segment_paths, output_path=concat_output, config=config):
                    logger.error("Sequence export concat failed")
                    return False
                if plan.music_path:
                    if progress_callback:
                        progress_callback(0.9, "Adding music track...")
                    if not self._mux_audio(concat_output, plan.music_path, staged, config, duration=float(plan.duration)):
                        return False
                if cancelled() or not staged.is_file() or staged.stat().st_size == 0:
                    return False
                self._validate_output(staged, plan.frame_count, plan.output_rate, config)
                plan.validate_media_unchanged()
                if cancelled():
                    return False
                os.replace(staged, config.output_path)
            if progress_callback:
                progress_callback(1.0, "Export complete!")
            return True
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            logger.error("Sequence export failed: %s", exc)
            if progress_callback:
                progress_callback(0, str(exc))
            return False

    def _validate_output(self, path, count, rate, config) -> None:
        from core.media_timing import probe_video_timing

        timing = probe_video_timing(path, config.cancel_check)
        if timing.frame_count != count or timing.variable or timing.rate != rate:
            raise ValueError("Encoded output does not match the render plan's frame count and rate")

    def _run_ffmpeg(self, command: list[str], config: ExportConfig, timeout: int) -> bool:
        """Run an encoder, terminating it before returning cancellation or timeout."""
        if config.cancel_check is None:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **get_subprocess_kwargs())
            if result.returncode:
                logger.error("FFmpeg failed: %s", (result.stderr or "")[-2000:])
            return result.returncode == 0
        if config.cancel_check():
            return False
        deadline = time.monotonic() + timeout
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **get_subprocess_kwargs())
        try:
            while True:
                if config.cancel_check() or time.monotonic() >= deadline:
                    return False
                try:
                    _, stderr = process.communicate(timeout=0.1)
                    if process.returncode:
                        logger.error("FFmpeg failed: %s", (stderr or "")[-2000:])
                    return process.returncode == 0 and not config.cancel_check()
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    # Maximum clip duration (seconds) for the reverse filter.
    # reverse buffers the entire clip in RAM (~900 MB for 5s of 1080p30).
    _REVERSE_MAX_DURATION = 15.0

    def _export_gap_segment(
        self,
        output_path: Path,
        duration_seconds: float,
        config: ExportConfig,
        audio_samples: int | None = None,
    ) -> bool:
        """Export a silent black segment for an empty timeline gap."""
        if duration_seconds <= 0:
            return True

        width = config.width or 1280
        height = config.height or 720
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={config.fps}",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", f"trim=end_frame={frame_boundary(duration_seconds, config.fps)},setpts=PTS-STARTPTS",
            "-af", f"atrim=end_sample={audio_samples if audio_samples is not None else frame_boundary(duration_seconds, 48000)},asetpts=PTS-STARTPTS",
            "-c:v", config.video_codec,
            "-preset", config.preset,
            "-crf", str(config.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
            str(output_path),
        ]

        return self._run_ffmpeg(cmd, config, 300)

    def _export_segment(
        self,
        source_path: Path,
        output_path: Path,
        start_frame: int,
        end_frame: int,
        source_fps: float,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]] = None,
        seq_clip: SequenceClip | RenderSegment | None = None,
        media_range: VideoRange | None = None,
        frame_count: int | None = None,
        audio_samples: int | None = None,
        source_origin: Fraction = Fraction(0),
        has_audio: bool | None = None,
    ) -> bool:
        """Render the exact source frame selection and planned raster length."""
        from models.media_time import VideoRange, frame_boundary, frame_rate
        media = media_range or VideoRange(start_frame, end_frame, frame_rate(source_fps))
        output_rate = frame_rate(config.fps)
        count = frame_count if frame_count is not None else frame_boundary(media.duration, output_rate)
        if count < 1:
            raise ValueError("Segment is shorter than one output frame")
        samples = audio_samples if audio_samples is not None else frame_boundary(count / output_rate, 48000)
        reverse = bool(seq_clip and seq_clip.reverse)
        if reverse and media.duration > self._REVERSE_MAX_DURATION:
            raise ValueError("Reverse segment exceeds the supported duration")
        filters = [f"trim=start_frame={start_frame}:end_frame={end_frame}", "setpts=PTS-STARTPTS"]
        seek = None
        if media.presentation_range is None and media.start > 5:
            # Decode a short preroll, then identify frames from their original
            # PTS. Counting frames after a seek would shift source coordinates.
            seek = float(source_origin + media.start - 5)
            filters = [
                f"trim=end={float(source_origin + media.end)}",
                f"select='between(round((pts*TB-({source_origin}))*{media.rate}),{start_frame},{end_frame - 1})'",
                "setpts=PTS-STARTPTS",
            ]
        if isinstance(seq_clip, RenderSegment):
            from math import lcm

            origin = seq_clip.timeline.start - Fraction(seq_clip.start_frame) / output_rate
            denominator = lcm(media.rate.numerator, output_rate.numerator, origin.denominator)
            if seq_clip.source_boundaries is not None:
                denominator = lcm(denominator, *(time.denominator for time in seq_clip.source_boundaries))
            if denominator > 2_000_000_000:
                raise ValueError("Media time grid exceeds the encoder timebase limit")
            filters.append(f"settb=expr=1/{denominator}")
            if seq_clip.source_boundaries is None:
                filters.append(f"setpts=N*{denominator * media.rate.denominator // media.rate.numerator}+({int(origin * denominator)})")
            else:
                filters.append(f"setpts=PTS-STARTPTS+({int(origin * denominator)})")
        effects = self._build_video_filter(config, bar_color, seq_clip, apply_reverse=False)
        if effects:
            filters.append(effects)
        filters.extend([
            f"fps={output_rate}:start_time=0:eof_action=pass", "tpad=stop_mode=clone:stop=2",
            f"trim=end_frame={count}",
        ])
        if reverse:
            filters.append("reverse")
        audio = [
            f"atrim=start={float(source_origin + media.start)}:end={float(source_origin + media.end)}",
            "asetpts=PTS-STARTPTS", "aresample=48000",
            "aformat=channel_layouts=stereo",
        ]
        if reverse:
            audio.append("areverse")
        audio.extend(["apad", f"atrim=end_sample={samples}"])
        command = [self.ffmpeg_path, "-y", "-copyts"]
        if seek is not None:
            command.extend(["-seek_timestamp", "1", "-noaccurate_seek", "-ss", str(seek)])
        command.extend(["-i", str(source_path)])
        source_has_audio = self._has_audio(source_path) if has_audio is None else has_audio
        if source_has_audio:
            command.extend(["-map", "0:v:0", "-map", "0:a:0"])
        else:
            command.extend([
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-map", "0:v:0", "-map", "1:a:0",
            ])
            audio = [f"atrim=end_sample={samples}", "asetpts=PTS-STARTPTS"]
        command.extend([
            "-vf", ",".join(filters), "-af", ",".join(audio),
            "-c:v", config.video_codec, "-preset", config.preset,
            "-crf", str(config.crf), "-pix_fmt", "yuv420p",
            "-c:a", config.audio_codec, "-b:a", config.audio_bitrate, str(output_path),
        ])
        return self._run_ffmpeg(command, config, 300)

    def _has_audio(self, path: Path) -> bool:
        """Inspect actual streams so every intermediate has one stereo track."""
        ffprobe = find_binary("ffprobe")
        if not ffprobe:
            raise ValueError("FFprobe is required to validate source streams")
        result = subprocess.run([
            ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
            "stream=index", "-of", "json", str(path),
        ], capture_output=True, text=True, timeout=30, **get_subprocess_kwargs())
        if result.returncode:
            raise ValueError(f"Cannot inspect audio streams in {path}")
        return bool(json.loads(result.stdout).get("streams"))

    def _export_frame_segment(
        self,
        frame_path: Path,
        output_path: Path,
        hold_seconds: float,
        fps: float,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]] = None,
        seq_clip: SequenceClip | RenderSegment | None = None,
        audio_samples: int | None = None,
    ) -> bool:
        """Export a still image as a video segment with silent audio.

        Creates a video from a single image, held for the specified duration,
        with a silent audio track for concat compatibility.
        """
        vf = self._build_video_filter(config, bar_color, seq_clip)
        vf_parts = [vf] if vf else []
        vf_parts.extend([f"fps={frame_rate(fps)}", f"trim=end_frame={frame_boundary(hold_seconds, fps)}", "setpts=PTS-STARTPTS"])

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-loop", "1",
            "-i", str(frame_path),
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-af", f"atrim=end_sample={audio_samples if audio_samples is not None else frame_boundary(hold_seconds, 48000)},asetpts=PTS-STARTPTS",
        ]

        if vf_parts:
            cmd.extend(["-vf", ",".join(vf_parts)])

        cmd.extend([
            "-c:v", config.video_codec,
            "-preset", config.preset,
            "-crf", str(config.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
            str(output_path),
        ])

        return self._run_ffmpeg(cmd, config, 300)

    def _build_video_filter(
        self,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]],
        seq_clip: SequenceClip | RenderSegment | None = None,
        apply_reverse: bool = False,
    ) -> Optional[str]:
        """Build ffmpeg video filter chain.

        Filter order: scale+pad -> hflip -> vflip -> reverse -> chromatic_bar
        Scaling preserves aspect ratio and pads with black to fill the target.
        """
        vf_parts = []
        if config.width and config.height:
            vf_parts.append(
                f"scale={config.width}:{config.height}"
                ":force_original_aspect_ratio=decrease"
                f",pad={config.width}:{config.height}"
                ":(ow-iw)/2:(oh-ih)/2:black"
            )
        if seq_clip and seq_clip.hflip:
            vf_parts.append("hflip")
        if seq_clip and seq_clip.vflip:
            vf_parts.append("vflip")
        if apply_reverse:
            vf_parts.append("reverse")
        chromatic_filter = self._chromatic_bar_filter(config=config, bar_color=bar_color)
        if chromatic_filter:
            vf_parts.append(chromatic_filter)
        if not vf_parts:
            return None
        return ",".join(vf_parts)

    def _chromatic_bar_filter(
        self,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]],
    ) -> Optional[str]:
        """Build a drawbox filter that paints the bottom chromatic bar."""
        if bar_color is None:
            return None

        ratio = max(0.001, float(config.chromatic_color_bar_height_ratio))
        min_height = max(1, int(config.chromatic_color_bar_min_height))
        # Escape comma for ffmpeg filtergraph parsing inside drawbox expressions.
        bar_h_expr = f"max({min_height}\\,ih*{ratio:.4f})"
        r, g, b = bar_color
        color_hex = f"0x{r:02x}{g:02x}{b:02x}"
        return (
            f"drawbox=x=0:y=ih-({bar_h_expr})"
            f":w=iw:h={bar_h_expr}:color={color_hex}@1.0:t=fill"
        )

    def _mux_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        config: ExportConfig,
        duration: float | None = None,
    ) -> bool:
        """Replace source audio with music, padding short tracks to video length."""
        audio_filter = "aresample=48000,apad"
        if duration is not None:
            audio_filter += f",atrim=duration={duration}"
        command = [
            self.ffmpeg_path, "-y", "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-af", audio_filter, "-c:a", config.audio_codec, "-b:a", config.audio_bitrate,
        ]
        if duration is not None:
            command.extend(["-t", str(duration)])
        else:
            command.append("-shortest")
        command.append(str(output_path))
        return self._run_ffmpeg(command, config, 600)

    def _concat_segments(
        self,
        segment_paths: list[Path],
        output_path: Path,
        config: ExportConfig,
    ) -> bool:
        """Concatenate segments using FFmpeg concat demuxer."""
        # Create concat list file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for path in segment_paths:
                # Validate path doesn't contain newlines (would break concat format)
                path_str = str(path.resolve())
                if "\n" in path_str or "\r" in path_str:
                    raise ValueError(f"Invalid path with newline characters: {path}")
                # Escape backslashes and single quotes for FFmpeg concat format
                escaped_path = path_str.replace("\\", "\\\\").replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
            concat_file = f.name

        try:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-map", "0:v:0", "-map", "0:a:0",
                "-vf", f"setpts=N/({frame_rate(config.fps)}*TB)",
                "-af", "asetpts=N/SR/TB",
                "-c:v", config.video_codec, "-preset", config.preset,
                "-crf", str(config.crf), "-pix_fmt", "yuv420p",
                "-c:a", config.audio_codec, "-b:a", config.audio_bitrate,
                str(output_path),
            ]

            return self._run_ffmpeg(cmd, config, 600)
        finally:
            Path(concat_file).unlink(missing_ok=True)


def export_sequence(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, tuple],
    output_path: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    frames: Optional[dict[str, "Frame"]] = None,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """
    Convenience function to export a sequence with default settings.

    Args:
        sequence: The Sequence to export
        sources: Dict of source_id -> Source
        clips: Dict of clip_id -> (Clip, Source)
        output_path: Where to save the output video
        progress_callback: Optional callback (progress 0-1, message)
        frames: Optional dict of frame_id -> Frame for frame-based entries

    Returns:
        True if export succeeded
    """
    # Resolve music_path if the sequence has one
    music_path = None
    raw_music = getattr(sequence, "music_path", None)
    if raw_music:
        p = Path(raw_music)
        if p.exists():
            music_path = p

    config = ExportConfig(
        output_path=output_path,
        fps=sequence.fps,
        show_chromatic_color_bar=(
            bool(getattr(sequence, "show_chromatic_color_bar", False))
            and sequence.algorithm == "color"
        ),
        music_path=music_path,
        cancel_check=cancel_check,
    )

    exporter = SequenceExporter()
    return exporter.export(
        sequence=sequence,
        sources=sources,
        clips=clips,
        config=config,
        progress_callback=progress_callback,
        frames=frames,
    )
