"""Export timeline sequences to video files."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING
from dataclasses import dataclass

from models.sequence import Sequence, SequenceClip
from models.clip import Source
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


class SequenceExporter:
    """Exports timeline sequences to video files using FFmpeg."""

    def __init__(self, ffmpeg_path: str = None):
        self.ffmpeg_path = ffmpeg_path or find_binary("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

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
        all_clips = sequence.get_all_clips()
        if not all_clips:
            return False

        # Auto-detect resolution from sources if not explicitly set.
        # Uses max width and max height across all sources (may come from
        # different clips) to avoid stretching any content.
        if config.width is None or config.height is None:
            max_w, max_h = 0, 0
            seen_sources: set[str] = set()
            for seq_clip in all_clips:
                source_id = getattr(seq_clip, "source_id", None)
                if source_id and source_id not in seen_sources:
                    seen_sources.add(source_id)
                    source = sources.get(source_id)
                    if source and source.width and source.height:
                        max_w = max(max_w, source.width)
                        max_h = max(max_h, source.height)
            if max_w > 0 and max_h > 0:
                config.width = max_w
                config.height = max_h
                logger.info("Auto-detected export resolution: %dx%d", max_w, max_h)

        logger.info(
            "Starting sequence export: clips=%d output=%s fps=%.3f resolution=%sx%s music=%s",
            len(all_clips),
            config.output_path,
            config.fps,
            config.width if config.width is not None else "source",
            config.height if config.height is not None else "source",
            config.music_path,
        )

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export each clip segment
            segment_paths = []
            total = len(all_clips)

            for i, seq_clip in enumerate(all_clips):
                if progress_callback:
                    progress_callback(i / total * 0.8, f"Processing clip {i+1}/{total}")

                segment_path = temp_path / f"segment_{i:04d}.mp4"
                bar_color = None
                if config.show_chromatic_color_bar:
                    bar_color = self._resolve_sequence_clip_color(
                        seq_clip=seq_clip,
                        clips=clips,
                        frames=frames,
                    ) or (0, 0, 0)

                if seq_clip.is_frame_entry:
                    # Frame-based entry: generate video from still image
                    if not frames:
                        logger.warning(
                            "Frame entry %s skipped: no frames dict provided",
                            seq_clip.id,
                        )
                        continue
                    frame = frames.get(seq_clip.frame_id)
                    if not frame:
                        logger.warning(
                            "Frame entry %s skipped: frame_id %s not found",
                            seq_clip.id,
                            seq_clip.frame_id,
                        )
                        continue
                    success = self._export_frame_segment(
                        frame_path=frame.file_path,
                        output_path=segment_path,
                        hold_seconds=seq_clip.hold_frames / config.fps,
                        fps=config.fps,
                        config=config,
                        bar_color=bar_color,
                    )
                else:
                    # Clip-based entry: extract from source video
                    clip_data = clips.get(seq_clip.source_clip_id)
                    if not clip_data:
                        continue

                    orig_clip, source = clip_data

                    # Use pre-rendered clip if available (transforms already baked in)
                    prerendered = getattr(seq_clip, "prerendered_path", None)
                    if prerendered and Path(prerendered).exists():
                        success = self._export_prerendered_segment(
                            prerendered_path=Path(prerendered),
                            output_path=segment_path,
                            config=config,
                            bar_color=bar_color,
                        )
                    else:
                        success = self._export_segment(
                            source_path=source.file_path,
                            output_path=segment_path,
                            start_frame=seq_clip.in_point,
                            end_frame=seq_clip.out_point,
                            source_fps=source.fps,
                            config=config,
                            bar_color=bar_color,
                            seq_clip=seq_clip,
                        )

                if success:
                    segment_paths.append(segment_path)
                else:
                    logger.error(
                        "Sequence export aborted: segment %d failed (seq_clip=%s, frame_entry=%s)",
                        i,
                        getattr(seq_clip, "id", None),
                        getattr(seq_clip, "is_frame_entry", False),
                    )
                    return False

            if not segment_paths:
                logger.error("Sequence export aborted: no segments were produced")
                return False

            # Concatenate all segments
            if progress_callback:
                progress_callback(0.8, "Concatenating clips...")

            # If music needs muxing, concat to a temp file first
            if config.music_path and config.music_path.exists():
                concat_output = temp_path / "concat_output.mp4"
            else:
                concat_output = config.output_path
                if config.music_path and not config.music_path.exists():
                    logger.warning(
                        "Music file not found, exporting without audio: %s",
                        config.music_path,
                    )

            success = self._concat_segments(
                segment_paths=segment_paths,
                output_path=concat_output,
                config=config,
            )
            if not success:
                logger.error(
                    "Sequence export concat failed: output=%s segment_count=%d",
                    concat_output,
                    len(segment_paths),
                )

            # Mux music audio onto the concatenated video
            if success and concat_output != config.output_path:
                if progress_callback:
                    progress_callback(0.9, "Adding music track...")
                success = self._mux_audio(
                    video_path=concat_output,
                    audio_path=config.music_path,
                    output_path=config.output_path,
                    config=config,
                )
                if not success:
                    logger.warning(
                        "Sequence export completed without requested music track: output=%s music=%s",
                        config.output_path,
                        config.music_path,
                    )

            if progress_callback:
                progress_callback(1.0, "Export complete!")

            if success:
                logger.info("Sequence export finished: output=%s", config.output_path)

            return success

    # Maximum clip duration (seconds) for the reverse filter.
    # reverse buffers the entire clip in RAM (~900 MB for 5s of 1080p30).
    _REVERSE_MAX_DURATION = 15.0

    def _export_segment(
        self,
        source_path: Path,
        output_path: Path,
        start_frame: int,
        end_frame: int,
        source_fps: float,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]] = None,
        seq_clip: Optional[SequenceClip] = None,
    ) -> bool:
        """Export a single segment from a source video."""
        start_seconds = start_frame / source_fps
        frame_duration = 1.0 / source_fps
        duration_seconds = (end_frame - start_frame) / source_fps - frame_duration

        # Check reverse safety limit
        apply_reverse = False
        if seq_clip and seq_clip.reverse:
            if duration_seconds <= self._REVERSE_MAX_DURATION:
                apply_reverse = True
            else:
                logger.warning(
                    "Skipping reverse for clip %s (%.1fs > %.0fs limit)",
                    seq_clip.id, duration_seconds, self._REVERSE_MAX_DURATION,
                )

        vf = self._build_video_filter(
            config=config, bar_color=bar_color, seq_clip=seq_clip,
            apply_reverse=apply_reverse,
        )
        normalized_vf = f"{vf},fps={config.fps}" if vf else f"fps={config.fps}"

        # Audio cutting is handled by the output-side -ss + -t below, which
        # is sample-accurate. Do NOT add atrim=0:duration here: the precise
        # -ss after -i shifts audio PTS forward, so atrim=0:N keeps a window
        # that no longer contains any samples — silently producing a
        # zero-byte audio track in every segment.
        af_parts = []
        if apply_reverse:
            af_parts.append("areverse")
        af = ",".join(af_parts) if af_parts else None

        # Use "double -ss" for frame-accurate seeking:
        # 1. -ss before -i: fast keyframe seek to ~5s before target (avoids
        #    decoding the entire file from the start)
        # 2. -ss after -i: precise frame-accurate seek from that keyframe
        # This prevents frozen-frame clips where keyframe seeking lands on a
        # single frame and the fps filter repeats it for the entire duration.
        coarse_seek = max(0, start_seconds - 5.0)
        precise_seek = start_seconds - coarse_seek

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(coarse_seek),
            "-i", str(source_path),
            "-ss", str(precise_seek),
            "-t", str(duration_seconds),
            "-c:v", config.video_codec,
            "-preset", config.preset,
            "-crf", str(config.crf),
            "-pix_fmt", "yuv420p",
        ]

        cmd.extend(["-vf", normalized_vf])
        if af:
            cmd.extend(["-af", af])

        cmd.extend([
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
            str(output_path),
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                **get_subprocess_kwargs())
        if result.returncode != 0:
            logger.error(
                "Sequence segment export failed: source=%s output=%s start_frame=%s end_frame=%s stderr=%s",
                source_path,
                output_path,
                start_frame,
                end_frame,
                (result.stderr or "").strip()[-1000:],
            )
        return result.returncode == 0

    def _export_prerendered_segment(
        self,
        prerendered_path: Path,
        output_path: Path,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]] = None,
    ) -> bool:
        """Export a pre-rendered clip segment.

        Transforms are already baked in. Always re-encodes with fps normalization
        to ensure consistent stream parameters across all segments for concat.
        """
        vf_parts = []
        if config.width and config.height:
            vf_parts.append(
                f"scale={config.width}:{config.height}"
                ":force_original_aspect_ratio=decrease"
                f",pad={config.width}:{config.height}"
                ":(ow-iw)/2:(oh-ih)/2:black"
            )
        chromatic_filter = self._chromatic_bar_filter(config=config, bar_color=bar_color)
        if chromatic_filter:
            vf_parts.append(chromatic_filter)
        # Always normalize FPS so concat -c copy works with consistent streams
        vf_parts.append(f"fps={config.fps}")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(prerendered_path),
            "-c:v", config.video_codec,
            "-preset", config.preset,
            "-crf", str(config.crf),
            "-pix_fmt", "yuv420p",
            "-vf", ",".join(vf_parts),
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
        ]

        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                **get_subprocess_kwargs())
        if result.returncode != 0:
            logger.error(
                "Prerendered segment export failed: input=%s output=%s stderr=%s",
                prerendered_path,
                output_path,
                (result.stderr or "").strip()[-1000:],
            )
        return result.returncode == 0

    def _export_frame_segment(
        self,
        frame_path: Path,
        output_path: Path,
        hold_seconds: float,
        fps: float,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]] = None,
    ) -> bool:
        """Export a still image as a video segment with silent audio.

        Creates a video from a single image, held for the specified duration,
        with a silent audio track for concat compatibility.
        """
        vf_parts = []
        if config.width and config.height:
            vf_parts.append(
                f"scale={config.width}:{config.height}"
                ":force_original_aspect_ratio=decrease"
                f",pad={config.width}:{config.height}"
                ":(ow-iw)/2:(oh-ih)/2:black"
            )
        chromatic_filter = self._chromatic_bar_filter(config=config, bar_color=bar_color)
        if chromatic_filter:
            vf_parts.append(chromatic_filter)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-loop", "1",
            "-i", str(frame_path),
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", str(hold_seconds),
            "-r", str(fps),
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
            "-shortest",
            str(output_path),
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                **get_subprocess_kwargs())
        if result.returncode != 0:
            logger.error(
                "Frame segment export failed: frame=%s output=%s hold=%.3fs stderr=%s",
                frame_path,
                output_path,
                hold_seconds,
                (result.stderr or "").strip()[-1000:],
            )
        return result.returncode == 0

    def _resolve_sequence_clip_color(
        self,
        seq_clip: SequenceClip,
        clips: dict[str, tuple],
        frames: Optional[dict[str, "Frame"]] = None,
    ) -> Optional[tuple[int, int, int]]:
        """Resolve the dominant color used for a sequence entry."""
        dominant_colors = None
        if seq_clip.is_frame_entry:
            frame = frames.get(seq_clip.frame_id) if frames and seq_clip.frame_id else None
            dominant_colors = getattr(frame, "dominant_colors", None) if frame else None
        else:
            clip_data = clips.get(seq_clip.source_clip_id)
            if clip_data:
                source_clip, _ = clip_data
                dominant_colors = source_clip.dominant_colors

        if not dominant_colors:
            return None

        color = dominant_colors[0]
        if len(color) < 3:
            return None
        r, g, b = color[0], color[1], color[2]
        return (
            max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
        )

    def _build_video_filter(
        self,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]],
        seq_clip: Optional[SequenceClip] = None,
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
    ) -> bool:
        """Mux a music audio track onto a video file.

        Uses -c:v copy to avoid re-encoding video. Audio is encoded
        with the configured codec. Uses -shortest to trim if audio
        is longer than video.
        """
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
            "-shortest",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                **get_subprocess_kwargs(),
            )
            if result.returncode != 0:
                logger.error("Audio mux failed: %s", result.stderr[-500:] if result.stderr else "")
                shutil.copy2(video_path, output_path)
                logger.warning("Exported without music due to mux failure")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("Audio mux timed out")
            shutil.copy2(video_path, output_path)
            return False
        except Exception as e:
            logger.error("Audio mux error: %s", e)
            shutil.copy2(video_path, output_path)
            return False

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
                "-c", "copy",  # Stream copy (fast, no re-encode)
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                    **get_subprocess_kwargs())
            if result.returncode != 0:
                logger.error(
                    "Sequence concat failed: output=%s segment_count=%d stderr=%s",
                    output_path,
                    len(segment_paths),
                    (result.stderr or "").strip()[-1000:],
                )
            return result.returncode == 0
        finally:
            Path(concat_file).unlink(missing_ok=True)


def export_sequence(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, tuple],
    output_path: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    frames: Optional[dict[str, "Frame"]] = None,
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
