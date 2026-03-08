"""Export timeline sequences to video files."""

import logging
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
                    success = self._export_segment(
                        source_path=source.file_path,
                        output_path=segment_path,
                        start_frame=seq_clip.in_point,
                        end_frame=seq_clip.out_point,
                        fps=config.fps,
                        config=config,
                        bar_color=bar_color,
                        seq_clip=seq_clip,
                    )

                if success:
                    segment_paths.append(segment_path)

            if not segment_paths:
                return False

            # Concatenate all segments
            if progress_callback:
                progress_callback(0.8, "Concatenating clips...")

            success = self._concat_segments(
                segment_paths=segment_paths,
                output_path=config.output_path,
                config=config,
            )

            if progress_callback:
                progress_callback(1.0, "Export complete!")

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
        fps: float,
        config: ExportConfig,
        bar_color: Optional[tuple[int, int, int]] = None,
        seq_clip: Optional[SequenceClip] = None,
    ) -> bool:
        """Export a single segment from a source video."""
        start_seconds = start_frame / fps
        duration_seconds = (end_frame - start_frame) / fps

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

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(start_seconds),
            "-i", str(source_path),
            "-t", str(duration_seconds),
            "-c:v", config.video_codec,
            "-preset", config.preset,
            "-crf", str(config.crf),
        ]

        if vf:
            cmd.extend(["-vf", vf])

        # Audio: apply areverse if reversing video
        if apply_reverse:
            cmd.extend(["-af", "areverse"])

        cmd.extend([
            "-c:a", config.audio_codec,
            "-b:a", config.audio_bitrate,
            str(output_path),
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                **get_subprocess_kwargs())
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
                ":(ow-iw)/2:(oh-ih)/2"
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

        Filter order: scale -> hflip -> vflip -> reverse -> chromatic_bar
        """
        vf_parts = []
        if config.width and config.height:
            vf_parts.append(f"scale={config.width}:{config.height}")
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
    config = ExportConfig(
        output_path=output_path,
        fps=sequence.fps,
        show_chromatic_color_bar=(
            bool(getattr(sequence, "show_chromatic_color_bar", False))
            and sequence.algorithm == "color"
        ),
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
