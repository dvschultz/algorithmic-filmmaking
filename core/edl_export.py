"""EDL (Edit Decision List) export for NLE workflows."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from models.clip import Source
from models.sequence import Sequence

if TYPE_CHECKING:
    from models.frame import Frame

logger = logging.getLogger(__name__)


@dataclass
class EDLExportConfig:
    """Configuration for EDL export."""

    output_path: Path
    title: str = "Scene Ripper Export"


def _sanitize_edl_string(value: str) -> str:
    """Remove format-breaking characters from EDL field values."""
    return value.replace("\n", " ").replace("\r", " ")[:255]


def frames_to_timecode(frames: int, fps: float) -> str:
    """Convert frame number to SMPTE timecode string.

    Args:
        frames: Frame number to convert
        fps: Frame rate

    Returns:
        Timecode string in HH:MM:SS:FF format (non-drop-frame)
    """
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    remaining_frames = int(frames % fps)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{remaining_frames:02d}"


def export_edl(
    sequence: Sequence,
    sources: dict[str, Source],
    config: EDLExportConfig,
    frames: Optional[dict[str, "Frame"]] = None,
) -> bool:
    """Export sequence as CMX 3600 EDL file.

    Args:
        sequence: The sequence to export
        sources: Dict mapping source_id to Source
        config: Export configuration
        frames: Optional dict of frame_id -> Frame for frame-based entries

    Returns:
        True if export succeeded, False otherwise
    """

    lines = []

    # Header
    lines.append(f"TITLE: {_sanitize_edl_string(config.title)}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    # Get all clips sorted by timeline position
    seq_clips = sequence.get_all_clips()

    if not seq_clips:
        return False

    record_frame = 0  # Running record position

    for i, seq_clip in enumerate(seq_clips):
        fps = sequence.fps

        if seq_clip.is_frame_entry:
            # Frame-based entry
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

            duration_frames = seq_clip.hold_frames
            # Source timecodes: still image, so 00:00:00:00 to duration
            src_in = frames_to_timecode(0, fps)
            src_out = frames_to_timecode(duration_frames, fps)
            clip_name = _sanitize_edl_string(frame.display_name())
        else:
            # Clip-based entry
            source = sources.get(seq_clip.source_id)
            if not source:
                continue
            fps = source.fps

            duration_frames = seq_clip.out_point - seq_clip.in_point
            src_in = frames_to_timecode(seq_clip.in_point, fps)
            src_out = frames_to_timecode(seq_clip.out_point, fps)
            clip_name = _sanitize_edl_string(source.filename)

        # Record timecodes (position on timeline)
        rec_in = frames_to_timecode(record_frame, sequence.fps)
        rec_out = frames_to_timecode(record_frame + duration_frames, sequence.fps)
        record_frame += duration_frames

        # Edit number and reel
        edit_num = f"{i + 1:03d}"
        reel = f"{i + 1:03d}"

        # EDL event line
        # Format: EDIT# REEL TRACK TRANS SRC_IN SRC_OUT REC_IN REC_OUT
        event_line = (
            f"{edit_num}  {reel}      V     C        "
            f"{src_in} {src_out} {rec_in} {rec_out}"
        )
        lines.append(event_line)

        # Source/frame filename comment
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append("")

    # Write to file
    try:
        output_path = config.output_path
        if not output_path.suffix.lower() == ".edl":
            output_path = output_path.with_suffix(".edl")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return True

    except (OSError, IOError):
        return False
