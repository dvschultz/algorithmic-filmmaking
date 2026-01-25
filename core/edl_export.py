"""EDL (Edit Decision List) export for NLE workflows."""

from dataclasses import dataclass
from pathlib import Path

from models.clip import Source
from models.sequence import Sequence


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
) -> bool:
    """Export sequence as CMX 3600 EDL file.

    Args:
        sequence: The sequence to export
        sources: Dict mapping source_id to Source
        config: Export configuration

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
        # Get source for this clip
        source = sources.get(seq_clip.source_id)
        if not source:
            continue

        fps = source.fps

        # Source timecodes (in/out points within source video)
        src_in = frames_to_timecode(seq_clip.in_point, fps)
        src_out = frames_to_timecode(seq_clip.out_point, fps)

        # Record timecodes (position on timeline)
        duration_frames = seq_clip.out_point - seq_clip.in_point
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

        # Source filename comment
        lines.append(f"* FROM CLIP NAME: {_sanitize_edl_string(source.filename)}")
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
