"""EDL (Edit Decision List) export for NLE workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from models.clip import Clip, Source
from models.sequence import Sequence, SequenceClip


@dataclass
class EDLExportConfig:
    """Configuration for EDL export."""

    output_path: Path
    title: str = "Scene Ripper Export"
    drop_frame: bool = False


def frames_to_timecode(frames: int, fps: float, drop_frame: bool = False) -> str:
    """Convert frame number to SMPTE timecode string.

    Args:
        frames: Frame number to convert
        fps: Frame rate
        drop_frame: Use drop-frame timecode format

    Returns:
        Timecode string in HH:MM:SS:FF format (or HH:MM:SS;FF for drop-frame)
    """
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    remaining_frames = int(frames % fps)

    separator = ";" if drop_frame else ":"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{remaining_frames:02d}"


def export_edl(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, tuple],
    config: EDLExportConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> bool:
    """Export sequence as CMX 3600 EDL file.

    Args:
        sequence: The sequence to export
        sources: Dict mapping source_id to Source
        clips: Dict mapping clip_id to (Clip, Source)
        config: Export configuration
        progress_callback: Optional (progress_0_to_1, message) callback

    Returns:
        True if export succeeded, False otherwise
    """
    if progress_callback:
        progress_callback(0.1, "Building EDL...")

    lines = []

    # Header
    lines.append(f"TITLE: {config.title}")
    fcm = "DROP FRAME" if config.drop_frame else "NON-DROP FRAME"
    lines.append(f"FCM: {fcm}")
    lines.append("")

    # Get all clips sorted by timeline position
    seq_clips = sequence.get_all_clips()
    total_clips = len(seq_clips)

    if total_clips == 0:
        if progress_callback:
            progress_callback(1.0, "No clips to export")
        return False

    record_frame = 0  # Running record position

    for i, seq_clip in enumerate(seq_clips):
        if progress_callback:
            progress = 0.1 + (0.8 * (i / max(total_clips, 1)))
            progress_callback(progress, f"Processing clip {i + 1}/{total_clips}")

        # Get source for this clip
        source = sources.get(seq_clip.source_id)
        if not source:
            continue

        fps = source.fps

        # Source timecodes (in/out points within source video)
        src_in = frames_to_timecode(seq_clip.in_point, fps, config.drop_frame)
        src_out = frames_to_timecode(seq_clip.out_point, fps, config.drop_frame)

        # Record timecodes (position on timeline)
        duration_frames = seq_clip.out_point - seq_clip.in_point
        rec_in = frames_to_timecode(record_frame, sequence.fps, config.drop_frame)
        rec_out = frames_to_timecode(
            record_frame + duration_frames, sequence.fps, config.drop_frame
        )
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
        lines.append(f"* FROM CLIP NAME: {source.filename}")
        lines.append("")

    if progress_callback:
        progress_callback(0.9, "Writing EDL file...")

    # Write to file
    try:
        output_path = config.output_path
        if not output_path.suffix.lower() == ".edl":
            output_path = output_path.with_suffix(".edl")

        output_path.write_text("\n".join(lines), encoding="utf-8")

        if progress_callback:
            progress_callback(1.0, f"Exported to {output_path.name}")

        return True

    except (OSError, IOError) as e:
        if progress_callback:
            progress_callback(1.0, f"Export failed: {e}")
        return False
