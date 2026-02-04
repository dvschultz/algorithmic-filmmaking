"""SRT (SubRip) export for sequence metadata.

Exports sequence clips as SRT subtitle files with metadata specific to
each sequence algorithm type (descriptions, colors, shot types, OCR text).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from models.clip import Clip, Source
from models.sequence import Sequence


@dataclass
class SRTExportConfig:
    """Configuration for SRT export."""

    output_path: Path


def _format_colors_hex(colors: Optional[list[tuple[int, int, int]]]) -> Optional[str]:
    """Format RGB tuples as hex color codes.

    Args:
        colors: List of (R, G, B) tuples

    Returns:
        Comma-separated hex codes like "#FF5733, #C70039", or None if empty
    """
    if not colors:
        return None
    return ", ".join(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timecode format HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds

    Returns:
        SRT timecode string (e.g., "00:01:30,500")
    """
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _sanitize_srt_text(text: str) -> str:
    """Sanitize text for SRT format.

    Removes/replaces characters that could break SRT parsing.

    Args:
        text: Raw text to sanitize

    Returns:
        Sanitized text safe for SRT format
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple newlines to single (blank lines break SRT entries)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text.strip()


# Metadata extraction functions by algorithm
ALGORITHM_METADATA_EXTRACTORS = {
    "storyteller": lambda clip: clip.description,
    "exquisite_corpus": lambda clip: clip.combined_text,
    "color": lambda clip: _format_colors_hex(clip.dominant_colors),
    "shot_type": lambda clip: clip.shot_type,
    "duration": lambda clip: clip.description,
    "shuffle": lambda clip: clip.description,
    "sequential": lambda clip: clip.description,
}


def export_srt(
    sequence: Sequence,
    clips_by_id: dict[str, Clip],
    sources_by_id: dict[str, Source],
    config: SRTExportConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[bool, int, int]:
    """Export sequence metadata as SRT subtitle file.

    Args:
        sequence: The sequence to export
        clips_by_id: Dict mapping clip_id to Clip
        sources_by_id: Dict mapping source_id to Source
        config: Export configuration
        progress_callback: Optional callback (progress 0-1, message)

    Returns:
        Tuple of (success, exported_count, skipped_count)
    """
    seq_clips = sequence.get_all_clips()
    if not seq_clips:
        return False, 0, 0

    # Determine metadata extractor based on algorithm
    algorithm = sequence.algorithm or "description_fallback"
    extractor = ALGORITHM_METADATA_EXTRACTORS.get(
        algorithm,
        lambda clip: clip.description  # Default fallback
    )

    entries = []
    entry_num = 1
    skipped = 0
    total = len(seq_clips)

    for i, seq_clip in enumerate(seq_clips):
        if progress_callback:
            progress_callback(i / total, f"Processing clip {i + 1}/{total}")

        # Get source clip for metadata
        clip = clips_by_id.get(seq_clip.source_clip_id)
        if not clip:
            skipped += 1
            continue

        # Extract metadata
        text = extractor(clip)
        if not text:
            skipped += 1
            continue

        # Calculate timecodes using sequence fps
        start_time = seq_clip.start_frame / sequence.fps
        duration = seq_clip.duration_frames / sequence.fps
        end_time = start_time + duration

        start_tc = _seconds_to_srt_time(start_time)
        end_tc = _seconds_to_srt_time(end_time)

        # Build SRT entry
        sanitized_text = _sanitize_srt_text(text)
        entry = f"{entry_num}\n{start_tc} --> {end_tc}\n{sanitized_text}\n"
        entries.append(entry)
        entry_num += 1

    # Write to file
    try:
        output_path = config.output_path
        if not output_path.suffix.lower() == ".srt":
            output_path = output_path.with_suffix(".srt")

        content = "\n".join(entries)
        output_path.write_text(content, encoding="utf-8")

        if progress_callback:
            progress_callback(1.0, "Export complete")

        return True, entry_num - 1, skipped

    except (OSError, IOError):
        return False, 0, 0
