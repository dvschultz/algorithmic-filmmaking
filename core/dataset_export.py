"""Dataset export functionality for exporting clip metadata to JSON."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from models.clip import Source, Clip


@dataclass
class DatasetExportConfig:
    """Configuration for dataset export."""

    output_path: Path
    include_thumbnails: bool = True
    pretty_print: bool = True


def export_dataset(
    source: Source,
    clips: list[Clip],
    config: DatasetExportConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Export clip metadata to a JSON file.

    Args:
        source: The source video
        clips: List of clips to export
        config: Export configuration
        progress_callback: Optional callback for progress updates (0-1, message)

    Returns:
        True if export succeeded, False otherwise
    """
    if progress_callback:
        progress_callback(0.1, "Building dataset...")

    # Build source metadata
    source_data = {
        "file_path": str(source.file_path),
        "filename": source.filename,
        "duration_seconds": source.duration_seconds,
        "fps": source.fps,
        "resolution": {
            "width": source.width,
            "height": source.height,
        },
    }

    # Build clips metadata
    clips_data = []
    total = len(clips)
    for i, clip in enumerate(clips):
        clip_data = {
            "id": clip.id,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "start_time": clip.start_time(source.fps),
            "end_time": clip.end_time(source.fps),
            "duration_seconds": clip.duration_seconds(source.fps),
        }

        # Add colors as RGB objects (convert numpy types to Python int)
        if clip.dominant_colors:
            clip_data["colors"] = [
                {"r": int(r), "g": int(g), "b": int(b)} for r, g, b in clip.dominant_colors
            ]

        # Add shot type
        if clip.shot_type:
            clip_data["shot_type"] = clip.shot_type

        # Add thumbnail path
        if config.include_thumbnails and clip.thumbnail_path:
            clip_data["thumbnail_path"] = str(clip.thumbnail_path)

        clips_data.append(clip_data)

        if progress_callback:
            progress_callback(0.1 + (0.7 * (i + 1) / total), f"Processing clip {i + 1}/{total}")

    # Build final dataset
    dataset = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "source": source_data,
        "clips": clips_data,
    }

    if progress_callback:
        progress_callback(0.9, "Writing JSON file...")

    # Write to file
    try:
        with open(config.output_path, "w", encoding="utf-8") as f:
            if config.pretty_print:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            else:
                json.dump(dataset, f, ensure_ascii=False)

        if progress_callback:
            progress_callback(1.0, "Export complete")

        return True
    except (OSError, IOError) as e:
        if progress_callback:
            progress_callback(0, f"Export failed: {e}")
        return False
