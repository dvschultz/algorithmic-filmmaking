"""Export project as a self-contained bundle.

Creates a folder containing the project file and all referenced assets
(source videos and frame images), with paths rewritten for portability.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.project import save_project, Project

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Summary of a completed bundle export."""

    dest_dir: Path = field(default_factory=Path)
    sources_copied: int = 0
    frames_copied: int = 0
    sources_skipped: list[str] = field(default_factory=list)
    frames_skipped: list[str] = field(default_factory=list)
    total_bytes: int = 0
    include_videos: bool = True


def _build_filename_map(
    paths: list[Path],
    subdirectory: str,
) -> dict[Path, str]:
    """Build a collision-safe mapping from absolute paths to bundle-relative paths.

    When multiple files share the same filename, a numeric suffix is appended:
    video.mp4, video_2.mp4, video_3.mp4, etc.

    Args:
        paths: Absolute paths of source files.
        subdirectory: Bundle subdirectory name (e.g. "sources", "frames").

    Returns:
        Mapping from absolute source path to bundle-relative posix path
        (e.g. "sources/video.mp4").
    """
    result: dict[Path, str] = {}
    # Track used names (case-insensitive for macOS/Windows compatibility)
    used_names: dict[str, int] = {}

    for path in paths:
        stem = path.stem
        suffix = path.suffix
        name_lower = path.name.lower()

        if name_lower not in used_names:
            # First occurrence — use original name
            used_names[name_lower] = 1
            bundle_name = path.name
        else:
            # Collision — increment counter and add suffix
            used_names[name_lower] += 1
            count = used_names[name_lower]
            bundle_name = f"{stem}_{count}{suffix}"

        result[path] = f"{subdirectory}/{bundle_name}"

    return result


def _strip_absolute_paths(project_json_path: Path) -> None:
    """Remove _absolute_path fields from an exported project JSON file.

    This ensures the bundle is truly portable — without these fields,
    the loader won't fall back to original machine paths.
    """
    with open(project_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _remove_absolute_paths(obj):
        if isinstance(obj, dict):
            obj.pop("_absolute_path", None)
            for value in obj.values():
                _remove_absolute_paths(value)
        elif isinstance(obj, list):
            for item in obj:
                _remove_absolute_paths(item)

    _remove_absolute_paths(data)

    with open(project_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_project_bundle(
    project: Project,
    dest_dir: Path,
    include_videos: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ExportResult:
    """Export a project as a self-contained bundle folder.

    Args:
        project: The project to export (uses in-memory state).
        dest_dir: Destination directory for the bundle. Must not exist
            (caller handles overwrite confirmation).
        include_videos: If True, copy source video files into the bundle.
        progress_callback: Optional callback(current, total, filename) for
            progress reporting.
        cancel_check: Optional callable returning True if the user requested
            cancellation.

    Returns:
        ExportResult with statistics about the export.

    Raises:
        FileExistsError: If dest_dir already exists.
        OSError: If directory creation or file copy fails.
    """
    if dest_dir.exists():
        raise FileExistsError(f"Destination already exists: {dest_dir}")

    result = ExportResult(dest_dir=dest_dir, include_videos=include_videos)

    # Create bundle directory structure
    sources_dir = dest_dir / "sources"
    frames_dir = dest_dir / "frames"
    dest_dir.mkdir(parents=True)
    sources_dir.mkdir()
    frames_dir.mkdir()

    try:
        # Build filename maps for collision resolution
        source_paths = [s.file_path for s in project.sources]
        frame_paths = [f.file_path for f in project.frames]

        source_name_map = _build_filename_map(source_paths, "sources")
        frame_name_map = _build_filename_map(frame_paths, "frames")

        # Count total files to copy for progress
        total_files = len(project.frames)
        if include_videos:
            total_files += len(project.sources)
        current_file = 0

        # Copy frame files
        for frame in project.frames:
            if cancel_check and cancel_check():
                _cleanup_partial_bundle(dest_dir)
                return result

            bundle_rel = frame_name_map.get(frame.file_path)
            if bundle_rel is None:
                continue

            dest_path = dest_dir / bundle_rel
            if frame.file_path.exists():
                shutil.copy2(frame.file_path, dest_path)
                result.frames_copied += 1
                result.total_bytes += frame.file_path.stat().st_size
            else:
                result.frames_skipped.append(str(frame.file_path))
                logger.warning(f"Frame file missing, skipped: {frame.file_path}")

            current_file += 1
            if progress_callback:
                progress_callback(current_file, total_files, frame.file_path.name)

        # Copy source video files (if requested)
        if include_videos:
            for source in project.sources:
                if cancel_check and cancel_check():
                    _cleanup_partial_bundle(dest_dir)
                    return result

                bundle_rel = source_name_map.get(source.file_path)
                if bundle_rel is None:
                    continue

                dest_path = dest_dir / bundle_rel
                if source.file_path.exists():
                    shutil.copy2(source.file_path, dest_path)
                    result.sources_copied += 1
                    result.total_bytes += source.file_path.stat().st_size
                else:
                    result.sources_skipped.append(str(source.file_path))
                    logger.warning(
                        f"Source video missing, skipped: {source.file_path}"
                    )

                current_file += 1
                if progress_callback:
                    progress_callback(
                        current_file, total_files, source.file_path.name
                    )

        # Build rewritten Source objects pointing into bundle subdirectories
        rewritten_sources = []
        for source in project.sources:
            bundle_rel = source_name_map.get(source.file_path, f"sources/{source.filename}")
            # Create a shallow copy with rewritten file_path
            from dataclasses import replace
            rewritten = replace(source, file_path=Path(bundle_rel))
            rewritten_sources.append(rewritten)

        # Build rewritten Frame objects
        rewritten_frames = []
        for frame in project.frames:
            bundle_rel = frame_name_map.get(frame.file_path, f"frames/{frame.file_path.name}")
            from dataclasses import replace
            rewritten = replace(frame, file_path=Path(bundle_rel))
            rewritten_frames.append(rewritten)

        # Save the project file into the bundle
        project_name = project.metadata.name or "Untitled Project"
        project_filename = f"{project_name}.sceneripper"
        project_path = dest_dir / project_filename

        # save_project uses base_path = filepath.parent for relative paths.
        # Since our rewritten sources already have paths like "sources/video.mp4",
        # we pass base_path=None via to_dict to avoid double-relativizing.
        # We write the JSON directly instead.
        _write_bundle_project_file(
            project_path,
            project,
            rewritten_sources,
            rewritten_frames,
        )

        # Strip _absolute_path fields from the exported JSON
        _strip_absolute_paths(project_path)

        if progress_callback:
            progress_callback(total_files, total_files, "Done")

    except Exception:
        # On error, leave partial bundle on disk (per spec) but re-raise
        raise

    return result


def _write_bundle_project_file(
    filepath: Path,
    project: Project,
    rewritten_sources,
    rewritten_frames,
) -> None:
    """Write the project JSON with rewritten paths.

    We call save_project() with the rewritten objects. The save_project
    function uses filepath.parent as base_path, so our rewritten paths
    (already relative like "sources/video.mp4") get stored correctly.
    """
    save_project(
        filepath=filepath,
        sources=rewritten_sources,
        clips=project.clips,
        sequence=project.sequence,
        ui_state=project.ui_state,
        metadata=project.metadata,
        frames=rewritten_frames,
    )


def _cleanup_partial_bundle(dest_dir: Path) -> None:
    """Remove a partially-created bundle directory on cancellation."""
    try:
        shutil.rmtree(dest_dir)
        logger.info(f"Cleaned up partial bundle: {dest_dir}")
    except OSError as e:
        logger.warning(f"Failed to clean up partial bundle {dest_dir}: {e}")
