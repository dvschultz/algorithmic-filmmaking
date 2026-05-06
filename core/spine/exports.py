"""Export-related spine impls (EDL + dataset JSON)."""

from __future__ import annotations

from typing import Optional

from core.settings import load_settings
from core.spine.security import validate_path


def export_edl(project, output_path: Optional[str] = None) -> dict:
    """Export the current sequence as an EDL file."""
    from core.edl_export import EDLExportConfig, export_edl as do_export

    if project.sequence is None or not project.sequence.get_all_clips():
        return {
            "success": False,
            "error": "No sequence to export. Add clips to the sequence first.",
        }

    if output_path:
        valid, error, validated_path = validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        edl_path = validated_path
    else:
        settings = load_settings()
        project_name = project.metadata.name or "untitled"
        edl_path = settings.export_dir / f"{project_name}.edl"

    edl_path.parent.mkdir(parents=True, exist_ok=True)

    config = EDLExportConfig(
        output_path=edl_path,
        title=project.metadata.name or "Scene Ripper Export",
    )

    success = do_export(
        sequence=project.sequence,
        sources=project.sources_by_id,
        config=config,
    )

    if success:
        return {
            "success": True,
            "output_path": str(edl_path),
            "clip_count": len(project.sequence.get_all_clips()),
            "message": (
                f"Exported {len(project.sequence.get_all_clips())} clips to EDL"
            ),
        }
    return {"success": False, "error": "Failed to write EDL file"}


def export_dataset(
    project,
    output_path: Optional[str] = None,
    include_thumbnails: bool = True,
    source_id: Optional[str] = None,
) -> dict:
    """Export clip metadata as a JSON dataset."""
    from core.dataset_export import (
        DatasetExportConfig,
        export_dataset as do_export,
    )

    if source_id:
        source = project.sources_by_id.get(source_id)
        if source is None:
            return {"success": False, "error": f"Source '{source_id}' not found"}
        clips = [c for c in project.clips if c.source_id == source_id]
        source_name = source.path.stem
    else:
        clips = project.clips
        source = project.sources[0] if project.sources else None
        source_name = "all_clips"

    if not clips:
        return {"success": False, "error": "No clips to export"}

    if source is None:
        return {"success": False, "error": "No source video found"}

    if output_path:
        valid, error, validated_path = validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        json_path = validated_path
    else:
        settings = load_settings()
        json_path = settings.export_dir / f"{source_name}_dataset.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)

    config = DatasetExportConfig(
        output_path=json_path,
        include_thumbnails=include_thumbnails,
        pretty_print=True,
    )

    success = do_export(source=source, clips=clips, config=config)

    if success:
        return {
            "success": True,
            "output_path": str(json_path),
            "clip_count": len(clips),
            "message": f"Exported {len(clips)} clips to JSON dataset",
        }
    return {"success": False, "error": "Failed to write dataset file"}


__all__ = ["export_dataset", "export_edl"]
