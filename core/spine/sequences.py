"""Sequence management shared by desktop agent and retained headless sessions."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Any

from models.sequence import Sequence
from core.spine.security import validate_path

if TYPE_CHECKING:
    from core.project import Project


def create_sequence(
    project: Project, name: str | None = None, fps: float | None = None
) -> dict:
    name = (name or project.metadata.name or "Untitled Sequence").strip()
    fps = (
        fps
        if fps is not None
        else (project.sources[0].fps if project.sources else 30.0)
    )
    if (
        not name
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not isfinite(fps)
        or fps <= 0
    ):
        return {
            "success": False,
            "error": "Provide a nonempty name and a finite positive fps",
        }
    sequence = Sequence(name=name, fps=fps)
    try:
        project.add_sequence(sequence, activate=True)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "message": f"Created sequence '{name}' at {fps} fps (now active)",
        "name": name,
        "fps": fps,
        "sequence_index": project.active_sequence_index,
        "sequence_id": sequence.id,
    }


def update_sequence(
    project: Project, sequence_id: str | None = None, **changes: Any
) -> dict:
    sequence = (
        project.sequence
        if sequence_id is None
        else next((s for s in project.sequences if s.id == sequence_id), None)
    )
    if sequence is None:
        return {"success": False, "error": "Sequence not found"}
    if "music_path" in changes and changes["music_path"] is not None:
        valid, error, path = validate_path(changes["music_path"], must_exist=True)
        if not valid:
            return {"success": False, "error": error}
        changes["music_path"] = str(path)
    try:
        project.update_sequence_metadata(sequence, **changes)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "message": f"Updated sequence: {', '.join(changes)}",
        "updated_fields": changes,
    }


def delete_sequence(project: Project, sequence_id: str) -> dict:
    index = next(
        (i for i, s in enumerate(project.sequences) if s.id == sequence_id), None
    )
    if index is None:
        return {"success": False, "error": "Sequence not found"}
    try:
        project.remove_sequence(index)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "deleted": sequence_id,
        "active_sequence_id": project.sequences[project.active_sequence_index].id,
    }


def list_sequences(project: Project) -> dict:
    return {
        "success": True,
        "sequences": [
            {
                "id": s.id,
                "name": s.name,
                "active": i == project.active_sequence_index,
                "clip_count": len(s.get_all_clips()),
            }
            for i, s in enumerate(project.sequences)
        ],
    }
