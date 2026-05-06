"""Frame-mutation spine impls."""

from __future__ import annotations

from typing import Optional

from core.constants import VALID_SHOT_TYPES


def update_frame(
    project,
    frame_id: str,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
    shot_type: Optional[str] = None,
) -> dict:
    """Update a frame's tags, notes, or shot_type."""
    frame = project.frames_by_id.get(frame_id)
    if frame is None:
        return {"success": False, "error": f"Frame not found: {frame_id}"}

    updated_fields: list[str] = []
    kwargs: dict = {}

    if shot_type is not None:
        if shot_type == "":
            kwargs["shot_type"] = None
            updated_fields.append("shot_type")
        elif shot_type in VALID_SHOT_TYPES:
            kwargs["shot_type"] = shot_type
            updated_fields.append("shot_type")
        else:
            return {
                "success": False,
                "error": (
                    f"Invalid shot type: '{shot_type}'. Must be one of: "
                    f"{', '.join(sorted(VALID_SHOT_TYPES))} or empty string to clear."
                ),
            }

    if kwargs:
        project.update_frame(frame_id, **kwargs)

    if tags is not None:
        frame.tags = list(tags)
        updated_fields.append("tags")

    if notes is not None:
        frame.notes = notes
        updated_fields.append("notes")

    if tags is not None or notes is not None:
        project._dirty = True
        project._notify_observers("frames_updated", [frame])

    return {
        "success": True,
        "frame_id": frame_id,
        "updated_fields": updated_fields,
        "message": (
            f"Updated {', '.join(updated_fields)}"
            if updated_fields
            else "No fields updated"
        ),
    }


__all__ = ["update_frame"]
