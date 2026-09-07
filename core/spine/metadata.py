"""Editorial clip metadata shared by adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.constants import VALID_SHOT_TYPES

if TYPE_CHECKING:
    from core.project import Project


def update_clip(project: Project, clip_id: str, **fields: Any) -> dict:
    if clip_id not in project.clips_by_id:
        return {"success": False, "error": f"Clip not found: {clip_id}"}
    changes = {key: value for key, value in fields.items() if value is not None}
    if "shot_type" in changes:
        value = changes["shot_type"]
        if value == "":
            changes["shot_type"] = None
        elif value not in VALID_SHOT_TYPES:
            return {
                "success": False,
                "error": f"Invalid shot type: '{value}'. Must be one of: {', '.join(sorted(VALID_SHOT_TYPES))} or empty string to clear.",
            }
    try:
        project.update_clip_metadata(clip_id, **changes)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    updated = list(changes)
    return {
        "success": True,
        "clip_id": clip_id,
        "updated_fields": updated,
        "message": f"Updated {', '.join(updated)}" if updated else "No fields updated",
    }


def edit_tags(
    project: Project, clip_ids: list[str], tags: list[str], *, remove: bool = False
) -> dict:
    if not clip_ids:
        return {"success": False, "error": "No clip IDs provided"}
    if not tags:
        return {"success": False, "error": "No tags provided"}
    updated: list[str] = []
    not_found: list[str] = []
    updates: dict[str, dict[str, Any]] = {}
    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None:
            not_found.append(clip_id)
            continue
        before = list(updates.get(clip_id, {}).get("tags", clip.tags))
        after = list(before)
        for tag in tags:
            if remove and tag in after:
                after.remove(tag)
            elif not remove and tag not in after:
                after.append(tag)
        if not remove or before != after:
            updated.append(clip_id)
            updates[clip_id] = {"tags": after}
    try:
        project.edit_metadata("clip", updates)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": bool(updated),
        "updated": updated,
        "not_found": not_found,
        "tags_removed" if remove else "tags_added": tags,
        "message": f"Removed tag(s) from {len(updated)} clip(s)"
        if remove
        else f"Added {len(tags)} tag(s) to {len(updated)} clip(s)",
    }
