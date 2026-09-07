"""Editorial clip metadata shared by adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

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


def add_tags_to_clip(project: Project, clip_id: str, tags: list[str]) -> dict:
    """Preserve the single-clip MCP tag normalization and response contract."""
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        raise ValueError(f"Clip not found: {clip_id}")
    existing = set(clip.tags or [])
    added = [tag.strip() for tag in tags if tag.strip() and tag.strip() not in existing]
    project.update_clip_metadata(
        clip_id, tags=list(dict.fromkeys((clip.tags or []) + added))
    )
    return {
        "success": True,
        "clip_id": clip_id,
        "tags_added": added,
        "all_tags": list(clip.tags),
    }


def remove_tags_from_clip(project: Project, clip_id: str, tags: list[str]) -> dict:
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        raise ValueError(f"Clip not found: {clip_id}")
    removed = {tag.strip() for tag in tags}
    original_count = len(clip.tags or [])
    project.update_clip_metadata(
        clip_id, tags=[tag for tag in (clip.tags or []) if tag not in removed]
    )
    return {
        "success": True,
        "clip_id": clip_id,
        "tags_removed": original_count - len(clip.tags),
        "all_tags": list(clip.tags),
    }


def set_clip_note(project: Project, clip_id: str, note: str) -> dict:
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        raise ValueError(f"Clip not found: {clip_id}")
    project.update_clip_metadata(clip_id, notes=note.strip())
    return {"success": True, "clip_id": clip_id, "note": clip.notes}


def update_clip_from_json(
    project: Project, clip_id: str, fields: dict[str, Any]
) -> dict:
    """Convert transport values before submitting one editorial metadata edit."""
    from math import isfinite
    from core.transcription_models import TranscriptSegment

    changes = dict(fields)
    if "transcript" in changes and changes["transcript"] is not None:
        raw_segments = changes["transcript"]
        if not isinstance(raw_segments, list):
            raise ValueError("transcript must be a list")

        def number(value: Any) -> TypeGuard[int | float]:
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and isfinite(value)
            )

        def validate_text_span(value: Any, start: str, end: str) -> None:
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise ValueError("Transcript segments and words require text")
            first, last = value.get(start), value.get(end)
            if not number(first) or not number(last) or not 0 <= first <= last:
                raise ValueError(
                    "Transcript times must be finite, nonnegative, and ordered"
                )

        for segment in raw_segments:
            validate_text_span(segment, "start_time", "end_time")
            if not number(segment.get("confidence", 0.0)):
                raise ValueError("Transcript confidence must be finite")
            if segment.get("language") is not None and not isinstance(
                segment["language"], str
            ):
                raise ValueError("Transcript language must be text")
            words = segment.get("words")
            if words is not None:
                if not isinstance(words, list):
                    raise ValueError("Transcript words must be a list")
                for word in words:
                    validate_text_span(word, "start", "end")
                    if word.get("probability") is not None and not number(
                        word["probability"]
                    ):
                        raise ValueError("Word probability must be finite")
        changes["transcript"] = [
            TranscriptSegment.from_dict(segment) for segment in raw_segments
        ]
    return update_clip(project, clip_id, **changes)
