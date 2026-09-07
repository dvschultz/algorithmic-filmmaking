"""Headless timeline operations using the project's reversible commands."""

from __future__ import annotations

from typing import Any

from core.project import Project
from models.sequence import Sequence, SequenceClip, Track


def _sequence(project: Project, sequence_id: str) -> Sequence:
    sequence = next(
        (item for item in project.sequences if item.id == sequence_id), None
    )
    if sequence is None:
        raise ValueError("Sequence not found")
    return sequence


def _track(sequence: Sequence, track_index: int) -> Track:
    if (
        isinstance(track_index, bool)
        or not isinstance(track_index, int)
        or not 0 <= track_index < len(sequence.tracks)
    ):
        raise ValueError("Invalid sequence track")
    return sequence.tracks[track_index]


def get_timeline(project: Project, sequence_id: str) -> dict:
    return {"success": True, "sequence": _sequence(project, sequence_id).to_dict()}


def insert_clips(
    project: Project,
    sequence_id: str,
    clip_ids: list[str],
    *,
    track_index: int = 0,
    start_frame: int | None = None,
) -> dict:
    """Insert enabled library clips as one edit, using absolute source ranges.

    A missing start appends after the furthest clip on the selected track.
    An explicit start places clips without rippling existing content.
    Every input is validated before publishing the batch.
    """
    sequence = _sequence(project, sequence_id)
    track = _track(sequence, track_index)
    if not clip_ids:
        raise ValueError("Provide at least one library clip ID")
    if start_frame is None:
        start_frame = max((clip.end_frame() for clip in track.clips), default=0)
    if (
        isinstance(start_frame, bool)
        or not isinstance(start_frame, int)
        or start_frame < 0
    ):
        raise ValueError("Start frame must be a nonnegative integer")
    entries = []
    position = start_frame
    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None or clip.source_id not in project.sources_by_id:
            raise ValueError(f"Library clip or source not found: {clip_id}")
        if clip.disabled:
            raise ValueError(f"Library clip is disabled: {clip_id}")
        entry = SequenceClip(
            source_clip_id=clip.id,
            source_id=clip.source_id,
            track_index=track_index,
            start_frame=position,
            in_point=clip.start_frame,
            out_point=clip.end_frame,
        )
        entries.append(entry)
        position += entry.duration_frames
    project.insert_sequence_clips(entries, sequence=sequence)
    return {"success": True, "added": [entry.id for entry in entries]}


def remove_clips(
    project: Project,
    sequence_id: str,
    clip_ids: list[str],
    *,
    ripple: bool = False,
) -> dict:
    sequence = _sequence(project, sequence_id)
    known = {clip.id for clip in sequence.get_all_clips()}
    if any(clip_id not in known for clip_id in clip_ids):
        raise ValueError("Timeline clip not found")
    removed = project.remove_from_sequence(clip_ids, sequence=sequence, ripple=ripple)
    return {"success": True, "removed": removed}


def reorder_clips(
    project: Project,
    sequence_id: str,
    clip_ids: list[str],
    *,
    track_index: int = 0,
) -> dict:
    sequence = _sequence(project, sequence_id)
    _track(sequence, track_index)
    if not project.reorder_sequence(
        clip_ids, sequence=sequence, track_index=track_index
    ):
        raise ValueError("Reorder requires unique existing timeline clip IDs")
    return {
        "success": True,
        "clip_order": [c.id for c in sequence.tracks[track_index].clips],
    }


def edit_clip(
    project: Project,
    sequence_id: str,
    clip_id: str,
    changes: dict[str, Any],
) -> dict:
    sequence = _sequence(project, sequence_id)
    project.update_sequence_clip(clip_id, sequence=sequence, **changes)
    return {"success": True, "clip_id": clip_id}


def clear_timeline(project: Project, sequence_id: str) -> dict:
    removed = project.clear_sequence(sequence=_sequence(project, sequence_id))
    return {"success": True, "clips_removed": removed}


def shuffle_clips(
    project: Project, sequence_id: str, method: str, *, track_index: int = 0
) -> dict:
    """Resolve an order once; redo reuses the stored command without reshuffling."""
    import random

    sequence = _sequence(project, sequence_id)
    track = _track(sequence, track_index)
    entries = list(track.clips)
    if method == "random":
        random.shuffle(entries)
    elif method == "reverse":
        entries.reverse()
    elif method == "by_color":

        def hue(entry):
            clip = project.clips_by_id.get(entry.source_clip_id)
            if clip and clip.dominant_colors:
                from core.analysis.color import rgb_to_hsv

                return rgb_to_hsv(clip.dominant_colors[0])[0]
            return 0

        entries.sort(key=hue)
    elif method == "by_shot_type":
        order = ["wide shot", "medium shot", "close-up", "extreme close-up", None]

        def shot_index(entry):
            clip = project.clips_by_id.get(entry.source_clip_id)
            shot = clip.shot_type if clip else None
            return order.index(shot) if shot in order else len(order)

        entries.sort(key=shot_index)
    else:
        raise ValueError(f"Unknown shuffle method: {method}")
    result = reorder_clips(
        project, sequence_id, [c.id for c in entries], track_index=track_index
    )
    return {**result, "method": method, "clips_shuffled": len(entries)}


def insert_legacy_clips(
    project: Project,
    clip_ids: list[str],
    *,
    track_index: int = 0,
    position: str | None = "end",
) -> dict:
    """Legacy insertion with implicit tracks, published as one reversible edit."""
    sequence = project.sequence
    if sequence is None:
        raise ValueError("No sequence in project")
    if (
        isinstance(track_index, bool)
        or not isinstance(track_index, int)
        or track_index < 0
    ):
        raise ValueError("Invalid sequence track")
    if track_index >= len(sequence.tracks) and track_index >= 256:
        raise ValueError("Track creation supports indices 0 through 255")
    existing = (
        sequence.tracks[track_index].clips if track_index < len(sequence.tracks) else []
    )
    if position == "end":
        start = max((clip.end_frame() for clip in existing), default=0)
    elif position == "start":
        start = 0
    else:
        try:
            start = int(position) if position is not None else -1
        except (ValueError, TypeError):
            raise ValueError(f"Invalid position: {position}") from None
    if start < 0:
        raise ValueError("Position must be a nonnegative frame")
    entries = []
    added = []
    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None or clip.source_id not in project.sources_by_id or clip.disabled:
            continue
        entry = SequenceClip(
            source_clip_id=clip.id,
            source_id=clip.source_id,
            track_index=track_index,
            start_frame=start,
            in_point=clip.start_frame,
            out_point=clip.end_frame,
        )
        entries.append(entry)
        added.append(
            {"clip_id": clip_id, "sequence_clip_id": entry.id, "start_frame": start}
        )
        start += entry.duration_frames
    project.insert_sequence_clips(entries, sequence=sequence, create_tracks=True)
    return {
        "success": True,
        "clips_added": len(added),
        "added": added,
        "sequence_duration": sequence.duration_seconds,
    }
