"""Reversible timeline membership and positioning edits, without Qt."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from models.sequence import Sequence, SequenceClip, Track

if TYPE_CHECKING:
    from core.project import Project


@dataclass(frozen=True)
class Placement:
    clip: SequenceClip
    start: int
    in_point: int
    out_point: int
    hold_frames: int
    track_index: int
    hflip: bool
    vflip: bool
    reverse: bool
    prerendered_path: str | None

    @classmethod
    def capture(cls, clip: SequenceClip, start: int | None = None) -> Placement:
        return cls(
            clip,
            clip.start_frame if start is None else start,
            clip.in_point,
            clip.out_point,
            clip.hold_frames,
            clip.track_index,
            clip.hflip,
            clip.vflip,
            clip.reverse,
            clip.prerendered_path,
        )

    def matches(self, clip: SequenceClip) -> bool:
        return clip is self.clip and self == Placement.capture(clip)


@dataclass(frozen=True)
class TrackEdit:
    track: Track
    index: int
    before: tuple[Placement, ...]
    after: tuple[Placement, ...]


@dataclass(frozen=True)
class EditSequenceClips:
    event_name = "sequence_changed"

    sequence: Sequence
    tracks: tuple[TrackEdit, ...]
    changed: tuple[SequenceClip, ...]
    label: str
    notification_ids: tuple[str, ...]

    @property
    def event_data(self) -> list[str]:
        return list(self.notification_ids)

    @classmethod
    def insert(cls, sequence: Sequence, clips: list[SequenceClip]) -> EditSequenceClips:
        edits = []
        existing = {clip.id for clip in sequence.get_all_clips()}
        for clip in clips:
            if clip.id in existing:
                raise ValueError("Sequence clip ID already exists")
            existing.add(clip.id)
            if not 0 <= clip.track_index < len(sequence.tracks):
                raise ValueError("Invalid sequence track")
            if clip.start_frame < 0 or clip.duration_frames <= 0:
                raise ValueError(
                    "Sequence clips require a nonnegative start and positive duration"
                )
        for index, track in enumerate(sequence.tracks):
            added = [clip for clip in clips if clip.track_index == index]
            if added:
                before = tuple(Placement.capture(c) for c in track.clips)
                after = tuple(
                    sorted(
                        (*before, *(Placement.capture(c) for c in added)),
                        key=lambda p: p.start,
                    )
                )
                edits.append(TrackEdit(track, index, before, after))
        return cls(
            sequence,
            tuple(edits),
            tuple(clips),
            f"Insert {len(clips)} sequence clip{'s' if len(clips) != 1 else ''}",
            tuple(c.frame_id or c.source_clip_id or c.id for c in clips),
        )

    @classmethod
    def remove(
        cls,
        sequence: Sequence,
        clip_ids: list[str],
        *,
        ripple: bool,
    ) -> EditSequenceClips:
        ids = set(clip_ids)
        edits = []
        removed = []
        for index, track in enumerate(sequence.tracks):
            deleted = [clip for clip in track.clips if clip.id in ids]
            if not deleted:
                continue
            removed.extend(deleted)
            before = tuple(Placement.capture(c) for c in track.clips)
            after = []
            position = 0
            for clip in track.clips:
                if clip.id not in ids:
                    after.append(
                        Placement.capture(
                            clip, position if ripple else clip.start_frame
                        )
                    )
                    position += clip.duration_frames
            edits.append(TrackEdit(track, index, before, tuple(after)))
        by_id = {clip.id: clip for clip in removed}
        removed = [
            by_id[clip_id] for clip_id in dict.fromkeys(clip_ids) if clip_id in by_id
        ]
        return cls(
            sequence,
            tuple(edits),
            tuple(removed),
            f"Remove {len(removed)} sequence clip{'s' if len(removed) != 1 else ''}",
            tuple(c.id for c in removed),
        )

    @classmethod
    def reorder(cls, sequence: Sequence, clip_ids: list[str]) -> EditSequenceClips:
        track = sequence.tracks[0]
        lookup = {c.id: c for c in track.clips}
        if len(set(clip_ids)) != len(clip_ids) or any(
            cid not in lookup for cid in clip_ids
        ):
            raise ValueError("Reorder requires unique existing sequence clip IDs")
        requested = set(clip_ids)
        ordered = [lookup[cid] for cid in clip_ids]
        ordered.extend(c for c in track.clips if c.id not in requested)
        before = tuple(Placement.capture(c) for c in track.clips)
        position = 0
        after = []
        for clip in ordered:
            after.append(Placement.capture(clip, position))
            position += clip.duration_frames
        edits = (
            (TrackEdit(track, 0, before, tuple(after)),)
            if before != tuple(after)
            else ()
        )
        return cls(
            sequence,
            edits,
            tuple(ordered) if edits else (),
            "Reorder sequence",
            tuple(clip_ids),
        )

    @classmethod
    def update(
        cls, sequence: Sequence, clip_id: str, changes: dict
    ) -> EditSequenceClips:
        allowed = {
            "in_point",
            "out_point",
            "start_frame",
            "track_index",
            "hold_frames",
            "hflip",
            "vflip",
            "reverse",
        }
        if not changes or set(changes) - allowed:
            raise ValueError("Provide supported sequence clip fields")
        target = next((c for c in sequence.get_all_clips() if c.id == clip_id), None)
        if target is None:
            raise ValueError(f"Sequence clip '{clip_id}' not found")
        for key, value in changes.items():
            if key in {"hflip", "vflip", "reverse"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a boolean")
            elif not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
        candidate = replace(target, **changes)
        if (
            candidate.start_frame < 0
            or candidate.in_point < 0
            or candidate.hold_frames < 1
        ):
            raise ValueError("Invalid sequence clip position or duration")
        if not candidate.is_frame_entry and candidate.out_point <= candidate.in_point:
            raise ValueError("out_point must be greater than in_point")
        if not 0 <= candidate.track_index < len(sequence.tracks):
            raise ValueError("track_index out of range")
        if any(
            getattr(candidate, key) != getattr(target, key)
            for key in ("in_point", "out_point", "hflip", "vflip", "reverse")
        ):
            # Cached media already bakes in the source range and transforms.
            # Placement retains the previous reference for undo.
            candidate.prerendered_path = None
        desired = replace(Placement.capture(candidate), clip=target)
        original_index = next(
            i
            for i, track in enumerate(sequence.tracks)
            if any(c is target for c in track.clips)
        )
        edits = []
        for index in sorted({original_index, candidate.track_index}):
            track = sequence.tracks[index]
            before = tuple(Placement.capture(c) for c in track.clips)
            if index == original_index == candidate.track_index:
                after = [desired if p.clip is target else p for p in before]
            else:
                after = [p for p in before if p.clip is not target]
                if index == candidate.track_index:
                    after.append(desired)
            after.sort(key=lambda p: p.start)
            if before != tuple(after):
                edits.append(TrackEdit(track, index, before, tuple(after)))
        return cls(
            sequence,
            tuple(edits),
            (target,) if edits else (),
            "Edit sequence clip",
            (clip_id,),
        )

    def apply(self, project: Project, *, undo: bool = False) -> list[SequenceClip]:
        if not any(sequence is self.sequence for sequence in project.sequences):
            raise ValueError("Sequence no longer belongs to this project")
        # Validate every affected track before changing any of them. Keep the
        # original objects: transforms and media references must survive undo.
        for edit in self.tracks:
            expected = edit.after if undo else edit.before
            if (
                edit.index >= len(self.sequence.tracks)
                or self.sequence.tracks[edit.index] is not edit.track
                or len(edit.track.clips) != len(expected)
                or any(not p.matches(c) for p, c in zip(expected, edit.track.clips))
            ):
                raise ValueError(
                    "Sequence changed since this edit; cannot apply history"
                )
        for edit in self.tracks:
            target = edit.before if undo else edit.after
            edit.track.clips[:] = [p.clip for p in target]
            for placement in target:
                placement.clip.start_frame = placement.start
                for field in (
                    "in_point",
                    "out_point",
                    "hold_frames",
                    "track_index",
                    "hflip",
                    "vflip",
                    "reverse",
                    "prerendered_path",
                ):
                    setattr(placement.clip, field, getattr(placement, field))
        return list(self.changed)
