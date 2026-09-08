"""Reversible timeline membership and positioning edits, without Qt."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from fractions import Fraction

from models.sequence import Sequence, SequenceClip, Track
from models.media_time import frame_boundary, frame_rate
from core.sequence_time import entry_duration, source_video_range

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
    timeline_start: str | None
    hold_duration: str | None
    source_presentation: tuple[str, str] | None

    @classmethod
    def capture(
        cls, clip: SequenceClip, start: int | None = None, *, exact_start: Fraction | None = None,
    ) -> Placement:
        hold_frames = clip.hold_frames
        if exact_start is not None and clip.is_frame_entry and clip.hold_duration is not None:
            rate = frame_rate(clip.timeline_rate or 30)
            hold_frames = frame_boundary(exact_start + clip.source_range.duration, rate) - frame_boundary(exact_start, rate)
        return cls(
            clip,
            clip.start_frame if start is None else start,
            clip.in_point,
            clip.out_point,
            hold_frames,
            clip.track_index,
            clip.hflip,
            clip.vflip,
            clip.reverse,
            clip.prerendered_path,
            clip.timeline_start if exact_start is None else str(exact_start),
            clip.hold_duration,
            clip.source_presentation,
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
    original_tracks: tuple[Track, ...] = ()
    created_tracks: tuple[Track, ...] = ()

    @property
    def event_data(self) -> list[str]:
        return list(self.notification_ids)

    @property
    def retained_sources(self) -> dict[str, str]:
        """Source references in both sides of the edit, including removed clips."""
        return {
            placement.clip.source_id: self.sequence.name
            for edit in self.tracks
            for placement in (*edit.before, *edit.after)
            if placement.clip.source_id
        }

    @classmethod
    def insert(
        cls,
        sequence: Sequence,
        clips: list[SequenceClip],
        *,
        create_tracks: bool = False,
    ) -> EditSequenceClips:
        edits = []
        planned_tracks = list(sequence.tracks)
        for clip in clips:
            index = clip.track_index
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Invalid sequence track")
            if create_tracks and index >= len(planned_tracks):
                if index >= 256:
                    raise ValueError("Track creation supports indices 0 through 255")
                while len(planned_tracks) <= index:
                    planned_tracks.append(
                        Track(name=f"Video {len(planned_tracks) + 1}")
                    )
        created = tuple(planned_tracks[len(sequence.tracks) :])
        existing = {clip.id for clip in sequence.get_all_clips()}
        for clip in clips:
            if clip.id in existing:
                raise ValueError("Sequence clip ID already exists")
            existing.add(clip.id)
            if not 0 <= clip.track_index < len(planned_tracks):
                raise ValueError("Invalid sequence track")
            if clip.start_frame < 0 or entry_duration(clip, sequence.fps) <= 0:
                raise ValueError(
                    "Sequence clips require a nonnegative start and positive duration"
                )
        for index, track in enumerate(planned_tracks):
            added = [clip for clip in clips if clip.track_index == index]
            if added or index >= len(sequence.tracks):
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
            tuple(sequence.tracks) if created else (),
            created,
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
            position = Fraction(0)
            for clip in track.clips:
                if clip.id not in ids:
                    after.append(
                        Placement.capture(
                            clip,
                            frame_boundary(position, frame_rate(sequence.fps)) if ripple else clip.start_frame,
                            exact_start=position if ripple else None,
                        )
                    )
                    position += entry_duration(clip, sequence.fps)
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
    def clear(cls, sequence: Sequence) -> EditSequenceClips:
        command = cls.remove(
            sequence, [clip.id for clip in sequence.get_all_clips()], ripple=False
        )
        return replace(command, label="Clear sequence", notification_ids=())

    @classmethod
    def reorder(
        cls,
        sequence: Sequence,
        clip_ids: list[str],
        *,
        track_index: int = 0,
    ) -> EditSequenceClips:
        if (
            isinstance(track_index, bool)
            or not isinstance(track_index, int)
            or not 0 <= track_index < len(sequence.tracks)
        ):
            raise ValueError("Invalid sequence track")
        track = sequence.tracks[track_index]
        lookup = {c.id: c for c in track.clips}
        if len(set(clip_ids)) != len(clip_ids) or any(
            cid not in lookup for cid in clip_ids
        ):
            raise ValueError("Reorder requires unique existing sequence clip IDs")
        requested = set(clip_ids)
        ordered = [lookup[cid] for cid in clip_ids]
        ordered.extend(c for c in track.clips if c.id not in requested)
        before = tuple(Placement.capture(c) for c in track.clips)
        position = Fraction(0)
        after = []
        for clip in ordered:
            after.append(Placement.capture(
                clip, frame_boundary(position, frame_rate(sequence.fps)), exact_start=position,
            ))
            position += entry_duration(clip, sequence.fps)
        edits = (
            (TrackEdit(track, track_index, before, tuple(after)),)
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
        cls, sequence: Sequence, clip_id: str, changes: dict, *, project: Project | None = None,
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
        if any(key in changes for key in ("in_point", "out_point")):
            if target.legacy_timing and target.legacy_timing.get("status") == "unresolved":
                raise ValueError("Resolve legacy timing before trimming this entry")
            if target.source_presentation is not None:
                source = project.sources_by_id.get(target.source_id) if project is not None else None
                if source is None or source.frame_timestamps is None:
                    raise ValueError("VFR trimming requires the source presentation timestamps")
                media = source_video_range(source, candidate.in_point, candidate.out_point)
                candidate.source_presentation = (str(media.start), str(media.end))
        if "hold_frames" in changes and changes["hold_frames"] != target.hold_frames:
            candidate.hold_duration = None
        if "start_frame" in changes and changes["start_frame"] != target.start_frame:
            candidate.timeline_start = None
        if (
            candidate.start_frame < 0
            or candidate.in_point < 0
            or candidate.hold_frames < 0
            or (candidate.is_frame_entry and entry_duration(candidate, sequence.fps) <= 0)
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
        validation_tracks = list(self.sequence.tracks)
        if self.created_tracks:
            expected_tracks = (
                self.original_tracks + self.created_tracks
                if undo
                else self.original_tracks
            )
            if len(expected_tracks) != len(validation_tracks) or any(
                actual is not expected
                for actual, expected in zip(validation_tracks, expected_tracks)
            ):
                raise ValueError("Sequence tracks changed since this edit")
            if not undo:
                validation_tracks.extend(self.created_tracks)
        # Validate every affected track before changing any of them. Keep the
        # original objects: transforms and media references must survive undo.
        for edit in self.tracks:
            expected = edit.after if undo else edit.before
            if (
                edit.index >= len(validation_tracks)
                or validation_tracks[edit.index] is not edit.track
                or len(edit.track.clips) != len(expected)
                or any(not p.matches(c) for p, c in zip(expected, edit.track.clips))
            ):
                raise ValueError(
                    "Sequence changed since this edit; cannot apply history"
                )
        for edit in self.tracks:
            target = edit.before if undo else edit.after
            expected = edit.after if undo else edit.before
            existing = {id(p.clip) for p in expected}
            for placement in target:
                clip = placement.clip
                if id(clip) in existing:
                    continue
                if (
                    (clip.source_id and clip.source_id not in project.sources_by_id)
                    or (
                        clip.source_clip_id
                        and clip.source_clip_id not in project.clips_by_id
                    )
                    or (clip.frame_id and clip.frame_id not in project.frames_by_id)
                ):
                    raise ValueError(
                        "Cannot restore sequence clip: referenced media was removed"
                    )
        if self.created_tracks and not undo:
            self.sequence.tracks.extend(self.created_tracks)
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
                    "timeline_start",
                    "hold_duration",
                    "source_presentation",
                ):
                    setattr(placement.clip, field, getattr(placement, field))
        if self.created_tracks and undo:
            self.sequence.tracks[len(self.original_tracks) :] = []
        return list(self.changed)
