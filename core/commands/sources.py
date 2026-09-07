"""Reversible source removal, including its library and sequence references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from models.clip import Clip, Source
from models.frame import Frame
from models.sequence import Sequence, SequenceClip, Track

if TYPE_CHECKING:
    from core.project import Project


def _same_objects(actual: list, expected: tuple) -> bool:
    return len(actual) == len(expected) and all(
        a is b for a, b in zip(actual, expected)
    )


@dataclass(frozen=True)
class RemovedTrackEntries:
    sequence: Sequence
    track: Track
    index: int
    before: tuple[SequenceClip, ...]
    after: tuple[SequenceClip, ...]


@dataclass(frozen=True)
class RemoveSources:
    event_name = "sources_changed"
    removed: tuple[Source, ...]
    before_sources: tuple[Source, ...]
    after_sources: tuple[Source, ...]
    before_clips: tuple[Clip, ...]
    after_clips: tuple[Clip, ...]
    before_frames: tuple[Frame, ...]
    after_frames: tuple[Frame, ...]
    tracks: tuple[RemovedTrackEntries, ...]

    @property
    def label(self) -> str:
        return "Remove source" if len(self.removed) == 1 else "Remove sources"

    @property
    def event_data(self) -> list[str]:
        return [source.id for source in self.removed]

    def notification_events(self, *, undo: bool) -> list[tuple[str, Any]]:
        # The aggregate event projects a fully committed library; legacy source
        # observers still receive their existing per-source notifications.
        return [(self.event_name, self.event_data)] + [
            ("source_added" if undo else "source_removed", source)
            for source in self.removed
        ]

    @classmethod
    def capture(cls, project: Project, source_ids: list[str]) -> RemoveSources:
        ids = set(source_ids)
        removed = tuple(source for source in project.sources if source.id in ids)
        ids = {source.id for source in removed}
        clips = tuple(project.clips)
        frames = tuple(project.frames)
        removed_clips = {clip.id for clip in clips if clip.source_id in ids}
        removed_frames = {
            frame.id
            for frame in frames
            if frame.source_id in ids or frame.clip_id in removed_clips
        }
        tracks = []
        for sequence in project.sequences:
            for index, track in enumerate(sequence.tracks):
                before = tuple(track.clips)
                after = tuple(
                    clip
                    for clip in before
                    if clip.source_id not in ids
                    and clip.source_clip_id not in removed_clips
                    and clip.frame_id not in removed_frames
                )
                if len(before) != len(after):
                    tracks.append(
                        RemovedTrackEntries(sequence, track, index, before, after)
                    )
        return cls(
            removed,
            tuple(project.sources),
            tuple(source for source in project.sources if source.id not in ids),
            clips,
            tuple(clip for clip in clips if clip.id not in removed_clips),
            frames,
            tuple(frame for frame in frames if frame.id not in removed_frames),
            tuple(tracks),
        )

    def apply(self, project: Project, *, undo: bool = False) -> list[Source]:
        libraries = (
            (
                project.sources,
                self.after_sources if undo else self.before_sources,
            ),
            (
                project.clips,
                self.after_clips if undo else self.before_clips,
            ),
            (
                project.frames,
                self.after_frames if undo else self.before_frames,
            ),
        )
        if any(
            not _same_objects(actual, expected) for actual, expected in libraries
        ):
            raise ValueError(
                "Library changed since source removal; cannot apply history"
            )
        for edit in self.tracks:
            expected = edit.after if undo else edit.before
            if (
                not any(sequence is edit.sequence for sequence in project.sequences)
                or edit.index >= len(edit.sequence.tracks)
                or edit.sequence.tracks[edit.index] is not edit.track
                or not _same_objects(edit.track.clips, expected)
            ):
                raise ValueError(
                    "Sequence changed since source removal; cannot apply history"
                )
        # Reject new references introduced outside history before deleting again.
        if not undo:
            ids = {source.id for source in self.removed}
            clip_ids = {clip.id for clip in self.before_clips} - {
                clip.id for clip in self.after_clips
            }
            frame_ids = {frame.id for frame in self.before_frames} - {
                frame.id for frame in self.after_frames
            }
            tracked = {id(edit.track) for edit in self.tracks}
            if any(
                clip.source_id in ids
                or clip.source_clip_id in clip_ids
                or clip.frame_id in frame_ids
                for sequence in project.sequences
                for track in sequence.tracks
                if id(track) not in tracked
                for clip in track.clips
            ):
                raise ValueError("Source is used by a new sequence edit")
        project.sources[:] = self.before_sources if undo else self.after_sources
        project.clips[:] = self.before_clips if undo else self.after_clips
        project.frames[:] = self.before_frames if undo else self.after_frames
        for edit in self.tracks:
            edit.track.clips[:] = edit.before if undo else edit.after
        project._invalidate_caches()
        return list(self.removed)
