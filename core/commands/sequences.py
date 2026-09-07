"""Reversible sequence collection and metadata changes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from models.sequence import Sequence

if TYPE_CHECKING:
    from core.project import Project


@dataclass(frozen=True)
class EditSequences:
    event_name = "sequences_changed"
    collection: list[Sequence]
    before: tuple[Sequence, ...]
    after: tuple[Sequence, ...]
    before_active: Sequence
    after_active: Sequence
    label: str

    @property
    def event_data(self) -> list[Sequence]:
        return list(self.collection)

    @property
    def retained_sequences(self) -> tuple[Sequence, ...]:
        return self.before + self.after

    @classmethod
    def add(cls, project: Project, sequence: Sequence, activate: bool) -> EditSequences:
        if any(s.id == sequence.id for s in project.sequences):
            raise ValueError("Sequence ID already exists")
        active = project.sequences[project.active_sequence_index]
        return cls(
            project.sequences,
            tuple(project.sequences),
            (*project.sequences, sequence),
            active,
            sequence if activate else active,
            "Create sequence",
        )

    @classmethod
    def remove(cls, project: Project, index: int) -> EditSequences:
        before = tuple(project.sequences)
        after = before[:index] + before[index + 1 :]
        if not after:
            after = (Sequence(),)
        active = before[project.active_sequence_index]
        return cls(
            project.sequences,
            before,
            after,
            active,
            after[0] if active is before[index] else active,
            "Delete sequence",
        )

    def apply(self, project: Project, *, undo: bool = False) -> list[Sequence]:
        expected, target = (
            (self.after, self.before) if undo else (self.before, self.after)
        )
        if (
            project.sequences is not self.collection
            or len(expected) != len(self.collection)
            or any(a is not b for a, b in zip(expected, self.collection))
        ):
            raise ValueError("Sequence collection changed since this edit")
        # A permanent clip/source deletion outside this history must not make
        # undo silently resurrect a sequence with dangling model references.
        for sequence in target:
            if any(current is sequence for current in expected):
                continue
            for clip in sequence.get_all_clips():
                if (
                    (clip.source_id and clip.source_id not in project.sources_by_id)
                    or (
                        clip.source_clip_id
                        and clip.source_clip_id not in project.clips_by_id
                    )
                    or (clip.frame_id and clip.frame_id not in project.frames_by_id)
                ):
                    raise ValueError(
                        "Cannot restore sequence: referenced media was removed"
                    )
        self.collection[:] = target
        active = self.before_active if undo else self.after_active
        project.active_sequence_index = next(
            i for i, s in enumerate(target) if s is active
        )
        return list(target)


@dataclass(frozen=True)
class EditSequenceMetadata:
    event_name = "sequences_changed"
    collection: list[Sequence]
    sequence: Sequence
    before: dict[str, Any]
    after: dict[str, Any]
    label: str = "Edit sequence settings"

    @property
    def event_data(self) -> list[Sequence]:
        return list(self.collection)

    @classmethod
    def capture(
        cls, project: Project, sequence: Sequence, changes: dict[str, Any]
    ) -> EditSequenceMetadata:
        allowed = {"name", "fps", "music_path", "allow_repeats"}
        if not changes or set(changes) - allowed:
            raise ValueError("Provide supported sequence settings")
        changes = dict(changes)
        if "name" in changes:
            if not isinstance(changes["name"], str) or not changes["name"].strip():
                raise ValueError("Sequence name cannot be empty")
            changes["name"] = changes["name"].strip()
        if "fps" in changes:
            fps = changes["fps"]
            if (
                isinstance(fps, bool)
                or not isinstance(fps, (int, float))
                or not isfinite(fps)
                or fps <= 0
            ):
                raise ValueError("fps must be finite and greater than zero")
        if "allow_repeats" in changes and not isinstance(
            changes["allow_repeats"], bool
        ):
            raise ValueError("allow_repeats must be a boolean")
        if "music_path" in changes and not isinstance(
            changes["music_path"], (str, type(None))
        ):
            raise ValueError("music_path must be a string or None")
        after = {k: v for k, v in changes.items() if getattr(sequence, k) != v}
        before = {k: getattr(sequence, k) for k in after}
        return cls(
            project.sequences,
            sequence,
            before,
            after,
            "Rename sequence" if set(changes) == {"name"} else "Edit sequence settings",
        )

    def apply(self, project: Project, *, undo: bool = False) -> list[Sequence]:
        expected, target = (
            (self.after, self.before) if undo else (self.before, self.after)
        )
        if not any(s is self.sequence for s in project.sequences) or any(
            getattr(self.sequence, key) != value for key, value in expected.items()
        ):
            raise ValueError("Sequence settings changed since this edit")
        for key, value in target.items():
            setattr(self.sequence, key, value)
        return [self.sequence] if target else []
