"""Reversible, owner-thread resolution of one preserved legacy entry."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from models.sequence import Sequence, SequenceClip

if TYPE_CHECKING:
    from core.project import Project


@dataclass(frozen=True)
class ResolveSequenceTiming:
    sequence: Sequence
    entry: SequenceClip
    before: dict
    after: dict
    source_input: tuple
    event_name = "sequence_changed"
    label = "Resolve legacy sequence timing"

    @property
    def event_data(self) -> list[str]:
        return [self.sequence.id]

    def apply(self, project: "Project", *, undo: bool = False) -> list[SequenceClip]:
        from core.sequence_time import sequence_source_input
        if not any(seq is self.sequence for seq in project.sequences) or not any(
            entry is self.entry for entry in self.sequence.get_all_clips()
        ):
            raise ValueError("Sequence entry no longer belongs to this project")
        if sequence_source_input(project, self.entry) != self.source_input:
            raise ValueError("Sequence source changed since timing resolution")
        expected, desired = (self.after, self.before) if undo else (self.before, self.after)
        if self.entry.to_dict() != expected:
            raise ValueError("Sequence entry changed since timing resolution")
        restored = SequenceClip.from_dict(desired)
        for name in (
            "in_point", "out_point", "source_rate", "timeline_rate", "timeline_start",
            "source_presentation", "legacy_timing", "hold_duration",
        ):
            setattr(self.entry, name, deepcopy(getattr(restored, name)))
        return [self.entry]
