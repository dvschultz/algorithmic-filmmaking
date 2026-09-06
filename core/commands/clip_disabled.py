"""Capture explicit clip states so undo never toggles an unexpected value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip


@dataclass(frozen=True)
class ClipDisabledChange:
    clip: Clip
    before: bool
    after: bool


@dataclass(frozen=True)
class SetClipsDisabled:
    event_name = "clips_updated"

    changes: tuple[ClipDisabledChange, ...]
    label: str

    @property
    def event_data(self) -> list[Clip]:
        return [change.clip for change in self.changes]

    @classmethod
    def capture(
        cls, project: Project, clip_ids: list[str], disabled: bool | None,
    ) -> SetClipsDisabled:
        if disabled is not None and not isinstance(disabled, bool):
            raise ValueError("disabled must be a boolean or None")
        changes = []
        for clip_id in dict.fromkeys(clip_ids):
            clip = project.clips_by_id.get(clip_id)
            if clip is None:
                continue
            after = not clip.disabled if disabled is None else disabled
            if after != clip.disabled:
                changes.append(ClipDisabledChange(clip, clip.disabled, after))
        verb = "Disable" if all(c.after for c in changes) else "Enable"
        if changes and any(c.after for c in changes) and not all(c.after for c in changes):
            verb = "Toggle"
        n = len(changes)
        return cls(tuple(changes), f"{verb} {n} clip{'s' if n != 1 else ''}")

    def apply(self, project: Project, *, undo: bool = False) -> list[Clip]:
        # Validate the entire batch before touching any target. Identity protects
        # a newly imported clip that happens to reuse a deleted clip's ID.
        for change in self.changes:
            expected = change.after if undo else change.before
            if (
                project.clips_by_id.get(change.clip.id) is not change.clip
                or change.clip.disabled != expected
            ):
                raise ValueError("Clip changed since this edit; cannot apply history")
        for change in self.changes:
            change.clip.disabled = change.before if undo else change.after
        return [change.clip for change in self.changes]
