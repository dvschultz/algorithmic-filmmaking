"""Owner-thread edit history for a live project, independent of Qt.

Clip state and sequence membership edits are migrated. Legacy model mutations still
advance the external revision, so undo never marks unsaved analysis as saved.
History is intentionally session-local and is not serialized into project files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import get_ident
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from core.project import Project
    from models.sequence import Sequence

logger = logging.getLogger(__name__)


T = TypeVar("T")


class EditCommand(Protocol[T]):
    @property
    def event_name(self) -> str: ...

    @property
    def event_data(self) -> list[Any]: ...

    @property
    def label(self) -> str: ...

    def apply(self, project: Project, *, undo: bool = False) -> list[T]: ...


@dataclass(frozen=True)
class HistoryEntry:
    command: EditCommand[Any]
    before: int
    after: int


class ProjectSession:
    """Serialize migrated edits and track their position relative to a save."""

    def __init__(self, project: Project) -> None:
        self.project = project
        self.session_id = uuid4().hex
        self._owner_thread = get_ident()
        self._busy = False
        self._closed = False
        self._undo: list[HistoryEntry] = []
        self._redo: list[HistoryEntry] = []
        self._position = 0
        self._next_position = 0
        self._external_revision = 0
        self._saved_state: tuple[int, int] | None = None if project.is_dirty else (0, 0)
        self._observers: list[Callable[[], None]] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo) and not self._closed

    @property
    def can_redo(self) -> bool:
        return bool(self._redo) and not self._closed

    @property
    def undo_text(self) -> str:
        return self._undo[-1].command.label if self.can_undo else ""

    @property
    def redo_text(self) -> str:
        return self._redo[-1].command.label if self.can_redo else ""

    def assert_owner(self) -> None:
        if self._closed:
            raise RuntimeError("Project session is closed")
        if get_ident() != self._owner_thread:
            raise RuntimeError("Project edits must run on the session owner thread")
        if self._busy:
            raise RuntimeError("A project edit is already being published")

    def add_observer(self, callback: Callable[[], None]) -> None:
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[], None]) -> None:
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify(self) -> None:
        for callback in tuple(self._observers):
            try:
                callback()
            except Exception:
                logger.exception("Project session observer failed")

    @property
    def retained_sequences(self) -> list[Sequence]:
        """Sequences whose media must remain available for undo or redo."""
        retained = {}
        for entry in (*self._undo, *self._redo):
            for sequence in getattr(entry.command, "retained_sequences", ()):
                retained[id(sequence)] = sequence
        return list(retained.values())

    def record_external_change(self) -> None:
        """Keep legacy edits and analysis outside editorial undo history."""
        self._external_revision += 1
        self._notify()

    @property
    def retained_sources(self) -> dict[str, list[str]]:
        """Source IDs and sequence names retained by timeline edit snapshots."""
        retained: dict[str, list[str]] = {}
        for entry in (*self._undo, *self._redo):
            for source_id, name in getattr(entry.command, "retained_sources", {}).items():
                names = retained.setdefault(source_id, [])
                if name not in names:
                    names.append(name)
        return retained

    def record_saved(self) -> None:
        self._saved_state = (self._position, self._external_revision)
        self._notify()

    def _publish(self, command: EditCommand[Any], *, undo: bool = False) -> None:
        self.project._mutation_generation += 1
        self.project._dirty = self._saved_state != (self._position, self._external_revision)
        notifications = getattr(command, "notification_events", None)
        events = notifications(undo=undo) if notifications else [(command.event_name, command.event_data)]
        for event, data in events:
            self.project._notify_observers(event, data)
        if command.event_name == "sequences_changed":
            self.project._notify_observers("active_sequence_changed", self.project.active_sequence_index)
        self._notify()

    def execute(self, command: EditCommand[T]) -> list[T]:
        self.assert_owner()
        self._busy = True
        try:
            clips = command.apply(self.project)
            if not clips:
                return []
            self._next_position += 1
            self._undo.append(HistoryEntry(command, self._position, self._next_position))
            self._position = self._next_position
            self._redo.clear()
            self._publish(command)
            return clips
        finally:
            self._busy = False

    def apply_external(self, apply: Callable[[], T]) -> T:
        """Apply an owner-thread result without adding editorial history.

        The callback owns validation, mutation, and normal dirty notifications.
        Observers cannot reset, close, or edit the session during application.
        This is a reentrancy guard, not a rollback transaction.
        """
        self.assert_owner()
        self._busy = True
        try:
            return apply()
        finally:
            self._busy = False

    def undo(self) -> str | None:
        return self._move_history(undo=True)

    def redo(self) -> str | None:
        return self._move_history(undo=False)

    def _move_history(self, *, undo: bool) -> str | None:
        self.assert_owner()
        source, destination = (self._undo, self._redo) if undo else (self._redo, self._undo)
        if not source:
            return None
        self._busy = True
        try:
            entry = source[-1]
            entry.command.apply(self.project, undo=undo)
            source.pop()
            destination.append(entry)
            self._position = entry.before if undo else entry.after
            self._publish(entry.command, undo=undo)
            return entry.command.label
        finally:
            self._busy = False

    def reset(self) -> None:
        """Invalidate pending results and history when New Project reuses a model."""
        self.assert_owner()
        self.session_id = uuid4().hex
        self._undo.clear()
        self._redo.clear()
        self._position = 0
        self._next_position = 0
        self._external_revision = 0
        self._saved_state = (0, 0)
        self._notify()

    def close(self) -> None:
        self.assert_owner()
        self._closed = True
        self._undo.clear()
        self._redo.clear()
        self._notify()
