"""Retained headless editorial sessions, called on one owning thread."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from core.project import Project
from core.project_revision import ProjectRevisionConflict
from core.spine.project_io import load_with_mtime, project_writer, save_with_mtime_check
from core.spine.sequences import list_sequences
from core.spine.security import validate_project_path


@dataclass
class _Session:
    path: Path
    project: Project | None = None
    mtime: float = 0.0

    def refresh(self) -> bool:
        valid, error, resolved = validate_project_path(str(self.path))
        if not valid:
            raise ValueError(error)
        if resolved != self.path:
            raise ValueError("Project session path was retargeted; close and reopen it")
        if self.project is not None:
            try:
                self.project.session.verify_file_revision()
                return False
            except ProjectRevisionConflict:
                self.discard()
        self.project, self.mtime = load_with_mtime(self.path)
        return True

    def discard(self) -> None:
        if self.project is not None:
            self.project.session.close()
            self.project = None


class ProjectSessions:
    """Keep history between calls; acquire disk ownership for each operation.

    All methods must run on the same owner thread. The MCP transport supplies
    a serial executor. Successful edits are saved before their results return.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def open(self, path: Path) -> dict:
        canonical = path.expanduser().resolve()
        for session_id, entry in self._sessions.items():
            if entry.path == canonical or (
                entry.path.exists() and entry.path.samefile(canonical)
            ):
                return self.inspect(session_id)
        session_id = uuid4().hex
        entry = _Session(canonical)
        with project_writer(canonical):
            entry.refresh()
            self._sessions[session_id] = entry
            return self._state(session_id, entry, reloaded=False)

    def _entry(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise ValueError("Unknown or closed project session") from None

    def _state(self, session_id: str, entry: _Session, *, reloaded: bool) -> dict:
        project = entry.project
        assert project is not None
        return {
            "success": True,
            "session_id": session_id,
            "project_path": str(entry.path),
            "history_reset": reloaded,
            "read_only": project.is_read_only,
            "can_undo": project.session.can_undo,
            "can_redo": project.session.can_redo,
            "undo_label": project.session.undo_text,
            "redo_label": project.session.redo_text,
            "sequences": list_sequences(project)["sequences"],
        }

    def inspect(self, session_id: str) -> dict:
        entry = self._entry(session_id)
        with project_writer(entry.path):
            reloaded = entry.refresh()
            return self._state(session_id, entry, reloaded=reloaded)

    def edit(self, session_id: str, operation: Callable[[Project], dict]) -> dict:
        entry = self._entry(session_id)
        with project_writer(entry.path):
            reloaded = entry.refresh()
            project = entry.project
            assert project is not None
            project._assert_writable()
            generation = project.mutation_generation
            try:
                result = operation(project)
                if not result.get("success"):
                    if project.mutation_generation != generation:
                        entry.discard()
                    return {
                        **result,
                        "session_id": session_id,
                        "history_reset": reloaded,
                    }
                if project.mutation_generation != generation:
                    save_with_mtime_check(project, entry.path, entry.mtime)
                    entry.mtime = entry.path.stat().st_mtime
                return {
                    **result,
                    "session": self._state(session_id, entry, reloaded=reloaded),
                }
            except BaseException:
                if project.mutation_generation != generation:
                    entry.discard()
                raise

    def close(self, session_id: str) -> dict:
        entry = self._entry(session_id)
        entry.discard()
        del self._sessions[session_id]
        return {"success": True, "closed_session_id": session_id}

    def close_all(self) -> None:
        for session_id in list(self._sessions):
            self.close(session_id)
