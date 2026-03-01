"""Undo command for toggling clip disabled state."""

from PySide6.QtGui import QUndoCommand

from core.project import Project


class ToggleClipDisabledCommand(QUndoCommand):
    """Toggle disabled state on one or more clips.

    Since toggle is self-inverse, undo() and redo() do the same thing.
    """

    def __init__(self, project: Project, clip_ids: list[str]):
        n = len(clip_ids)
        super().__init__(f"Disable {n} clip{'s' if n != 1 else ''}")
        self._project = project
        self._clip_ids = clip_ids

    def redo(self) -> None:
        self._project.toggle_clips_disabled(self._clip_ids)

    def undo(self) -> None:
        self._project.toggle_clips_disabled(self._clip_ids)
