"""Qt actions projecting the shared project history, with no second undo stack."""

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction

from core.project_session import ProjectSession
from core.spine import history


class SessionHistoryAdapter(QObject):
    changed = Signal()
    action_failed = Signal(str)

    def __init__(self, session: ProjectSession, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session = session
        session.add_observer(self._on_changed)

    def _on_changed(self) -> None:
        self.changed.emit()

    def set_session(self, session: ProjectSession) -> None:
        self._session.remove_observer(self._on_changed)
        self._session = session
        session.add_observer(self._on_changed)
        self.changed.emit()

    def canUndo(self) -> bool:
        return self._session.can_undo

    def canRedo(self) -> bool:
        return self._session.can_redo

    def undoText(self) -> str:
        return self._session.undo_text

    def redoText(self) -> str:
        return self._session.redo_text

    def undo(self) -> dict:
        result = history.undo(self._session.project)
        if not result["success"]:
            self.action_failed.emit(result["error"])
        return result

    def redo(self) -> dict:
        result = history.redo(self._session.project)
        if not result["success"]:
            self.action_failed.emit(result["error"])
        return result

    def createUndoAction(self, parent: QObject, prefix: str = "Undo") -> QAction:
        return self._create_action(parent, prefix, undo=True)

    def createRedoAction(self, parent: QObject, prefix: str = "Redo") -> QAction:
        return self._create_action(parent, prefix, undo=False)

    def _create_action(self, parent: QObject, prefix: str, *, undo: bool) -> QAction:
        action = QAction(parent)

        def refresh() -> None:
            label = self.undoText() if undo else self.redoText()
            action.setText(f"{prefix} {label}" if label else prefix)
            action.setEnabled(self.canUndo() if undo else self.canRedo())

        self.changed.connect(refresh)
        action.triggered.connect(self.undo if undo else self.redo)
        refresh()
        return action
