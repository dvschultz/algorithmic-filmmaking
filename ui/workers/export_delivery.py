"""Owner-thread export delivery with project and worker identity checks."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject


class ExportDelivery(RetiringQObject):
    def __init__(
        self,
        window: Any,
        attribute: str,
        worker: Any,
        handlers: dict[str, Callable],
    ) -> None:
        super().__init__(window)
        self.window = window
        self.attribute = attribute
        self.worker = worker
        self.handlers = handlers
        self.session_id = window.project.session.session_id
        self._completed = False
        worker.progress.connect(
            self.progress if attribute == "export_worker" else self.bundle_progress
        )
        worker.export_completed.connect(self.result)
        worker.error.connect(self.error)
        worker.finished.connect(self.finished)

    def _current(self) -> bool:
        return (
            getattr(self.window, self.attribute, None) is self.worker
            and self.window.project.session.session_id == self.session_id
        )

    @Slot(float, str)
    def progress(self, value: float, message: str) -> None:
        if self._current() and not self._completed:
            self.handlers["progress"](value, message)

    @Slot(int, int, str)
    def bundle_progress(self, current: int, total: int, filename: str) -> None:
        if self._current() and not self._completed:
            self.handlers["progress"](current, total, filename)

    @Slot(object)
    def result(self, result: Any) -> None:
        if self._current() and not self._completed:
            self._completed = True
            self.handlers["result"](result)

    @Slot(str)
    def error(self, error: str) -> None:
        if self._current() and not self._completed:
            self._completed = True
            self.handlers["error"](error)

    @Slot()
    def finished(self) -> None:
        if getattr(self.window, self.attribute, None) is self.worker:
            setattr(self.window, self.attribute, None)
        self.worker.deleteLater()
        self.retire()
