"""Deliver download signals only to the requesting project and worker channel."""

from collections.abc import Callable

from PySide6.QtCore import QObject, Slot


class DownloadDelivery(QObject):
    """Owner-thread relay that retains replaced workers until they finish.

    A custom finished hook must eventually call `_finished()` to release the
    channel. Give subclass Qt slots distinct names to preserve sender identity.
    """

    def __init__(
        self,
        window,
        attribute: str,
        worker,
        handlers: dict[str, Callable],
        *,
        finished: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.attribute = attribute
        self.worker = worker
        worker.gui_tool_reply = getattr(window, "_dispatch_gui_reply", None)
        self.session_id = window.project.session.session_id
        self.handlers = handlers
        previous = window._download_deliveries.get(attribute)
        if previous is not None:
            previous.worker.cancel()
        window._download_deliveries[attribute] = self
        window._active_download_workers.add(worker)
        worker.setParent(self)
        worker.finished.connect(finished or self._finished)

    def bind_signal(self, signal, handler: str) -> None:
        signal.connect(getattr(self, handler))

    def _deliver(self, handler: str, *args) -> None:
        if (
            self.window.project.session.session_id == self.session_id
            and self.window._download_deliveries.get(self.attribute) is self
        ):
            self.handlers[handler](*args)

    @Slot(float, str)
    def progress(self, value: float, message: str) -> None:
        self._deliver("progress", value, message)

    @Slot(int, int, str)
    def bulk_progress(self, current: int, total: int, message: str) -> None:
        self._deliver("bulk_progress", current, total, message)

    @Slot(object)
    def result(self, result) -> None:
        self._deliver("result", result)

    @Slot(str, object)
    def url_result(self, url: str, result) -> None:
        self._deliver("url_result", url, result)

    @Slot(str)
    def error(self, error: str) -> None:
        self._deliver("error", error)

    @Slot(str, str)
    def video_error(self, video_id: str, error: str) -> None:
        self._deliver("video_error", video_id, error)

    @Slot()
    def completed(self) -> None:
        self._deliver("completed")

    @Slot(list)
    def bulk_completed(self, results: list) -> None:
        self._deliver("bulk_completed", results)

    @Slot()
    def _finished(self) -> None:
        self.window._active_download_workers.discard(self.worker)
        if self.window._download_deliveries.get(self.attribute) is self:
            del self.window._download_deliveries[self.attribute]
        if getattr(self.window, self.attribute, None) is self.worker:
            setattr(self.window, self.attribute, None)
        self.worker.deleteLater()
        self.deleteLater()
