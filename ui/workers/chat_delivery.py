"""Deliver chat signals only to their requesting conversation and project."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Slot

from ui.workers.gui_tool_cancellation import cancel_gui_tool_work


def stop_chat_workers(window: Any, *, wait: bool = False) -> None:
    """Invalidate delivery immediately; retain threads until cooperative completion."""
    # A timeout may already have closed the mailbox while its queued signal
    # is still waiting. Retire captured native work before detaching the chat.
    cancel_gui_tool_work(window)
    window._chat_worker = None
    workers = tuple(getattr(window, "_active_chat_workers", ()))
    for worker in workers:
        worker.stop()
    if wait:
        for worker in workers:
            worker.wait()


class ChatDelivery(QObject):
    """Owner-thread relay whose lifetime includes its worker's shutdown."""

    def __init__(
        self, window: Any, worker: QThread, handlers: dict[str, Callable[..., None]]
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.handlers = handlers
        self.session_id = window.project.session.session_id
        window._active_chat_workers.add(worker)
        worker.setParent(self)
        worker.finished.connect(self._finished)

    def _deliver(self, name: str, *args) -> None:
        if (
            self.window._chat_worker is self.worker
            and self.window.project.session.session_id == self.session_id
        ):
            self.handlers[name](*args)

    @Slot()
    def _finished(self) -> None:
        self.window._active_chat_workers.discard(self.worker)
        if self.window._chat_worker is self.worker:
            self.window._chat_worker = None
        self.worker.deleteLater()
        self.deleteLater()

    @Slot(str)
    def text_chunk(self, value: str) -> None:
        self._deliver("text_chunk", value)

    @Slot()
    def clear_current_bubble(self) -> None:
        self._deliver("clear_current_bubble")

    @Slot(str, dict)
    def tool_called(self, name: str, args: dict) -> None:
        self._deliver("tool_called", name, args)

    @Slot(str, dict, bool)
    def tool_result(self, name: str, result: dict, success: bool) -> None:
        self._deliver("tool_result", name, result, success)

    @Slot(str, dict, str)
    def gui_tool_requested(self, name: str, args: dict, call_id: str) -> None:
        # Cancel may arrive after emission but before queued owner-thread delivery.
        if not getattr(self.worker, "_stop_requested", False):
            self._deliver("gui_tool_requested", name, args, call_id)

    @Slot(str, str)
    def gui_tool_cancelled(self, name: str, token: str) -> None:
        self._deliver("gui_tool_cancelled", name, token)

    @Slot(str, list)
    def complete(self, response: str, history: list) -> None:
        self._deliver("complete", response, history)

    @Slot(str)
    def error(self, message: str) -> None:
        self._deliver("error", message)

    @Slot(str, str)
    def auth_failed(self, provider: str, message: str) -> None:
        self._deliver("auth_failed", provider, message)

    @Slot(str, int, int)
    def workflow_progress(self, name: str, current: int, total: int) -> None:
        self._deliver("workflow_progress", name, current, total)

    @Slot(str, list)
    def youtube_search_completed(self, query: str, videos: list) -> None:
        self._deliver("youtube_search_completed", query, videos)

    @Slot(str, dict)
    def video_download_completed(self, url: str, result: dict) -> None:
        self._deliver("video_download_completed", url, result)
