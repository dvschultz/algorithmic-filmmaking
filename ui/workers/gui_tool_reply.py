"""Capture GUI tool reply ownership at dispatch, including eager worker starts."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, Slot


@dataclass(frozen=True)
class GuiToolReply:
    worker: Any
    session_id: str
    name: str
    token: str

    @classmethod
    def capture(cls, window: Any, name: str, token: str) -> "GuiToolReply":
        return cls(window._chat_worker, window.project.session.session_id, name, token)

    def is_current(self, window: Any) -> bool:
        """Check whether GUI work still belongs to the live conversation."""
        return (
            self.worker is not None
            and window._chat_worker is self.worker
            and window.project.session.session_id == self.session_id
            and getattr(self.worker, "_stop_requested", False) is not True
        )

    def send(self, window: Any, result: dict) -> bool:
        """Send only to the original live requester, with captured identity."""
        if not self.is_current(window):
            return False
        return bool(
            self.worker.set_gui_tool_result(
                {**result, "tool_call_id": self.token, "name": self.name}
            )
        )


@contextmanager
def gui_reply_scope(window: Any, reply: GuiToolReply) -> Iterator[None]:
    """Expose the current dispatch to worker constructors, restoring nested calls."""
    previous = getattr(window, "_dispatch_gui_reply", None)
    window._dispatch_gui_reply = reply
    try:
        yield
    finally:
        window._dispatch_gui_reply = previous


class AgentAnalysisCompletion(QObject):
    """Owner-thread callback tied to an analysis worker and its request."""

    def __init__(
        self,
        window: Any,
        worker: QThread,
        attribute: str,
        handler: Callable[..., None],
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.attribute = attribute
        self.handler = handler
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.session_id = window.project.session.session_id
        self._delivered = False
        worker.finished.connect(self._finished)

    @Slot()
    def _finished(self) -> None:
        if getattr(self.window, self.attribute, None) is self.worker:
            setattr(self.window, self.attribute, None)
        self.worker.deleteLater()
        self.deleteLater()

    @Slot()
    def completed(self) -> None:
        if (
            self._delivered
            or self.window.project.session.session_id != self.session_id
            or getattr(self.window, self.attribute, None) is not self.worker
        ):
            return
        self._delivered = True
        self.handler(reply=self.reply)
