"""Agent-owned legacy decisions, tied to the initiating request and session."""

from typing import Any

from PySide6.QtCore import Slot

from core.operations.colors import ColorApplication
from ui.workers.legacy_reuse_worker import LegacyReuseWorker
from ui.workers.qt_lifetime import RetiringQObject


class AgentLegacyReuse(RetiringQObject):
    def __init__(self, window: Any, operation: str, clip_ids: list[str]) -> None:
        super().__init__(window)
        self.window = window
        self.project = window.project
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.worker = LegacyReuseWorker(self.project, operation, clip_ids, window)
        self.worker.gui_tool_reply = self.reply
        self.result: dict | None = None
        self.worker.result_ready.connect(self.apply_result)
        self.worker.error.connect(self.failed)
        self.worker.finished.connect(self.finished)

    def start(self) -> bool:
        if self.reply is None or not self.reply.is_current(self.window):
            self.worker.deleteLater()
            self.retire()
            return False
        if not hasattr(self.window, "_active_legacy_reuses"):
            self.window._active_legacy_reuses = set()
        self.window._active_legacy_reuses.add(self.worker)
        try:
            self.worker.start()
        except Exception as exc:
            self.failed(str(exc))
            self.finished()
        return True

    @Slot(object)
    def apply_result(self, result: Any) -> None:
        if (
            self.result is not None or self.worker.is_cancelled()
            or self.window.project is not self.project
            or self.reply is None or not self.reply.is_current(self.window)
        ):
            return
        try:
            accepted, failed = [], []
            if isinstance(self.worker.application, ColorApplication):
                outcomes = self.worker.application.apply(result).outcomes
                for outcome in outcomes:
                    if outcome.status == "succeeded":
                        accepted.append(outcome.target_id)
                    else:
                        failed.append({"clip_id": outcome.target_id, "message": outcome.message or outcome.code})
            else:
                for outcome in result:
                    if outcome.status == "succeeded" and self.worker.application.apply(self.project, outcome):
                        accepted.append(outcome.clip_id)
                    else:
                        failed.append({"clip_id": outcome.clip_id, "message": outcome.message or outcome.code or "Target changed"})
            self.result = {"success": not failed, "result": {"accepted": accepted, "failed": failed, "provenance": "unknown", "saved": False}}
            self.window._update_window_title()
        except Exception as exc:
            self.failed(str(exc))

    @Slot(str)
    def failed(self, message: str) -> None:
        self.result = {"success": False, "error": message}

    @Slot()
    def finished(self) -> None:
        if self.reply is not None:
            self.reply.send(self.window, self.result or {"success": False, "error": "Legacy reuse cancelled or target changed"})
        self.window._active_legacy_reuses.discard(self.worker)
        self.worker.deleteLater()
        self.retire()
