"""In-app log viewing widgets and bridges."""

from __future__ import annotations

import logging
import threading
from collections import deque

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.theme import Spacing


class InAppLogBridge(QObject):
    """Forward Python log records into a bounded in-app buffer."""

    line_emitted = Signal(str)

    def __init__(self, max_lines: int = 1000):
        super().__init__()
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._handler = _InAppLogHandler(self)
        self._installed_on: set[int] = set()

    def install(self, logger_obj: logging.Logger | None = None) -> None:
        """Attach the bridge handler to the provided logger once."""
        target = logger_obj or logging.getLogger()
        if self._handler in target.handlers:
            self._installed_on.add(id(target))
            return

        target.addHandler(self._handler)
        self._installed_on.add(id(target))

    def recent_lines(self) -> list[str]:
        """Return buffered log lines in emission order."""
        with self._lock:
            return list(self._lines)

    def clear_buffer(self) -> None:
        """Clear the in-app buffer without touching on-disk log files."""
        with self._lock:
            self._lines.clear()

    def append_line(self, line: str) -> None:
        """Append a formatted line and emit it to live viewers."""
        with self._lock:
            self._lines.append(line)
        self.line_emitted.emit(line)

    @property
    def handler(self) -> logging.Handler:
        return self._handler


class _InAppLogHandler(logging.Handler):
    """Logging handler that forwards formatted lines into a bridge."""

    def __init__(self, bridge: InAppLogBridge):
        super().__init__(level=logging.DEBUG)
        self._bridge = bridge
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._bridge.append_line(line)


class LogViewerWidget(QWidget):
    """Simple read-only widget for viewing recent live logs."""

    def __init__(self, bridge: InAppLogBridge, parent=None):
        super().__init__(parent)
        self._bridge = bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)

        controls = QHBoxLayout()
        controls.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_view)
        controls.addWidget(clear_btn)
        layout.addLayout(controls)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(1000)
        layout.addWidget(self.text_edit, stretch=1)

        initial_lines = self._bridge.recent_lines()
        if initial_lines:
            self.text_edit.setPlainText("\n".join(initial_lines))

        self._bridge.line_emitted.connect(self._append_line)

    def _append_line(self, line: str) -> None:
        scrollbar = self.text_edit.verticalScrollBar()
        should_autoscroll = scrollbar.value() >= max(0, scrollbar.maximum() - 4)
        self.text_edit.appendPlainText(line)
        if should_autoscroll:
            scrollbar.setValue(scrollbar.maximum())

    def _clear_view(self) -> None:
        self._bridge.clear_buffer()
        self.text_edit.clear()


_bridge_singleton: InAppLogBridge | None = None


def get_in_app_log_bridge() -> InAppLogBridge:
    """Return the shared in-app log bridge singleton."""
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = InAppLogBridge()
    return _bridge_singleton
