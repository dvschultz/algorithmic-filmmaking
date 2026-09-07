"""Chat deliveries belong to one conversation and project session."""

import os
from pathlib import Path
import subprocess
import sys


def test_chat_delivery_ownership_and_lifetime():
    code = """
import threading, time
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.project import Project
from ui.workers.chat_delivery import ChatDelivery, stop_chat_workers
app = QCoreApplication([])
class Worker(QThread):
    text_chunk = Signal(str)
    video_download_completed = Signal(str, dict)
    gui_tool_requested = Signal(str, dict, str)
    gui_tool_cancelled = Signal(str, str)
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.stopped = False
    def stop(self):
        self.stopped = True
        self._stop_requested = True
    def run(self):
        assert self.release.wait(3)
        self.text_chunk.emit('text')
        self.video_download_completed.emit('url', {'file_path': 'old.mp4'})
        self.gui_tool_requested.emit('edit', {}, 'call-id')
        self.gui_tool_cancelled.emit('edit', 'call-id')
window = QObject()
window.project = Project.new(name='test')
window._active_chat_workers = set()
window._chat_worker = None
cancelled = []
window._on_gui_tool_cancelled = cancelled.append
received = []
owner = threading.get_ident()
def bind(worker):
    stop_chat_workers(window)
    window._chat_worker = worker
    relay = ChatDelivery(window, worker, {
        'text_chunk': lambda value: received.append((value, threading.get_ident())),
        'video_download_completed': lambda *args: received.append(args),
        'gui_tool_requested': lambda *args: received.append(args),
        'gui_tool_cancelled': lambda *args: cancelled.append(args),
    })
    worker.text_chunk.connect(relay.text_chunk)
    worker.video_download_completed.connect(relay.video_download_completed)
    worker.gui_tool_requested.connect(relay.gui_tool_requested)
    worker.gui_tool_cancelled.connect(relay.gui_tool_cancelled)
    worker.start()
def drain(worker):
    worker.release.set()
    end = time.monotonic() + 5
    while worker in window._active_chat_workers and time.monotonic() < end:
        app.processEvents(); time.sleep(.001)
    assert worker not in window._active_chat_workers
old, new = Worker(), Worker()
bind(old); bind(new)
assert old.stopped and old in window._active_chat_workers
drain(old)
assert not received and window._chat_worker is new
assert not cancelled
window.project.clear()
drain(new)
assert not received
current = Worker()
bind(current); drain(current)
assert received == [('text', owner), ('url', {'file_path': 'old.mp4'}),
                    ('edit', {}, 'call-id')]
assert cancelled == [('edit', 'call-id')]
received.clear()
cleared = Worker()
bind(cleared)
from unittest.mock import Mock
from ui.workers.gui_tool_reply import GuiToolReply
native = Mock(gui_tool_reply=GuiToolReply.capture(window, 'download_video', 'native-token'))
window._active_download_workers = {native}
stop_chat_workers(window)
assert cancelled == [('edit', 'call-id')]
native.cancel.assert_called_once()
assert window._chat_worker is None and cleared in window._active_chat_workers
drain(cleared)
assert not received
assert cancelled == [('edit', 'call-id')]
native.cancel.assert_called_once()
closing = Worker()
bind(closing)
closing.release.set()
stop_chat_workers(window, wait=True)
assert not closing.isRunning()
drain(closing)
assert not received
cancelled_worker = Worker()
bind(cancelled_worker)
cancelled_worker.stop()
drain(cancelled_worker)
assert not any(len(item) == 3 for item in received), received
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
