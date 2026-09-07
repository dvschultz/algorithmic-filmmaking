"""Download completion must retain submission ownership across Qt delivery."""

import os
from pathlib import Path
import subprocess
import sys


def test_download_delivery_session_replacement_and_cleanup():
    code = """
import time, threading
from PySide6.QtCore import QCoreApplication, QObject, Signal
from core.project import Project
from ui.workers.base import CancellableWorker
from ui.workers.download_delivery import DownloadDelivery
app = QCoreApplication([])
class Worker(CancellableWorker):
    result = Signal(object)
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
    def run(self):
        assert self.release.wait(3)
        self.result.emit('done')
window = QObject()
window.project = Project.new(name='test')
window._download_deliveries = {}
window._active_download_workers = set()
owner = threading.get_ident()
received = []
def bind(worker):
    window.download_worker = worker
    relay = DownloadDelivery(window, 'download_worker', worker,
        {'result': lambda value: received.append((value, threading.get_ident()))})
    relay.bind_signal(worker.result, 'result')
    worker.start()
    return relay
def spin(predicate):
    end = time.monotonic() + 5
    while not predicate() and time.monotonic() < end:
        app.processEvents(); time.sleep(.001)
    assert predicate()
from ui.workers.gui_tool_reply import GuiToolReply
window._chat_worker = object()
request = GuiToolReply.capture(window, 'download_videos', 'captured')
window._dispatch_gui_reply = request
old, new = Worker(), Worker()
bind(old)
assert old.gui_tool_reply is request
window._dispatch_gui_reply = None
bind(new)
assert new.gui_tool_reply is None
assert old.gui_tool_reply is request
assert old.is_cancelled()
old.release.set()
spin(lambda: old not in window._active_download_workers)
assert not received and window.download_worker is new
# A result queued by the current worker after project reset is also stale.
window.project.clear()
new.release.set()
spin(lambda: not window._active_download_workers)
assert not received and window.download_worker is None
current = Worker()
bind(current); current.release.set()
spin(lambda: not window._active_download_workers)
assert received == [('done', owner)]
# Normal reset must not terminate a download while it can own receipt writes.
from types import SimpleNamespace, MethodType
from unittest.mock import Mock
from ui.main_window import MainWindow
worker = Mock()
worker.isRunning.return_value = True
worker.wait.return_value = False
window = SimpleNamespace(_download_deliveries={'download_worker': object()},
    _active_download_workers={worker}, _source_import_queue=Mock(),
    _chat_worker=None, download_worker=worker)
window._cancel_download_workers = MethodType(MainWindow._cancel_download_workers, window)
window._stop_worker_safely = MethodType(MainWindow._stop_worker_safely, window)
MainWindow._stop_all_workers(window)
worker.cancel.assert_called_once()
worker.wait.assert_not_called()
worker.terminate.assert_not_called()
assert window._active_download_workers == {worker}
assert not window._download_deliveries
MainWindow._cancel_download_workers(window, wait=True)
worker.wait.assert_called_once_with()
worker.terminate.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
