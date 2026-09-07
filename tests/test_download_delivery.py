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
old, new = Worker(), Worker()
bind(old); bind(new)
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
