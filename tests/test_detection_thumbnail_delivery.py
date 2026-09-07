"""Detection thumbnails retain their detection generation and project session."""

import os
from pathlib import Path
import subprocess
import sys


def test_thumbnail_delivery_rejects_replaced_detection_and_old_session():
    code = """
import threading
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.project import Project
from ui.workers.detection_thumbnail_delivery import DetectionThumbnailDelivery
app = QCoreApplication([])
class Worker(QThread):
    progress = Signal(int, int)
    thumbnail_ready = Signal(str, str)
    def run(self):
        self.progress.emit(1, 1)
        self.thumbnail_ready.emit('clip', 'thumb.jpg')
window = QObject()
window.project = Project.new()
received = []
owner = threading.get_ident()
window._on_thumbnail_progress = lambda *args: received.append(('progress', threading.get_ident()))
window._on_thumbnail_ready = lambda *args: received.append(('thumb', threading.get_ident()))
def bind():
    guard = SimpleNamespace(session_id=window.project.session.session_id)
    window._active_detection_guard = guard
    window.thumbnail_worker = worker = Worker()
    DetectionThumbnailDelivery(window, worker, guard, lambda: received.append(('done', threading.get_ident())))
    worker.start()
    assert worker.wait(3000)
    return worker
def drain():
    for _ in range(10): app.processEvents()
bind()
window._active_detection_guard = object()
drain()
assert not received and window.thumbnail_worker is None
bind()
window.project.clear()
drain()
assert not received
bind(); drain()
assert received == [('progress', owner), ('thumb', owner), ('done', owner)]
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
