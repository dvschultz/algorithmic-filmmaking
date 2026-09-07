"""Detection thumbnails retain their detection generation and project session."""

import os
from pathlib import Path
import subprocess
import sys


def test_thumbnail_delivery_rejects_replaced_detection_and_old_session():
    code = """
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from models.clip import Clip, Source
from ui.workers.thumbnail_worker import ThumbnailWorker
from ui.workers.detection_thumbnail_delivery import DetectionThumbnailDelivery
app = QCoreApplication([])
temporary = tempfile.TemporaryDirectory()
directory = Path(temporary.name)
video = directory / 'video.mp4'; video.write_bytes(b'video')
def generate(self, **kwargs):
    path = kwargs['output_path']; path.write_bytes(b'thumbnail'); return path
patcher = patch('core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail', generate)
patcher.start()
window = QObject()
window.project = Project.new()
received = []
owner = threading.get_ident()
window._on_thumbnail_progress = lambda *args: received.append(('progress', threading.get_ident()))
window._on_thumbnail_ready = lambda *args: received.append(('thumb', threading.get_ident()))
def bind():
    if not window.project.sources:
        window.project.add_source(Source(id='source', file_path=video, fps=30))
        window.project.add_clips([Clip(id='clip', source_id='source', start_frame=0, end_frame=30)])
    guard = SimpleNamespace(session_id=window.project.session.session_id)
    window._active_detection_guard = guard
    window.thumbnail_worker = worker = ThumbnailWorker(window.project.sources[0], window.project.clips,
        directory, project=window.project)
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
