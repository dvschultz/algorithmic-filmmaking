"""Use real queued Qt signals to test face publication context."""

import os
import subprocess
import sys


def test_queued_face_delivery():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.faces import FaceTask
from ui.workers.face_delivery import FaceDelivery
from tests.test_description_operations import project_with_thumbnails
app = QCoreApplication([])
class Worker(QThread):
    faces_ready = Signal(str, list)
    def __init__(self, task):
        super().__init__(); self.tasks=(task,); self.cancelled=False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.faces_ready.emit('c-0', [])
        self.faces_ready.emit('c-0', [])
window = QObject()
with TemporaryDirectory() as root:
    for mode in ('current', 'worker', 'project', 'session', 'edit', 'cancel', 'pipeline', 'reply'):
        project=project_with_thumbnails(Path(root),1)
        clip=project.clips[0]; source=project.sources[0]
        worker=Worker(FaceTask(clip.id,clip.source_id,source.file_path,clip.start_frame,clip.end_frame,source.fps))
        window.project=project; window.face_detection_worker=worker; window._analysis_run=None
        window._on_face_detection_error=Mock()
        reply=SimpleNamespace(is_current=lambda _: True); window._dispatch_gui_reply=reply
        delivery=FaceDelivery(window,worker,pipeline=True)
        if mode=='worker': window.face_detection_worker=object()
        if mode=='project': window.project=project_with_thumbnails(Path(root),1)
        if mode=='session': project.clear()
        if mode=='edit': clip.face_embeddings=[{'manual': True}]
        if mode=='cancel': worker.cancelled=True
        if mode=='pipeline': window._analysis_run=object()
        if mode=='reply': reply.is_current=lambda _:False
        worker.start(); assert worker.wait(5000); app.processEvents()
        assert clip.face_embeddings == ([] if mode=='current' else [{'manual': True}] if mode=='edit' else None)
        if mode=='edit': window._on_face_detection_error.assert_called_once()
        else: window._on_face_detection_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
