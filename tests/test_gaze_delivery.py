"""Use real queued Qt signals to test gaze publication context."""

import os
import subprocess
import sys


def test_queued_gaze_delivery():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.gaze import GazeTask
from ui.workers.gaze_delivery import GazeDelivery
from tests.test_description_operations import project_with_thumbnails
app = QCoreApplication([])
class Worker(QThread):
    gaze_ready = Signal(str, float, float, str)
    def __init__(self, task):
        super().__init__(); self.tasks=(task,); self.cancelled=False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.gaze_ready.emit('c-0', 2., 1., 'at_camera')
        self.gaze_ready.emit('c-0', 2., 1., 'at_camera')
window = QObject()
with TemporaryDirectory() as root:
    for mode in ('current', 'worker', 'project', 'session', 'edit', 'cancel', 'pipeline', 'reply'):
        project=project_with_thumbnails(Path(root),1)
        clip=project.clips[0]; source=project.sources[0]
        worker=Worker(GazeTask(clip.id,clip.source_id,source.file_path,clip.start_frame,clip.end_frame,source.fps))
        window.project=project; window._gaze_worker=worker; window._analysis_run=None
        window._on_gaze_error=Mock(); window._on_gaze_ready=Mock()
        reply=SimpleNamespace(is_current=lambda _: True); window._dispatch_gui_reply=reply
        delivery=GazeDelivery(window,worker,pipeline=True)
        if mode=='worker': window._gaze_worker=object()
        if mode=='project': window.project=project_with_thumbnails(Path(root),1)
        if mode=='session': project.clear()
        if mode=='edit': clip.gaze_category='looking_left'
        if mode=='cancel': worker.cancelled=True
        if mode=='pipeline': window._analysis_run=object()
        if mode=='reply': reply.is_current=lambda _:False
        worker.start(); assert worker.wait(5000); app.processEvents()
        assert clip.gaze_category == ('at_camera' if mode=='current' else 'looking_left' if mode=='edit' else None)
        if mode=='current': window._on_gaze_ready.assert_called_once()
        else: window._on_gaze_ready.assert_not_called()
        if mode=='edit': window._on_gaze_error.assert_called_once()
        else: window._on_gaze_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
