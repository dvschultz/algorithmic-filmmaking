"""Exercise real queued Qt delivery against changing launch context."""

import os
import subprocess
import sys


def test_queued_cinematography_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.cinematography import CinematographyTask
from models.cinematography import CinematographyAnalysis
from models.frame import Frame
from ui.workers.cinematography_delivery import CinematographyDelivery
app = QCoreApplication([])
class Worker(QThread):
    clip_completed = Signal(str, object)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.clip_completed.emit('c-0', CinematographyAnalysis(shot_size='CU'))
        self.clip_completed.emit('c-0', CinematographyAnalysis(shot_size='CU'))
window = QObject()
with TemporaryDirectory() as directory:
    for target_type in ('clip', 'frame'):
        for mode in ('current', 'worker', 'session', 'project', 'edit', 'cancel', 'pipeline', 'reply'):
            window._analysis_run = None
            window.project = project_with_thumbnails(Path(directory), 1)
            clip = window.project.clips[0]
            source = window.project.sources[0]
            if target_type == 'frame':
                target = Frame(id=clip.id, file_path=clip.thumbnail_path)
                window.project.add_frames([target])
            else: target = clip
            task = CinematographyTask(clip.id, clip.thumbnail_path,
                source.file_path if source.file_path.exists() else None,
                clip.start_frame, clip.end_frame, source.fps, target_type=target_type)
            worker = Worker(task)
            window.cinematography_worker = worker
            window._on_cinematography_clip_ready = Mock()
            window._on_cinematography_error = Mock()
            reply = SimpleNamespace(is_current=lambda _: True)
            window._dispatch_gui_reply = reply
            delivery = CinematographyDelivery(window, worker, pipeline=True)
            if mode == 'worker': window.cinematography_worker = object()
            if mode == 'session': window.project.clear()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': target.shot_type = 'wide'
            if mode == 'cancel': worker.cancelled = True
            if mode == 'pipeline': window._analysis_run = object()
            if mode == 'reply': reply.is_current = lambda _: False
            worker.start(); assert worker.wait(5000)
            app.processEvents()
            if mode == 'current':
                assert target.shot_type == 'close-up'
                assert target.cinematography.shot_size == 'CU'
                if target_type == 'clip': window._on_cinematography_clip_ready.assert_called_once()
                else:
                    assert clip.cinematography is None
                    window._on_cinematography_clip_ready.assert_not_called()
            else:
                window._on_cinematography_clip_ready.assert_not_called()
                assert target.cinematography is None
            if mode == 'edit': window._on_cinematography_error.assert_called_once()
            else: window._on_cinematography_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
