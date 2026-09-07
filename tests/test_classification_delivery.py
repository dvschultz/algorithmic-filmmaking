"""Exercise real queued Qt delivery against changing launch context."""

import os
import subprocess
import sys


def test_queued_classification_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.classification import ClassificationTask
from models.frame import Frame
from ui.workers.classification_delivery import ClassificationDelivery
app = QCoreApplication([])
class Worker(QThread):
    labels_ready = Signal(str, list)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.labels_ready.emit('c-0', [('person', .9)])
        self.labels_ready.emit('c-0', [('person', .9)])
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
            task = ClassificationTask(clip.id, clip.thumbnail_path,
                target_type=target_type)
            worker = Worker(task)
            window.classification_worker = worker
            window._on_classification_error = Mock()
            reply = SimpleNamespace(is_current=lambda _: True)
            window._dispatch_gui_reply = reply
            delivery = ClassificationDelivery(window, worker, pipeline=True)
            if mode == 'worker': window.classification_worker = object()
            if mode == 'session': window.project.clear()
            if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
            if mode == 'edit': target.object_labels = ['user edit']
            if mode == 'cancel': worker.cancelled = True
            if mode == 'pipeline': window._analysis_run = object()
            if mode == 'reply': reply.is_current = lambda _: False
            worker.start(); assert worker.wait(5000)
            app.processEvents()
            if mode == 'current':
                assert target.object_labels == ['person']
                if target_type == 'frame': assert clip.object_labels is None
            else:
                assert target.object_labels == (['user edit'] if mode == 'edit' else None)
            if mode == 'edit': window._on_classification_error.assert_called_once()
            else: window._on_classification_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
