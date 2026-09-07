"""Real Qt queued description delivery stays on the project owner thread."""

import os
import subprocess
import sys


def test_description_delivery_guards_queued_results():
    code = r"""
import tempfile, threading
from pathlib import Path
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from ui.workers.description_worker import DescriptionWorker
from ui.workers.description_delivery import DescriptionDelivery
app = QCoreApplication([])
owner = threading.get_ident()
class Worker(QThread):
    description_ready = Signal(str, str, str)
    def __init__(self, tasks):
        super().__init__(); self.tasks = tasks; self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.description_ready.emit('c-0', 'Generated', 'model')
        self.description_ready.emit('c-0', 'Duplicate', 'model')
window = QObject()
with tempfile.TemporaryDirectory() as directory:
    for mode in ('current', 'worker', 'session', 'edit', 'cancel'):
        window.project = project_with_thumbnails(Path(directory), 1)
        target = window.project.clips[0]
        tasks = DescriptionWorker(window.project.clips, sources=window.project.sources_by_id, tier='cloud').tasks
        worker = Worker(tasks)
        window.description_worker = worker
        callbacks = []
        window._on_description_ready = lambda *args: callbacks.append(threading.get_ident())
        window._on_description_error = Mock()
        delivery = DescriptionDelivery(window, worker)
        if mode == 'worker': window.description_worker = object()
        if mode == 'session': window.project.clear()
        if mode == 'edit': target.description = 'User edit'
        if mode == 'cancel': worker.cancelled = True
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        if mode == 'current':
            assert target.description == 'Generated'
            assert callbacks == [owner]
        else:
            assert not callbacks
            assert target.description == ('User edit' if mode == 'edit' else None)
        if mode == 'edit':
            window._on_description_error.assert_called_once()
            assert 'discarded' in window._on_description_error.call_args.args[1]
        else:
            window._on_description_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
