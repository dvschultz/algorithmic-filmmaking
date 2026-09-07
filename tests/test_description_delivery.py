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
    for mode in ('current', 'worker', 'session', 'edit', 'cancel', 'receipt', 'payload', 'save_as'):
        window.project = project_with_thumbnails(Path(directory), 1)
        target = window.project.clips[0]
        tasks = DescriptionWorker(window.project.clips, sources=window.project.sources_by_id, tier='cloud').tasks
        worker = Worker(tasks)
        if mode in ('receipt', 'payload', 'save_as'):
            from dataclasses import asdict
            from hashlib import sha256
            import json
            from types import SimpleNamespace
            from core.jobs.gui_results import GuiResultReceipt
            from core.operations.description import DescriptionOutcome
            window.project.save(Path(directory) / 'project.json')
            payload = json.dumps(asdict(DescriptionOutcome('c-0', 'succeeded', 'Generated', 'model')))
            worker.cache = SimpleNamespace(path=window.project.path.resolve(), results={'c-0': GuiResultReceipt('a' * 64, sha256(payload.encode()).hexdigest(), payload)})
            if mode == 'payload': worker.cache.results['c-0'] = GuiResultReceipt('a' * 64, 'b' * 64, payload.replace('Generated', 'Other'))
            if mode == 'save_as': window.project.save(Path(directory) / 'copy.json')
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
        if mode in ('current', 'receipt'):
            assert target.description == 'Generated'
            assert callbacks == [owner]
            assert bool(window.project.metadata.job_results) == (mode == 'receipt')
        else:
            assert not callbacks
            assert target.description == ('User edit' if mode == 'edit' else None)
        if mode in ('edit', 'payload', 'save_as'):
            window._on_description_error.assert_called_once()
            assert not window.project.metadata.job_results
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
