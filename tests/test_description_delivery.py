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


def test_queued_verified_reuse_and_failure_delivery():
    code = r"""
import tempfile
from pathlib import Path
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.description import DescriptionApplication, DescriptionOptions, description_task, run_description
from tests.test_description_operations import project_with_thumbnails
from ui.workers.description_delivery import DescriptionDelivery
app = QCoreApplication([])
class Worker(QThread):
    outcome_ready = Signal(object)
    def is_cancelled(self): return False
    def run(self):
        self.outcome_ready.emit(self.outcome)
        self.outcome_ready.emit(self.outcome)
with tempfile.TemporaryDirectory() as directory:
    for mode in ('reuse', 'failure', 'tampered'):
        window = QObject()
        project = window.project = project_with_thumbnails(Path(directory), 1)
        options = DescriptionOptions('cloud', model='test-model', input_mode='frame')
        task = description_task(project.clips[0], project.sources[0])
        with patch('core.analysis.description.describe_frame', return_value=('Generated', 'test-model')):
            first = run_description((task,), options)[0]
        assert DescriptionApplication(project, (task,), options).apply(project, first)
        assert project.save(Path(directory) / 'project.json')
        task = description_task(project.clips[0], project.sources[0])
        if mode == 'failure':
            options = replace(options, prompt='New prompt')
            with patch('core.analysis.description.describe_frame', side_effect=RuntimeError('Invalid input')):
                outcome = run_description((task,), options)[0]
        else:
            outcome = run_description((task,), options)[0]
        worker = Worker()
        worker.tasks = (task,)
        worker.options = options
        worker.outcome = outcome
        worker.cache = SimpleNamespace(path=project.path.resolve(), results={}, transient_outcomes={task.clip_id: asdict(outcome)})
        if mode == 'tampered': worker.cache.transient_outcomes[task.clip_id]['model'] = 'other'
        window.description_worker = worker
        window._on_description_ready = Mock()
        window._on_description_error = Mock()
        delivery = DescriptionDelivery(window, worker)
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        assert project.clips[0].description == 'Generated'
        assert not project.metadata.job_results
        if mode == 'reuse': window._on_description_ready.assert_called_once()
        else: window._on_description_ready.assert_not_called()
        if mode == 'failure': assert project.clips[0].analysis_records['describe'].state == 'failed'
        if mode == 'tampered': window._on_description_error.assert_called_once()
        else: window._on_description_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
