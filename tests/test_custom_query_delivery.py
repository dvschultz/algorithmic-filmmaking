"""Real Qt queued custom-query replies remain bound to their launch context."""

import os
import subprocess
import sys


def test_queued_query_delivery_guards():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_description_operations import project_with_thumbnails
from core.operations.custom_query import CustomQueryTask
from ui.workers.custom_query_delivery import CustomQueryDelivery
app = QCoreApplication([])
class Worker(QThread):
    query_result_ready = Signal(str, str, bool, float, str)
    def __init__(self, task):
        super().__init__(); self.tasks = (task,); self.cancelled = False
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.query_result_ready.emit('c-0', 'person', True, .9, 'model')
        self.query_result_ready.emit('c-0', 'person', True, .9, 'model')
window = QObject()
with TemporaryDirectory() as directory:
    for mode in ('current', 'worker', 'session', 'project', 'edit', 'cancel', 'pipeline', 'reply', 'receipt', 'payload', 'save_as'):
        window._analysis_run = None
        window.project = project_with_thumbnails(Path(directory), 1)
        clip = window.project.clips[0]
        worker = Worker(CustomQueryTask(clip.id, clip.thumbnail_path, 'person'))
        if mode in ('receipt', 'payload', 'save_as'):
            from dataclasses import asdict
            import json
            from core.jobs.gui_results import GuiResultReceipt
            from core.operations.custom_query import CustomQueryOutcome
            window.project.save(Path(directory) / 'project.json')
            payload = json.dumps(asdict(CustomQueryOutcome('c-0', 'person', 'succeeded', True, .9, 'model')))
            worker.cache = SimpleNamespace(path=window.project.path.resolve(), results={'c-0':GuiResultReceipt('a'*64, 'b'*64, payload)})
            if mode == 'payload': worker.cache.results['c-0'] = GuiResultReceipt('a'*64, 'b'*64, payload.replace('person','other'))
            if mode == 'save_as': window.project.save(Path(directory) / 'copy.json')
        window.custom_query_worker = worker
        window._on_custom_query_ready = Mock()
        window._on_custom_query_error = Mock()
        reply = SimpleNamespace(is_current=lambda _: True)
        window._dispatch_gui_reply = reply
        delivery = CustomQueryDelivery(window, worker)
        if mode == 'worker': window.custom_query_worker = object()
        if mode == 'session': window.project.clear()
        if mode == 'project': window.project = project_with_thumbnails(Path(directory), 1)
        if mode == 'edit': clip.custom_queries = [{'query':'manual'}]
        if mode == 'cancel': worker.cancelled = True
        if mode == 'pipeline': window._analysis_run = object()
        if mode == 'reply': reply.is_current = lambda _: False
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        if mode in ('current', 'receipt'):
            assert len(clip.custom_queries) == 1
            window._on_custom_query_ready.assert_called_once()
            assert bool(window.project.metadata.job_results) == (mode == 'receipt')
        else:
            window._on_custom_query_ready.assert_not_called()
            assert clip.custom_queries == ([{'query':'manual'}] if mode == 'edit' else None)
        if mode in ('edit', 'payload', 'save_as'): window._on_custom_query_error.assert_called_once()
        else: window._on_custom_query_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_queued_query_records_reuse_and_failures():
    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.operations.custom_query import CustomQueryApplication, CustomQueryOptions, custom_query_task, custom_query_record_key, run_custom_query
from tests.test_description_operations import project_with_thumbnails
from ui.workers.custom_query_delivery import CustomQueryDelivery
app = QCoreApplication([])
class Worker(QThread):
    outcome_ready = Signal(object)
    def is_cancelled(self): return False
    def run(self):
        self.outcome_ready.emit(self.outcome)
        self.outcome_ready.emit(self.outcome)
with TemporaryDirectory() as directory:
    for mode in ('success', 'reuse', 'failure', 'tampered', 'edit'):
        window = QObject()
        project = window.project = project_with_thumbnails(Path(directory), 1)
        options = CustomQueryOptions('cloud', 'model')
        task = custom_query_task(project.clips[0], project.sources[0], 'person')
        with patch('core.analysis.custom_query.evaluate_custom_query', return_value=(False, 0.0, 'model')):
            first = run_custom_query((task,), options)[0]
        assert CustomQueryApplication(project, (task,), options).apply(project, first)
        assert project.save(Path(directory) / 'project.json')
        task = custom_query_task(project.clips[0], project.sources[0], 'person', skip_existing=mode != 'failure')
        if mode == 'failure':
            with patch('core.analysis.custom_query.evaluate_custom_query', side_effect=ValueError('Invalid answer')):
                outcome = run_custom_query((task,), options)[0]
        elif mode == 'success':
            task = custom_query_task(project.clips[0], project.sources[0], 'person')
            with patch('core.analysis.custom_query.evaluate_custom_query', return_value=(True, 0.9, 'model')):
                outcome = run_custom_query((task,), options)[0]
        else:
            outcome = run_custom_query((task,), options)[0]
        worker = Worker()
        worker.tasks = (task,)
        worker.options = options
        worker.outcome = outcome
        worker.cache = SimpleNamespace(path=project.path.resolve(), results={}, transient_outcomes={task.clip_id: asdict(outcome)})
        if mode == 'success':
            import json
            from core.jobs.gui_results import GuiResultReceipt
            worker.cache.results[task.clip_id] = GuiResultReceipt('a'*64, 'b'*64, json.dumps(asdict(outcome)))
        if mode == 'tampered': worker.cache.transient_outcomes[task.clip_id]['model'] = 'other'
        window.custom_query_worker = worker
        window._on_custom_query_ready = Mock()
        window._on_custom_query_error = Mock()
        delivery = CustomQueryDelivery(window, worker)
        if mode == 'edit': project.sources[0].fps += 1
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        clip = project.clips[0]
        assert len(clip.custom_queries) == (2 if mode == 'success' else 1)
        assert bool(project.metadata.job_results) == (mode == 'success')
        if mode in ('success', 'reuse'): window._on_custom_query_ready.assert_called_once()
        else: window._on_custom_query_ready.assert_not_called()
        if mode == 'failure': assert clip.analysis_records[custom_query_record_key('person')].state == 'failed'
        if mode in ('tampered', 'edit'): window._on_custom_query_error.assert_called_once()
        else: window._on_custom_query_error.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
