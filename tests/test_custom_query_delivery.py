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
