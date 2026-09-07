"""Queued alignment signals respect model ownership and run identity."""

import os
import subprocess
import sys


def test_alignment_delivery_rejects_stale_runs_and_edits():
    code = r"""
import tempfile, threading
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace
from core.jobs.gui_alignment import AlignmentReceipt
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.transcription_models import TranscriptSegment
from core.operations.alignment import snapshot_alignment_tasks
from tests.test_spine_analyze import _build_project
from ui.workers.alignment_delivery import AlignmentDelivery
app = QCoreApplication([])
owner = threading.get_ident()
retained = []
class Worker(QThread):
    progress = Signal(int, int)
    error = Signal(str)
    clip_aligned = Signal(str, list)
    alignment_completed = Signal()
    def run(self):
        self.progress.emit(1, 1)
        self.clip_aligned.emit('c-0', [])
        self.clip_aligned.emit('c-0', [])
        self.alignment_completed.emit()
with tempfile.TemporaryDirectory() as directory:
    for mode in ('current', 'edit', 'project', 'clear', 'worker', 'location', 'payload'):
        tab = QObject()
        tab.project = _build_project(Path(directory), 1)
        clip = tab.project.clips[0]
        clip.transcript = [TranscriptSegment(0, 1, 'hello', language='en')]
        tab._alignment_generation = 1
        tab._project_provider = lambda: tab.project
        tab._on_alignment_progress = Mock()
        tab._on_alignment_error = Mock()
        tab._on_alignment_completed = Mock()
        tab._on_alignment_thread_finished = Mock()
        calls = []
        tab._on_clip_aligned = lambda *args: calls.append(threading.get_ident())
        worker = Worker()
        worker.tasks = snapshot_alignment_tasks(tab.project.clips, tab.project.sources_by_id)
        if mode in ('location', 'payload'):
            tab.project.path = Path(directory) / 'original.json'
            worker.cache = SimpleNamespace(path=tab.project.path.resolve(), results={'c-0': AlignmentReceipt('0' * 64, '0' * 64, '{}')})
        tab._forced_alignment_worker = worker
        delivery = AlignmentDelivery(tab, worker, tab.project)
        retained.append((tab, worker, delivery))
        if mode == 'edit': clip.transcript[0].text = 'changed'
        if mode == 'project': tab.project.clear()
        if mode == 'clear': tab._alignment_generation += 1
        if mode == 'worker': tab._forced_alignment_worker = object()
        if mode == 'location': tab.project.path = Path(directory) / 'new.json'
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        if mode == 'current':
            assert calls == [owner]
            assert clip.transcript[0].words == []
            tab._on_alignment_error.assert_not_called()
        else:
            assert not calls and clip.transcript[0].words is None
        if mode in ('edit', 'payload', 'location'): tab._on_alignment_error.assert_called_once()
        if mode in ('project', 'clear', 'worker'):
            tab._on_alignment_progress.assert_not_called()
            tab._on_alignment_completed.assert_not_called()
        if mode == 'worker': tab._on_alignment_thread_finished.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
