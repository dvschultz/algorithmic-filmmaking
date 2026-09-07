"""Real queued transcription signals cannot mutate a replaced request."""

import os
import subprocess
import sys


def test_transcription_delivery_guards_model_and_ui():
    code = r"""
import tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from tests.test_spine_analyze import _build_project
from core.operations.transcription import snapshot_tasks
from ui.workers.transcription_delivery import TranscriptionDelivery
from ui.workers.gui_tool_reply import GuiToolReply
app = QCoreApplication([])
owner_thread = threading.get_ident()
class Worker(QThread):
    job_started = Signal(str, str)
    progress = Signal(int, int)
    status = Signal(str)
    error = Signal(str)
    transcript_ready = Signal(str, list)
    def __init__(self, tasks):
        super().__init__(); self.tasks = tasks
    def run(self):
        self.job_started.emit('task-id', 'session_only')
        self.progress.emit(1, 1); self.status.emit('status'); self.error.emit('error')
        self.transcript_ready.emit('c-0', [])
        self.transcript_ready.emit('c-0', [])
window = QObject()
with tempfile.TemporaryDirectory() as directory:
    for mode in ('current', 'worker', 'project', 'edit', 'expired'):
        window.project = _build_project(Path(directory), 1)
        original = window.project.clips[0]
        window._chat_worker = SimpleNamespace(_stop_requested=False, is_gui_tool_pending=lambda *_: True)
        reply = GuiToolReply.capture(window, 'transcribe', 'token')
        callbacks = []
        window._on_transcript_ready = lambda *args: callbacks.append(threading.get_ident())
        window._on_transcription_progress = Mock()
        window._on_transcription_error = Mock()
        window.status_bar = Mock()
        worker = Worker(snapshot_tasks(window.project.clips, window.project.sources_by_id))
        window.transcription_worker = worker
        delivery = TranscriptionDelivery(window, worker, reply=reply)
        if mode == 'worker': window.transcription_worker = object()
        if mode == 'project': window.project.clear()
        if mode == 'edit': original.start_frame += 1
        if mode == 'expired': window._chat_worker.is_gui_tool_pending = lambda *_: False
        worker.start(); assert worker.wait(5000)
        app.processEvents()
        if mode == 'current':
            assert callbacks == [owner_thread]
            assert original.transcript == []
            window._on_transcription_progress.assert_called_once_with(1, 1)
        else:
            assert not callbacks and original.transcript is None
        if mode == 'edit':
            assert window._on_transcription_error.call_count == 2
            assert 'discarded' in window._on_transcription_error.call_args.args[0]
        if mode in ('worker', 'project', 'expired'):
            window._on_transcription_progress.assert_not_called()
            window._on_transcription_error.assert_not_called()
            window.status_bar.showMessage.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
