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
from core.jobs.gui_results import GuiResultReceipt
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
    for mode in ('current', 'worker', 'project', 'edit', 'expired', 'location', 'payload'):
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
        if mode in ('location', 'payload'):
            window.project.path = Path(directory) / 'original.json'
            worker.cache = SimpleNamespace(path=window.project.path.resolve(), results={'c-0': GuiResultReceipt('0' * 64, '0' * 64, '{}')})
        window.transcription_worker = worker
        delivery = TranscriptionDelivery(window, worker, reply=reply)
        if mode == 'worker': window.transcription_worker = object()
        if mode == 'project': window.project.clear()
        if mode == 'edit': original.start_frame += 1
        if mode == 'expired': window._chat_worker.is_gui_tool_pending = lambda *_: False
        if mode == 'location': window.project.path = Path(directory) / 'new.json'
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
        if mode in ('location', 'payload'):
            assert window._on_transcription_error.call_count == 2
            assert not window.project.metadata.job_results
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


def test_verified_transient_delivery_rejects_cancelled_or_modified_outcomes():
    code = r"""
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject, Signal
from core.operations.transcription import TranscriptionApplication, TranscriptionOptions, run_transcription
from core.operations.transcription_records import transcription_task
from tests.test_spine_analyze import _build_project
from ui.workers.transcription_delivery import TranscriptionDelivery
app = QCoreApplication([])
class Worker(QObject):
    finished = Signal()
    progress = Signal(int, int)
    status = Signal(str)
    error = Signal(str)
    job_started = Signal(str, str)
    outcome_ready = Signal(object)
    def is_cancelled(self): return self.cancelled
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    for mode in ('reuse', 'failure', 'tampered', 'cancelled', 'changed'):
        project = _build_project(path, 1)
        project.path = path / 'project.json'
        options = TranscriptionOptions(backend='faster-whisper')
        with patch('core.transcription._has_audio_stream', return_value=True), patch('core.transcription.transcribe_clip', return_value=[]):
            first = transcription_task(project.clips[0], project.sources[0])
            result = run_transcription((first,), options)[0]
            assert TranscriptionApplication(project, (first,), options).apply(project, result)
            task = transcription_task(project.clips[0], project.sources[0], skip_existing=mode != 'failure')
            with patch('core.transcription.transcribe_clip', side_effect=RuntimeError('offline')):
                outcome = run_transcription((task,), options)[0]
        worker = Worker(); worker.tasks = (task,); worker._options = options; worker.cancelled = mode == 'cancelled'
        worker.cache = SimpleNamespace(path=project.path.resolve(), results={}, transient_outcomes={outcome.clip_id: asdict(outcome)})
        window = QObject(); window.project = project; window.transcription_worker = worker
        window.status_bar = Mock(); window._on_transcription_progress = Mock(); window._on_transcription_error = Mock(); window._on_transcript_ready = Mock()
        delivery = TranscriptionDelivery(window, worker)
        if mode == 'tampered': outcome = replace(outcome, record_json='{}')
        if mode == 'changed': project.clips[0].end_frame += 1
        worker.outcome_ready.emit(outcome); worker.outcome_ready.emit(outcome)
        if mode == 'reuse':
            window._on_transcript_ready.assert_called_once()
            window._on_transcription_error.assert_not_called()
        elif mode == 'failure':
            assert project.clips[0].analysis_records['transcribe'].state == 'failed'
            assert project.clips[0].transcript == []
            window._on_transcript_ready.assert_not_called()
        else:
            window._on_transcript_ready.assert_not_called()
            assert project.clips[0].analysis_records['transcribe'].state == 'succeeded'
        assert not project.metadata.job_results
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
