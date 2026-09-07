"""GUI transcription observes session jobs without claiming durable saves."""

from unittest.mock import Mock

import pytest

from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project
from ui.workers.transcription_worker import TranscriptionWorker


@pytest.fixture
def worker(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 1)
    monkeypatch.setattr(
        "core.binary_resolver.find_binary", lambda *_: "/usr/bin/ffmpeg"
    )
    monkeypatch.setattr(
        "core.transcription_storage.validate_transcription_disk_space", lambda *_: None
    )
    monkeypatch.setattr("core.transcription.get_model", lambda *_: object())
    worker = TranscriptionWorker(
        project.clips,
        project.sources[0],
        backend="faster-whisper",
        model_cache_dir=tmp_path,
        project=project,
    )
    return worker


def test_session_job_preserves_word_results_and_closes(worker, monkeypatch):
    segment = TranscriptSegment(
        0, 1, "hello", 0.8, [WordTimestamp(0, 1, "hello", 0.9)], "en"
    )
    monkeypatch.setattr("core.transcription.transcribe_clip", lambda **_: [segment])
    completed, delivered, started = [], [], []
    worker.transcription_completed.connect(lambda: completed.append(True))
    worker.transcript_ready.connect(
        lambda cid, segments: delivered.append((cid, segments))
    )
    worker.job_started.connect(
        lambda task, persistence: started.append((task, persistence))
    )
    worker.run()
    assert worker.task_id and worker.job_status == "completed"
    assert started == [(worker.task_id, "session_only")]
    assert worker.operation.session_id
    assert worker.operation.persistence == "session_only"
    assert completed == [True]
    assert len(delivered) == 1
    assert worker.result[0].segments[0].to_dict() == segment.to_dict()
    assert worker._runtime is None


def test_cancelled_before_start_records_unprocessed_targets(worker, monkeypatch):
    preload = Mock()
    monkeypatch.setattr("core.transcription.get_model", preload)
    worker.cancel()
    worker.run()
    preload.assert_not_called()
    assert worker.job_status == "cancelled"
    assert len(worker.result) == 1
    assert worker.result[0].status == "unprocessed"
    assert worker.result[0].code == "cancelled"
    assert worker._runtime is None


def test_cancel_during_model_load_does_not_start_asr(worker, monkeypatch):
    asr = Mock()
    monkeypatch.setattr("core.transcription.get_model", lambda *_: worker.cancel())
    monkeypatch.setattr("core.transcription.transcribe_clip", asr)
    completed = []
    worker.transcription_completed.connect(lambda: completed.append(True))
    worker.run()
    assert worker.job_status == "cancelled"
    assert completed == [True]
    asr.assert_not_called()
    assert worker._runtime is None


def test_preflight_failure_is_failed_job_with_one_completion(worker, monkeypatch):
    monkeypatch.setattr("core.binary_resolver.find_binary", lambda *_: None)
    completed, errors = [], []
    worker.transcription_completed.connect(lambda: completed.append(True))
    worker.error.connect(errors.append)
    worker.run()
    assert worker.job_status == "failed"
    assert completed == [True]
    assert len(errors) == 1 and "FFmpeg" in errors[0]
    assert worker._runtime is None


@pytest.mark.parametrize("saved", [False, True])
def test_real_job_delivers_on_project_owner_thread(saved):
    import os
    import subprocess
    import sys

    code = r"""
import tempfile, threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from tests.test_spine_analyze import _build_project
from core.project import Project
from ui.workers.transcription_worker import TranscriptionWorker
from ui.workers.transcription_delivery import TranscriptionDelivery
app = QCoreApplication([])
window = QObject()
owner = threading.get_ident()
calls = []
window.status_bar = Mock()
window._on_transcription_progress = Mock()
window._on_transcription_error = Mock()
window._on_transcript_ready = lambda *args: calls.append(threading.get_ident())
with tempfile.TemporaryDirectory() as directory:
    window.project = _build_project(Path(directory), 1)
    if SAVED:
        assert window.project.save(Path(directory) / 'project.json')
    worker = TranscriptionWorker(window.project.clips, window.project.sources[0], backend='faster-whisper', project=window.project)
    window.transcription_worker = worker
    delivery = TranscriptionDelivery(window, worker)
    with patch.object(worker, '_prepare', return_value=True), patch('core.transcription.transcribe_clip', return_value=[]), patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=Path(directory))):
        worker.start()
        assert worker.wait(5000)
    assert worker.task_id and worker.job_status == 'completed'
    assert worker._runtime is None
    assert window.project.clips[0].transcript is None
    app.processEvents()
    assert calls == [owner]
    assert window.project.clips[0].transcript == []
    window._on_transcription_error.assert_not_called()
    assert any('unsaved' in call.args[0] for call in window.status_bar.showMessage.call_args_list)
    if SAVED:
        assert len(window.project.metadata.job_results) == 1
        assert Project.load(window.project.path).clips[0].transcript is None
        assert window.project.save()
        assert Project.load(window.project.path).clips[0].transcript == []
"""
    result = subprocess.run(
        [sys.executable, "-c", f"SAVED={saved!r}\n" + code],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
