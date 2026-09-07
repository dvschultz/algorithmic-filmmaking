"""GUI alignment observes session jobs and retains typed word outcomes."""

from unittest.mock import Mock

import pytest

from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project
from ui.workers.forced_alignment_worker import ForcedAlignmentWorker


@pytest.fixture
def worker(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 1)
    project.clips[0].transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda *_: (True, [])
    )

    def extract(*args, **kwargs):
        path = tmp_path / "audio.wav"
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", extract)
    monkeypatch.setattr(
        "core.analysis.alignment.align_words",
        lambda *_, **kw: [WordTimestamp(0, 1, "hello", 0.9)],
    )
    return ForcedAlignmentWorker(project.clips, project.sources_by_id, project=project)


def test_session_job_retains_words_and_closes(worker):
    completed, delivered = [], []
    worker.alignment_completed.connect(lambda: completed.append(True))
    worker.clip_aligned.connect(lambda cid, words: delivered.append((cid, words)))
    worker.run()
    assert worker.task_id and worker.job_status == "completed"
    assert worker.operation.persistence == "session_only"
    assert worker.operation.session_id
    assert worker.result[0].words == (WordTimestamp(0, 1, "hello", 0.9),)
    assert len(delivered) == 1
    assert completed == [True]
    assert worker._runtime is None


def test_completion_follows_session_store_close(worker, monkeypatch):
    from core.jobs import JobRuntime

    runtime = JobRuntime.for_session(max_workers=1)
    monkeypatch.setattr(JobRuntime, "for_session", lambda **_: runtime)
    closed_at_completion = []

    def completed():
        with pytest.raises(RuntimeError, match="closed"):
            runtime.store.get(worker.task_id)
        closed_at_completion.append(True)

    worker.alignment_completed.connect(completed)
    worker.run()
    assert closed_at_completion == [True]


def test_cancel_before_start_skips_dependency_work(worker, monkeypatch):
    ready = Mock()
    monkeypatch.setattr("core.feature_registry.check_feature_ready", ready)
    worker.cancel()
    worker.run()
    ready.assert_not_called()
    assert worker.job_status == "cancelled"
    assert worker.result[0].status == "unprocessed"
    assert worker._runtime is None


def test_cancel_during_preflight_does_not_install(worker, monkeypatch):
    def ready(*args):
        worker.cancel()
        return False, ["missing"]

    install = Mock()
    monkeypatch.setattr("core.feature_registry.check_feature_ready", ready)
    monkeypatch.setattr("core.feature_registry.install_for_feature", install)
    worker.run()
    install.assert_not_called()
    assert worker.job_status == "cancelled"
    assert worker.result[0].code == "cancelled"


def test_preflight_failure_records_failed_job_once(worker, monkeypatch):
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        Mock(side_effect=RuntimeError("unavailable")),
    )
    completed, errors = [], []
    worker.alignment_completed.connect(lambda: completed.append(True))
    worker.error.connect(errors.append)
    worker.run()
    assert worker.job_status == "failed"
    assert completed == [True]
    assert len(errors) == 1 and "unavailable" in errors[0]
    assert worker._runtime is None


@pytest.mark.parametrize("saved", [False, True])
def test_real_job_applies_on_project_owner_thread(saved):
    import os
    import subprocess
    import sys

    code = r"""
import tempfile, threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import QCoreApplication, QObject
from core.transcription_models import TranscriptSegment
from core.project import Project
from core.jobs import JobStore
from tests.test_spine_analyze import _build_project
from ui.workers.forced_alignment_worker import ForcedAlignmentWorker
from ui.workers.alignment_delivery import AlignmentDelivery
app = QCoreApplication([])
tab = QObject()
owner = threading.get_ident()
calls = []
tab._alignment_generation = 1
tab._on_alignment_progress = Mock()
tab._on_alignment_error = Mock()
tab._on_alignment_completed = Mock()
tab._on_alignment_thread_finished = Mock()
tab._on_clip_aligned = lambda *args: calls.append(threading.get_ident())
with tempfile.TemporaryDirectory() as directory:
    tab.project = _build_project(Path(directory), 1)
    tab._project_provider = lambda: tab.project
    tab.project.clips[0].transcript = [TranscriptSegment(0, 1, 'hello', language='en')]
    if SAVED:
        assert tab.project.save(Path(directory) / 'project.json')
        tab.project = Project.load(tab.project.path, retain_writer=True)
    worker = ForcedAlignmentWorker(tab.project.clips, tab.project.sources_by_id, project=tab.project)
    tab._forced_alignment_worker = worker
    delivery = AlignmentDelivery(tab, worker, tab.project)
    wav = Path(directory) / 'audio.wav'
    wav.write_bytes(b'fake')
    with patch.object(worker, '_prepare', return_value=True), patch('core.analysis.alignment.extract_audio_to_wav', return_value=wav), patch('core.analysis.alignment.align_words', return_value=[]), patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=Path(directory))):
        worker.start()
        assert worker.wait(5000)
    assert worker.task_id and worker.job_status == 'completed'
    assert worker._runtime is None
    assert tab.project.clips[0].transcript[0].words is None
    app.processEvents()
    assert calls == [owner]
    assert tab.project.clips[0].transcript[0].words == []
    tab._on_alignment_error.assert_not_called()
    tab._on_alignment_completed.assert_called_once()
    if SAVED:
        history = JobStore(Path(directory) / 'jobs.db')
        row = history.get(worker.task_id)
        assert row.status == 'completed' and row.project_path == str(tab.project.path)
        assert row.result['publication'] == 'explicit_project_save'
        assert worker.operation.publication == 'owner_thread'
        assert len(tab.project.metadata.job_results) == 1
        assert Project.load(tab.project.path).clips[0].transcript[0].words is None
        assert tab.project.save()
        assert Project.load(tab.project.path).clips[0].transcript[0].words == []
        tab.project.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", f"SAVED={saved!r}\n" + code],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
