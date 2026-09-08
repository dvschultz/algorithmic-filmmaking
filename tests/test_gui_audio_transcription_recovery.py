"""Standalone audio uses shared jobs and journals before explicit GUI saves."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.audio_transcription import AudioTranscriptionApplication
from core.project import Project
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.audio_source import AudioSource
from ui.workers.audio_transcribe_worker import AudioTranscribeWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"audio")
    project = Project.new()
    project.add_audio_source(AudioSource(id="audio", file_path=path))
    project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr(
        "core.jobs.gui_audio_transcription.audio_transcription_runtime",
        lambda *args: {"runtime": 1},
    )
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda path: True)
    provider = Mock(
        return_value=[
            TranscriptSegment(
                0.123456789,
                1.23456789,
                "hello",
                -0.123456789,
                words=[WordTimestamp(0.123456789, 1.23456789, "hello", 0.87654321)],
            )
        ]
    )
    monkeypatch.setattr("core.transcription.transcribe_video", provider)
    yield project, provider
    project.close_writer()


def worker_for(project, **kwargs):
    return AudioTranscribeWorker(
        project.audio_sources[0], project=project, backend="faster-whisper", **kwargs
    )


@pytest.mark.parametrize(
    "change", ["id", "status", "segments", "text", "time", "range"]
)
def test_malformed_durable_outcomes_are_rejected(change):
    from core.operations.audio_transcription import AudioTranscriptionOutcome

    data = {
        "audio_source_id": "audio",
        "status": "succeeded",
        "segments": [
            {"start_time": 0.0, "end_time": 1.0, "text": "hello", "confidence": -0.5},
        ],
    }
    if change == "id":
        data["audio_source_id"] = ""
    elif change == "status":
        data["status"] = "unknown"
    elif change == "segments":
        data["segments"] = None
    elif change == "text":
        data["segments"][0]["text"] = None
    elif change == "time":
        data["segments"][0]["start_time"] = True
    else:
        data["segments"][0]["end_time"] = -1
    with pytest.raises(ValueError):
        AudioTranscriptionOutcome.from_dict(data)


@pytest.mark.parametrize("empty", [False, True])
def test_reopen_reuses_inference_and_explicit_save_acknowledges(setup, empty):
    project, provider = setup
    if empty:
        provider.return_value = []
    worker = worker_for(project)
    worker.run()
    assert worker.job_status == "completed", worker.result
    assert worker.result.status == "succeeded"
    assert project.audio_sources[0].transcript is None
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        again = worker_for(reopened)
        application = AudioTranscriptionApplication(reopened, again.task)
        again.run()
        assert again.result == worker.result
        assert provider.call_count == 1
        receipt = again.cache.results[again.task.audio_source_id]
        assert receipt.matches(again.result)
        assert application.apply(reopened, again.result)
        reopened.record_job_result(receipt.result_id, receipt.digest)
        store = JobStore(project.path.parent / "jobs.db")
        assert not store.get_result(receipt.result_id)["committed"]
        assert reopened.save()
        assert store.get_result(receipt.result_id)["committed"]
        expected = [s.to_dict() for s in again.result.segments]
        assert [s.to_dict() for s in reopened.audio_sources[0].transcript] == expected
        store.close()
    finally:
        reopened.close_writer()


def test_unsaved_audio_uses_session_job(setup):
    project, provider = setup
    unsaved = Project.new()
    unsaved.add_audio_source(project.audio_sources[0])
    worker = worker_for(unsaved)
    worker.run()
    assert worker.cache is None
    assert worker.operation.persistence == "session_only"
    assert worker.job_status == "completed"
    assert worker.task_id
    assert provider.call_count == 1
    assert unsaved.audio_sources[0].transcript is None


@pytest.mark.parametrize("empty", [False, True])
def test_verified_saved_audio_reuses_without_inference(setup, empty):
    project, provider = setup
    if empty:
        provider.return_value = []
    worker = worker_for(project)
    application = AudioTranscriptionApplication(project, worker.task, worker.options)
    worker.run()
    assert worker.result.record_json is not None
    assert application.apply(project, worker.result)
    assert project.save()
    again = worker_for(project)
    again.run()
    assert again.result.status == "skipped"
    assert again.result.record_json is not None
    assert again.cache.transient_outcomes[again.task.audio_source_id]
    assert provider.call_count == 1


def test_failed_gui_audio_preserves_existing_transcript_with_failure_record(setup):
    project, provider = setup
    audio = project.audio_sources[0]
    audio.transcript = [TranscriptSegment(0, 1, "existing", -.5)]
    worker = worker_for(project)
    application = AudioTranscriptionApplication(project, worker.task, worker.options)
    provider.side_effect = RuntimeError("provider unavailable")
    worker.run()
    assert worker.result.record_json is not None
    assert worker.cache.transient_outcomes[audio.id]
    assert application.apply(project, worker.result)
    assert audio.transcript[0].text == "existing"
    assert audio.analysis_records["transcribe"].state == "failed"


@pytest.mark.parametrize("change", ["media", "model", "runtime", "prior"])
def test_changed_inputs_do_not_reuse_result(setup, monkeypatch, change):
    project, provider = setup
    worker_for(project).run()
    kwargs = {}
    if change == "media":
        project.audio_sources[0].file_path.write_bytes(b"changed audio")
    if change == "model":
        kwargs["model_name"] = "medium.en"
    if change == "runtime":
        monkeypatch.setattr(
            "core.jobs.gui_audio_transcription.audio_transcription_runtime",
            lambda *args: {"runtime": 2},
        )
    if change == "prior":
        project.audio_sources[0].transcript = []
    worker = worker_for(project, **kwargs)
    worker.run()
    assert worker.result.status == "succeeded"
    assert provider.call_count == 2


@pytest.mark.parametrize("change", ["media", "runtime"])
def test_queued_input_change_rejects_computation(setup, monkeypatch, change):
    project, provider = setup
    worker = worker_for(project)
    if change == "media":
        project.audio_sources[0].file_path.write_bytes(b"changed audio")
    else:
        monkeypatch.setattr(
            "core.jobs.gui_audio_transcription.audio_transcription_runtime",
            lambda *args: {"runtime": 2},
        )
    worker.run()
    provider.assert_not_called()
    assert worker.job_status == "failed"
    assert not worker.cache.results


def test_precancelled_replay_never_delivers(setup):
    project, provider = setup
    worker_for(project).run()
    worker = worker_for(project)
    delivered = []
    worker.transcript_ready.connect(lambda *args: delivered.append(args))
    worker.cancel()
    worker.run()
    assert worker.job_status == "cancelled"
    assert delivered == []
    assert not project.metadata.job_results
    assert provider.call_count == 1


def test_provider_failure_is_not_cached_as_silence(setup):
    project, provider = setup
    provider.side_effect = RuntimeError("provider failed")
    worker = worker_for(project)
    worker.run()
    assert worker.result.status == "failed"
    assert not worker.cache.results
    provider.side_effect = None
    worker_for(project).run()
    assert provider.call_count == 2


@pytest.mark.parametrize("change", ["edit", "save_as", "record"])
def test_changed_saved_output_does_not_acknowledge_receipt(setup, change):
    project, _ = setup
    worker = worker_for(project)
    application = AudioTranscriptionApplication(project, worker.task)
    worker.run()
    assert application.apply(project, worker.result)
    receipt = worker.cache.results[worker.task.audio_source_id]
    project.record_job_result(receipt.result_id, receipt.digest)
    if change == "edit":
        project.audio_sources[0].transcript = []
    if change == "record":
        project.audio_sources[0].analysis_records.clear()
    project.save(
        project.path.parent / "other.json" if change == "save_as" else project.path
    )
    store = JobStore(worker.cache.path.parent / "jobs.db")
    try:
        assert not store.get_result(receipt.result_id)["committed"]
    finally:
        store.close()


def test_corrupt_cached_payload_is_not_recomputed_or_delivered(setup):
    import sqlite3

    project, provider = setup
    worker = worker_for(project)
    worker.run()
    receipt = worker.cache.results[worker.task.audio_source_id]
    database = project.path.parent / "jobs.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE job_results SET payload_json = '{}' WHERE result_id = ?",
            (receipt.result_id,),
        )
    again = worker_for(project)
    again.run()
    assert again.job_status == "failed"
    assert provider.call_count == 1
    assert not again.cache.results


def test_same_clip_id_cannot_checkpoint_audio_result(setup):
    from models.clip import Clip, Source

    project, _ = setup
    worker = worker_for(project)
    worker.run()
    receipt = worker.cache.results[worker.task.audio_source_id]
    source = Source(id="video", file_path=project.audio_sources[0].file_path)
    project.add_source(source)
    clip = Clip(id="audio", source_id=source.id, start_frame=0, end_frame=10)
    clip.transcript = list(worker.result.segments)
    project.add_clips([clip])
    project.record_job_result(receipt.result_id, receipt.digest)
    project.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not store.get_result(receipt.result_id)["committed"]
    finally:
        store.close()


def test_real_queued_audio_recovery_delivery():
    import os
    import subprocess
    import sys

    code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from dataclasses import replace
from threading import get_ident
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from models.audio_source import AudioSource
from core.jobs.store import JobStore
from ui.workers.audio_transcribe_worker import AudioTranscribeWorker
from ui.workers.audio_transcription_delivery import AudioTranscriptionDelivery
from core.operations.audio_transcription import AudioTranscriptionApplication
from core.transcription_models import TranscriptSegment
app = QCoreApplication([])
owner = get_ident()
owners = []
with TemporaryDirectory() as directory:
    for mode in ('current', 'edit', 'save_as', 'payload', 'project', 'cancel', 'reuse', 'failure', 'transient_payload'):
        folder = Path(directory) / mode; folder.mkdir()
        path = folder / 'voice.wav'; path.write_bytes(b'audio')
        window = QObject(); owners.append(window)
        window.project = Project.new()
        audio = AudioSource(id='audio', file_path=path)
        window.project.add_audio_source(audio)
        window.project.save(folder / 'project.json')
        original = window.project
        window._on_audio_transcript_ready = Mock(side_effect=lambda *a: get_ident() == owner or (_ for _ in ()).throw(AssertionError('wrong thread')))
        window._on_audio_transcribe_error = Mock()
        with patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=folder)), patch('core.transcription._has_audio_stream', return_value=True), patch('core.transcription.transcribe_video', return_value=[]) as provider:
            if mode in ('reuse', 'transient_payload'):
                first = AudioTranscribeWorker(audio, project=original, backend='faster-whisper', parent=window)
                application = AudioTranscriptionApplication(original, first.task, first.options)
                first.run()
                assert application.apply(original, first.result)
                assert original.save()
            if mode == 'failure':
                audio.transcript = [TranscriptSegment(0, 1, 'existing')]
                provider.side_effect = RuntimeError('provider unavailable')
            worker = AudioTranscribeWorker(audio, project=original, backend='faster-whisper', parent=window)
            window._active_audio_transcribes = {worker}
            delivery = AudioTranscriptionDelivery(window, worker)
            worker.start(); assert worker.wait(10000)
            assert worker.job_status == ('failed' if mode == 'failure' else 'completed'), worker.result
            receipt = worker.cache.results.get(audio.id)
            if mode == 'edit': audio.transcript = []
            if mode == 'save_as': original.path = folder / 'other.json'
            if mode == 'payload': worker.cache.results[audio.id] = replace(receipt, payload_json='{}')
            if mode == 'transient_payload': worker.cache.transient_outcomes[audio.id] = {}
            if mode == 'project': window.project = Project.new()
            if mode == 'cancel': worker.cancel()
            app.processEvents()
            assert not window._active_audio_transcribes
            if mode == 'current':
                assert audio.transcript == []
                assert original.metadata.job_results == {receipt.result_id: receipt.digest}
                window._on_audio_transcript_ready.assert_called_once()
                assert original.save()
                store = JobStore(folder / 'jobs.db')
                assert store.get_result(receipt.result_id)['committed']
                store.close()
            elif mode == 'reuse':
                assert worker.result.status == 'skipped'
                assert provider.call_count == 1
                window._on_audio_transcript_ready.assert_called_once()
                assert not original.metadata.job_results
            elif mode == 'failure':
                assert audio.transcript[0].text == 'existing'
                assert audio.analysis_records['transcribe'].state == 'failed'
                window._on_audio_transcript_ready.assert_not_called()
                window._on_audio_transcribe_error.assert_called_once()
                assert not original.metadata.job_results
            else:
                assert not original.metadata.job_results
                window._on_audio_transcript_ready.assert_not_called()
                if mode not in ('edit', 'transient_payload'): assert audio.transcript is None
        original.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
