"""Audio import recovery preserves identity before explicit project save."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.jobs.store import JobStore
from core.operations.audio_import import (
    AudioImportApplication,
    AudioImportTask,
    AudioImportOutcome,
)
from core.project import Project
from ui.workers.audio_import_worker import AudioImportWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    media = tmp_path / "voice.wav"
    media.write_bytes(b"audio")
    project = Project.new()
    project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr(
        "core.jobs.gui_audio_import.audio_import_runtime", lambda: {"runtime": 1}
    )
    processor = Mock(ffprobe_available=True)
    processor.get_audio_info.return_value = dict(
        duration=1.23456789, sample_rate=48000, channels=2
    )
    monkeypatch.setattr("core.ffmpeg.FFmpegProcessor", lambda: processor)
    yield project, media, processor.get_audio_info
    project.close_writer()


def publish(project, worker):
    receipt = worker.cache.results[str(worker.task.path)]
    recovered = AudioImportTask.from_dict(worker.cache.recorded.task)
    application = AudioImportApplication(project, worker.task)
    assert application.apply(project, worker.result, recovered_task=recovered)
    project.record_job_result(receipt.result_id, receipt.digest)
    return receipt


def test_reopen_reuses_probe_identity_and_save_acknowledges(setup):
    project, media, probe = setup
    worker = AudioImportWorker(media, project=project)
    worker.run()
    assert worker.job_status == "completed", worker.result
    assert not project.audio_sources and not project.metadata.job_results
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        retry = AudioImportWorker(media, project=reopened)
        assert retry.task.audio_source_id != worker.task.audio_source_id
        retry.run()
        assert retry.result == worker.result
        assert probe.call_count == 1
        receipt = publish(reopened, retry)
        store = JobStore(project.path.parent / "jobs.db")
        try:
            assert not store.get_result(receipt.result_id)["committed"]
            assert reopened.save()
            assert store.get_result(receipt.result_id)["committed"]
            assert reopened.audio_sources[0].id == worker.result.audio_source_id
            assert reopened.audio_sources[0].duration_seconds == 1.23456789
        finally:
            store.close()
    finally:
        reopened.close_writer()


def test_unsaved_import_uses_session_history(setup):
    _, media, probe = setup
    project = Project.new()
    worker = AudioImportWorker(media, project=project)
    worker.run()
    assert worker.cache is None
    assert worker.operation.persistence == "session_only"
    assert worker.job_status == "completed" and worker.task_id
    assert not project.audio_sources
    probe.assert_called_once()


@pytest.mark.parametrize("change", ["media", "runtime"])
def test_changed_inputs_require_new_probe(setup, monkeypatch, change):
    project, media, probe = setup
    first = AudioImportWorker(media, project=project)
    first.run()
    if change == "media":
        media.write_bytes(b"changed audio")
    else:
        monkeypatch.setattr(
            "core.jobs.gui_audio_import.audio_import_runtime", lambda: {"runtime": 2}
        )
    retry = AudioImportWorker(media, project=project)
    retry.run()
    assert retry.result.status == "succeeded"
    assert retry.result.audio_source_id != first.result.audio_source_id
    assert probe.call_count == 2


@pytest.mark.parametrize("change", ["media", "runtime"])
def test_queued_changes_do_not_probe(setup, monkeypatch, change):
    project, media, probe = setup
    worker = AudioImportWorker(media, project=project)
    if change == "media":
        media.write_bytes(b"changed")
    else:
        monkeypatch.setattr(
            "core.jobs.gui_audio_import.audio_import_runtime", lambda: {"runtime": 2}
        )
    worker.run()
    assert worker.job_status == "failed"
    assert not worker.cache.results
    probe.assert_not_called()


def test_cancelled_replay_never_delivers(setup):
    project, media, probe = setup
    AudioImportWorker(media, project=project).run()
    worker = AudioImportWorker(media, project=project)
    received = []
    worker.outcome_ready.connect(received.append)
    worker.cancel()
    worker.run()
    assert worker.job_status == "cancelled"
    assert not received and not worker.cache.results
    probe.assert_called_once()


def test_failed_probe_is_not_replayable(setup):
    project, media, probe = setup
    probe.side_effect = RuntimeError("probe failed")
    worker = AudioImportWorker(media, project=project)
    worker.run()
    assert worker.result.status == "failed" and not worker.cache.results
    probe.side_effect = None
    retry = AudioImportWorker(media, project=project)
    retry.run()
    assert retry.result.status == "succeeded" and probe.call_count == 2


def test_failed_save_retains_unacknowledged_reusable_probe(setup):
    project, media, probe = setup
    worker = AudioImportWorker(media, project=project)
    worker.run()
    receipt = publish(project, worker)
    with patch("core.project.save_project", return_value=False):
        assert not project.save()
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        assert not reopened.audio_sources
        retry = AudioImportWorker(media, project=reopened)
        retry.run()
        assert retry.result.audio_source_id == worker.result.audio_source_id
        publish(reopened, retry)
        assert reopened.save()
        store = JobStore(project.path.parent / "jobs.db")
        assert store.get_result(receipt.result_id)["committed"]
        store.close()
        probe.assert_called_once()
    finally:
        reopened.close_writer()


def test_failed_checkpoint_retries_without_probe(setup):
    project, media, probe = setup
    worker = AudioImportWorker(media, project=project)
    worker.run()
    receipt = publish(project, worker)
    with patch.object(
        JobStore, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        assert project.save()
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        assert len(reopened.audio_sources) == 1
        assert reopened.save()
        store = JobStore(project.path.parent / "jobs.db")
        assert store.get_result(receipt.result_id)["committed"]
        store.close()
        probe.assert_called_once()
    finally:
        reopened.close_writer()


@pytest.mark.parametrize("change", ["metadata", "removed", "save_as"])
def test_changed_saved_output_is_not_acknowledged(setup, change):
    project, media, _ = setup
    worker = AudioImportWorker(media, project=project)
    worker.run()
    receipt = publish(project, worker)
    if change == "metadata":
        project.audio_sources[0].duration_seconds = 99
    if change == "removed":
        project.remove_audio_source(worker.result.audio_source_id)
    project.save(
        project.path.parent / "other.json" if change == "save_as" else project.path
    )
    store = JobStore(worker.cache.path.parent / "jobs.db")
    assert not store.get_result(receipt.result_id)["committed"]
    store.close()


def test_removed_then_reimported_audio_has_new_generation(setup):
    project, media, probe = setup
    first = AudioImportWorker(media, project=project)
    first.run()
    publish(project, first)
    project.save()
    project.remove_audio_source(first.result.audio_source_id)
    retry = AudioImportWorker(media, project=project)
    retry.run()
    assert retry.result.audio_source_id != first.result.audio_source_id
    assert probe.call_count == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("audio_source_id", ""),
        ("status", "invalid"),
        ("duration", float("nan")),
        ("duration", 0),
        ("channels", True),
        ("sample_rate", -1),
    ],
)
def test_invalid_serialized_audio_metadata_is_rejected(field, value):
    data = asdict(AudioImportOutcome("audio", "succeeded", 10, 48000, 2))
    data[field] = value
    with pytest.raises(ValueError):
        AudioImportOutcome.from_dict(data)


def test_real_queued_recovery_validates_receipt_and_original_owner():
    import os
    import subprocess
    import sys

    code = r"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from threading import get_ident
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.jobs.store import JobStore
from ui.workers.audio_import_worker import AudioImportWorker
from ui.workers.audio_import_delivery import AudioImportDelivery
app = QCoreApplication([])
owner = get_ident(); owners = []
with TemporaryDirectory() as directory:
    for mode in ('current', 'payload', 'project', 'save_as', 'cancel', 'media'):
        folder = Path(directory) / mode; folder.mkdir()
        media = folder / 'voice.wav'; media.write_bytes(b'audio')
        window = QObject(); owners.append(window)
        window.project = Project.new(); window.project.save(folder / 'project.json')
        original = window.project
        window._on_audio_imported = Mock(side_effect=lambda *a: get_ident() == owner or (_ for _ in ()).throw(AssertionError('wrong thread')))
        window._on_audio_import_error = Mock()
        with patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=folder)), patch('core.ffmpeg.FFmpegProcessor') as processor:
            processor.return_value.ffprobe_available = True
            processor.return_value.get_audio_info.return_value = dict(duration=10, sample_rate=48000, channels=2)
            first = AudioImportWorker(media, project=original); first.run()
            worker = AudioImportWorker(media, parent=window, project=original)
            window._active_audio_imports = {worker}
            delivery = AudioImportDelivery(window, worker)
            worker.start(); assert worker.wait(10000)
            assert worker.job_status == 'completed', worker.result
            assert worker.result.audio_source_id == first.result.audio_source_id
            assert worker.task.audio_source_id != first.task.audio_source_id
            processor.return_value.get_audio_info.assert_called_once()
            receipt = worker.cache.results[str(worker.task.path)]
            if mode == 'payload': worker.cache.results[str(worker.task.path)] = replace(receipt, payload_json='{}')
            if mode == 'project': window.project = Project.new()
            if mode == 'save_as': original.path = folder / 'other.json'
            if mode == 'cancel': worker.cancel()
            if mode == 'media': media.write_bytes(b'changed')
            app.processEvents()
            assert not window._active_audio_imports
            if mode == 'current':
                assert original.audio_sources[0].id == first.result.audio_source_id
                assert original.metadata.job_results == {receipt.result_id: receipt.digest}
                window._on_audio_imported.assert_called_once()
                assert original.save()
                store = JobStore(folder / 'jobs.db')
                assert store.get_result(receipt.result_id)['committed']
                store.close()
            else:
                assert not original.audio_sources and not original.metadata.job_results
                window._on_audio_imported.assert_not_called()
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


def test_corrupt_cached_probe_is_not_recomputed(setup):
    import sqlite3

    project, media, probe = setup
    first = AudioImportWorker(media, project=project)
    first.run()
    receipt = first.cache.results[str(media)]
    with sqlite3.connect(project.path.parent / "jobs.db") as connection:
        connection.execute(
            "UPDATE job_results SET payload_json = '{}' WHERE result_id = ?",
            (receipt.result_id,),
        )
    retry = AudioImportWorker(media, project=project)
    retry.run()
    assert retry.job_status == "failed"
    assert not retry.cache.results
    probe.assert_called_once()
