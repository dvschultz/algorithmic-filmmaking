"""Still-image recovery preserves artifacts and IDs through explicit saves."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.jobs.store import JobStore
from core.operations.image_import import (
    ImageImportApplication,
    ImageImportTask,
    ImageImportOutcome,
)
from core.project import Project
from tests.test_image_import_operations import make_image
from ui.workers.image_import_worker import ImageImportWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = [
        make_image(tmp_path / "one" / "same.png"),
        make_image(tmp_path / "two" / "same.png", (17, 23)),
    ]
    project = Project.new()
    project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr(
        "core.jobs.gui_image_import.image_import_runtime", lambda: {"runtime": 1}
    )
    from core.thumbnail import generate_image_thumbnail

    provider = Mock(wraps=generate_image_thumbnail)
    monkeypatch.setattr("core.thumbnail.generate_image_thumbnail", provider)
    yield project, paths, provider
    project.close_writer()


def worker_for(project, paths, *, copy_files=True):
    return ImageImportWorker(
        paths, paths[0].parent.parent / "frames", project=project, copy_files=copy_files
    )


def publish(project, worker):
    recovered = ImageImportTask.from_dict(worker.cache.recorded.task)
    receipt = worker.cache.results[worker.cache.batch_id]
    assert ImageImportApplication(project, worker.task).apply(
        project, worker.result, recovered_task=recovered
    )
    project.record_job_result(receipt.result_id, receipt.digest)
    return receipt


@pytest.mark.parametrize("copy_files", [False, True])
def test_reopen_reuses_artifacts_and_ids_then_save_acknowledges(setup, copy_files):
    project, paths, provider = setup
    first = worker_for(project, paths, copy_files=copy_files)
    first.run()
    assert first.job_status == "completed", first.result
    assert not project.frames
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        again = worker_for(reopened, paths, copy_files=copy_files)
        again.run()
        assert again.result == first.result
        assert again.task.request_id != first.task.request_id
        assert provider.call_count == 2
        assert not again.task.artifact_dir.exists()
        receipt = publish(reopened, again)
        store = JobStore(project.path.parent / "jobs.db")
        assert not store.get_result(receipt.result_id)["committed"]
        assert reopened.save()
        assert store.get_result(receipt.result_id)["committed"]
        store.close()
        assert [f.id for f in reopened.frames] == [f.id for f in first.result.frames]
        assert [(f.width, f.height) for f in reopened.frames] == [(400, 300), (17, 23)]
    finally:
        reopened.close_writer()


def test_unsaved_project_has_session_only_job(setup):
    _, paths, provider = setup
    project = Project.new()
    worker = worker_for(project, paths)
    worker.run()
    assert worker.operation.persistence == "session_only" and worker.cache is None
    assert worker.job_status == "completed" and worker.task_id
    assert not project.frames and provider.call_count == 2


@pytest.mark.parametrize("change", ["media", "runtime"])
def test_queued_change_rejects_computation(setup, monkeypatch, change):
    project, paths, provider = setup
    worker = worker_for(project, paths)
    if change == "media":
        paths[1].write_bytes(b"changed")
    else:
        monkeypatch.setattr(
            "core.jobs.gui_image_import.image_import_runtime", lambda: {"runtime": 2}
        )
    worker.run()
    assert worker.job_status == "failed"
    assert not worker.cache.results
    provider.assert_not_called()


@pytest.mark.parametrize("change", ["media", "runtime", "policy", "order"])
def test_changed_inputs_do_not_reuse_recorded_batch(setup, monkeypatch, change):
    project, paths, provider = setup
    first = worker_for(project, paths)
    first.run()
    if change == "media":
        make_image(paths[1], (33, 44))
    if change == "runtime":
        monkeypatch.setattr(
            "core.jobs.gui_image_import.image_import_runtime", lambda: {"runtime": 2}
        )
    if change == "order":
        paths = paths[::-1]
    again = worker_for(project, paths, copy_files=change != "policy")
    again.run()
    assert again.result.status == "succeeded"
    assert [f.id for f in again.result.frames] != [f.id for f in first.result.frames]
    assert provider.call_count == 4


def test_partial_invalid_inputs_are_preserved_without_losing_success(setup):
    project, paths, provider = setup
    bad = paths[0].parent / "bad.png"
    bad.write_bytes(b"invalid image")
    paths = [paths[0], bad, paths[0].parent]
    first = worker_for(project, paths)
    first.run()
    assert first.result.status == "succeeded" and len(first.result.frames) == 1
    assert len(first.result.errors) == 2
    again = worker_for(project, paths)
    again.run()
    assert again.result == first.result
    provider.assert_called_once()


def test_invalid_directory_does_not_invalidate_success_when_artifacts_are_created(setup):
    project, paths, provider = setup
    worker = worker_for(project, [paths[0], project.path.parent])
    worker.run()
    assert worker.result.status == "succeeded", worker.result.errors
    assert len(worker.result.frames) == 1 and len(worker.result.errors) == 1
    again = worker_for(project, [paths[0], project.path.parent])
    again.run()
    assert again.result == worker.result
    provider.assert_called_once()


def test_failed_save_reuses_files_and_checkpoint_retry_avoids_work(setup):
    project, paths, provider = setup
    first = worker_for(project, paths)
    first.run()
    receipt = publish(project, first)
    with patch("core.project.save_project", return_value=False):
        assert not project.save()
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        assert not reopened.frames
        again = worker_for(reopened, paths)
        again.run()
        assert again.result == first.result
        publish(reopened, again)
        with patch.object(
            JobStore, "checkpoint_results", side_effect=OSError("checkpoint failed")
        ):
            assert reopened.save()
        store = JobStore(project.path.parent / "jobs.db")
        assert not store.get_result(receipt.result_id)["committed"]
        assert reopened.save()
        assert store.get_result(receipt.result_id)["committed"]
        assert provider.call_count == 2
        store.close()
    finally:
        reopened.close_writer()


@pytest.mark.parametrize("change", ["removed", "dimensions", "save_as"])
def test_saved_output_changes_prevent_acknowledgment(setup, change):
    project, paths, _ = setup
    worker = worker_for(project, paths)
    worker.run()
    receipt = publish(project, worker)
    if change == "removed":
        project.remove_frames([worker.result.frames[0].id])
    if change == "dimensions":
        project.frames[0].width = 99
    project.save(
        project.path.parent / "other.json" if change == "save_as" else project.path
    )
    store = JobStore(worker.cache.path.parent / "jobs.db")
    assert not store.get_result(receipt.result_id)["committed"]
    store.close()


@pytest.mark.parametrize("change", ["copy", "thumbnail", "payload"])
def test_corrupt_artifacts_or_cache_cannot_replay(setup, change):
    import sqlite3

    project, paths, provider = setup
    first = worker_for(project, paths)
    first.run()
    receipt = first.cache.results[first.cache.batch_id]
    if change == "copy":
        first.result.frames[0].path.write_bytes(b"changed")
    if change == "thumbnail":
        first.result.frames[0].thumbnail_path.write_bytes(b"changed")
    if change == "payload":
        with sqlite3.connect(project.path.parent / "jobs.db") as connection:
            connection.execute(
                "UPDATE job_results SET payload_json = '{}' WHERE result_id = ?",
                (receipt.result_id,),
            )
    again = worker_for(project, paths)
    again.run()
    assert again.job_status == "failed"
    assert provider.call_count == 2 and not project.frames


def test_precancelled_replay_does_not_publish(setup):
    project, paths, provider = setup
    worker_for(project, paths).run()
    again = worker_for(project, paths)
    again.cancel()
    again.run()
    assert again.job_status == "cancelled" and again.result.status == "unprocessed"
    assert not again.cache.results and provider.call_count == 2


def test_published_batch_starts_new_generation(setup):
    project, paths, provider = setup
    first = worker_for(project, paths)
    first.run()
    publish(project, first)
    project.save()
    again = worker_for(project, paths)
    again.run()
    assert [f.id for f in again.result.frames] != [f.id for f in first.result.frames]
    assert provider.call_count == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "invalid"),
        ("status", "unknown"),
        ("frames", []),
        ("errors", [None]),
    ],
)
def test_malformed_outcomes_are_rejected(setup, field, value):
    project, paths, _ = setup
    first = worker_for(project, paths)
    first.run()
    payload = json.loads(json.dumps(first.result.to_dict()))
    payload[field] = value
    with pytest.raises(ValueError):
        ImageImportOutcome.from_dict(payload)


def test_real_queued_recovery_preserves_ids_and_checks_receipt():
    import os
    import subprocess
    import sys

    code = r"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject
from core.project import Project
from core.jobs.store import JobStore
from ui.workers.image_import_worker import ImageImportWorker
from ui.workers.image_import_delivery import ImageImportDelivery
app = QCoreApplication([]); owners = []
with TemporaryDirectory() as directory:
    for mode in ('current', 'payload', 'project', 'save_as', 'cancel', 'artifact'):
        folder = Path(directory).resolve() / mode; folder.mkdir()
        media = folder / 'image.png'; Image.new('RGB', (400, 300), 'red').save(media)
        window = QObject(); owners.append(window)
        window.project = Project.new(); window.project.save(folder / 'project.json')
        original = window.project
        window.frames_tab = SimpleNamespace(update_frame_browser=Mock())
        window.status_bar = SimpleNamespace(showMessage=Mock())
        window._update_chat_project_state = Mock()
        with patch('core.settings.load_settings', return_value=SimpleNamespace(cache_dir=folder)):
            first = ImageImportWorker([media], folder / 'frames', project=original, copy_files=True); first.run()
            worker = ImageImportWorker([media], folder / 'frames', parent=window, project=original, copy_files=True)
            window._image_import_worker = worker
            delivery = ImageImportDelivery(window, worker)
            with patch('core.thumbnail.generate_image_thumbnail', side_effect=AssertionError('recomputed')):
                worker.start(); assert worker.wait(10000)
            assert worker.job_status == 'completed', worker.result
            assert worker.result == first.result
            assert worker.task.request_id != first.task.request_id
            receipt = worker.cache.results[worker.cache.batch_id]
            if mode == 'payload': worker.cache.results[worker.cache.batch_id] = replace(receipt, payload_json='{}')
            if mode == 'project': window.project = Project.new()
            if mode == 'save_as': original.path = folder / 'other.json'
            if mode == 'cancel': worker.cancel()
            if mode == 'artifact': worker.result.frames[0].path.write_bytes(b'changed')
            app.processEvents()
            assert window._image_import_worker is None
            if mode == 'current':
                assert original.frames[0].id == first.result.frames[0].id
                assert original.metadata.job_results == {receipt.result_id: receipt.digest}
                window.frames_tab.update_frame_browser.assert_called_once()
                assert original.save()
                store = JobStore(folder / 'jobs.db')
                assert store.get_result(receipt.result_id)['committed']
                store.close()
            else:
                assert not original.frames and not original.metadata.job_results
                window.frames_tab.update_frame_browser.assert_not_called()
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
