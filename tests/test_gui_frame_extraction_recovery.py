"""Saved extraction replays artifacts and checkpoints only published frames."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.frame_extraction import (
    FrameExtractionApplication,
    FrameExtractionTask,
    FrameExtractionOutcome,
)
from core.project import Project
from models.clip import Source
from tests.test_frame_extraction_operations import extract
from ui.workers.frame_extraction_worker import FrameExtractionWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    project = Project.new()
    project.add_source(Source(id="source", file_path=path, width=20, height=12))
    project.save(tmp_path / "project.sceneripper")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr(
        "core.jobs.gui_frame_extraction.frame_extraction_runtime",
        lambda: {"runtime": 1},
    )
    provider = Mock(side_effect=extract)
    monkeypatch.setattr("core.ffmpeg.extract_frames_batch", provider)
    yield project, provider
    project.close_writer()


def worker_for(project, **kwargs):
    return FrameExtractionWorker(
        project.sources[0],
        None,
        "interval",
        kwargs.get("interval", 5),
        project.path.parent / "frames",
        project=project,
    )


@pytest.mark.parametrize("empty", [False, True])
def test_reopen_reuses_files_and_save_acknowledges_receipt(setup, empty):
    project, provider = setup
    if empty:
        provider.side_effect = lambda *a, **kw: []
    first = worker_for(project)
    first.run()
    assert first.job_status == "completed", first.result
    assert first.result.status == "succeeded", first.result.message
    assert not project.frames
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        again = worker_for(reopened)
        application = FrameExtractionApplication(reopened, again.task)
        again.run()
        assert again.result == first.result
        assert provider.call_count == 1
        assert not again.task.artifact_dir.exists()
        record = again.cache.recorded
        receipt = again.cache.results[again.task.source_id]
        assert receipt.matches(record)
        assert application.apply(
            reopened,
            again.result,
            recovered_task=FrameExtractionTask.from_dict(record.task),
        )
        reopened.record_job_result(receipt.result_id, receipt.digest)
        store = JobStore(project.path.parent / "jobs.db")
        assert not store.get_result(receipt.result_id)["committed"]
        assert reopened.save()
        assert store.get_result(receipt.result_id)["committed"]
        assert [frame.id for frame in reopened.frames] == [
            frame.id for frame in first.result.frames
        ]
        fresh = worker_for(reopened)
        fresh.run()
        assert fresh.result.status == "succeeded"
        assert provider.call_count == 2
        assert fresh.result.request_id != first.result.request_id
        store.close()
    finally:
        reopened.close_writer()


@pytest.mark.parametrize("change", ["media", "runtime", "interval"])
def test_changed_inputs_create_new_extraction(setup, monkeypatch, change):
    project, provider = setup
    first = worker_for(project)
    first.run()
    if change == "media":
        project.sources[0].file_path.write_bytes(b"changed video")
    if change == "runtime":
        monkeypatch.setattr(
            "core.jobs.gui_frame_extraction.frame_extraction_runtime",
            lambda: {"runtime": 2},
        )
    second = worker_for(project, interval=1 if change == "interval" else 5)
    second.run()
    assert second.result.status == "succeeded", second.result.message
    assert provider.call_count == 2
    assert second.result.frames[0].path != first.result.frames[0].path


@pytest.mark.parametrize("change", ["media", "runtime"])
def test_queued_changes_do_not_start_provider(setup, monkeypatch, change):
    project, provider = setup
    worker = worker_for(project)
    if change == "media":
        project.sources[0].file_path.write_bytes(b"changed video")
    else:
        monkeypatch.setattr(
            "core.jobs.gui_frame_extraction.frame_extraction_runtime",
            lambda: {"runtime": 2},
        )
    worker.run()
    assert worker.result.status == "failed"
    provider.assert_not_called()


def test_changed_cached_artifact_fails_closed(setup):
    project, provider = setup
    first = worker_for(project)
    first.run()
    first.result.frames[0].path.write_bytes(b"replacement")
    again = worker_for(project)
    again.run()
    assert again.result.status == "failed"
    assert provider.call_count == 1
    assert not project.frames


def test_cancelled_replay_does_not_publish_or_compute(setup):
    project, provider = setup
    worker_for(project).run()
    again = worker_for(project)
    again.cancel()
    again.run()
    assert again.result.status == "unprocessed"
    assert provider.call_count == 1
    assert not project.frames


def test_serialized_outcome_rejects_invalid_frame_metadata(setup):
    project, _ = setup
    worker = worker_for(project)
    worker.run()
    payload = worker.result.to_dict()
    payload["frames"][0]["frame_number"] = True
    with pytest.raises(ValueError):
        FrameExtractionOutcome.from_dict(payload)


def test_deleted_frame_prevents_checkpoint(setup):
    project, _ = setup
    worker = worker_for(project)
    application = FrameExtractionApplication(project, worker.task)
    worker.run()
    assert application.apply(project, worker.result)
    receipt = worker.cache.results[worker.task.source_id]
    project.record_job_result(receipt.result_id, receipt.digest)
    project.remove_frames([worker.result.frames[0].id])
    project.save()
    store = JobStore(project.path.parent / "jobs.db")
    assert not store.get_result(receipt.result_id)["committed"]
    store.close()
