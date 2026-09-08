"""Saved GUI embedding batches recover without saving unrelated desktop edits."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.embeddings import EmbeddingApplication
from core.project import Project
from tests.test_description_operations import project_with_thumbnails
from ui.workers.embedding_worker import EmbeddingAnalysisWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 3)
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    compute = Mock(side_effect=lambda paths: [[0.123456789] * 768 for _ in paths])
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", compute
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    return project, compute


def worker_for(project, **kwargs):
    return EmbeddingAnalysisWorker(project.clips, project=project, **kwargs)


def run(project, *, apply=False, prepare=lambda: True, cancel=None):
    worker = worker_for(project)
    application = EmbeddingApplication(project, worker.tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            receipt = worker.cache.results[outcome.clip_id]
            assert receipt.matches(outcome)
            assert application.apply(project, outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return worker.cache.run(
        worker.tasks, cancel or Event(), prepare, deliver, lambda *_: None
    )


def test_restart_recovers_precision_and_explicit_save_checkpoints(setup):
    project, compute = setup
    first = run(project)
    reopened = Project.load(project.path)
    assert (
        run(reopened, apply=True, prepare=Mock(side_effect=AssertionError("cache hit")))
        == first
    )
    compute.assert_called_once()
    assert reopened.clips[0].embedding == [0.123456789] * 768
    assert Project.load(project.path).clips[0].embedding is None
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert reopened.save()
        assert all(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert Project.load(project.path).clips[0].embedding == [0.123456789] * 768
    finally:
        store.close()


def test_worker_uses_durable_history_and_remains_detached(setup):
    project, compute = setup
    worker = worker_for(project)
    worker.run()
    assert worker.job_status == "completed"
    assert all(c.embedding is None for c in project.clips)
    retry = worker_for(Project.load(project.path))
    retry.run()
    assert retry.result == worker.result
    assert all(o.status == "succeeded" for o in retry.result)
    compute.assert_called_once()


def test_whole_batch_is_recorded_before_first_delivery(setup):
    project, compute = setup
    worker = worker_for(project)

    def fail_delivery(outcome):
        assert len(worker.cache.results) == 3
        raise RuntimeError("GUI went away")

    with pytest.raises(RuntimeError, match="GUI went away"):
        worker.cache.run(
            worker.tasks, Event(), lambda: True, fail_delivery, lambda *_: None
        )
    assert len(run(Project.load(project.path), apply=True)) == 3
    compute.assert_called_once()


def test_model_retained_across_batches_and_cache_hits_do_not_unload(setup, monkeypatch):
    project, compute = setup
    unload = Mock()
    monkeypatch.setattr("core.analysis.embeddings.unload_model", unload)
    for _ in range(2):
        worker = worker_for(project, chunk_size=2)
        worker.run()
        assert worker.job_status == "completed"
    assert [len(c.args[0]) for c in compute.call_args_list] == [2, 1]
    unload.assert_called_once()


def test_fatal_model_failure_stops_later_batches(setup):
    project, compute = setup
    compute.side_effect = RuntimeError("unavailable")
    worker = worker_for(project, chunk_size=1)
    worker.run()
    assert [o.status for o in worker.result] == ["failed", "unprocessed", "unprocessed"]
    compute.assert_called_once()
    assert worker.cache.results == {}


@pytest.mark.parametrize("change", ["source", "image", "runtime"])
def test_changed_inputs_rejected_before_inference(setup, monkeypatch, change):
    from core.jobs.commits import StaleJobResult

    project, compute = setup

    def prepare():
        if change == "runtime":
            monkeypatch.setattr(
                "core.jobs.gui_embeddings._runtime", lambda: {"changed": True}
            )
        else:
            path = (
                project.sources[0].file_path
                if change == "source"
                else project.clips[0].thumbnail_path
            )
            path.write_bytes(b"changed")
        return True

    with pytest.raises(StaleJobResult):
        run(project, prepare=prepare)
    compute.assert_not_called()


@pytest.mark.parametrize("change", ["source", "range", "image"])
def test_changed_target_invalidates_unpublished_cache(setup, change):
    project, compute = setup
    run(project)
    if change == "range":
        for clip in project.clips:
            clip.start_frame += 1
    else:
        path = (
            project.sources[0].file_path
            if change == "source"
            else project.clips[0].thumbnail_path
        )
        path.write_bytes(b"changed")
    run(project)
    assert compute.call_count == 2


@pytest.mark.parametrize("change", ["edit", "save_as"])
def test_changed_saved_vector_or_path_does_not_checkpoint(setup, change):
    project, _ = setup
    run(project, apply=True)
    path = project.path
    if change == "edit":
        for clip in project.clips:
            clip.embedding[0] = 0.99
    assert project.save(path.parent / "copy.json" if change == "save_as" else path)
    store = JobStore(path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )
    finally:
        store.close()


def test_cancelled_cache_recovery_does_not_publish(setup):
    project, compute = setup
    run(project)
    cancel = Event()
    cancel.set()
    assert all(
        o.status == "unprocessed" for o in run(project, apply=True, cancel=cancel)
    )
    assert not project.metadata.job_results
    compute.assert_called_once()


def test_invalid_vector_is_not_journaled(setup):
    project, compute = setup
    compute.side_effect = None
    compute.return_value = [[0.0] * 768, [0.1] * 768, [0.2] * 768]
    assert [o.status for o in run(project, apply=True)] == [
        "failed",
        "succeeded",
        "succeeded",
    ]
    assert len(project.metadata.job_results) == 2
    assert project.clips[0].embedding is None


def test_unrelated_unsaved_edits_are_not_saved_by_worker(setup):
    project, _ = setup
    project.clips[0].notes = "unsaved editorial note"
    worker = worker_for(project)
    worker.run()
    assert Project.load(project.path).clips[0].notes != "unsaved editorial note"
    assert project.clips[0].notes == "unsaved editorial note"


def test_unsaved_worker_is_session_only(setup):
    project, _ = setup
    project.path = None
    worker = worker_for(project)
    assert worker.cache is None
    assert worker.operation.persistence == "session_only"


def test_launcher_enables_saved_project_recovery(setup):
    from core.settings import Settings
    from ui.workers.clip_analysis_work import create_clip_analysis_worker

    project, _ = setup
    worker, application = create_clip_analysis_worker(
        project,
        Settings(),
        "embeddings",
        project.clips,
    )
    assert worker.cache.path == project.path.resolve()
    assert application.project is project


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
def test_corrupt_saved_receipt_fails_without_inference(setup, column):
    import sqlite3

    project, compute = setup
    run(project, apply=True)
    assert project.save()
    rid = next(iter(project.metadata.job_results))
    with sqlite3.connect(project.path.parent / "jobs.db") as connection:
        connection.execute(
            f"UPDATE job_results SET {column} = ? WHERE result_id = ?", ("{}", rid)
        )
    worker = worker_for(Project.load(project.path))
    worker.run()
    assert worker.job_status == "failed"
    compute.assert_called_once()


def test_recorded_vector_is_detached_when_deserialized():
    from dataclasses import asdict
    from core.operations.embeddings import EmbeddingOutcome

    payload = asdict(EmbeddingOutcome.from_vector("clip", [0.1] * 768))
    payload["vector"] = list(payload["vector"])
    outcome = EmbeddingOutcome.from_dict(payload)
    payload["vector"][0] = 0.9
    assert outcome.vector[0] == 0.1
