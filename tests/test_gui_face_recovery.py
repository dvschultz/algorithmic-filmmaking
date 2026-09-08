"""Saved GUI face jobs recover computations without saving unrelated edits."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.faces import FaceApplication
from core.project import Project
from tests.test_description_operations import project_with_thumbnails
from ui.workers.face_detection_worker import FaceDetectionWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    compute = Mock(
        return_value=[
            {"bbox": [0, 0, 1, 1], "embedding": [0.123456789] * 512, "confidence": 0.9}
        ]
    )
    monkeypatch.setattr("core.analysis.faces.extract_faces_from_clip", compute)
    monkeypatch.setattr("core.analysis.faces._load_insightface", Mock())
    monkeypatch.setattr("core.analysis.faces.unload_model", Mock())
    return project, compute


def worker_for(project, **kwargs):
    return FaceDetectionWorker(
        project.clips, project.sources_by_id, project=project, **kwargs
    )


def run(project, *, apply=False, cancel=None, prepare=lambda: True):
    worker = worker_for(project)
    application = FaceApplication(project, worker.tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            assert application.apply(project, outcome)
            receipt = worker.cache.results[outcome.clip_id]
            assert receipt.matches(outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return worker.cache.run(
        worker.tasks, cancel or Event(), prepare, deliver, lambda *_: None
    )


def test_restart_recovers_full_precision_then_checkpoints_saved_precision(setup):
    project, compute = setup
    first = run(project)
    reopened = Project.load(project.path)
    assert (
        run(reopened, apply=True, prepare=Mock(side_effect=AssertionError("cache hit")))
        == first
    )
    assert compute.call_count == 2
    assert reopened.clips[0].face_embeddings[0]["embedding"][0] == 0.123456789
    assert Project.load(project.path).clips[0].face_embeddings is None
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert reopened.save()
        assert all(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert (
            Project.load(project.path).clips[0].face_embeddings[0]["embedding"][0]
            == 0.12346
        )
    finally:
        store.close()


def test_skipped_offline_source_does_not_block_other_results(setup):
    project, compute = setup
    project.clips[0].face_embeddings = []
    project.clips[0].source_id = "missing"
    assert [o.status for o in run(project)] == ["skipped", "succeeded"]
    compute.assert_called_once()


def test_missing_source_is_item_failure(setup):
    project, compute = setup
    project.clips[0].source_id = "missing"
    assert [o.status for o in run(project)] == ["failed", "succeeded"]
    compute.assert_called_once()


def test_cancelled_recovery_does_not_publish(setup):
    project, compute = setup
    run(project)
    cancel = Event()
    cancel.set()
    assert all(
        o.status == "unprocessed" for o in run(project, apply=True, cancel=cancel)
    )
    assert not project.metadata.job_results
    assert compute.call_count == 2


def test_worker_uses_durable_runtime_and_does_not_mutate_project(setup):
    project, compute = setup
    worker = worker_for(project)
    worker.run()
    assert worker.job_status == "completed"
    assert len(worker.result) == 2
    assert all(o.status == "succeeded" for o in worker.result)
    assert all(c.face_embeddings is None for c in project.clips)
    again = worker_for(Project.load(project.path))
    again.run()
    assert again.result == worker.result
    assert compute.call_count == 2


@pytest.mark.parametrize("change", ["edit", "save_as"])
def test_changed_saved_output_or_path_does_not_checkpoint(setup, change):
    project, _ = setup
    run(project, apply=True)
    path = project.path
    if change == "edit":
        for clip in project.clips:
            clip.face_embeddings = []
    assert project.save(path.parent / "copy.json" if change == "save_as" else path)
    store = JobStore(path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )
    finally:
        store.close()


def test_pipeline_launcher_enables_recovery(setup):
    from core.settings import Settings
    from ui.workers.clip_analysis_work import create_clip_analysis_worker

    project, _ = setup
    worker, application = create_clip_analysis_worker(
        project,
        Settings(),
        "face_embeddings",
        project.clips,
    )
    assert worker.cache.path == project.path.resolve()
    assert application.project is project


def test_model_session_is_reused_and_cache_hits_do_not_load(setup, monkeypatch):
    project, _ = setup
    load, unload = Mock(), Mock()
    monkeypatch.setattr("core.analysis.faces._load_insightface", load)
    monkeypatch.setattr("core.analysis.faces.unload_model", unload)
    run(project)
    run(project)
    load.assert_called_once()
    unload.assert_called_once()


@pytest.mark.parametrize("change", ["media", "runtime"])
def test_changed_inference_inputs_are_rejected(setup, monkeypatch, change):
    from core.jobs.commits import StaleJobResult

    project, compute = setup

    def prepare():
        if change == "media":
            project.sources[0].file_path.write_bytes(b"changed")
        else:
            monkeypatch.setattr(
                "core.jobs.gui_faces._runtime", lambda: {"changed": True}
            )
        return True

    with pytest.raises(StaleJobResult):
        run(project, prepare=prepare)
    compute.assert_not_called()


def test_failed_and_empty_results_remain_distinct(setup):
    project, compute = setup
    compute.side_effect = [ValueError("decode failed"), []]
    outcomes = run(project, apply=True)
    assert [o.status for o in outcomes] == ["failed", "succeeded"]
    assert project.clips[0].face_embeddings is None
    assert project.clips[1].face_embeddings == []
    assert len(project.metadata.job_results) == 1
    assert project.save()


def test_face_outcome_detaches_cached_arrays():
    from core.operations.faces import FaceOutcome

    face = {"bbox": [0, 0, 1, 1], "embedding": [0.1] * 512, "confidence": 0.9}
    outcome = FaceOutcome.from_dict(
        {"clip_id": "c", "status": "succeeded", "faces": [face]}
    )
    face["embedding"][0] = 0.9
    assert outcome.faces[0].embedding[0] == 0.1
