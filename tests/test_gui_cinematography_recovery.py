"""GUI cinematography results survive restart without implicitly saving project edits."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis_target import AnalysisTarget
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.cinematography import (
    CinematographyApplication,
    CinematographyOptions,
)
from core.project import Project
from models.frame import Frame
from models.cinematography import CinematographyAnalysis
from tests.test_description_operations import project_with_thumbnails
from ui.workers.cinematography_worker import CinematographyWorker

OPTIONS = CinematographyOptions(
    tier="cloud",
    model="test-model",
    local_model="local-model",
    mode="frame",
    parallelism=2,
)


@pytest.fixture(params=["clip", "frame"])
def setup(request, tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    if request.param == "frame":
        project.add_frames(
            [
                Frame(id=f"f-{i}", file_path=c.thumbnail_path)
                for i, c in enumerate(project.clips)
            ]
        )
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr(
        "ui.workers.cinematography_worker.resolve_options", lambda *args: OPTIONS
    )
    compute = Mock(return_value=CinematographyAnalysis(shot_size="CU"))
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", compute)
    return project, compute


def worker_for(project):
    return CinematographyWorker(
        project.clips,
        sources_by_id=project.sources_by_id,
        project=project,
        skip_existing=False,
        analysis_targets=[AnalysisTarget.from_frame(f) for f in project.frames] or None,
    )


def run(project, *, apply=False, prepare=lambda: True, cancel=None, limit=None):
    worker = worker_for(project)
    tasks = worker.tasks[:limit]
    application = CinematographyApplication(project, tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            assert application.apply(project, outcome)
            receipt = worker.cache.results[outcome.clip_id]
            assert receipt.matches(outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return worker.cache.run(tasks, cancel or Event(), prepare, deliver, lambda *_: None)


def test_restart_reuses_results_and_waits_for_explicit_save(setup):
    project, compute = setup
    first = run(project)
    reopened = Project.load(project.path)
    assert (
        run(reopened, apply=True, prepare=Mock(side_effect=AssertionError("cache hit")))
        == first
    )
    assert compute.call_count == 2
    assert all(
        t.cinematography is None
        for t in (Project.load(project.path).frames or Project.load(project.path).clips)
    )
    store = JobStore(project.path.parent / "jobs.db")
    assert not any(
        store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
    )
    assert reopened.save()
    assert all(
        store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
    )


def test_mixed_hits_compute_only_missing_target(setup):
    project, compute = setup
    run(project, limit=1)
    prepare = Mock(return_value=True)
    assert all(o.status == "succeeded" for o in run(project, prepare=prepare))
    assert compute.call_count == 2
    prepare.assert_called_once()


def test_changed_model_does_not_reuse_results(setup, monkeypatch):
    project, compute = setup
    run(project)
    monkeypatch.setattr(
        "ui.workers.cinematography_worker.resolve_options",
        lambda *args: replace(OPTIONS, model="other"),
    )
    run(project)
    assert compute.call_count == 4


def test_preflight_media_change_prevents_inference(setup):
    project, compute = setup

    def prepare():
        project.clips[0].thumbnail_path.write_bytes(b"changed")
        return True

    with pytest.raises(StaleJobResult, match="media changed"):
        run(project, prepare=prepare)
    compute.assert_not_called()


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


@pytest.mark.parametrize("change", ["edit", "save_as", "failed_save", "checkpoint"])
def test_save_acknowledgement_requires_exact_published_output(
    setup, monkeypatch, change
):
    project, compute = setup
    run(project, apply=True)
    store = JobStore(project.path.parent / "jobs.db")
    if change == "edit":
        for target in project.frames or project.clips:
            target.shot_type = "wide"
        assert project.save()
    elif change == "save_as":
        assert project.save(project.path.parent / "copy.json")
    elif change == "failed_save":
        with monkeypatch.context() as patch:
            patch.setattr("core.project.save_project", lambda **_: False)
            assert not project.save()
        reopened = Project.load(project.path)
        run(reopened, apply=True)
        assert compute.call_count == 2
    else:
        with monkeypatch.context() as patch:
            patch.setattr(
                JobStore,
                "checkpoint_results",
                Mock(side_effect=RuntimeError("checkpoint")),
            )
            assert project.save()
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )
    if change == "checkpoint":
        assert project.save()
        assert all(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )


def test_saved_worker_retains_history_and_completes_once(setup):
    project, compute = setup
    worker = worker_for(project)
    completed = []
    worker.analysis_completed.connect(lambda _: completed.append(True))
    worker.run()
    assert completed == [True]
    assert len(worker.result) == 2
    assert all(outcome.status == "succeeded" for outcome in worker.result)
    row = JobStore(project.path.parent / "jobs.db").get(worker.task_id)
    assert row.status == "completed"
    assert row.result["publication"] == "explicit_project_save"
    assert not project.metadata.job_results
    assert compute.call_count == 2


def test_partial_cache_success_survives_preparation_failure(setup, monkeypatch):
    project, compute = setup
    run(project, limit=1)
    worker = worker_for(project)
    monkeypatch.setattr(
        worker, "_prepare", Mock(side_effect=RuntimeError("preflight failed"))
    )
    worker.run()
    assert [outcome.status for outcome in worker.result] == ["succeeded", "failed"]
    assert worker.job_status == "failed"
    assert compute.call_count == 1


def test_unsaved_worker_is_session_only_and_cancellation_settles(setup):
    project, compute = setup
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    assert worker.operation.persistence == "session_only"
    assert worker.cache is None
    worker.cancel()
    completed = []
    worker.analysis_completed.connect(lambda value: completed.append(value))
    worker.run()
    assert completed == [{}]
    assert worker.job_status == "cancelled"
    assert all(outcome.status == "unprocessed" for outcome in worker.result)
    compute.assert_not_called()


def test_explicit_refresh_after_save_gets_new_generation(setup):
    project, compute = setup
    run(project, apply=True)
    assert project.save()
    run(project, apply=True)
    assert compute.call_count == 4
    assert len(project.metadata.job_results) == 4


def test_changed_media_during_inference_is_not_recorded(setup):
    project, compute = setup
    worker = worker_for(project)
    image = worker.tasks[0].thumbnail_path

    def change(**_):
        image.write_bytes(b"changed during inference")
        return CinematographyAnalysis()

    compute.side_effect = change
    worker.run()
    assert not worker.cache.results
    assert all(outcome.status == "failed" for outcome in worker.result)
    assert not project.metadata.job_results
