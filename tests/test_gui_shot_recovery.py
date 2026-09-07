"""Desktop shot recovery retains inference without saving editorial state."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis_target import AnalysisTarget
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.shots import ShotTypeApplication
from core.project import Project
from models.frame import Frame
from tests.test_description_operations import project_with_thumbnails
from ui.workers.shot_type_worker import ShotTypeWorker


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
        "core.settings.load_settings",
        lambda: SimpleNamespace(
            cache_dir=tmp_path,
            shot_classifier_tier="cpu",
            shot_classifier_cloud_model=None,
        ),
    )
    compute = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    return project, compute


def worker_for(project):
    return ShotTypeWorker(
        project.clips,
        project.sources_by_id,
        project=project,
        skip_existing=False,
        analysis_targets=[AnalysisTarget.from_frame(f) for f in project.frames] or None,
    )


def run(project, *, apply=False, prepare=lambda: True, cancel=None, limit=None):
    worker = worker_for(project)
    tasks = worker.tasks[:limit]
    application = ShotTypeApplication(project, tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            assert application.apply(project, outcome)
            receipt = worker.cache.receipt(outcome)
            assert receipt.matches(outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return worker.cache.run(tasks, cancel or Event(), prepare, deliver, lambda *_: None)


def test_restart_reuses_inference_until_explicit_save(setup):
    project, compute = setup
    first = run(project)
    reopened = Project.load(project.path)
    before = project.path.read_bytes()
    assert (
        run(reopened, apply=True, prepare=Mock(side_effect=AssertionError("cache hit")))
        == first
    )
    assert project.path.read_bytes() == before
    assert compute.call_count == 2
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert reopened.save()
        assert all(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
    finally:
        store.close()


def test_mixed_hits_only_compute_missing_targets(setup):
    project, compute = setup
    run(project, limit=1)
    assert all(outcome.status == "succeeded" for outcome in run(project))
    assert compute.call_count == 2


@pytest.mark.parametrize(
    "change", ["edit", "target", "save_as", "failed_save", "checkpoint"]
)
def test_only_matching_saved_results_are_acknowledged(setup, monkeypatch, change):
    project, compute = setup
    run(project, apply=True)
    store = JobStore(project.path.parent / "jobs.db")
    try:
        if change == "edit":
            for target in project.frames or project.clips:
                target.shot_type = "manual"
            assert project.save()
        elif change == "target":
            if project.frames:
                for frame in project.frames:
                    frame.frame_number = 123
            else:
                for clip in project.clips:
                    clip.start_frame += 1
            assert project.save()
        elif change == "save_as":
            assert project.save(project.path.parent / "copy.json")
        elif change == "failed_save":
            with monkeypatch.context() as patch:
                patch.setattr("core.project.save_project", lambda **_: False)
                assert not project.save()
            run(Project.load(project.path), apply=True)
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
                store.get_result(rid)["committed"]
                for rid in project.metadata.job_results
            )
    finally:
        store.close()


def test_runtime_history_does_not_implicitly_publish(setup):
    project, compute = setup
    before = project.path.read_bytes()
    worker = worker_for(project)
    completed = []
    worker.analysis_completed.connect(lambda: completed.append(True))
    worker.run()
    assert completed == [True]
    assert len(worker.result) == 2
    assert all(outcome.status == "succeeded" for outcome in worker.result)
    assert not project.metadata.job_results
    assert project.path.read_bytes() == before
    store = JobStore(project.path.parent / "jobs.db")
    try:
        row = store.get(worker.task_id)
        assert row.status == "completed"
        assert row.result["publication"] == "explicit_project_save"
    finally:
        store.close()
    assert compute.call_count == 2


def test_session_only_cancelled_worker_settles_once(setup):
    project, compute = setup
    project.path = None
    worker = worker_for(project)
    assert worker.cache is None
    assert worker.operation.persistence == "session_only"
    worker.cancel()
    worker.run()
    assert worker.job_status == "cancelled"
    assert len(worker.result) == 2
    assert all(outcome.status == "unprocessed" for outcome in worker.result)
    compute.assert_not_called()


def test_recovered_success_survives_later_preparation_failure(setup, monkeypatch):
    project, compute = setup
    run(project, limit=1)
    worker = worker_for(project)
    monkeypatch.setattr(
        worker, "_prepare", Mock(side_effect=RuntimeError("preparation failed"))
    )
    worker.run()
    assert [outcome.status for outcome in worker.result] == ["succeeded", "failed"]
    assert worker.job_status == "failed"
    assert compute.call_count == 1


def test_inference_media_change_never_records_success(setup):
    project, compute = setup
    worker = worker_for(project)

    def change(*_):
        worker.tasks[0].thumbnail_path.write_bytes(b"changed")
        return "wide", 0.9

    compute.side_effect = change
    worker.run()
    assert worker.result[0].status == "failed"
    assert (
        worker.tasks[0].clip_id
        not in worker.cache.journals[worker.tasks[0].target_type].results
    )


def test_runtime_change_does_not_reuse_old_computation(setup, monkeypatch):
    project, compute = setup
    run(project)
    monkeypatch.setattr("core.jobs.gui_shots._runtime", lambda: {"changed": True})
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


@pytest.mark.parametrize("change", ["confidence", "target_type", "digest"])
def test_invalid_cached_outcome_is_not_recomputed_or_published(setup, change):
    from hashlib import sha256
    import json
    import sqlite3

    project, compute = setup
    run(project)
    with sqlite3.connect(project.path.parent / "jobs.db") as connection:
        rid, raw = connection.execute(
            "SELECT result_id, payload_json FROM job_results LIMIT 1"
        ).fetchone()
        payload = json.loads(raw)
        if change == "confidence":
            del payload["confidence"]
        elif change == "target_type":
            payload["target_type"] = (
                "frame" if payload["target_type"] == "clip" else "clip"
            )
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = "invalid" if change == "digest" else sha256(raw.encode()).hexdigest()
        connection.execute(
            "UPDATE job_results SET payload_json = ?, payload_digest = ? WHERE result_id = ?",
            (raw, digest, rid),
        )
    with pytest.raises((ValueError, StaleJobResult)):
        run(project, apply=True)
    assert not project.metadata.job_results
    assert compute.call_count == 2


def test_cancelled_recovery_does_not_publish(setup):
    project, compute = setup
    run(project)
    cancel = Event()
    cancel.set()
    assert all(
        outcome.status == "unprocessed"
        for outcome in run(project, apply=True, cancel=cancel)
    )
    assert not project.metadata.job_results
    assert compute.call_count == 2


def test_refresh_after_save_creates_new_generation(setup):
    project, compute = setup
    run(project, apply=True)
    assert project.save()
    run(project, apply=True)
    assert compute.call_count == 4
    assert len(project.metadata.job_results) == 4


def test_clip_and_frame_with_same_id_keep_distinct_receipts(setup):
    project, compute = setup
    if project.frames:
        return
    clip = project.clips[0]
    frame = Frame(id=clip.id, file_path=clip.thumbnail_path)
    project.add_frames([frame])
    assert project.save()
    targets = [
        AnalysisTarget.from_clip(clip, project.sources_by_id[clip.source_id]),
        AnalysisTarget.from_frame(frame),
    ]
    worker = ShotTypeWorker([], {}, project=project, analysis_targets=targets)
    compute.side_effect = [("wide", 0.9), ("close-up", 0.8)]
    application = ShotTypeApplication(project, worker.tasks)

    def deliver(outcome):
        assert application.apply(project, outcome)
        receipt = worker.cache.receipt(outcome)
        assert not receipt.matches(
            replace(
                outcome,
                target_type="frame" if outcome.target_type == "clip" else "clip",
            )
        )
        project.record_job_result(receipt.result_id, receipt.digest)

    outcomes = worker.cache.run(
        worker.tasks, Event(), lambda: True, deliver, lambda *_: None
    )
    assert len(outcomes) == len(project.metadata.job_results) == 2
    assert clip.shot_type == "wide" and frame.shot_type == "close-up"
    assert project.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert all(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )
    finally:
        store.close()
