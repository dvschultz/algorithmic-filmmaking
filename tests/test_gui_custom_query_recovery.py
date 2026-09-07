"""GUI query receipts recover inference but wait for an explicit project save."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.custom_query import CustomQueryApplication, CustomQueryOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails
from ui.workers.custom_query_worker import CustomQueryWorker

OPTIONS = CustomQueryOptions("cloud", "model", 2)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr(
        "ui.workers.custom_query_worker.resolve_options", lambda *args: OPTIONS
    )
    provider = Mock(return_value=(True, 0.9, "model"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    return project, provider


def worker_for(project):
    return CustomQueryWorker(
        project.clips, "person", project.sources_by_id, tier="cloud", project=project
    )


def run(project, *, apply=False, prepare=lambda: True, limit=None, cancel=None):
    worker = worker_for(project)
    tasks = worker.tasks[:limit]
    application = CustomQueryApplication(project, tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            assert application.apply(project, outcome)
            receipt = worker.cache.results[outcome.clip_id]
            assert receipt.matches(outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return worker.cache.run(tasks, cancel or Event(), prepare, deliver, lambda *_: None)


def test_restart_recovers_without_preparation_or_implicit_save(setup):
    project, provider = setup
    first = run(project)
    reopened = Project.load(project.path)
    assert (
        run(reopened, apply=True, prepare=Mock(side_effect=AssertionError("cache hit")))
        == first
    )
    assert provider.call_count == 2
    assert Project.load(project.path).clips[0].custom_queries is None
    store = JobStore(project.path.parent / "jobs.db")
    assert not any(
        store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
    )
    assert reopened.save()
    assert all(
        store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
    )


def test_multiple_unsaved_appends_are_acknowledged_together(setup):
    project, provider = setup
    run(project, apply=True)
    run(project, apply=True)
    assert provider.call_count == 4
    assert project.save()
    assert all(len(c.custom_queries) == 2 for c in Project.load(project.path).clips)
    store = JobStore(project.path.parent / "jobs.db")
    assert all(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


@pytest.mark.parametrize("change", ["edit", "save_as", "failed_save", "checkpoint"])
def test_saved_receipts_require_matching_published_output(setup, monkeypatch, change):
    project, provider = setup
    run(project, apply=True)
    store = JobStore(project.path.parent / "jobs.db")
    if change == "edit":
        for clip in project.clips:
            clip.custom_queries[0]["match"] = False
        assert project.save()
    elif change == "save_as":
        assert project.save(project.path.parent / "copy.json")
    elif change == "failed_save":
        with monkeypatch.context() as patch:
            patch.setattr("core.project.save_project", lambda **_: False)
            assert not project.save()
        reopened = Project.load(project.path)
        run(reopened, apply=True)
        assert provider.call_count == 2
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


def test_mixed_cache_hits_and_preflight_failure_preserve_success(setup, monkeypatch):
    project, provider = setup
    run(project, limit=1)
    worker = worker_for(project)
    monkeypatch.setattr(worker, "_prepare", Mock(side_effect=RuntimeError("preflight")))
    worker.run()
    assert [o.status for o in worker.result] == ["succeeded", "failed"]
    assert worker.job_status == "failed"
    assert provider.call_count == 1


def test_media_changed_during_preparation_is_not_computed(setup):
    project, provider = setup

    def prepare():
        project.sources[0].file_path.write_bytes(b"changed")
        return True

    with pytest.raises(StaleJobResult, match="source media"):
        run(project, prepare=prepare)
    provider.assert_not_called()


def test_cancelled_cache_recovery_does_not_publish(setup):
    project, provider = setup
    run(project)
    cancel = Event()
    cancel.set()
    assert all(
        o.status == "unprocessed" for o in run(project, apply=True, cancel=cancel)
    )
    assert not project.metadata.job_results
    assert provider.call_count == 2


def test_saved_worker_retains_computation_history(setup):
    project, _ = setup
    worker = worker_for(project)
    completed = []
    worker.analysis_completed.connect(lambda: completed.append(True))
    worker.run()
    assert completed == [True]
    row = JobStore(project.path.parent / "jobs.db").get(worker.task_id)
    assert row.status == "completed"
    assert row.result["publication"] == "explicit_project_save"
    assert not project.metadata.job_results


def test_corrupted_committed_identity_blocks_cached_history(setup):
    project, provider = setup
    run(project, apply=True)
    store = JobStore(project.path.parent / "jobs.db")
    with store._connect() as connection:
        row = connection.execute(
            "SELECT result_id,spec_json FROM job_results LIMIT 1"
        ).fetchone()
        spec = json.loads(row[1])
        spec["arguments"]["model"] = "tampered"
        connection.execute(
            "UPDATE job_results SET spec_json=? WHERE result_id=?",
            (json.dumps(spec), row[0]),
        )
    with pytest.raises(StaleJobResult, match="identity"):
        run(project)
    assert provider.call_count == 2
