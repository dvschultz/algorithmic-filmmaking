"""GUI query receipts recover inference but wait for an explicit project save."""

import json
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.custom_query import (
    CustomQueryApplication,
    CustomQueryOptions,
    custom_query_record_key,
)
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


def worker_for(project, **kwargs):
    return CustomQueryWorker(
        project.clips,
        "person",
        project.sources_by_id,
        tier="cloud",
        project=project,
        **kwargs,
    )


def run(
    project, *, apply=False, prepare=lambda: True, limit=None, cancel=None, **kwargs
):
    worker = worker_for(project, **kwargs)
    tasks = worker.tasks[:limit]
    application = CustomQueryApplication(project, tasks, worker.options)

    def deliver(outcome):
        if apply and outcome.can_apply:
            assert application.apply(project, outcome)
            receipt = worker.cache.results.get(outcome.clip_id)
            if receipt is not None:
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


def test_verified_gui_reuse_does_not_append_or_require_old_receipts(setup):
    project, provider = setup
    run(project, apply=True)
    assert project.save()
    store = JobStore(project.path.parent / "jobs.db")
    with store._connect() as connection:
        connection.execute("DELETE FROM job_results")
    reopened = Project.load(project.path)
    outcomes = run(reopened, apply=True, skip_existing=True)
    assert len(outcomes) == 2
    assert all(o.status == "skipped" and o.code == "valid_analysis" for o in outcomes)
    assert provider.call_count == 2
    assert all(len(clip.custom_queries) == 1 for clip in reopened.clips)


def test_legacy_query_history_is_computed_even_when_skipping(setup):
    project, provider = setup
    for clip in project.clips:
        clip.custom_queries = [{"query": "person", "match": True}]
    outcomes = run(project, apply=True, skip_existing=True)
    assert len(outcomes) == provider.call_count == 2
    assert all(o.status == "succeeded" and o.record_json for o in outcomes)
    assert all(len(clip.custom_queries) == 2 for clip in project.clips)


def test_gui_failure_records_preserve_query_history(setup):
    project, provider = setup
    run(project, apply=True)
    provider.side_effect = ValueError("Invalid answer")
    outcomes = run(project, apply=True)
    assert all(o.status == "failed" and o.record_json for o in outcomes)
    assert all(len(clip.custom_queries) == 1 for clip in project.clips)
    assert all(
        clip.analysis_records[custom_query_record_key("person")].state == "failed"
        for clip in project.clips
    )


def test_gui_query_receipt_reuse_ignores_parallelism(setup):
    project, provider = setup
    first = run(project)
    assert (
        run(
            project,
            options=replace(OPTIONS, parallelism=4),
            prepare=Mock(side_effect=AssertionError("must reuse")),
        )
        == first
    )
    assert provider.call_count == 2


def test_runtime_change_invalidates_gui_query_receipt(setup, monkeypatch):
    project, provider = setup
    run(project)
    monkeypatch.setattr(
        "core.operations.description.model_runtime",
        lambda *args: {"packages": {"test": "changed"}},
    )
    run(project)
    assert provider.call_count == 4


def test_query_checkpoint_requires_exact_analysis_record(setup):
    project, _ = setup
    run(project, apply=True)
    key = custom_query_record_key("person")
    for clip in project.clips:
        clip.analysis_records[key] = replace(
            clip.analysis_records[key], input_json="{}"
        )
    assert project.save()
    store = JobStore(project.path.parent / "jobs.db")
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


def test_changed_answer_keeps_superseded_receipts_uncommitted(setup):
    project, provider = setup
    run(project, apply=True)
    previous_receipts = set(project.metadata.job_results)
    provider.return_value = False, 0.0, "model"
    run(project, apply=True)
    assert project.save()
    reopened = Project.load(project.path)
    assert all(len(clip.custom_queries) == 2 for clip in reopened.clips)
    store = JobStore(project.path.parent / "jobs.db")
    for result_id in project.metadata.job_results:
        assert store.get_result(result_id)["committed"] == (
            result_id not in previous_receipts
        )


def test_verified_local_query_reuse_does_not_load_weights(setup, monkeypatch):
    project, provider = setup
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    from core.analysis_model_identity import LOCAL_DESCRIPTION_FALLBACK

    provider.return_value = False, 0.0, LOCAL_DESCRIPTION_FALLBACK
    loader = Mock()
    monkeypatch.setattr("core.analysis.description._load_local_model", loader)
    options = replace(OPTIONS, tier="local", model="mlx-community/Qwen-test")
    worker = worker_for(project, options=options)
    application = CustomQueryApplication(project, worker.tasks, options)
    worker.run()
    assert all(application.apply(project, o) for o in worker.result)
    assert loader.call_count == 2
    worker = worker_for(project, options=options, skip_existing=True)
    worker.run()
    assert len(worker.result) == 2
    assert all(o.status == "skipped" for o in worker.result)
    assert loader.call_count == provider.call_count == 2


def test_analysis_target_retains_query_records_for_reuse(setup):
    from core.analysis_target import AnalysisTarget

    project, provider = setup
    run(project, apply=True)
    targets = [
        AnalysisTarget.from_clip(clip, project.sources[0]) for clip in project.clips
    ]
    worker = worker_for(project, analysis_targets=targets, skip_existing=True)
    application = CustomQueryApplication(project, worker.tasks, OPTIONS)
    worker.run()
    assert len(worker.result) == 2
    assert all(o.status == "skipped" for o in worker.result)
    assert all(application.apply(project, o) for o in worker.result)
    assert provider.call_count == 2


def test_media_changed_during_local_loading_is_not_evaluated(setup, monkeypatch):
    project, provider = setup
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    monkeypatch.setattr(
        "core.analysis.description._load_local_model",
        lambda *_: project.sources[0].file_path.write_bytes(b"changed while loading"),
    )
    outcomes = run(project, options=replace(OPTIONS, tier="local"))
    assert all(o.status == "failed" and o.record_json is None for o in outcomes)
    provider.assert_not_called()
