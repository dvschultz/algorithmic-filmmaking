"""GUI descriptions survive restart without implicitly saving project edits."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis_target import AnalysisTarget
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.description import DescriptionApplication, DescriptionOptions
from core.project import Project
from models.frame import Frame
from tests.test_description_operations import project_with_thumbnails
from ui.workers.description_worker import DescriptionWorker

OPTIONS = DescriptionOptions(
    tier="cloud", model="test-model", input_mode="frame", parallelism=2
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
        "ui.workers.description_worker.resolve_options", lambda *args: OPTIONS
    )
    compute = Mock(return_value=("Generated", "test-model"))
    monkeypatch.setattr("core.analysis.description.describe_frame", compute)
    return project, compute


def worker_for(project, *, skip_existing=False):
    return DescriptionWorker(
        project.clips,
        sources=project.sources_by_id,
        tier="cloud",
        project=project,
        skip_existing=skip_existing,
        analysis_targets=[AnalysisTarget.from_frame(f) for f in project.frames] or None,
    )


def run(
    project,
    *,
    apply=False,
    prepare=lambda: True,
    cancel=None,
    limit=None,
    skip_existing=False,
):
    worker = worker_for(project, skip_existing=skip_existing)
    tasks = worker.tasks[:limit]
    application = DescriptionApplication(project, tasks)

    def deliver(outcome):
        if apply and outcome.can_apply:
            assert application.apply(project, outcome)
            receipt = worker.cache.results.get(outcome.clip_id)
            if receipt is not None:
                assert receipt.matches(outcome)
                project.record_job_result(receipt.result_id, receipt.digest)
            else:
                from dataclasses import asdict

                assert worker.cache.transient_outcomes[outcome.clip_id] == asdict(
                    outcome
                )

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
        t.description is None
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
        "ui.workers.description_worker.resolve_options",
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
            target.description = "Manual edit"
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
    worker.description_completed.connect(lambda: completed.append(True))
    worker.run()
    assert completed == [True]
    assert worker.success_count == 2
    row = JobStore(project.path.parent / "jobs.db").get(worker.task_id)
    assert row.status == "completed"
    assert row.result["publication"] == "explicit_project_save"
    assert not project.metadata.job_results
    assert compute.call_count == 2


def test_unsaved_media_changed_during_model_load_prevents_provider_call(
    setup, monkeypatch
):
    project, compute = setup
    worker = DescriptionWorker(
        project.clips,
        sources=project.sources_by_id,
        options=replace(OPTIONS, tier="local"),
    )
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    monkeypatch.setattr("core.analysis.description.is_model_loaded", lambda *_: False)
    monkeypatch.setattr(
        "core.analysis.description._load_local_model",
        lambda *_: project.clips[0].thumbnail_path.write_bytes(b"changed"),
    )
    worker.run()
    compute.assert_not_called()
    assert worker.error_count == 2


def test_partial_cache_success_survives_preparation_failure(setup, monkeypatch):
    project, compute = setup
    run(project, limit=1)
    worker = worker_for(project)
    monkeypatch.setattr(
        worker, "_prepare", Mock(side_effect=RuntimeError("preflight failed"))
    )
    worker.run()
    assert worker.success_count == 1
    assert worker.error_count == 1
    assert [outcome.status for outcome in worker.result] == ["succeeded", "failed"]
    assert worker.job_status == "failed"
    assert compute.call_count == 1


def test_verified_gui_records_survive_missing_receipt_cache(setup, monkeypatch):
    project, compute = setup
    run(project, apply=True)
    assert project.save()
    reopened = Project.load(project.path)
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(cache_dir=project.path.parent / "fresh-cache"),
    )
    outcomes = run(reopened, apply=True, skip_existing=True)
    assert all(o.status == "skipped" and o.code == "valid_analysis" for o in outcomes)
    assert compute.call_count == 2


def test_legacy_gui_text_is_recomputed(setup):
    project, compute = setup
    for target in project.frames or project.clips:
        target.description = "Legacy text"
    outcomes = run(project, apply=True, skip_existing=True)
    assert all(o.status == "succeeded" and o.record_json for o in outcomes)
    assert compute.call_count == 2


def test_gui_failure_retains_display_and_does_not_add_receipts(setup, monkeypatch):
    project, compute = setup
    run(project, apply=True)
    receipts = dict(project.metadata.job_results)
    compute.side_effect = RuntimeError("Provider rejected request")
    monkeypatch.setattr(
        "ui.workers.description_worker.resolve_options",
        lambda *args: replace(OPTIONS, prompt="Changed prompt"),
    )
    outcomes = run(project, apply=True, skip_existing=True)
    assert all(o.status == "failed" and o.record_json for o in outcomes)
    assert project.metadata.job_results == receipts
    assert all(
        t.description == "Generated"
        and t.analysis_records["describe"].state == "failed"
        for t in project.frames or project.clips
    )


def test_checkpoint_requires_exact_analysis_record(setup):
    project, _ = setup
    run(project, apply=True)
    for target in project.frames or project.clips:
        target.analysis_records["describe"] = replace(
            target.analysis_records["describe"], input_json="{}"
        )
    assert project.save()
    store = JobStore(project.path.parent / "jobs.db")
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


def test_runtime_change_invalidates_gui_receipt(setup, monkeypatch):
    project, compute = setup
    run(project)
    monkeypatch.setattr(
        "core.operations.description.model_runtime",
        lambda *args: {"packages": {"test": "changed"}},
    )
    run(project)
    assert compute.call_count == 4


def test_verified_local_reuse_does_not_load_weights(setup, monkeypatch):
    project, compute = setup
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    loader = Mock()
    monkeypatch.setattr("core.analysis.description._load_local_model", loader)
    options = replace(OPTIONS, tier="local")
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, options=options
    )
    application = DescriptionApplication(project, worker.tasks, options)
    worker.run()
    assert all(application.apply(project, outcome) for outcome in worker.result)
    assert loader.call_count == 2
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, options=options
    )
    worker.run()
    assert all(o.status == "skipped" for o in worker.result)
    assert loader.call_count == compute.call_count == 2


def test_gui_receipt_reuse_ignores_parallelism(setup, monkeypatch):
    project, compute = setup
    first = run(project)
    monkeypatch.setattr(
        "ui.workers.description_worker.resolve_options",
        lambda *args: replace(OPTIONS, parallelism=4),
    )
    assert run(project, prepare=Mock(side_effect=AssertionError("must reuse"))) == first
    assert compute.call_count == 2
