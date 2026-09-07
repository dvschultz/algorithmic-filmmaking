"""Saved GUI gaze jobs recover computations without saving unrelated edits."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.gaze import GazeApplication
from core.project import Project
from tests.test_description_operations import project_with_thumbnails
from ui.workers.gaze_worker import GazeAnalysisWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    compute = Mock(
        return_value={
            "gaze_yaw": 2.123456,
            "gaze_pitch": -1.23456,
            "gaze_category": "at_camera",
        }
    )
    monkeypatch.setattr("core.analysis.gaze.extract_gaze_from_clip", compute)
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", Mock())
    monkeypatch.setattr("core.analysis.gaze.unload_model", Mock())
    return project, compute


def worker_for(project, **kwargs):
    return GazeAnalysisWorker(
        project.clips, project.sources_by_id, project=project, **kwargs
    )


def run(project, *, apply=False, cancel=None, prepare=lambda: True):
    worker = worker_for(project)
    application = GazeApplication(project, worker.tasks)

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
    assert reopened.clips[0].gaze_yaw == 2.123456
    assert Project.load(project.path).clips[0].gaze_category is None
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert reopened.save()
        assert all(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
        assert Project.load(project.path).clips[0].gaze_yaw == 2.12
    finally:
        store.close()


def test_skipped_offline_source_does_not_block_other_results(setup):
    project, compute = setup
    project.clips[0].gaze_category = "at_camera"
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
    assert all(c.gaze_category is None for c in project.clips)
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
            clip.gaze_category = "looking_left"
    assert project.save(path.parent / "copy.json" if change == "save_as" else path)
    store = JobStore(path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )
    finally:
        store.close()


def test_pipeline_launcher_enables_recovery(setup, monkeypatch):
    from PySide6.QtCore import QObject
    from ui.main_window import MainWindow

    project, _ = setup
    window = QObject()
    window.project = project
    window.sources = project.sources
    for name in (
        "_reset_analysis_run_error",
        "_on_gaze_progress",
        "_on_gaze_error",
        "_on_pipeline_gaze_finished",
    ):
        setattr(window, name, Mock())
    monkeypatch.setattr(GazeAnalysisWorker, "start", Mock())
    MainWindow._launch_gaze_worker(window, project.clips)
    assert window._gaze_worker.cache.path == project.path


def test_model_session_is_reused_and_cache_hits_do_not_load(setup, monkeypatch):
    project, _ = setup
    load, unload = Mock(), Mock()
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", load)
    monkeypatch.setattr("core.analysis.gaze.unload_model", unload)
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
                "core.jobs.gui_gaze._runtime", lambda: {"changed": True}
            )
        return True

    with pytest.raises(StaleJobResult):
        run(project, prepare=prepare)
    compute.assert_not_called()


def test_failed_and_empty_results_remain_distinct(setup):
    project, compute = setup
    compute.side_effect = [ValueError("decode failed"), None]
    outcomes = run(project, apply=True)
    assert [o.status for o in outcomes] == ["failed", "succeeded"]
    assert project.clips[0].gaze_category is None
    assert project.clips[1].gaze_category is None
    assert len(project.metadata.job_results) == 1
    assert project.save()


def test_saved_empty_observation_is_skipped_without_recomputation(setup):
    project, compute = setup
    compute.return_value = None
    run(project, apply=True)
    assert project.save()
    reopened = Project.load(project.path)
    outcomes = run(reopened, apply=True)
    assert compute.call_count == 2
    assert all(o.status == "skipped" for o in outcomes)
    assert not reopened.is_dirty


def test_explicit_refresh_recomputes_saved_empty_observation(setup):
    project, compute = setup
    compute.return_value = None
    run(project, apply=True)
    assert project.save()
    worker = worker_for(Project.load(project.path), skip_existing=False)
    worker.run()
    assert compute.call_count == 4
    assert all(
        o.status == "succeeded" and o.code == "no_gaze_detected" for o in worker.result
    )


def test_changed_media_invalidates_saved_empty_observation(setup):
    project, compute = setup
    compute.return_value = None
    run(project, apply=True)
    assert project.save()
    project.sources[0].file_path.write_bytes(b"changed")
    run(Project.load(project.path))
    assert compute.call_count == 4


def test_restart_recovers_unpublished_empty_observation(setup):
    project, compute = setup
    compute.return_value = None
    first = run(project)
    reopened = Project.load(project.path)
    assert (
        run(reopened, apply=True, prepare=Mock(side_effect=AssertionError("cache hit")))
        == first
    )
    assert compute.call_count == 2
    assert len(reopened.metadata.job_results) == 2


@pytest.mark.parametrize("change", ["runtime", "options", "manual_angle"])
def test_changed_inputs_do_not_reuse_saved_empty_observation(
    setup, monkeypatch, change
):
    project, compute = setup
    compute.return_value = None
    run(project, apply=True)
    assert project.save()
    reopened = Project.load(project.path)
    kwargs = {}
    if change == "runtime":
        monkeypatch.setattr("core.jobs.gui_gaze._runtime", lambda: {"version": "new"})
    elif change == "options":
        kwargs["sample_interval"] = 2.0
    else:
        for clip in reopened.clips:
            clip.gaze_yaw = 15.0
    worker = worker_for(reopened, **kwargs)
    worker.run()
    assert worker.job_status == "completed"
    assert compute.call_count == 4
    assert all(o.status == "succeeded" for o in worker.result)


def test_missing_saved_receipt_fails_without_recomputation(setup):
    from core.jobs.commits import StaleJobResult

    project, compute = setup
    compute.return_value = None
    run(project, apply=True)
    assert project.save()
    # Preserve project receipt references while simulating a lost cache.
    (project.path.parent / "jobs.db").unlink()
    with pytest.raises(StaleJobResult, match="payload is missing"):
        run(Project.load(project.path))
    assert compute.call_count == 2


def test_empty_observation_refreshes_existing_gui_badges():
    from ui.main_window import MainWindow

    window = SimpleNamespace(
        clips_by_id={"clip": object()},
        cut_tab=Mock(),
        analyze_tab=Mock(),
        clip_details_sidebar=Mock(),
    )
    MainWindow._on_gaze_ready(window, "clip", None, None, None)
    window.cut_tab.update_clip_gaze.assert_called_once_with("clip", None)
    window.analyze_tab.update_clip_gaze.assert_called_once_with("clip", None)
    window.clip_details_sidebar.refresh_gaze_if_showing.assert_called_once_with("clip")
