"""GUI and agent scalar requests use the shared analysis controller."""

import time

import pytest

from tests import test_scalar_records

scalar_setup = test_scalar_records.setup


@pytest.fixture
def saved(scalar_setup, tmp_path, monkeypatch):
    from core.settings import Settings

    project, operation, provider = scalar_setup
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: Settings(cache_dir=tmp_path)
    )
    project.save(tmp_path / "project.sceneripper")
    return project, operation, provider


def controller_for(saved):
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.settings import Settings
    from ui.workers.clip_analysis import ClipAnalysisController

    app = QApplication.instance() or QApplication([])
    project, operation, _ = saved
    window = QObject()
    window.project = project
    window.settings = Settings(cache_dir=project.path.parent)
    controller = ClipAnalysisController(window, project.clips, [operation])
    results = []
    controller.completed.connect(lambda owner, result: results.append(result))
    return app, window, controller, results


def finish(app, controller):
    deadline = time.monotonic() + 10
    while not controller.finished and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    assert controller.finished


def test_agent_scalar_registration_dispatches_without_computing(saved):
    from types import SimpleNamespace
    from core.chat_tools import analyze_all_live
    from core.analysis_operations import OPERATIONS_BY_KEY

    project, operation, provider = saved
    request = analyze_all_live(
        SimpleNamespace(project=project), [project.clips[0].id], [operation]
    )
    assert request["_wait_for_worker"] == "analyze_all"
    assert request["operations"] == [operation]
    assert not OPERATIONS_BY_KEY[operation].default_enabled
    provider.assert_not_called()


def test_gui_scalar_controller_publishes_and_reuses(saved):
    project, operation, provider = saved
    for _ in range(2):
        app, window, controller, results = controller_for(saved)
        controller.start()
        finish(app, controller)
        assert results[0]["succeeded"] == [project.clips[0].id], results
        assert not results[0]["errors"]
        assert operation in project.clips[0].analysis_records
        assert project.metadata.job_results
        assert not window._active_clip_analyses
    assert provider.call_count == 1


def test_gui_scalar_controller_publishes_owned_failure(saved):
    project, operation, provider = saved
    setattr(project.clips[0], test_scalar_records.FIELDS[operation], 0.1)
    provider.side_effect = RuntimeError("decode failed")
    app, window, controller, results = controller_for(saved)
    controller.start()
    finish(app, controller)
    assert results[0]["operations"][operation][project.clips[0].id] == "failed"
    assert project.clips[0].analysis_records[operation].state == "failed"
    assert getattr(project.clips[0], test_scalar_records.FIELDS[operation]) == 0.1
    assert not window._active_clip_analyses


def test_scalar_headless_analysis_plan_runs_registered_operation(saved):
    from core.spine.analyze import analyze_clips

    project, operation, provider = saved
    result = analyze_clips(project, operations=[operation])
    assert result["success"], result
    assert provider.call_count == 1
    assert operation in project.clips[0].analysis_records


def test_scalar_durable_analysis_plan_uses_journal(saved):
    from threading import Event
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.jobs.store import JobStore
    from core.project import Project

    project, operation, provider = saved
    store = JobStore(project.path.parent / "jobs.db")
    try:
        for _ in range(2):
            current = Project.load(project.path)
            spec = analysis_job_spec(
                current, arguments={"operations": [operation], "clip_ids": None}
            )
            result = run_analysis_job(
                store, project.path, spec, lambda *_: None, Event()
            )
            assert result["success"], result
        assert provider.call_count == 1
        assert operation in Project.load(project.path).clips[0].analysis_records
    finally:
        store.close()


@pytest.mark.parametrize("change", ["cancel", "range", "project", "path"])
def test_gui_scalar_controller_rejects_late_results(saved, change, tmp_path):
    from threading import Event
    from core.project import Project

    project, operation, provider = saved
    started, release = Event(), Event()

    def compute(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return 0.5

    provider.side_effect = compute
    app, window, controller, results = controller_for(saved)
    controller.start()
    try:
        assert started.wait(3)
        if change == "cancel":
            controller.cancel()
        elif change == "range":
            project.clips[0].end_frame += 1
        elif change == "project":
            window.project = Project.new()
        else:
            project.path = tmp_path / "other.sceneripper"
    finally:
        release.set()
        finish(app, controller)
    assert not results[0]["succeeded"]
    assert operation not in project.clips[0].analysis_records
