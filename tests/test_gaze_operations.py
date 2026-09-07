"""Gaze computation must not mutate editor state or publish cancelled work."""

from unittest.mock import Mock

import pytest

from tests.test_description_operations import project_with_thumbnails
from ui.workers.gaze_worker import GazeAnalysisWorker
from core.analysis.gaze import extract_gaze_from_clip as extract_video_gaze
from core.operations.gaze import GazeTask, GazeOptions, GazeApplication, run_gaze


def tasks(project):
    return tuple(
        GazeTask(
            c.id,
            c.source_id,
            project.sources_by_id[c.source_id].file_path,
            c.start_frame,
            c.end_frame,
            project.sources_by_id[c.source_id].fps,
        )
        for c in project.clips
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    load = Mock()
    provider = Mock(
        return_value={"gaze_yaw": 2.0, "gaze_pitch": 1.0, "gaze_category": "at_camera"}
    )
    monkeypatch.setattr("core.analysis.gaze.is_model_loaded", lambda: False)
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", load)
    monkeypatch.setattr("core.analysis.gaze.unload_model", Mock())
    monkeypatch.setattr("core.analysis.gaze.extract_gaze_from_clip", provider)
    return project, load, provider


def test_worker_keeps_project_detached(setup):
    project, _, _ = setup
    worker = GazeAnalysisWorker(project.clips, project.sources_by_id)
    results = []
    worker.gaze_ready.connect(lambda *args: results.append(args))
    worker.run()
    assert len(results) == 2
    assert all(c.gaze_category is None for c in project.clips)


def test_precancelled_worker_does_not_load_model(setup):
    project, load, provider = setup
    worker = GazeAnalysisWorker(project.clips, project.sources_by_id)
    worker.cancel()
    worker.run()
    load.assert_not_called()
    provider.assert_not_called()


def test_cancel_during_inference_rejects_late_result(setup):
    project, _, provider = setup
    worker = GazeAnalysisWorker(project.clips, project.sources_by_id)
    result = provider.return_value
    provider.side_effect = lambda **_: (worker.cancel() or result)
    signals = []
    worker.gaze_ready.connect(lambda *args: signals.append(args))
    worker.run()
    assert not signals
    assert all(c.gaze_category is None for c in project.clips)


@pytest.mark.parametrize("failure", ["open", "dimensions", "read"])
def test_unreadable_video_is_failure_not_absent_gaze(setup, monkeypatch, failure):
    cap = Mock()
    cap.isOpened.return_value = failure != "open"
    cap.get.return_value = 0 if failure == "dimensions" else 100
    cap.read.return_value = (False, None)
    monkeypatch.setattr("core.analysis.gaze.cv2.VideoCapture", lambda _: cap)
    with pytest.raises(ValueError):
        extract_video_gaze("video.mp4", 0, 30, 30.0)
    cap.release.assert_called_once()


@pytest.mark.parametrize(
    "change", ["none", "range", "source", "gaze", "media", "session"]
)
def test_application_rejects_stale_targets(setup, change):
    project, _, _ = setup
    snapshot = tasks(project)
    application = GazeApplication(project, snapshot)
    outcome = run_gaze(snapshot, GazeOptions())[0]
    clip = project.clips[0]
    if change == "range":
        clip.start_frame += 1
    elif change == "source":
        project.sources[0].fps += 1
    elif change == "gaze":
        clip.gaze_yaw = 99
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "session":
        project.clear()
    assert application.apply(project, outcome) == (change == "none")
    assert not application.apply(project, outcome)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "2"])
def test_invalid_provider_angles_are_failures(setup, value):
    project, _, provider = setup
    provider.return_value["gaze_yaw"] = value
    outcomes = run_gaze(tasks(project), GazeOptions())
    assert all(o.status == "failed" for o in outcomes)


def test_no_gaze_keeps_legacy_result_distinct_from_decode_error(setup):
    from core.spine.analyze import gaze

    project, _, provider = setup
    provider.side_effect = [None, ValueError("decode failed")]
    result = gaze(project)["result"]
    assert [o["code"] for o in result["failed"]] == ["no_gaze_detected", "gaze_failed"]
    assert all(c.gaze_category is None for c in project.clips)


def test_spine_uses_sampling_and_guarded_application(setup):
    from core.spine.analyze import gaze

    project, _, provider = setup
    result = gaze(project, sample_interval=0.4)["result"]
    assert len(result["succeeded"]) == 2
    assert all(c.gaze_category == "at_camera" for c in project.clips)
    assert provider.call_args.kwargs["sample_interval"] == 0.4


def test_waiting_job_cancels_without_unloading_active_model(setup, monkeypatch):
    from threading import Thread, Lock, Event, current_thread

    project, _, provider = setup
    entered, release, waiting, cancel = Event(), Event(), Event(), Event()
    lock = Lock()
    results = []

    class ObservedLock:
        def acquire(self, **kwargs):
            if current_thread() is second:
                waiting.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    monkeypatch.setattr("core.operations.gaze._inference_lock", ObservedLock())
    unload = Mock()
    monkeypatch.setattr("core.analysis.gaze.unload_model", unload)
    result = provider.return_value

    def compute(**kwargs):
        entered.set()
        assert release.wait(5)
        return result

    provider.side_effect = compute
    first = Thread(target=lambda: run_gaze(tasks(project)[:1], GazeOptions()))
    second = Thread(
        target=lambda: results.extend(
            run_gaze(tasks(project)[:1], GazeOptions(), cancel_event=cancel)
        )
    )
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert waiting.wait(5)
        cancel.set()
        second.join(5)
        assert not second.is_alive()
        assert results[0].status == "unprocessed"
        unload.assert_not_called()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    unload.assert_called_once()


def test_pipeline_launcher_applies_through_owner(setup, monkeypatch):
    from PySide6.QtCore import QObject
    from ui.main_window import MainWindow

    project, _, _ = setup
    window = QObject()
    window.project = project
    window.sources = project.sources
    for name in (
        "_reset_analysis_run_error",
        "_on_gaze_progress",
        "_on_gaze_ready",
        "_on_gaze_error",
        "_on_pipeline_gaze_finished",
    ):
        setattr(window, name, Mock())
    monkeypatch.setattr(GazeAnalysisWorker, "start", Mock())
    MainWindow._launch_gaze_worker(window, project.clips)
    window._gaze_worker.gaze_ready.emit(project.clips[0].id, 2.0, 1.0, "at_camera")
    assert project.clips[0].gaze_category == "at_camera"
    window._on_gaze_ready.assert_called_once()
    window._on_gaze_error.assert_not_called()


def test_spine_does_not_overwrite_edit_during_inference(setup):
    from core.spine.analyze import gaze

    project, _, provider = setup
    result = provider.return_value

    def edited(**kwargs):
        project.clips[0].gaze_category = "looking_left"
        return result

    provider.side_effect = edited
    outcome = gaze(project)["result"]
    assert outcome["failed"][0]["code"] == "stale_input"
    assert project.clips[0].gaze_category == "looking_left"
    assert len(outcome["succeeded"]) == 1
