"""Face analysis keeps failures and cancellation out of project data."""

from threading import Event
from unittest.mock import Mock

import pytest

from tests.test_description_operations import project_with_thumbnails
from ui.workers.face_detection_worker import FaceDetectionWorker
from core.operations.faces import FaceTask, FaceOptions, FaceApplication, run_faces
from core.analysis.faces import extract_faces_from_clip as extract_video_faces


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    monkeypatch.setattr("core.analysis.faces._load_insightface", Mock())
    monkeypatch.setattr("core.analysis.faces.unload_model", Mock())
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.analysis.faces.extract_faces_from_clip", provider)
    return project, provider


def tasks(project):
    return tuple(
        FaceTask(
            c.id,
            c.source_id,
            project.sources_by_id[c.source_id].file_path,
            c.start_frame,
            c.end_frame,
            project.sources_by_id[c.source_id].fps,
        )
        for c in project.clips
    )


def test_worker_failure_preserves_missing_analysis_and_completes(setup):
    project, provider = setup
    provider.side_effect = ValueError("bad video")
    worker = FaceDetectionWorker(project.clips, project.sources_by_id)
    completed = []
    worker.detection_completed.connect(lambda: completed.append(True))
    worker.run()
    assert all(c.face_embeddings is None for c in project.clips)
    assert completed == [True]


def test_precancelled_worker_does_not_load_model_and_completes(setup, monkeypatch):
    project, provider = setup
    load = Mock()
    monkeypatch.setattr("core.analysis.faces._load_insightface", load)
    worker = FaceDetectionWorker(project.clips, project.sources_by_id)
    completed = []
    worker.detection_completed.connect(lambda: completed.append(True))
    worker.cancel()
    worker.run()
    assert completed == [True]
    load.assert_not_called()
    provider.assert_not_called()


def test_model_load_failure_completes_worker(setup, monkeypatch):
    project, _ = setup
    monkeypatch.setattr(
        "core.analysis.faces._load_insightface",
        Mock(side_effect=RuntimeError("load failed")),
    )
    worker = FaceDetectionWorker(project.clips, project.sources_by_id)
    completed = []
    worker.detection_completed.connect(lambda: completed.append(True))
    worker.run()
    assert completed == [True]


def test_empty_faces_roundtrip(setup):
    project, _ = setup
    clip = project.clips[0]
    clip.face_embeddings = []
    assert type(clip).from_dict(clip.to_dict()).face_embeddings == []


def test_cancel_during_inference_rejects_late_reply(setup):
    project, provider = setup
    cancel = Event()
    provider.side_effect = lambda **_: (cancel.set() or [])
    delivered = []
    results = run_faces(
        tasks(project), FaceOptions(), cancel_event=cancel, on_outcome=delivered.append
    )
    assert all(o.status == "unprocessed" for o in results)
    assert not delivered
    assert provider.call_count == 1


@pytest.mark.parametrize(
    "change", ["none", "range", "source", "faces", "media", "session"]
)
def test_stale_face_application(setup, change):
    project, provider = setup
    snapshot = tasks(project)
    application = FaceApplication(project, snapshot)
    outcome = run_faces(snapshot, FaceOptions())[0]
    clip = project.clips[0]
    if change == "range":
        clip.start_frame += 1
    elif change == "source":
        project.sources[0].fps += 1
    elif change == "faces":
        clip.face_embeddings = [{"manual": True}]
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "session":
        project.clear()
    assert application.apply(project, outcome) == (change == "none")
    assert not application.apply(project, outcome)


def test_faces_are_detached_and_worker_does_not_publish(setup):
    project, provider = setup
    raw = [
        {
            "bbox": [0, 0, 10, 10],
            "embedding": [0.1] * 512,
            "confidence": 0.8,
            "frame_number": 0,
        }
    ]
    provider.return_value = raw
    worker = FaceDetectionWorker(project.clips, project.sources_by_id)
    emitted = []
    worker.faces_ready.connect(lambda cid, faces: emitted.append(faces))
    worker.run()
    raw[0]["embedding"][0] = 99
    emitted[0][0]["bbox"][0] = 99
    assert worker.result[0].faces[0].embedding[0] == 0.1
    assert worker.result[0].faces[0].bbox[0] == 0
    assert all(c.face_embeddings is None for c in project.clips)


@pytest.mark.parametrize(
    "raw", [None, [{"bbox": [0, 0, 1, 1], "embedding": [], "confidence": 0.8}]]
)
def test_invalid_provider_result_is_failed(setup, raw):
    project, provider = setup
    provider.return_value = raw
    assert all(o.status == "failed" for o in run_faces(tasks(project), FaceOptions()))


def test_spine_uses_same_results_and_notifies(setup):
    from core.spine.analyze import face_embeddings

    project, provider = setup
    result = face_embeddings(project, sample_interval=0.5)["result"]
    assert len(result["succeeded"]) == 2
    assert all(c.face_embeddings == [] for c in project.clips)
    assert provider.call_args.kwargs["sample_interval"] == 0.5


@pytest.mark.parametrize("opened", [False, True])
def test_unreadable_video_is_not_an_empty_success(setup, monkeypatch, opened):
    import sys
    from types import SimpleNamespace
    from core.analysis import faces

    project, _ = setup
    monkeypatch.setattr(faces, "_load_insightface", Mock())
    cap = Mock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (False, None)
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=lambda *_: cap, CAP_PROP_POS_FRAMES=1),
    )
    with pytest.raises(ValueError):
        extract_video_faces(project.sources[0].file_path, 0, 60, 30)
    cap.release.assert_called_once()


def test_waiting_face_job_cancels_without_loading_or_unloading_active_model(
    setup, monkeypatch
):
    from threading import Thread, Lock, current_thread

    project, provider = setup
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

    monkeypatch.setattr("core.operations.faces._inference_lock", ObservedLock())
    unload = Mock()
    monkeypatch.setattr("core.analysis.faces.unload_model", unload)

    def compute(**kwargs):
        entered.set()
        assert release.wait(5)
        return []

    provider.side_effect = compute
    first = Thread(target=lambda: run_faces(tasks(project)[:1], FaceOptions()))
    second = Thread(
        target=lambda: results.extend(
            run_faces(tasks(project)[:1], FaceOptions(), cancel_event=cancel)
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


def test_pipeline_completion_does_not_unload_another_active_job(setup, monkeypatch):
    from threading import Thread
    from PySide6.QtCore import QObject
    from core.settings import Settings
    from ui.workers.clip_analysis import ClipAnalysisController

    project, provider = setup
    entered, release = Event(), Event()
    unload = Mock()
    monkeypatch.setattr("core.analysis.faces.unload_model", unload)
    results = []

    def compute(**kwargs):
        entered.set()
        assert release.wait(5)
        return []

    provider.side_effect = compute
    snapshot = tasks(project)[:1]
    active = Thread(target=lambda: results.extend(run_faces(snapshot, FaceOptions())))
    window = QObject()
    window.project = project
    window.settings = Settings()
    monkeypatch.setattr(FaceDetectionWorker, "start", Mock())
    controller = ClipAnalysisController(window, project.clips, ["face_embeddings"])
    controller.start()
    worker = controller.workers["face_embeddings"]
    active.start()
    try:
        assert entered.wait(5)
        # A prior worker's queued completion can arrive after this job starts.
        worker.finished.emit()
        assert "face_embeddings" in controller.plan.results
        controller.cancel()
        unload.assert_not_called()
    finally:
        release.set()
        active.join(5)
    assert not active.is_alive()
    assert results[0].status == "succeeded"
    unload.assert_called_once()
