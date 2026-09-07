"""Shared object detection must preserve immutable results and cancellation."""

from threading import Event
from unittest.mock import Mock

import pytest

from core.operations.object_detection import (
    ObjectDetectionOptions,
    ObjectDetectionTask,
    run_object_detection,
)
from tests.test_description_operations import project_with_thumbnails
from ui.workers.object_detection_worker import ObjectDetectionWorker


def test_pre_cancelled_worker_does_not_prepare_model(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    prepare = Mock()
    monkeypatch.setattr(
        "core.analysis.detection.ensure_default_detection_model_loaded", prepare
    )
    worker = ObjectDetectionWorker(project.clips)
    completed = []
    worker.detection_completed.connect(lambda: completed.append(True))
    worker.cancel()
    worker.run()
    prepare.assert_not_called()
    assert completed == [True]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [{"label": "person", "confidence": float("nan"), "bbox": [0, 0, 1, 1]}],
        [{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1]}],
    ],
)
def test_invalid_response_is_a_failure(tmp_path, monkeypatch, raw):
    image = project_with_thumbnails(tmp_path, 1).clips[0].thumbnail_path
    monkeypatch.setattr(
        "core.analysis.detection.detect_objects", lambda *args, **kwargs: raw
    )
    result = run_object_detection(
        (ObjectDetectionTask("clip", image),), ObjectDetectionOptions()
    )
    assert result[0].status == "failed"
    assert result[0].code == "detection_failed"


def test_results_do_not_share_provider_or_output_containers(tmp_path, monkeypatch):
    image = project_with_thumbnails(tmp_path, 1).clips[0].thumbnail_path
    raw = [{"label": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4]}]
    monkeypatch.setattr(
        "core.analysis.detection.detect_objects", lambda *args, **kwargs: raw
    )
    result = run_object_detection(
        (ObjectDetectionTask("clip", image),), ObjectDetectionOptions()
    )[0]
    raw[0]["bbox"][0] = 99
    emitted = result.detection_dicts()
    emitted[0]["bbox"][1] = 100
    assert result.detections[0].bbox == (1, 2, 3, 4)
    assert result.person_count == 1


def test_cancellation_during_inference_suppresses_reply_and_later_work(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 2)
    cancel = Event()
    compute = Mock(side_effect=lambda *args, **kwargs: (cancel.set() or []))
    monkeypatch.setattr("core.analysis.detection.detect_objects", compute)
    delivered = []
    result = run_object_detection(
        tuple(ObjectDetectionTask(c.id, c.thumbnail_path) for c in project.clips),
        ObjectDetectionOptions(),
        cancel_event=cancel,
        on_outcome=delivered.append,
    )
    assert [o.status for o in result] == ["unprocessed", "unprocessed"]
    assert delivered == []
    assert compute.call_count == 1


@pytest.mark.parametrize("detect_all", [False, True])
def test_gui_and_spine_share_parameters_and_results(tmp_path, monkeypatch, detect_all):
    from core.spine.analyze import detect_objects

    project = project_with_thumbnails(tmp_path, 1)
    objects = Mock(
        return_value=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    )
    people = Mock(return_value=2)
    monkeypatch.setattr("core.analysis.detection.detect_objects", objects)
    monkeypatch.setattr("core.analysis.detection.count_people", people)
    worker = ObjectDetectionWorker(
        project.clips, confidence=0.4, detect_all=detect_all, parallelism=4
    )
    emitted = []
    worker.objects_ready.connect(
        lambda cid, detections, count: emitted.append((cid, detections, count))
    )
    worker.run()
    result = detect_objects(project, confidence=0.4, detect_all=detect_all)["result"]
    provider = objects if detect_all else people
    assert provider.call_count == 2
    assert provider.call_args.kwargs == {"confidence_threshold": 0.4}
    assert project.clips[0].detected_objects == (emitted[0][1] if detect_all else None)
    assert (
        result["succeeded"][0]["person_count"]
        == emitted[0][2]
        == (1 if detect_all else 2)
    )


@pytest.mark.parametrize("count", [-1, True, None, 1.5])
def test_invalid_people_count_is_not_a_success(tmp_path, monkeypatch, count):
    image = project_with_thumbnails(tmp_path, 1).clips[0].thumbnail_path
    monkeypatch.setattr(
        "core.analysis.detection.count_people", lambda *args, **kwargs: count
    )
    outcome = run_object_detection(
        (ObjectDetectionTask("clip", image),), ObjectDetectionOptions(detect_all=False)
    )[0]
    assert outcome.status == "failed"


def test_download_failure_does_not_repeat_for_every_target(tmp_path, monkeypatch):
    from core.errors import ModelDownloadError

    project = project_with_thumbnails(tmp_path, 2)
    provider = Mock(side_effect=ModelDownloadError("network down"))
    monkeypatch.setattr("core.analysis.detection.detect_objects", provider)
    result = run_object_detection(
        tuple(ObjectDetectionTask(c.id, c.thumbnail_path) for c in project.clips),
        ObjectDetectionOptions(),
    )
    assert provider.call_count == 1
    assert [o.status for o in result] == ["failed", "unprocessed"]
    assert result[0].code == "model_load_failed"


def test_failed_item_and_empty_detection_are_distinct(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    monkeypatch.setattr(
        "core.analysis.detection.detect_objects",
        Mock(side_effect=[ValueError("bad image"), []]),
    )
    result = run_object_detection(
        tuple(ObjectDetectionTask(c.id, c.thumbnail_path) for c in project.clips),
        ObjectDetectionOptions(),
    )
    assert [o.status for o in result] == ["failed", "succeeded"]
    assert result[1].detections == ()
    assert result[1].person_count == 0


def test_worker_reports_model_failure_after_an_earlier_item_error(tmp_path, monkeypatch):
    from core.errors import ModelDownloadError

    project = project_with_thumbnails(tmp_path, 2)
    monkeypatch.setattr(
        "core.analysis.detection.detect_objects",
        Mock(side_effect=[ValueError("bad image"), ModelDownloadError("model unavailable")]),
    )
    worker = ObjectDetectionWorker(project.clips)
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors == ["model unavailable"]


def test_frame_identity_survives_worker_task_building(tmp_path):
    from core.analysis_target import AnalysisTarget
    from models.frame import Frame

    image = project_with_thumbnails(tmp_path, 1).clips[0].thumbnail_path
    worker = ObjectDetectionWorker(
        [],
        analysis_targets=[
            AnalysisTarget.from_frame(Frame(id="frame", file_path=image))
        ],
    )
    assert worker.tasks[0].target_type == "frame"


def test_waiting_job_can_cancel_without_concurrent_inference(tmp_path, monkeypatch):
    from threading import Thread, Lock, current_thread

    image = project_with_thumbnails(tmp_path, 1).clips[0].thumbnail_path
    entered, release, waiting, cancel = Event(), Event(), Event(), Event()
    results = []
    lock = Lock()

    class ObservedLock:
        def acquire(self, **kwargs):
            if current_thread() is second:
                waiting.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    monkeypatch.setattr(
        "core.operations.object_detection._inference_lock", ObservedLock()
    )

    def provider(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return []

    compute = Mock(side_effect=provider)
    monkeypatch.setattr("core.analysis.detection.detect_objects", compute)
    tasks = (ObjectDetectionTask("clip", image),)
    first = Thread(target=lambda: run_object_detection(tasks, ObjectDetectionOptions()))

    def second_run():
        results.extend(
            run_object_detection(tasks, ObjectDetectionOptions(), cancel_event=cancel)
        )

    second = Thread(target=second_run)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert waiting.wait(5)
        cancel.set()
        second.join(5)
        assert not second.is_alive()
        assert results[0].status == "unprocessed"
        assert compute.call_count == 1
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)


@pytest.mark.parametrize("command", ["objects", "people"])
def test_cli_uses_shared_inference_and_preserves_analysis_image_size(
    tmp_path, monkeypatch, command
):
    import json
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.project import Project
    from core.settings import Settings

    project = project_with_thumbnails(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=tmp_path)
    )
    generator = Mock()
    generator.generate_clip_thumbnail.return_value = project.clips[0].thumbnail_path
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", lambda **kwargs: generator)
    objects = Mock(
        return_value=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    )
    people = Mock(return_value=3)
    monkeypatch.setattr("core.analysis.detection.detect_objects", objects)
    monkeypatch.setattr("core.analysis.detection.count_people", people)
    register_commands()
    result = CliRunner().invoke(
        cli, ["--json", "analyze", command, str(path), "--confidence", "0.4"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["analyzed_clips"] == 1
    assert generator.generate_clip_thumbnail.call_args.kwargs["width"] == 320
    assert (objects if command == "objects" else people).call_args.kwargs == {
        "confidence_threshold": 0.4
    }
    assert Project.load(path).clips[0].person_count == (
        1 if command == "objects" else 3
    )
