"""Combined GUI controllers publish verified failures without erasing projections."""

import time
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("saved", [False, True])
def test_combined_embedding_reuse_refreshes_file_bindings(tmp_path, monkeypatch, saved):
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.settings import Settings
    from core.spine.analyze import embeddings
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.clip_analysis import ClipAnalysisController

    app = QApplication.instance() or QApplication([])
    window = QObject()
    window.project = project_with_thumbnails(tmp_path, 1)
    window.settings = Settings(cache_dir=tmp_path / "cache")
    monkeypatch.setattr("core.settings.load_settings", lambda: window.settings)
    provider = Mock(return_value=[[0.1] * 768])
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", provider
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", lambda: None)
    embeddings(window.project)
    clip = window.project.clips[0]
    original = clip.analysis_records["embeddings"]
    moved = tmp_path / "moved.jpg"
    moved.write_bytes(clip.thumbnail_path.read_bytes())
    clip.thumbnail_path = moved
    if saved:
        assert window.project.save(tmp_path / "project.json")
    controller = ClipAnalysisController(window, [clip], ["embeddings"])
    try:
        controller.start()
        deadline = time.monotonic() + 10
        while not controller.finished and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.002)
        assert controller.finished
        assert set(controller.plan.results["embeddings"].values()) == {"skipped"}
        assert clip.analysis_records["embeddings"].identity == original.identity
        assert clip.analysis_records["embeddings"].input_json != original.input_json
        assert provider.call_count == 1
    finally:
        controller.cancel()
        app.processEvents()
        window.project.close_writer()


@pytest.mark.parametrize("operation", ["classify", "detect_objects", "shots"])
def test_frame_inference_uses_image_when_original_video_is_offline(
    tmp_path, monkeypatch, operation
):
    from core.analysis_target import AnalysisTarget
    from models.frame import Frame
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.classification_worker import ClassificationWorker
    from ui.workers.object_detection_worker import ObjectDetectionWorker
    from ui.workers.shot_type_worker import ShotTypeWorker
    from core.operations.shots import ShotTypeOptions

    project = project_with_thumbnails(tmp_path, 1)
    frame = Frame(
        id="frame",
        source_id=project.sources[0].id,
        file_path=project.clips[0].thumbnail_path,
    )
    project.add_frames([frame])
    project.sources[0].file_path.unlink()
    provider = Mock(return_value=("wide", 0.9) if operation == "shots" else [])
    paths = {
        "classify": "core.analysis.classification.classify_frame",
        "detect_objects": "core.analysis.detection.detect_objects",
        "shots": "core.analysis.shots.classify_shot_type",
    }
    monkeypatch.setattr(paths[operation], provider)
    worker_type = {
        "classify": ClassificationWorker,
        "detect_objects": ObjectDetectionWorker,
        "shots": ShotTypeWorker,
    }[operation]
    extra = (
        {"sources_by_id": project.sources_by_id, "options": ShotTypeOptions()}
        if operation == "shots"
        else {}
    )
    worker = worker_type(
        [],
        project=project,
        analysis_targets=[AnalysisTarget.from_frame(frame)],
        **extra,
    )
    worker.run()
    assert worker.result[0].status == "succeeded"
    provider.assert_called_once()


def test_combined_ocr_force_reaches_worker(tmp_path, monkeypatch):
    from core.settings import Settings
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.clip_analysis_work import create_clip_analysis_worker

    project = project_with_thumbnails(tmp_path, 1)
    settings = Settings(cache_dir=tmp_path / "cache")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    worker, _ = create_clip_analysis_worker(
        project, settings, "extract_text", project.clips, force_rerun=True
    )
    assert worker.tasks and all(not task.skip for task in worker.tasks)


@pytest.mark.parametrize("operation", ["classify", "detect_objects", "extract_text"])
def test_combined_controller_rechecks_requested_options(
    tmp_path, monkeypatch, operation
):
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.operations.clip_analysis import ClipAnalysisOptions
    from core.settings import Settings
    from core.spine.analyze import classify_content, detect_objects, extract_text
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.clip_analysis import ClipAnalysisController

    app = QApplication.instance() or QApplication([])
    window = QObject()
    window.project = project_with_thumbnails(tmp_path, 1)
    window.settings = Settings(
        cache_dir=tmp_path / "cache",
        text_extraction_method="hybrid",
        text_extraction_vlm_model="next",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: window.settings)
    provider = Mock(return_value=[])
    paths = {
        "classify": "core.analysis.classification.classify_frame",
        "detect_objects": "core.analysis.detection.detect_objects",
        "extract_text": "core.analysis.ocr.extract_text_from_clip",
    }
    monkeypatch.setattr(paths[operation], provider)
    if operation == "classify":
        classify_content(window.project)
        options = ClipAnalysisOptions(top_k=2)
    elif operation == "detect_objects":
        detect_objects(window.project)
        options = ClipAnalysisOptions(confidence=0.9)
    else:
        extract_text(window.project, vlm_model="original")
        options = ClipAnalysisOptions()
    assert window.project.save(tmp_path / "project.json")
    controller = ClipAnalysisController(
        window, window.project.clips, [operation], options=options
    )
    try:
        controller.start()
        deadline = time.monotonic() + 10
        while not controller.finished and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.002)
        assert controller.finished
        assert provider.call_count == 2
        assert set(controller.plan.results[operation].values()) == {"succeeded"}
    finally:
        controller.cancel()
        app.processEvents()
        window.project.close_writer()


@pytest.mark.parametrize(
    "kind,operation",
    [
        (kind, operation)
        for kind in ("clip", "frame")
        for operation in ("classify", "detect_objects", "extract_text", "shots")
    ]
    + [("clip", "gaze")],
)
def test_combined_controller_retains_failed_attempt(
    tmp_path, monkeypatch, kind, operation
):
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QApplication
    from core.settings import Settings
    from models.frame import Frame
    from tests.test_description_operations import project_with_thumbnails
    from ui.workers.clip_analysis import ClipAnalysisController
    from ui.workers.frame_analysis import FrameAnalysisController

    app = QApplication.instance() or QApplication([])
    window = QObject()
    window.project = project_with_thumbnails(tmp_path, 1)
    window.settings = Settings(cache_dir=tmp_path / "cache")
    monkeypatch.setattr("core.settings.load_settings", lambda: window.settings)
    target = window.project.clips[0]
    if kind == "frame":
        target = Frame(id="frame", file_path=target.thumbnail_path)
        window.project.add_frames([target])
    target.object_labels = ["old label"]
    target.detected_objects = []
    target.person_count = 0
    target.extracted_texts = []
    target.shot_type = "old shot"
    target.gaze_yaw = 1.23
    target.gaze_pitch = 2.34
    target.gaze_category = "at_camera"
    assert window.project.save(tmp_path / "project.json")
    provider = Mock(side_effect=RuntimeError("provider unavailable"))
    paths = {
        "classify": "core.analysis.classification.classify_frame",
        "detect_objects": "core.analysis.detection.detect_objects",
        "extract_text": "core.analysis.ocr.extract_text_from_" + kind,
        "shots": "core.analysis.shots.classify_shot_type",
        "gaze": "core.analysis.gaze.extract_gaze_from_clip",
    }
    monkeypatch.setattr(paths[operation], provider)
    if operation == "gaze":
        monkeypatch.setattr("core.analysis.gaze.load_face_mesh", Mock())
        monkeypatch.setattr("core.analysis.gaze.unload_model", Mock())
    controller = (
        FrameAnalysisController(window, [target.id], [operation])
        if kind == "frame"
        else ClipAnalysisController(window, [target], [operation], force_rerun=True)
    )
    try:
        controller.start()
        deadline = time.monotonic() + 10
        while not controller.finished and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.002)
        assert controller.finished
        provider.assert_called_once()
        assert target.analysis_records[operation].state == "failed"
        assert target.object_labels == ["old label"]
        assert target.detected_objects == [] and target.person_count == 0
        assert target.extracted_texts == []
        assert target.shot_type == "old shot"
        assert (target.gaze_yaw, target.gaze_pitch, target.gaze_category) == (
            1.23,
            2.34,
            "at_camera",
        )
        assert not window.project.metadata.job_results
    finally:
        controller.cancel()
        app.processEvents()
        window.project.close_writer()
