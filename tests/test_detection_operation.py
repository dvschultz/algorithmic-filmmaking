"""Contracts for the shared, project-independent detection computation."""

from threading import Event
from unittest.mock import Mock

import pytest

from core.operations.detection import DetectionCancelled, DetectionRequest, run_detection
from core.scene_detect import DetectionConfig, KaraokeDetectionConfig


@pytest.mark.parametrize("mode", ["adaptive", "content", "karaoke"])
def test_settings_are_detached_and_rebuilt_for_each_execution(tmp_path, monkeypatch, mode):
    config = DetectionConfig(threshold=4, use_adaptive=mode != "content")
    karaoke = KaraokeDetectionConfig(language="fr")
    request = DetectionRequest.build(tmp_path / "v.mp4", config, mode=mode, karaoke_config=karaoke)
    config.threshold = 9
    karaoke.language = "en"
    seen = []
    result = (object(), [])

    def create_detector(config):
        seen.append(config.threshold)
        assert config.use_adaptive == (mode != "content")
        config.threshold = 8  # Backend mutation must not alter a retry's request.
        detector = Mock()
        detector.detect_scenes_with_progress.return_value = result

        def detect_karaoke(path, progress, settings):
            assert settings.language == "fr"
            return result

        detector.detect_karaoke_scenes_with_progress.side_effect = detect_karaoke
        return detector

    monkeypatch.setattr("core.scene_detect.SceneDetector", create_detector)
    assert run_detection(request) == result
    assert run_detection(request) == result
    assert seen == [4, 4]


@pytest.mark.parametrize("cancel_before", [True, False])
def test_cancelled_operation_never_delivers_result(tmp_path, monkeypatch, cancel_before):
    cancel = Event()
    detector = Mock()

    def detect(path, progress):
        cancel.set()
        return object(), []

    detector.detect_scenes_with_progress.side_effect = detect
    constructor = Mock(return_value=detector)
    monkeypatch.setattr("core.scene_detect.SceneDetector", constructor)
    if cancel_before:
        cancel.set()
    with pytest.raises(DetectionCancelled):
        run_detection(DetectionRequest.build(tmp_path / "v.mp4"), cancel_event=cancel)
    assert constructor.call_count == (0 if cancel_before else 1)


def test_progress_and_errors_are_preserved(tmp_path, monkeypatch):
    detector = Mock()
    failure = RuntimeError("decoder failed")

    def detect(path, progress):
        progress(0.5, "Analyzing")
        raise failure

    detector.detect_scenes_with_progress.side_effect = detect
    monkeypatch.setattr("core.scene_detect.SceneDetector", Mock(return_value=detector))
    progress = Mock()
    with pytest.raises(RuntimeError) as raised:
        run_detection(DetectionRequest.build(tmp_path / "v.mp4"), progress_callback=progress)
    assert raised.value is failure
    progress.assert_called_once_with(0.5, "Analyzing")


@pytest.mark.parametrize("when", ["before", "during"])
def test_changed_media_is_not_delivered(tmp_path, monkeypatch, when):
    path = tmp_path / "v.mp4"
    path.write_bytes(b"original")
    request = DetectionRequest.build(path)
    detector = Mock()

    def detect(path, progress):
        path.write_bytes(b"changed media")
        return object(), []

    detector.detect_scenes_with_progress.side_effect = detect
    constructor = Mock(return_value=detector)
    monkeypatch.setattr("core.scene_detect.SceneDetector", constructor)
    if when == "before":
        path.write_bytes(b"changed before dispatch")
    with pytest.raises(RuntimeError, match="media changed"):
        run_detection(request)
    assert constructor.call_count == (0 if when == "before" else 1)


def test_guard_preserves_unrelated_edits_but_rejects_target_edits(tmp_path):
    from core.operations.detection import DetectionGuard, StaleDetectionResult
    from core.project import Project
    from models.clip import Clip, Source

    project = Project.new(name="target")
    source = Source(file_path=tmp_path / "v.mp4")
    project.add_source(source)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project.add_clips([clip])
    guard = DetectionGuard.capture(project, source.file_path)
    project.add_source(Source(file_path=tmp_path / "unrelated.mp4"))
    guard.validate(project)
    clip.notes = "preserve this edit"
    with pytest.raises(StaleDetectionResult):
        guard.validate(project)
    with pytest.raises(StaleDetectionResult):
        guard.validate(Project.new(name="different session"))


def test_detection_preserves_verified_timing_on_existing_source(tmp_path):
    from core.operations.detection import DetectionApplication, DetectionGuard
    from core.project import Project
    from models.clip import Clip, Source

    path = tmp_path / "variable.mp4"
    path.write_bytes(b"source")
    project = Project.new()
    original = Source(file_path=path)
    project.add_source(original)
    application = DetectionApplication(project, DetectionGuard.capture(project, path))
    detected = Source(file_path=path, fps=24, variable_frame_rate=True, frame_timestamps=("0", "1/24", "1/8"))
    clip = Clip(source_id=detected.id, start_frame=0, end_frame=2)
    assert application.apply(detected, [clip], still_current=lambda: True) is original
    assert original.variable_frame_rate
    assert original.frame_timestamps == detected.frame_timestamps
    project.add_to_sequence([clip.id])
    assert str(project.sequence.get_all_clips()[0].source_range.duration) == "1/8"
