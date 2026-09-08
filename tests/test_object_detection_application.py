"""Object publication preserves edits, empty results and people-only analysis."""

import pytest

from core.analysis_target import AnalysisTarget
from models.frame import Frame
from tests.test_description_operations import project_with_thumbnails


@pytest.mark.parametrize("kind", ["clip", "frame"])
def test_empty_objects_and_zero_people_roundtrip(tmp_path, kind):
    project = project_with_thumbnails(tmp_path, 1)
    target = project.clips[0] if kind == "clip" else Frame(file_path=tmp_path / "f.jpg")
    target.detected_objects = []
    target.person_count = 0
    restored = type(target).from_dict(target.to_dict())
    assert restored.detected_objects == []
    assert restored.person_count == 0
    analysis = (
        AnalysisTarget.from_clip(restored, None)
        if kind == "clip"
        else AnalysisTarget.from_frame(restored)
    )
    assert analysis.person_count == 0


def test_people_only_spine_preserves_objects(tmp_path, monkeypatch):
    from core.spine.analyze import detect_objects

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.detected_objects = [{"label": "cat", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]
    monkeypatch.setattr("core.analysis.detection.count_people", lambda *a, **kw: 2)
    output = detect_objects(project, detect_all=False)
    assert len(output["result"]["succeeded"]) == 1
    assert clip.person_count == 2
    assert clip.detected_objects[0]["label"] == "cat"


@pytest.mark.parametrize("kind", ["clip", "frame"])
def test_people_only_worker_revalidates_legacy_zero(tmp_path, kind):
    from ui.workers.object_detection_worker import ObjectDetectionWorker

    project = project_with_thumbnails(tmp_path, 1)
    target = (
        project.clips[0]
        if kind == "clip"
        else Frame(file_path=project.clips[0].thumbnail_path)
    )
    target.person_count = 0
    worker = (
        ObjectDetectionWorker([target], detect_all=False)
        if kind == "clip"
        else ObjectDetectionWorker(
            [], detect_all=False, analysis_targets=[AnalysisTarget.from_frame(target)]
        )
    )
    assert len(worker.tasks) == 1
    assert worker.tasks[0].analysis_json is not None


@pytest.mark.parametrize("kind", ["clip", "frame"])
@pytest.mark.parametrize(
    "change",
    ["none", "objects", "count", "image", "path", "identity", "replace", "session"],
)
def test_rejects_stale_or_duplicate_publication(tmp_path, kind, change):
    from dataclasses import replace
    from core.operations.object_detection import (
        ObjectDetectionApplication,
        ObjectDetectionTask,
        ObjectDetectionOutcome,
        DetectedObject,
    )

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    target = clip
    if kind == "frame":
        target = Frame(id=clip.id, file_path=clip.thumbnail_path)
        project.add_frames([target])
    target.detected_objects = [{"label": "cat", "bbox": [0, 0, 1, 1]}]
    application = ObjectDetectionApplication(
        project,
        (ObjectDetectionTask(target.id, clip.thumbnail_path, target_type=kind),),
    )
    if change == "objects":
        target.detected_objects[0]["bbox"][0] = 99
    elif change == "count":
        target.person_count = 99
    elif change == "image":
        clip.thumbnail_path.write_bytes(b"changed")
    elif change == "path":
        setattr(
            target,
            "file_path" if kind == "frame" else "thumbnail_path",
            tmp_path / "new.jpg",
        )
    elif change == "identity":
        if kind == "frame":
            target.frame_number = 99
        else:
            target.end_frame += 1
    elif change == "replace":
        if kind == "frame":
            project.remove_frames([target.id])
            project.add_frames([replace(target)])
        else:
            project.remove_clips([target.id])
            project.add_clips([replace(target)])
    elif change == "session":
        project.clear()
    outcome = ObjectDetectionOutcome(
        target.id, "succeeded", (DetectedObject("person", 0.9, (0, 0, 1, 1)),), 1
    )
    assert application.apply(project, outcome) == (change == "none")
    assert not application.apply(project, outcome)
    if change == "none":
        assert target.person_count == 1
        if kind == "frame":
            assert clip.person_count is None


@pytest.mark.parametrize("kind", ["clip", "frame"])
def test_people_only_accepts_unrelated_object_edit(tmp_path, kind):
    from core.operations.object_detection import (
        ObjectDetectionApplication,
        ObjectDetectionTask,
        ObjectDetectionOutcome,
        ObjectDetectionOptions,
    )

    project = project_with_thumbnails(tmp_path, 1)
    target = project.clips[0]
    if kind == "frame":
        target = Frame(file_path=target.thumbnail_path)
        project.add_frames([target])
    image = target.file_path if kind == "frame" else target.thumbnail_path
    app = ObjectDetectionApplication(
        project,
        (ObjectDetectionTask(target.id, image, target_type=kind),),
        ObjectDetectionOptions(detect_all=False),
    )
    target.detected_objects = [{"label": "user edit"}]
    target.notes = "keep"
    assert app.apply(
        project, ObjectDetectionOutcome(target.id, "succeeded", person_count=0)
    )
    assert target.detected_objects == [{"label": "user edit"}]
    assert target.person_count == 0
    assert target.notes == "keep"


def test_spine_rejects_edit_during_inference(tmp_path, monkeypatch):
    from core.spine.analyze import detect_objects

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]

    def provider(*a, **kw):
        clip.person_count = 99
        return []

    monkeypatch.setattr("core.analysis.detection.detect_objects", provider)
    result = detect_objects(project)["result"]
    assert result["failed"] == [{"clip_id": clip.id, "code": "stale_result"}]
    assert clip.person_count == 99
    assert clip.detected_objects is None


def test_publication_requires_owner_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from core.operations.object_detection import (
        ObjectDetectionApplication,
        ObjectDetectionTask,
        ObjectDetectionOutcome,
    )

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    application = ObjectDetectionApplication(
        project, (ObjectDetectionTask(clip.id, clip.thumbnail_path),)
    )
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError):
            pool.submit(
                application.apply,
                project,
                ObjectDetectionOutcome(clip.id, "succeeded", person_count=0),
            ).result()
    assert clip.person_count is None
