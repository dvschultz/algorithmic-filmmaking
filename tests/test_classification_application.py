"""Frame storage and guarded classification publication."""

import pytest

from core.analysis_target import AnalysisTarget
from models.frame import Frame
from core.operations.classification import (
    ClassificationApplication,
    ClassificationTask,
    ClassificationOutcome,
)
from tests.test_description_operations import project_with_thumbnails


def result(cid):
    return ClassificationOutcome(cid, "succeeded", (("person", 0.9),))


@pytest.mark.parametrize("labels", [[], ["person", "car"]])
def test_frame_classification_roundtrip_preserves_empty_result(tmp_path, labels):
    frame = Frame(
        file_path=tmp_path / "frame.jpg",
        detected_objects=[{"label": "cat", "confidence": 0.8}],
    )
    frame.object_labels = labels
    encoded = frame.to_dict()
    assert encoded["object_labels"] == labels
    restored = Frame.from_dict(encoded)
    assert restored.object_labels == labels
    assert restored.detected_objects == frame.detected_objects
    assert AnalysisTarget.from_frame(restored).object_labels == labels


def test_legacy_frame_has_no_classification():
    frame = Frame.from_dict({"id": "old", "file_path": "/tmp/old.jpg"})
    assert frame.object_labels is None


@pytest.mark.parametrize("kind", ["clip", "frame"])
@pytest.mark.parametrize(
    "change", ["none", "labels", "image", "path", "identity", "replace", "session"]
)
def test_classification_rejects_stale_delivery(tmp_path, kind, change):
    from dataclasses import replace

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    if kind == "frame":
        target = Frame(
            id=clip.id,
            file_path=clip.thumbnail_path,
            detected_objects=[{"label": "cat"}],
        )
        project.add_frames([target])
    else:
        target = clip
    target.object_labels = []
    task = ClassificationTask(target.id, clip.thumbnail_path, target_type=kind)
    application = ClassificationApplication(project, (task,))
    if change == "labels":
        target.object_labels.append("user edit")
    elif change == "image":
        clip.thumbnail_path.write_bytes(b"changed")
    elif change == "path":
        if kind == "frame":
            target.file_path = tmp_path / "new.jpg"
        else:
            target.thumbnail_path = tmp_path / "new.jpg"
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
    assert application.apply(project, result(target.id)) == (change == "none")
    assert not application.apply(project, result(target.id))
    if change == "none":
        assert target.object_labels == ["person"]
        if kind == "frame":
            assert target.detected_objects == [{"label": "cat"}]
            assert clip.object_labels is None


def test_external_analysis_thumbnail_preserves_display_thumbnail(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    original = clip.thumbnail_path
    external = tmp_path / "cli-analysis.jpg"
    external.write_bytes(b"external")
    application = ClassificationApplication(
        project, (ClassificationTask(clip.id, external),)
    )
    clip.notes = "Keep my note"
    assert application.apply(project, result(clip.id))
    assert clip.thumbnail_path == original
    assert clip.notes == "Keep my note"


def test_spine_does_not_overwrite_edits_during_analysis(tmp_path, monkeypatch):
    from core.spine.analyze import classify_content

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    monkeypatch.setattr(
        "core.analysis.classification.classify_frame",
        lambda *a, **kw: [("person", 0.9)],
    )
    output = classify_content(
        project,
        progress_callback=lambda *_: setattr(clip, "object_labels", ["user edit"]),
    )
    assert output["result"]["failed"] == [{"clip_id": clip.id, "code": "stale_result"}]
    assert clip.object_labels == ["user edit"]


def test_publication_requires_owner_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    application = ClassificationApplication(
        project, (ClassificationTask(clip.id, clip.thumbnail_path),)
    )
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError):
            pool.submit(application.apply, project, result(clip.id)).result()
    assert clip.object_labels is None


def test_empty_frame_result_is_saved_and_skipped_on_next_run(tmp_path):
    from core.project import Project
    from ui.workers.classification_worker import ClassificationWorker

    project = project_with_thumbnails(tmp_path, 1)
    frame = Frame(file_path=project.clips[0].thumbnail_path)
    project.add_frames([frame])
    application = ClassificationApplication(
        project, (ClassificationTask(frame.id, frame.file_path, target_type="frame"),)
    )
    assert application.apply(project, ClassificationOutcome(frame.id, "succeeded"))
    assert project.save(tmp_path / "project.json")
    restored = Project.load(project.path)
    assert restored.frames[0].object_labels == []
    worker = ClassificationWorker(
        [], analysis_targets=[AnalysisTarget.from_frame(restored.frames[0])]
    )
    assert worker.tasks == ()
