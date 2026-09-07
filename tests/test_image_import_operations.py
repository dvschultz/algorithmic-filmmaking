"""Still-image imports validate media and isolate generated artifacts."""

from dataclasses import replace
from threading import Event

from PIL import Image
import pytest

from core.operations.image_import import (
    ImageImportTask,
    ImageImportApplication,
    run_image_import,
)
from core.project import Project


def make_image(path, size=(400, 300)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)
    return path


def test_copied_image_named_thumbnail_is_not_overwritten(tmp_path):
    media = make_image(tmp_path / "thumbnail.jpg")
    original = media.read_bytes()
    task = ImageImportTask.from_paths([media], tmp_path / "out", copy_files=True)
    outcome = run_image_import(task)
    assert outcome.status == "succeeded", outcome.errors
    frame = outcome.frames[0]
    assert frame.path != frame.thumbnail_path
    assert frame.path.read_bytes() == original
    assert media.read_bytes() == original
    with Image.open(frame.path) as image:
        assert image.size == (400, 300)


@pytest.mark.parametrize("copy_files", [False, True])
def test_valid_images_keep_order_dimensions_and_distinct_names(tmp_path, copy_files):
    paths = [
        make_image(tmp_path / "one" / "same.png"),
        make_image(tmp_path / "two" / "same.png", (17, 23)),
    ]
    project = Project.new()
    task = ImageImportTask.from_paths(paths, tmp_path / "out", copy_files=copy_files)
    application = ImageImportApplication(project, task)
    result = run_image_import(task)
    assert result.status == "succeeded"
    assert [(f.width, f.height) for f in result.frames] == [(400, 300), (17, 23)]
    assert all(f.thumbnail_path.is_file() for f in result.frames)
    assert len({f.path for f in result.frames}) == 2
    assert [f.path == p for f, p in zip(result.frames, paths)] == [not copy_files] * 2
    assert not project.frames
    assert application.apply(project, result)
    assert not application.apply(project, result)
    assert [f.id for f in project.frames] == [f.id for f in result.frames]
    assert all(f.source_id is None and f.frame_number is None for f in project.frames)


def test_partial_failure_does_not_publish_invalid_image_or_leave_copy(tmp_path):
    good = make_image(tmp_path / "good.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    task = ImageImportTask.from_paths([good, bad], tmp_path / "out", copy_files=True)
    result = run_image_import(task)
    assert result.status == "succeeded" and len(result.frames) == 1
    assert len(result.errors) == 1
    assert not (task.artifact_dir / task.items[1].id).exists()
    assert bad.read_bytes() == b"not an image"


@pytest.mark.parametrize("when", ["before", "progress"])
def test_cancellation_removes_only_new_artifacts(tmp_path, when):
    media = make_image(tmp_path / "original.png")
    before = media.read_bytes()
    root = tmp_path / "out"
    root.mkdir()
    sentinel = root / "existing.txt"
    sentinel.write_text("keep")
    task = ImageImportTask.from_paths([media, media], root, copy_files=True)
    cancel = Event()
    if when == "before":
        cancel.set()
    result = run_image_import(
        task, cancel_event=cancel, progress=lambda *_: cancel.set()
    )
    assert result.status == "unprocessed" and not result.frames
    assert not task.artifact_dir.exists()
    assert sentinel.read_text() == "keep" and media.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["project", "session", "save_as", "source", "copy", "thumbnail", "id"]
)
def test_changed_owner_or_artifacts_cannot_publish(tmp_path, change):
    media = make_image(tmp_path / "original.png")
    project = Project.new()
    task = ImageImportTask.from_paths([media], tmp_path / "out", copy_files=True)
    application = ImageImportApplication(project, task)
    result = run_image_import(task)
    if change == "project":
        project = Project.new()
    if change == "session":
        project.clear()
    if change == "save_as":
        project.path = tmp_path / "new.sceneripper"
    if change == "source":
        media.write_bytes(b"changed")
    if change == "copy":
        result.frames[0].path.write_bytes(b"changed")
    if change == "thumbnail":
        result.frames[0].thumbnail_path.write_bytes(b"changed")
    if change == "id":
        result = replace(result, frames=(replace(result.frames[0], id="other"),))
    if change in ("project", "session", "save_as"):
        assert not application.apply(project, result)
    else:
        with pytest.raises(ValueError):
            application.apply(project, result)
    assert not project.frames


def test_existing_request_directory_is_never_replaced(tmp_path):
    media = make_image(tmp_path / "image.png")
    task = ImageImportTask.from_paths([media], tmp_path / "out", copy_files=True)
    task.artifact_dir.mkdir(parents=True)
    sentinel = task.artifact_dir / "keep.txt"
    sentinel.write_text("keep")
    result = run_image_import(task)
    assert result.status == "failed"
    assert sentinel.read_text() == "keep"


def test_agent_path_policy_reports_rejected_inputs(tmp_path):
    media = make_image(tmp_path / "image.png")
    task = ImageImportTask.from_paths(
        [media, tmp_path / ".." / "image.png"],
        tmp_path / "out",
        copy_files=True,
        validate_paths=True,
    )
    result = run_image_import(task)
    assert result.status == "succeeded" and len(result.frames) == 1
    assert "traversal" in result.errors[0]


@pytest.mark.parametrize("when", ["queued", "during"])
def test_changed_source_cannot_leave_import_artifacts(tmp_path, monkeypatch, when):
    media = make_image(tmp_path / "image.png")
    task = ImageImportTask.from_paths([media], tmp_path / "out", copy_files=True)
    if when == "queued":
        media.write_bytes(b"changed")
    else:
        from core.thumbnail import generate_image_thumbnail

        def changed(*args, **kwargs):
            result = generate_image_thumbnail(*args, **kwargs)
            media.write_bytes(b"changed")
            return result

        monkeypatch.setattr("core.thumbnail.generate_image_thumbnail", changed)
    result = run_image_import(task)
    assert result.status == "failed" and not result.frames
    assert not task.artifact_dir.exists()
    assert media.read_bytes() == b"changed"
