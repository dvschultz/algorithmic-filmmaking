"""Extraction owns its inputs, output files, and publication context."""

from dataclasses import replace
from threading import Event
from unittest.mock import patch

from PIL import Image
import pytest

from core.operations.frame_extraction import (
    FrameExtractionApplication,
    FrameExtractionTask,
    run_frame_extraction,
)
from core.project import Project
from models.clip import Clip, Source


def setup_request(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"source")
    project = Project.new()
    source = Source(file_path=path, width=20, height=12, fps=30)
    project.add_source(source)
    task = FrameExtractionTask.from_source(
        source, None, "interval", 5, tmp_path / "output"
    )
    return project, source, task


def extract(path, directory, fps, **kwargs):
    directory.mkdir(parents=True)
    paths = [directory / f"frame_{number:06d}.png" for number in (0, 5, 10)]
    for target in paths:
        Image.new("RGB", (20, 12), "red").save(target)
    return paths


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "project",
        "session",
        "save_as",
        "source",
        "source_id",
        "media",
        "artifact",
    ],
)
def test_publication_requires_original_inputs_and_artifacts(tmp_path, change):
    project, source, task = setup_request(tmp_path)
    application = FrameExtractionApplication(project, task)
    with patch("core.ffmpeg.extract_frames_batch", side_effect=extract):
        result = run_frame_extraction(task)
    assert result.status == "succeeded", result.message
    assert [frame.frame_number for frame in result.frames] == [0, 5, 10]
    assert all((frame.width, frame.height) == (20, 12) for frame in result.frames)
    if change == "project":
        project = Project.new()
    if change == "session":
        project.clear()
    if change == "save_as":
        project.path = tmp_path / "other.sceneripper"
    if change == "source":
        source.fps = 24
    if change == "source_id":
        source.id = "edited-id"
    if change == "media":
        source.file_path.write_bytes(b"changed")
    if change == "artifact":
        result.frames[0].path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="artifacts changed"):
            application.apply(project, result)
    else:
        assert application.apply(project, result) == (change == "none")
    assert len(project.frames) == (3 if change == "none" else 0)
    assert not application.apply(project, result)


@pytest.mark.parametrize("failure", ["cancel", "error", "changed"])
def test_failed_extraction_removes_only_its_own_workspace(tmp_path, failure):
    project, source, task = setup_request(tmp_path)
    other = task.artifact_dir.parent / "previous.png"
    other.parent.mkdir()
    other.write_bytes(b"published")
    cancel = Event()

    def provider(*args, **kwargs):
        paths = extract(*args, **kwargs)
        if failure == "cancel":
            cancel.set()
        if failure == "error":
            raise RuntimeError("provider failed")
        if failure == "changed":
            source.file_path.write_bytes(b"changed")
        return paths

    with patch("core.ffmpeg.extract_frames_batch", side_effect=provider):
        result = run_frame_extraction(task, cancel_event=cancel)
    assert result.status == ("unprocessed" if failure == "cancel" else "failed")
    assert not task.artifact_dir.exists()
    assert other.read_bytes() == b"published"
    assert not project.frames


def test_each_request_preserves_previously_extracted_files(tmp_path):
    _, source, task = setup_request(tmp_path)
    second = FrameExtractionTask.from_source(
        source, None, "interval", 5, task.artifact_dir.parent
    )
    assert second.artifact_dir != task.artifact_dir
    with patch("core.ffmpeg.extract_frames_batch", side_effect=extract):
        first_result = run_frame_extraction(task)
        second_result = run_frame_extraction(second)
        repeated = run_frame_extraction(task)
    assert first_result.status == second_result.status == "succeeded"
    assert repeated.status == "failed"
    assert all(
        frame.path.exists() for frame in first_result.frames + second_result.frames
    )


def test_application_rejects_duplicate_frame_numbers(tmp_path):
    project, _, task = setup_request(tmp_path)
    application = FrameExtractionApplication(project, task)
    with patch("core.ffmpeg.extract_frames_batch", side_effect=extract):
        result = run_frame_extraction(task)
    result = replace(result, frames=(result.frames[0], result.frames[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        application.apply(project, result)
    assert not project.frames


def test_clip_range_change_discards_extraction(tmp_path):
    project, source, original = setup_request(tmp_path)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=15)
    project.add_clips([clip])
    task = FrameExtractionTask.from_source(
        source, clip, "interval", 5, original.artifact_dir.parent
    )
    application = FrameExtractionApplication(project, task)
    with patch("core.ffmpeg.extract_frames_batch", side_effect=extract):
        result = run_frame_extraction(task)
    assert result.status == "succeeded"
    clip.start_frame = 5
    assert not application.apply(project, result)
    assert not project.frames
