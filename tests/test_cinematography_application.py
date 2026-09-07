"""Cinematography publication preserves edits made during inference."""

import json

import pytest

from core.operations.cinematography import (
    CinematographyApplication,
    CinematographyOutcome,
    CinematographyTask,
)
from models.cinematography import CinematographyAnalysis
from models.frame import Frame
from tests.test_description_operations import project_with_thumbnails
from ui.workers.cinematography_worker import CinematographyWorker


def outcome(target_id):
    return CinematographyOutcome(
        target_id,
        "succeeded",
        json.dumps(CinematographyAnalysis(shot_size="CU").to_dict()),
    )


def test_frame_publication_with_colliding_clip_id(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    frame = Frame(id=clip.id, file_path=clip.thumbnail_path)
    project.add_frames([frame])
    task = CinematographyTask(
        frame.id, frame.file_path, None, 0, 0, 30, target_type="frame"
    )
    application = CinematographyApplication(project, (task,))
    assert application.apply(project, outcome(frame.id))
    assert frame.shot_type == "close-up"
    assert frame.cinematography.shot_size == "CU"
    assert clip.cinematography is None
    assert not application.apply(project, outcome(frame.id))


@pytest.mark.parametrize(
    "edit", ["shot", "analysis", "range", "fps", "image", "source"]
)
def test_clip_edits_reject_delivery(tmp_path, edit):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.cinematography = CinematographyAnalysis(shot_size="MS")
    worker = CinematographyWorker(
        project.clips, project.sources_by_id, skip_existing=False
    )
    application = CinematographyApplication(project, worker.tasks)
    if edit == "shot":
        clip.shot_type = "wide"
    elif edit == "analysis":
        clip.cinematography.shot_size = "LS"
    elif edit == "range":
        clip.end_frame += 1
    elif edit == "fps":
        project.sources[0].fps += 1
    elif edit == "image":
        clip.thumbnail_path.write_bytes(b"changed image")
    else:
        project.sources[0].file_path = tmp_path / "different.mp4"
    assert not application.apply(project, outcome(clip.id))
    assert clip.cinematography.shot_size != "CU"


def test_clip_accepts_once_and_preserves_unrelated_edits(tmp_path):
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    application = CinematographyApplication(project, worker.tasks)
    clip.notes = "Keep my note"
    assert application.apply(project, outcome(clip.id))
    assert clip.shot_type == "close-up"
    assert clip.notes == "Keep my note"
    assert not application.apply(project, outcome(clip.id))


def test_spine_preserves_edit_during_computation(tmp_path, monkeypatch):
    from core.spine.analyze import cinematography

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography",
        lambda **_: CinematographyAnalysis(shot_size="CU"),
    )
    result = cinematography(
        project, progress_callback=lambda *_: setattr(clip, "shot_type", "wide")
    )
    assert result["result"]["failed"] == [{"clip_id": clip.id, "code": "stale_result"}]
    assert clip.cinematography is None
    assert clip.shot_type == "wide"


@pytest.mark.parametrize(
    "edit", ["image", "path", "identity", "shot", "analysis", "replacement"]
)
def test_frame_edits_reject_delivery(tmp_path, edit):
    from dataclasses import replace

    project = project_with_thumbnails(tmp_path, 1)
    frame = Frame(
        file_path=project.clips[0].thumbnail_path,
        cinematography=CinematographyAnalysis(shot_size="MS"),
    )
    project.add_frames([frame])
    task = CinematographyTask(
        frame.id, frame.file_path, None, 0, 0, 30, target_type="frame"
    )
    application = CinematographyApplication(project, (task,))
    if edit == "image":
        frame.file_path.write_bytes(b"changed")
    elif edit == "path":
        frame.file_path = tmp_path / "other.jpg"
    elif edit == "identity":
        frame.frame_number = 99
    elif edit == "shot":
        frame.shot_type = "wide"
    elif edit == "analysis":
        frame.cinematography.shot_size = "LS"
    else:
        project.remove_frames([frame.id])
        project.add_frames([replace(frame)])
    assert not application.apply(project, outcome(frame.id))
    assert project.frames_by_id[frame.id].cinematography.shot_size != "CU"


def test_publication_requires_owner_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    project = project_with_thumbnails(tmp_path, 1)
    worker = CinematographyWorker(project.clips, project.sources_by_id)
    application = CinematographyApplication(project, worker.tasks)
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError):
            pool.submit(
                application.apply, project, outcome(project.clips[0].id)
            ).result()
    assert project.clips[0].cinematography is None
