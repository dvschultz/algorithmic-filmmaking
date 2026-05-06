"""Tests for the scene-detection spine.

Covers ``detect_scenes_for_source``, ``detect_scenes_for_video``,
``detect_scenes_new_project``, and ``detect_scenes_bulk``. The MCP
``start_detect_scenes_*`` job wrappers are exercised through these spine
fns plus the runtime tests in ``test_jobs_runtime.py``.

Most tests stub ``core.scene_detect.SceneDetector.detect_scenes`` to avoid
running the real OpenCV/PySceneDetect pipeline — the spine fns' job is to
orchestrate the model calls (replace_source_clips, add_source, save), not
to test the detector itself.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from core.spine.detect import (
    detect_scenes_bulk,
    detect_scenes_for_source,
    detect_scenes_for_video,
    detect_scenes_new_project,
)


def _build_project_with_source(tmp_path: Path, source_id: str = "src-1"):
    """Create a Project with a single source pointing to a real on-disk file
    (so the file_path.exists() check passes)."""
    from core.project import Project
    from models.clip import Source

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    project = Project.new(name="test")
    source = Source(
        id=source_id,
        file_path=video_path,
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)
    return project, source, video_path


def _stub_detect_returns(source, clips):
    """Patch SceneDetector.detect_scenes to return a fixed (source, clips)."""

    def fake_detect(self, video_path):
        return source, clips

    return patch(
        "core.scene_detect.SceneDetector.detect_scenes", new=fake_detect
    )


def test_detect_scenes_for_source_replaces_clips(tmp_path):
    from models.clip import Clip

    project, source, _ = _build_project_with_source(tmp_path)
    new_clips = [
        Clip(id=f"c-{i}", source_id=source.id, start_frame=i * 60, end_frame=(i + 1) * 60)
        for i in range(3)
    ]

    with _stub_detect_returns(source, new_clips):
        result = detect_scenes_for_source(project, source.id, sensitivity=3.0)

    assert result["success"] is True
    assert result["result"]["source_id"] == source.id
    assert result["result"]["clip_count"] == 3
    assert {c.id for c in project.clips} == {"c-0", "c-1", "c-2"}


def test_detect_scenes_for_source_unknown_source(tmp_path):
    project, _, _ = _build_project_with_source(tmp_path)
    result = detect_scenes_for_source(project, "not-a-real-source", sensitivity=3.0)
    assert result["success"] is False
    assert result["error"]["code"] == "unknown_source_id"


def test_detect_scenes_for_source_missing_file(tmp_path):
    from core.project import Project
    from models.clip import Source

    project = Project.new(name="t")
    source = Source(
        id="src-1",
        file_path=tmp_path / "does-not-exist.mp4",
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)

    result = detect_scenes_for_source(project, "src-1", sensitivity=3.0)
    assert result["success"] is False
    assert result["error"]["code"] == "source_file_missing"


def test_detect_scenes_for_video_creates_new_source(tmp_path):
    from core.project import Project
    from models.clip import Clip, Source

    project = Project.new(name="t")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    detected_source = Source(
        id="src-new",
        file_path=video,
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    detected_clips = [
        Clip(id="c-1", source_id=detected_source.id, start_frame=0, end_frame=30)
    ]

    with _stub_detect_returns(detected_source, detected_clips):
        result = detect_scenes_for_video(project, video, sensitivity=3.0)

    assert result["success"] is True
    assert result["result"]["is_new_source"] is True
    assert len(project.sources) == 1
    assert len(project.clips) == 1


def test_detect_scenes_for_video_replaces_existing_source_clips(tmp_path):
    from models.clip import Clip

    project, source, video = _build_project_with_source(tmp_path)
    # Pre-populate with stale clips that should be replaced.
    stale = [Clip(id="stale", source_id=source.id, start_frame=0, end_frame=10)]
    project.add_clips(stale)

    new_clips = [
        Clip(id="fresh-1", source_id=source.id, start_frame=0, end_frame=30),
        Clip(id="fresh-2", source_id=source.id, start_frame=30, end_frame=60),
    ]
    with _stub_detect_returns(source, new_clips):
        result = detect_scenes_for_video(project, video, sensitivity=3.0)

    assert result["success"] is True
    assert result["result"]["is_new_source"] is False
    assert {c.id for c in project.clips} == {"fresh-1", "fresh-2"}


def test_detect_scenes_for_video_missing_file(tmp_path):
    from core.project import Project

    project = Project.new(name="t")
    result = detect_scenes_for_video(project, tmp_path / "nope.mp4")
    assert result["success"] is False
    assert result["error"]["code"] == "source_file_missing"


def test_detect_scenes_new_project_creates_and_saves(tmp_path):
    from models.clip import Clip, Source

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "new.sceneripper"

    detected_source = Source(
        id="src-new",
        file_path=video,
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    detected_clips = [
        Clip(id="c-1", source_id=detected_source.id, start_frame=0, end_frame=30)
    ]

    with _stub_detect_returns(detected_source, detected_clips):
        result = detect_scenes_new_project(video, output, sensitivity=3.0)

    assert result["success"] is True
    assert result["result"]["clip_count"] == 1
    assert output.exists()


def test_detect_scenes_bulk_aggregates_failures(tmp_path):
    from core.project import Project
    from models.clip import Clip, Source

    project = Project.new(name="t")
    # Source A — exists on disk; will succeed.
    a_video = tmp_path / "a.mp4"
    a_video.write_bytes(b"fake")
    a_src = Source(id="A", file_path=a_video, duration_seconds=60.0, fps=30.0, width=1920, height=1080)
    project.add_source(a_src)
    # Source B — file missing; will fail.
    b_src = Source(
        id="B",
        file_path=tmp_path / "b.mp4",  # never created
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(b_src)

    new_clips = [Clip(id="A-1", source_id="A", start_frame=0, end_frame=30)]
    with _stub_detect_returns(a_src, new_clips):
        result = detect_scenes_bulk(project, ["A", "B", "unknown"], sensitivity=3.0)

    assert result["success"] is True
    payload = result["result"]
    assert len(payload["succeeded"]) == 1
    assert payload["succeeded"][0]["source_id"] == "A"
    assert len(payload["failed"]) == 2
    failed_ids = {f["source_id"] for f in payload["failed"]}
    assert failed_ids == {"B", "unknown"}


def test_detect_scenes_bulk_cancellation(tmp_path):
    from core.project import Project
    from models.clip import Clip, Source

    project = Project.new(name="t")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    src_a = Source(id="A", file_path=video, duration_seconds=60.0, fps=30.0, width=1920, height=1080)
    src_b = Source(id="B", file_path=video, duration_seconds=60.0, fps=30.0, width=1920, height=1080)
    project.add_source(src_a)
    project.add_source(src_b)

    cancel = threading.Event()
    cancel.set()  # cancel immediately

    with _stub_detect_returns(src_a, []):
        result = detect_scenes_bulk(
            project, ["A", "B"], sensitivity=3.0, cancel_event=cancel
        )

    assert result["success"] is True
    # Cancel was set before we started — both sources land in cancelled.
    assert "A" in result["result"]["cancelled"]
    assert "B" in result["result"]["cancelled"]


def test_progress_callback_is_invoked(tmp_path):
    from models.clip import Clip

    project, source, _ = _build_project_with_source(tmp_path)
    new_clips = [Clip(id="c-1", source_id=source.id, start_frame=0, end_frame=30)]
    progress_calls = []

    def cb(progress, message):
        progress_calls.append((progress, message))

    with _stub_detect_returns(source, new_clips):
        detect_scenes_for_source(
            project,
            source.id,
            sensitivity=3.0,
            progress_callback=cb,
        )

    assert progress_calls
    # Final callback should report 1.0.
    assert any(p == 1.0 for p, _ in progress_calls)
