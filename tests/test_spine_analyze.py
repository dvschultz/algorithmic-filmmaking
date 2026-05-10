"""Tests for the analysis-op spine.

Covers ``analyze_colors``, ``analyze_shots``, and ``transcribe`` happy
paths plus skip-existing semantics and per-clip cancellation. Heavy
ML-backend calls (color k-means, CLIP zero-shot, Whisper) are stubbed
out — these tests focus on the spine fn's orchestration: which clips it
visits, what it skips, what it aggregates into the result.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from core.spine.analyze import analyze_colors, analyze_shots, transcribe


def _build_project(tmp_path: Path, n_clips: int = 3, populate_colors: int = 0):
    """Project with one source and ``n_clips`` clips. The first
    ``populate_colors`` clips have ``dominant_colors`` pre-set to exercise
    skip-existing."""
    from core.project import Project
    from models.clip import Clip, Source

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    project = Project.new(name="test")
    source = Source(
        id="src-1",
        file_path=video,
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)

    clips = [
        Clip(
            id=f"c-{i}",
            source_id=source.id,
            start_frame=i * 60,
            end_frame=(i + 1) * 60,
        )
        for i in range(n_clips)
    ]
    for clip in clips[:populate_colors]:
        clip.dominant_colors = [(255, 0, 0)]
    project.add_clips(clips)
    return project


# -------- analyze_colors --------


def test_analyze_colors_happy_path(tmp_path):
    project = _build_project(tmp_path, n_clips=2)

    with patch(
        "core.analysis.color.extract_dominant_colors",
        return_value=[(10, 20, 30), (40, 50, 60)],
    ):
        result = analyze_colors(project)

    assert result["success"] is True
    assert len(result["result"]["succeeded"]) == 2
    assert result["result"]["failed"] == []
    # Clips received the colors.
    for clip in project.clips:
        assert clip.dominant_colors == [(10, 20, 30), (40, 50, 60)]


def test_analyze_colors_skip_existing(tmp_path):
    project = _build_project(tmp_path, n_clips=3, populate_colors=2)
    extract_calls = []

    def fake_extract(**kwargs):
        extract_calls.append(kwargs)
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=fake_extract):
        result = analyze_colors(project)

    # Two clips already had colors -> skipped; only the third was processed.
    assert len(result["result"]["skipped"]) == 2
    assert len(result["result"]["succeeded"]) == 1
    assert len(extract_calls) == 1


def test_analyze_colors_skip_existing_off(tmp_path):
    project = _build_project(tmp_path, n_clips=3, populate_colors=2)
    extract_calls = []

    def fake_extract(**kwargs):
        extract_calls.append(kwargs)
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=fake_extract):
        result = analyze_colors(project, skip_existing=False)

    assert len(result["result"]["succeeded"]) == 3
    assert len(extract_calls) == 3


def test_analyze_colors_per_clip_failure_aggregated(tmp_path):
    project = _build_project(tmp_path, n_clips=3)
    call_count = [0]

    def fake_extract(**kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("transient kmeans failure")
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=fake_extract):
        result = analyze_colors(project)

    assert len(result["result"]["succeeded"]) == 2
    assert len(result["result"]["failed"]) == 1
    failure = result["result"]["failed"][0]
    assert failure["code"] == "extraction_failed"
    assert "transient" in failure["message"]


def test_analyze_colors_missing_source_file(tmp_path):
    from core.project import Project
    from models.clip import Clip, Source

    project = Project.new(name="t")
    source = Source(
        id="src-1",
        file_path=tmp_path / "missing.mp4",
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)
    project.add_clips(
        [Clip(id="c-1", source_id="src-1", start_frame=0, end_frame=30)]
    )

    result = analyze_colors(project)
    assert result["result"]["failed"][0]["code"] == "source_file_missing"


def test_analyze_colors_cancellation(tmp_path):
    project = _build_project(tmp_path, n_clips=5)
    cancel = threading.Event()
    cancel.set()

    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        result = analyze_colors(project, cancel_event=cancel)

    # Cancel was set before the loop entered — nothing should have processed.
    assert result["result"]["succeeded"] == []
    assert result["result"]["skipped"] == []


def test_analyze_colors_progress_callback(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    calls = []

    def cb(p, msg):
        calls.append((p, msg))

    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        analyze_colors(project, progress_callback=cb)

    assert calls
    assert any(p == 1.0 for p, _ in calls)


def test_analyze_colors_specific_clip_ids(tmp_path):
    project = _build_project(tmp_path, n_clips=4)
    extract_calls = []

    def fake_extract(**kwargs):
        extract_calls.append(kwargs)
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=fake_extract):
        result = analyze_colors(project, clip_ids=["c-0", "c-2"])

    assert len(result["result"]["succeeded"]) == 2
    assert len(extract_calls) == 2


# -------- analyze_shots --------


def test_analyze_shots_happy_path(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    fake_thumb = tmp_path / "thumb.png"
    fake_thumb.write_bytes(b"fake")
    for clip in project.clips:
        clip.thumbnail_path = str(fake_thumb)

    with patch(
        "core.analysis.shots.classify_shot_type",
        return_value=("close-up", 0.9),
    ):
        result = analyze_shots(project)

    assert len(result["result"]["succeeded"]) == 2
    for clip in project.clips:
        assert clip.shot_type == "close-up"


def test_analyze_shots_skip_existing(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    fake_thumb = tmp_path / "thumb.png"
    fake_thumb.write_bytes(b"fake")
    for clip in project.clips:
        clip.thumbnail_path = str(fake_thumb)
    project.clips[0].shot_type = "wide"

    classify_calls = []

    def fake_classify(path):
        classify_calls.append(path)
        return ("close-up", 0.9)

    with patch("core.analysis.shots.classify_shot_type", side_effect=fake_classify):
        result = analyze_shots(project)

    assert len(result["result"]["skipped"]) == 1
    assert len(result["result"]["succeeded"]) == 1
    assert len(classify_calls) == 1


def test_analyze_shots_thumbnail_missing(tmp_path):
    project = _build_project(tmp_path, n_clips=1)
    # No clip.thumbnail_path set — should surface thumbnail_missing.
    result = analyze_shots(project)

    assert result["result"]["failed"][0]["code"] == "thumbnail_missing"


# -------- transcribe --------


def test_transcribe_happy_path(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    fake_segments = [
        {"start_time": 0.0, "end_time": 1.0, "text": "hello", "confidence": 0.9}
    ]

    with patch("core.transcription.transcribe_clip", return_value=fake_segments):
        result = transcribe(project)

    assert len(result["result"]["succeeded"]) == 2
    for clip in project.clips:
        assert clip.transcript == fake_segments


def test_transcribe_skip_existing(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    project.clips[0].transcript = [{"text": "old"}]

    with patch("core.transcription.transcribe_clip", return_value=[{"text": "new"}]):
        result = transcribe(project)

    assert len(result["result"]["skipped"]) == 1
    assert len(result["result"]["succeeded"]) == 1


def test_transcribe_empty_segments_treated_as_success(tmp_path):
    project = _build_project(tmp_path, n_clips=1)

    with patch("core.transcription.transcribe_clip", return_value=[]):
        result = transcribe(project)

    # Silent clips are valid (segment_count=0), not failures.
    assert len(result["result"]["succeeded"]) == 1
    assert result["result"]["succeeded"][0]["segment_count"] == 0
    assert project.clips[0].transcript == []
