"""Tests for the analysis-op spine.

Covers ``analyze_colors``, ``analyze_shots``, and ``transcribe`` happy
paths plus skip-existing semantics and per-clip cancellation. Heavy
ML-backend calls (color k-means, CLIP zero-shot, Whisper) are stubbed
out — these tests focus on the spine fn's orchestration: which clips it
visits, what it skips, what it aggregates into the result.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

from core.spine.analyze import (
    analyze_clips,
    analyze_colors,
    analyze_shots,
    face_embeddings,
    gaze,
    transcribe,
)


from tests.project_fixtures import build_project as _build_project  # noqa: E402


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
    project = _build_project(tmp_path, n_clips=3)
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        analyze_colors(project, clip_ids=["c-0", "c-1"])
    extract_calls = []

    def fake_extract(**kwargs):
        extract_calls.append(kwargs)
        return [(1, 2, 3)]

    with patch("core.analysis.color.extract_dominant_colors", side_effect=fake_extract):
        result = analyze_colors(project)

    # Two verified records are reusable; only the third clip needs computation.
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


def test_analyze_shots_recomputes_legacy_then_reuses_verified_results(tmp_path):
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
        assert len(result["result"]["succeeded"]) == 2
        result = analyze_shots(project)

    assert len(result["result"]["skipped"]) == 2
    assert len(result["result"]["succeeded"]) == 0
    assert len(classify_calls) == 2


def test_analyze_shots_thumbnail_missing(tmp_path):
    project = _build_project(tmp_path, n_clips=1)
    # No clip.thumbnail_path set — should surface thumbnail_missing.
    result = analyze_shots(project)

    assert result["result"]["failed"][0]["code"] == "thumbnail_missing"


# -------- transcribe --------


def test_transcribe_happy_path(tmp_path):
    from core.transcription_models import TranscriptSegment
    project = _build_project(tmp_path, n_clips=2)
    fake_segments = [
        TranscriptSegment(0.0, 1.0, "hello", 0.9)
    ]

    with patch("core.transcription.transcribe_clip", return_value=fake_segments):
        result = transcribe(project)

    assert len(result["result"]["succeeded"]) == 2
    for clip in project.clips:
        assert clip.transcript == fake_segments


def test_transcribe_skip_existing(tmp_path):
    from core.transcription_models import TranscriptSegment
    project = _build_project(tmp_path, n_clips=2)
    project.clips[0].transcript = [TranscriptSegment(0, 1, "old")]

    with patch("core.transcription.transcribe_clip", return_value=[TranscriptSegment(0, 1, "new")]):
        result = transcribe(project)

    assert len(result["result"]["skipped"]) == 0
    assert len(result["result"]["succeeded"]) == 2


def test_transcribe_empty_segments_treated_as_success(tmp_path):
    project = _build_project(tmp_path, n_clips=1)

    with patch("core.transcription.transcribe_clip", return_value=[]):
        result = transcribe(project)

    # Silent clips are valid (segment_count=0), not failures.
    assert len(result["result"]["succeeded"]) == 1
    assert result["result"]["succeeded"][0]["segment_count"] == 0
    assert project.clips[0].transcript == []


# Regression: GH #109 — headless spine paths must unload heavy models after a
# job so long-lived MCP servers don't accumulate InsightFace / MediaPipe state.
def test_face_embeddings_unloads_model_on_success(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    with patch("core.analysis.faces._load_insightface"), \
         patch("core.analysis.faces.extract_faces_from_clip", return_value=[]), \
         patch("core.analysis.faces.unload_model") as mock_unload:
        face_embeddings(project)
    assert mock_unload.call_count == 1


def test_face_embeddings_unloads_model_on_exception(tmp_path):
    project = _build_project(tmp_path, n_clips=1)
    with patch(
        "core.analysis.faces.extract_faces_from_clip",
        side_effect=RuntimeError("boom"),
    ), patch("core.analysis.faces._load_insightface"), \
         patch("core.analysis.faces.unload_model") as mock_unload:
        face_embeddings(project)
    # Per-clip errors are aggregated, not raised — but unload still runs.
    assert mock_unload.call_count == 1


def test_precancelled_face_embeddings_does_not_unload_another_job_model(tmp_path):
    project = _build_project(tmp_path, n_clips=3)
    cancel = threading.Event()
    cancel.set()
    with patch("core.analysis.faces.extract_faces_from_clip", return_value=[]), \
         patch("core.analysis.faces.unload_model") as mock_unload:
        face_embeddings(project, cancel_event=cancel)
    assert mock_unload.call_count == 0


def test_gaze_unloads_model_on_success(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    fake_result = {"gaze_yaw": 0.0, "gaze_pitch": 0.0, "gaze_category": "at_camera"}
    with patch("core.analysis.gaze.load_face_mesh"), \
         patch("core.analysis.gaze.extract_gaze_from_clip", return_value=fake_result), \
         patch("core.analysis.gaze.unload_model") as mock_unload:
        result = gaze(project)["result"]
    assert len(result["succeeded"]) == 2
    assert not result["failed"]
    assert mock_unload.call_count == 1


def test_gaze_unloads_model_on_exception(tmp_path):
    project = _build_project(tmp_path, n_clips=1)
    with patch(
        "core.analysis.gaze.extract_gaze_from_clip",
        side_effect=RuntimeError("boom"),
    ), patch("core.analysis.gaze.load_face_mesh"), patch("core.analysis.gaze.unload_model") as mock_unload:
        gaze(project)
    assert mock_unload.call_count == 1


def test_gaze_unloads_model_on_cancel(tmp_path):
    project = _build_project(tmp_path, n_clips=3)
    cancel = threading.Event()
    cancel.set()
    fake_result = {"gaze_yaw": 0.0, "gaze_pitch": 0.0, "gaze_category": "center"}
    with patch("core.analysis.gaze.extract_gaze_from_clip", return_value=fake_result), \
         patch("core.analysis.gaze.unload_model") as mock_unload:
        gaze(project, cancel_event=cancel)
    assert mock_unload.call_count == 0


def test_analyze_clips_maps_operation_progress_into_parent_range(tmp_path):
    project = _build_project(tmp_path, n_clips=2)
    progress = []

    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]), \
         patch("core.analysis.shots.classify_shot_type", return_value=("wide", 0.9)):
        result = analyze_clips(
            project,
            operations=["colors", "shots"],
            progress_callback=lambda pct, msg: progress.append((pct, msg)),
        )

    assert result["success"] is True
    assert any(0.0 < pct < 0.5 and "colors:" in msg for pct, msg in progress)
    assert any(0.5 < pct < 1.0 and "shots:" in msg for pct, msg in progress)
    assert progress[-1][0] == 1.0
