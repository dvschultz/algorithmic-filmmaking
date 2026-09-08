"""GUI agent summaries distinguish verified empty results from missing analysis."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.project import Project
from models.clip import Clip
from tests.analysis_fixtures import verify_clip_analysis
from tests import test_scalar_records, test_face_records, test_transcription_completion
from ui.main_window import MainWindow

scalar_inputs = test_scalar_records.setup
face_inputs = test_face_records.setup
transcribed = test_transcription_completion.analyzed


def window(project, settings=None):
    return SimpleNamespace(
        project=project, settings=settings,
        _build_agent_clip_context=lambda clip: {"clip_id": clip.id},
        _truncate_for_agent=MainWindow._truncate_for_agent,
        _summarize_detections_for_agent=MainWindow._summarize_detections_for_agent,
        _rgb_to_hex=MainWindow._rgb_to_hex,
    )


@pytest.mark.parametrize("operation,options", [
    ("classify", {"classify": True}), ("detect_objects", {"objects": True}),
    ("extract_text", {"ocr": True}), ("gaze", {"gaze": True}),
])
def test_summary_includes_verified_empty_results(tmp_path, monkeypatch, operation, options):
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    project = Project(sources=[source], clips=[clip])
    harness = window(project)
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", Mock(side_effect=AssertionError("summary must not hash")))
    summary = MainWindow._build_agent_analysis_summary(harness, [clip], [operation])[operation]
    assert summary["analyzed_count"] == 1
    assert summary["clips"][0]["clip_id"] == clip.id
    if operation == "gaze":
        assert summary["distribution"] == {}
        assert summary["clips"][0]["gaze_category"] is None
    clip.analysis_records[operation] = replace(clip.analysis_records[operation], state="failed")
    assert MainWindow._build_agent_analysis_summary(harness, [clip], [operation])[operation]["analyzed_count"] == 0


def test_empty_transcript_is_reported_and_stale_transcript_is_not(transcribed):
    project, settings = transcribed
    harness = window(project, settings)
    result = MainWindow._build_agent_analysis_summary(harness, project.clips, ["transcribe"])["transcribe"]
    assert result["analyzed_count"] == 1
    assert result["clips"][0]["segment_count"] == 0
    assert result["clips"][0]["transcript_excerpt"] is None
    project.clips[0].end_frame += 1
    assert MainWindow._build_agent_analysis_summary(harness, project.clips, ["transcribe"])["transcribe"]["analyzed_count"] == 0


def test_scalar_summary_includes_zero_and_verified_no_audio(scalar_inputs):
    project, operation, provider = scalar_inputs
    provider.return_value = None if operation == "volume" else 0.0
    test_scalar_records.run(scalar_inputs)
    harness = window(project)
    result = MainWindow._build_agent_analysis_summary(harness, project.clips, [operation])[operation]
    assert result["analyzed_count"] == 1
    assert result["clips"][0][test_scalar_records.FIELDS[operation]] == provider.return_value
    project.clips[0].analysis_records.clear()
    assert MainWindow._build_agent_analysis_summary(harness, project.clips, [operation])[operation]["analyzed_count"] == 0


def test_face_summary_includes_verified_no_faces(face_inputs):
    project, provider, _ = face_inputs
    execute = provider.side_effect

    def no_faces(**kwargs):
        execute(**kwargs)  # Preserve the model-execution provenance callback.
        return []

    provider.side_effect = no_faces
    outcome, _ = test_face_records.run(project)
    assert outcome.status == "succeeded"
    result = MainWindow._build_agent_analysis_summary(window(project), project.clips, ["face_embeddings"])["face_embeddings"]
    assert result["analyzed_count"] == 1
    assert result["total_faces"] == 0
    assert result["clips"][0]["face_count"] == 0


def test_summary_does_not_count_legacy_nonempty_projection(tmp_path):
    clip = Clip(source_id="source", start_frame=0, end_frame=30, dominant_colors=[(1, 2, 3)])
    project = Project(clips=[clip])
    assert MainWindow._build_agent_analysis_summary(window(project), [clip], ["colors"])["colors"]["analyzed_count"] == 0


def test_query_summary_rejects_stale_and_legacy_results(tmp_path, monkeypatch):
    from core.settings import Settings
    from tests.test_description_operations import project_with_thumbnails
    from core.spine.analyze import custom_query

    project = project_with_thumbnails(tmp_path, 1)
    settings = Settings(description_model_tier="cloud", description_model_cloud="model")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", Mock(return_value=(False, 0.0, "model")))
    assert custom_query(project, query="person")["result"]["succeeded"]
    harness = window(project, settings)
    summary = MainWindow._build_custom_query_agent_summary(harness, project.clips, " person ")
    assert summary["non_match_count"] == 1
    assert summary["missing_result_count"] == 0
    project.clips[0].analysis_records.clear()
    summary = MainWindow._build_custom_query_agent_summary(harness, project.clips, "person")
    assert summary["non_match_count"] == 0
    assert summary["missing_result_count"] == 1
