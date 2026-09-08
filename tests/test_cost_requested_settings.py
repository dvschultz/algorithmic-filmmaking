"""Cost gates verify the requested provider settings, including unsaved edits."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.cost_estimates import estimate_sequence_cost
from core.settings import Settings
from models.clip import Clip
from tests.analysis_fixtures import verify_clip_analysis
from tests import test_transcription_completion

transcribed = test_transcription_completion.analyzed


@pytest.mark.parametrize("operation,options,field,value", [
    ("describe", {"descriptions": True}, "description_model_cloud", "other-model"),
    ("describe", {"descriptions": True}, "description_input_mode", "video"),
    ("cinematography", {"cinematography": True}, "cinematography_model", "other-model"),
    ("cinematography", {"cinematography": True}, "cinematography_input_mode", "video"),
    ("shots", {"shots": True}, "shot_classifier_cloud_model", "other-model"),
    ("extract_text", {"ocr": True}, "description_model_cloud", "other-model"),
])
def test_cost_uses_requested_settings(tmp_path, monkeypatch, operation, options, field, value):
    configured = Settings(
        description_model_tier="cloud", description_model_cloud="gpt-test",
        description_input_mode="frame", cinematography_tier="cloud",
        cinematography_model="gpt-test", cinematography_input_mode="frame",
        shot_classifier_tier="cloud", shot_classifier_cloud_model="gpt-test",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: configured)
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    if operation == "shots":
        from core.operations.shots import shot_task, ShotTypeOptions, ShotTypeApplication, run_shot_types
        from core.project import Project

        monkeypatch.setattr("core.analysis.shots.classify_shot_type_tiered", Mock(return_value=("wide shot", 0.9)))
        project = Project(sources=[source], clips=[clip])
        task = shot_task(clip, source, skip_existing=False)
        shot_options = ShotTypeOptions.from_settings()
        outcome = run_shot_types((task,), shot_options)[0]
        assert ShotTypeApplication(project, (task,), shot_options).apply(project, outcome)
    original = configured

    def estimate(settings):
        return estimate_sequence_cost(
            "test", [clip], settings=settings, override_required=[operation],
            sources_by_id={source.id: source},
        )

    assert estimate(original) == []
    requested = replace(original, **{field: value})
    assert estimate(requested)[0].clips_needing == 1
    # A matching explicit request remains reusable when global settings differ.
    configured = requested
    assert estimate(original) == []


@pytest.mark.parametrize("operation", ["transcribe", "transcription_with_words"])
def test_transcription_cost_uses_requested_model(transcribed, monkeypatch, operation):
    project, settings = transcribed

    def estimate(requested):
        return estimate_sequence_cost(
            "test", project.clips, settings=requested, override_required=[operation],
            sources_by_id=project.sources_by_id,
        )

    assert estimate(settings) == []
    requested = replace(settings, transcription_model="large-v3")
    assert estimate(requested)[0].clips_needing == 1
    monkeypatch.setattr("core.settings.load_settings", lambda: requested)
    assert estimate(settings) == []


def test_parallelism_changes_do_not_require_new_analysis(tmp_path, monkeypatch):
    settings = Settings(description_model_tier="cloud", description_model_cloud="gpt-test")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, descriptions=True)
    monkeypatch.setattr(
        "core.jobs.media.MediaFingerprints.get",
        Mock(side_effect=AssertionError("cost checks must not hash media")),
    )
    requested = replace(settings, description_parallelism=settings.description_parallelism + 1)
    assert estimate_sequence_cost(
        "test", [clip], settings=requested, override_required=["describe"],
        sources_by_id={source.id: source},
    ) == []


def test_cloud_transcription_cost_uses_requested_model(tmp_path, monkeypatch):
    settings = Settings(transcription_backend="groq", transcription_cloud_model="whisper-large-v3-turbo")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, transcriptions=True)
    assert estimate_sequence_cost(
        "test", [clip], settings=settings, override_required=["transcribe"],
        sources_by_id={source.id: source},
    ) == []
    requested = replace(settings, transcription_cloud_model="whisper-large-v3")
    assert estimate_sequence_cost(
        "test", [clip], settings=requested, override_required=["transcribe"],
        sources_by_id={source.id: source},
    )[0].clips_needing == 1
