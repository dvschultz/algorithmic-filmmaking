"""Sequence cost gates use the same verified completion checks as analysis."""

import pytest

from core.cost_estimates import estimate_sequence_cost
from models.clip import Clip
from tests.analysis_fixtures import verify_clip_analysis
from tests import test_transcription_completion

transcribed = test_transcription_completion.analyzed


@pytest.mark.parametrize(
    "operation,options",
    [
        ("colors", {}),
        ("shots", {"shots": True}),
        ("extract_text", {"ocr": True}),
        ("embeddings", {"embeddings": True}),
        ("boundary_embeddings", {"boundary": True}),
        ("gaze", {"gaze": True}),
    ],
)
def test_verified_cost_matches_records_and_inputs(
    tmp_path, monkeypatch, operation, options
):
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    sources = {source.id: source}

    def blocked(*args, **kwargs):
        raise AssertionError("cost checks must not hash")

    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", blocked)

    def estimate():
        return estimate_sequence_cost(
            "test", [clip], override_required=[operation], sources_by_id=sources
        )

    assert estimate() == []
    record = clip.analysis_records.pop(operation)
    assert estimate()[0].clips_needing == 1
    clip.analysis_records[operation] = record
    clip.end_frame += 1
    assert estimate()[0].clips_needing == 1


@pytest.mark.parametrize(
    "operation,field,value",
    [
        ("colors", "dominant_colors", [(1, 2, 3)]),
        ("shots", "shot_type", "wide shot"),
        ("describe", "description", "Legacy description"),
        ("embeddings", "embedding", [0.1] * 768),
        ("boundary_embeddings", "first_frame_embedding", [0.1] * 768),
        ("gaze", "gaze_category", "center"),
    ],
)
def test_legacy_projections_require_analysis(operation, field, value):
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    setattr(clip, field, value)
    estimates = estimate_sequence_cost("test", [clip], override_required=[operation])
    assert estimates[0].clips_needing == 1


@pytest.mark.parametrize("operation", ["transcribe", "transcription_with_words"])
def test_verified_empty_transcription_satisfies_cost(transcribed, operation):
    project, settings = transcribed

    def estimate():
        return estimate_sequence_cost(
            "test",
            project.clips,
            override_required=[operation],
            sources_by_id=project.sources_by_id,
            settings=settings,
        )

    assert estimate() == []
    project.clips[0].analysis_records.clear()
    assert estimate()[0].clips_needing == 1


@pytest.mark.parametrize(
    "operation,options",
    [
        ("describe", {"descriptions": True}),
        ("cinematography", {"cinematography": True}),
    ],
)
def test_cloud_analysis_cost_checks_verified_projections(
    tmp_path, monkeypatch, operation, options
):
    from core.settings import Settings

    settings = Settings(
        description_model_tier="cloud",
        description_model_cloud="gpt-test",
        cinematography_tier="cloud",
        cinematography_model="gpt-test",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    clip = Clip(source_id="source", start_frame=0, end_frame=30)
    source = verify_clip_analysis(clip, tmp_path, **options)
    estimates = estimate_sequence_cost(
        "test",
        [clip],
        override_required=[operation],
        sources_by_id={source.id: source},
        settings=settings,
    )
    assert estimates == [], clip.analysis_records
    estimates = estimate_sequence_cost(
        "test",
        [clip],
        override_required=[operation],
        sources_by_id={source.id: source},
        settings=settings,
        tier_overrides={operation: "local"},
    )
    assert estimates[0].clips_needing == 1
    clip.analysis_records.clear()
    estimates = estimate_sequence_cost(
        "test",
        [clip],
        override_required=[operation],
        sources_by_id={source.id: source},
        settings=settings,
    )
    assert estimates[0].clips_needing == 1
