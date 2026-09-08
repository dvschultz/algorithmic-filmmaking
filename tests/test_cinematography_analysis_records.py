"""Verified cinematography reuse tracks both its analysis and derived shot type."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.operations.cinematography import (
    CinematographyApplication,
    CinematographyOptions,
    cinematography_task,
    run_cinematography,
)
from core.project import Project
from models.cinematography import CinematographyAnalysis
from tests.test_description_operations import project_with_thumbnails

OPTIONS = CinematographyOptions("cloud", "frame", "model", "local")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(
        return_value=CinematographyAnalysis(shot_size="CU", analysis_model="model")
    )
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", provider)
    return project, provider


def evaluate(project, *, options=OPTIONS, reuse=True):
    task = cinematography_task(
        project.clips[0], project.sources[0], skip_existing=reuse
    )
    outcome = run_cinematography((task,), options)[0]
    assert CinematographyApplication(project, (task,), options).apply(project, outcome)
    return outcome


def test_verified_record_reuses_and_survives_save(setup, tmp_path):
    project, provider = setup
    assert evaluate(project).status == "succeeded"
    assert project.save(tmp_path / "project.json")
    assert evaluate(Project.load(project.path)).status == "skipped"
    provider.assert_called_once()


@pytest.mark.parametrize(
    "change",
    ["image", "source", "range", "fps", "model", "projection", "shot_type", "runtime"],
)
def test_changed_inputs_or_display_require_recomputation(setup, monkeypatch, change):
    project, provider = setup
    evaluate(project)
    clip = project.clips[0]
    options = OPTIONS
    if change == "image":
        clip.thumbnail_path.write_bytes(b"changed")
    elif change == "source":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "range":
        clip.end_frame += 1
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "model":
        options = replace(options, model="other")
    elif change == "projection":
        clip.cinematography.shot_size = "MS"
    elif change == "shot_type":
        clip.shot_type = "wide shot"
    else:
        monkeypatch.setattr(
            "core.operations.cinematography.model_runtime",
            lambda *args: {"packages": {"test": "changed"}},
        )
    assert evaluate(project, options=options).status == "succeeded"
    assert provider.call_count == 2


def test_parallelism_does_not_invalidate_reuse(setup):
    project, provider = setup
    evaluate(project)
    assert (
        evaluate(project, options=replace(OPTIONS, parallelism=4)).status == "skipped"
    )
    provider.assert_called_once()


def test_legacy_projection_needs_verification(setup):
    project, provider = setup
    project.clips[0].cinematography = CinematographyAnalysis(shot_size="MS")
    assert evaluate(project).status == "succeeded"
    provider.assert_called_once()


def test_failed_attempt_keeps_display_and_requires_retry(setup):
    project, provider = setup
    evaluate(project)
    original = project.clips[0].cinematography.to_dict()
    provider.side_effect = ValueError("Invalid answer")
    assert evaluate(project, reuse=False).status == "failed"
    clip = project.clips[0]
    assert clip.analysis_records["cinematography"].state == "failed"
    assert clip.cinematography.to_dict() == original
    provider.side_effect = None
    assert evaluate(project).status == "succeeded"
    assert provider.call_count == 3


def test_frame_fallback_cannot_satisfy_video_request(setup):
    project, provider = setup
    options = replace(OPTIONS, mode="video")

    def fallback(**kwargs):
        kwargs["on_execution"](
            {"backend": "cloud", "model": "model", "input_mode": "frame"}
        )
        return CinematographyAnalysis(
            shot_size="CU", analysis_model="model", analysis_mode="frame"
        )

    provider.side_effect = fallback
    evaluate(project, options=options)
    record = project.clips[0].analysis_records["cinematography"]
    assert record.identity.to_dict()["model"]["execution"]["input_mode"] == "frame"
    assert evaluate(project, options=options).status == "succeeded"
    assert provider.call_count == 2


def test_changed_media_during_inference_cannot_publish(setup):
    project, provider = setup

    def mutate(**kwargs):
        project.sources[0].file_path.write_bytes(b"changed")
        return CinematographyAnalysis(analysis_model="model")

    provider.side_effect = mutate
    task = cinematography_task(project.clips[0], project.sources[0])
    result = run_cinematography((task,), OPTIONS)[0]
    assert result.status == "failed"
    assert result.record_json is None


def test_direct_headless_entry_uses_verified_records(setup, monkeypatch):
    from core.spine.analyze import cinematography

    project, provider = setup
    monkeypatch.setattr(
        "core.operations.cinematography.resolve_options", lambda *args: OPTIONS
    )
    assert len(cinematography(project)["result"]["succeeded"]) == 1
    assert cinematography(project)["result"]["skipped"] == [
        {"clip_id": "c-0", "reason": "valid_analysis"}
    ]
    provider.assert_called_once()


def test_frame_reuses_without_video_input(setup):
    from models.frame import Frame

    project, provider = setup
    frame = Frame(id="frame", file_path=project.clips[0].thumbnail_path)
    project.add_frames([frame])
    options = replace(OPTIONS, mode="video")
    for status in ("succeeded", "skipped"):
        task = cinematography_task(frame)
        outcome = run_cinematography((task,), options)[0]
        assert outcome.status == status
        assert CinematographyApplication(project, (task,), options).apply(
            project, outcome
        )
    assert provider.call_args.kwargs["mode"] == "frame"
    provider.assert_called_once()


def test_local_reuse_never_enters_provider(setup, monkeypatch):
    project, provider = setup
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: True)
    options = replace(OPTIONS, tier="local")
    provider.return_value = CinematographyAnalysis(
        shot_size="CU", analysis_model="local"
    )
    evaluate(project, options=options)
    provider.side_effect = AssertionError(
        "Verified reuse must not load weights or run inference"
    )
    assert evaluate(project, options=options).status == "skipped"
    provider.assert_called_once()


def test_failed_fallback_records_actual_frame_prompt(setup):
    project, provider = setup

    def fail(**kwargs):
        kwargs["on_execution"](
            {"backend": "cloud", "model": "model", "input_mode": "frame"}
        )
        raise ValueError("Invalid answer")

    provider.side_effect = fail
    assert evaluate(project, options=replace(OPTIONS, mode="video")).status == "failed"
    record = project.clips[0].analysis_records["cinematography"]
    assert record.identity.to_dict()["model"]["execution"]["input_mode"] == "frame"
    assert record.state == "failed"


def test_owner_rejects_changed_range_after_computation(setup):
    project, _ = setup
    task = cinematography_task(project.clips[0], project.sources[0])
    application = CinematographyApplication(project, (task,), OPTIONS)
    outcome = run_cinematography((task,), OPTIONS)[0]
    project.clips[0].end_frame += 1
    assert not application.apply(project, outcome)
    assert not project.clips[0].analysis_records


def test_owner_rejects_record_with_different_prompt(setup):
    import json

    project, _ = setup
    task = cinematography_task(project.clips[0], project.sources[0])
    outcome = run_cinematography((task,), OPTIONS)[0]
    record = json.loads(outcome.record_json)
    record["identity"]["prompt_sha256"] = "0" * 64
    outcome = replace(outcome, record_json=json.dumps(record))
    assert not CinematographyApplication(project, (task,), OPTIONS).apply(
        project, outcome
    )


def test_legacy_delivery_cannot_retain_previous_verification(setup):
    project, _ = setup
    evaluate(project)
    task = replace(
        cinematography_task(project.clips[0], project.sources[0], skip_existing=False),
        snapshot_json=None,
    )
    outcome = run_cinematography((task,), OPTIONS)[0]
    assert CinematographyApplication(project, (task,)).apply(project, outcome)
    assert project.clips[0].analysis_records["cinematography"].provenance == "unknown"
