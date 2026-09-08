"""Completion needs current cinematography provenance without loading VLM runtimes."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.analysis_availability import compute_disabled_operations
from core.operations.cinematography import (
    CinematographyApplication,
    cinematography_task,
    resolve_options,
    run_cinematography,
)
from core.project import Project
from core.settings import Settings
from models.cinematography import CinematographyAnalysis
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def analyzed(tmp_path, monkeypatch):
    settings = Settings(
        cinematography_tier="cloud",
        cinematography_model="model",
        cinematography_input_mode="frame",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.cinematography.analyze_cinematography",
        Mock(
            return_value=CinematographyAnalysis(shot_size="CU", analysis_model="model")
        ),
    )
    task = cinematography_task(project.clips[0], project.sources[0])
    options = resolve_options()
    outcome = run_cinematography((task,), options)[0]
    assert CinematographyApplication(project, (task,), options).apply(project, outcome)
    return project, settings


def complete(project):
    return "cinematography" in compute_disabled_operations(
        project.clips, ["cinematography"], sources_by_id=project.sources_by_id
    )


def test_verified_analysis_is_complete_after_save(analyzed, tmp_path):
    project, _ = analyzed
    assert complete(project)
    assert project.save(tmp_path / "project.json")
    assert complete(Project.load(project.path))


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "failed",
        "model",
        "mode",
        "tier",
        "fps",
        "range",
        "source",
        "image",
        "analysis",
        "shot_type",
        "prompt",
        "runtime",
    ],
)
def test_invalidated_analysis_stays_available(analyzed, monkeypatch, tmp_path, change):
    project, settings = analyzed
    clip = project.clips[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "failed":
        clip.analysis_records["cinematography"] = replace(
            clip.analysis_records["cinematography"], state="failed", error="Failed"
        )
    elif change == "model":
        settings.cinematography_model = "other"
    elif change == "mode":
        settings.cinematography_input_mode = "video"
    elif change == "tier":
        settings.cinematography_tier = "local"
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "range":
        clip.end_frame += 1
    elif change == "source":
        project.sources[0].file_path = tmp_path / "other.mp4"
    elif change == "image":
        clip.thumbnail_path.write_bytes(b"changed")
    elif change == "analysis":
        clip.cinematography.shot_size = "MS"
    elif change == "shot_type":
        clip.shot_type = "wide"
    elif change == "runtime":
        monkeypatch.setattr(
            "core.operations.cinematography.model_runtime",
            lambda *args: {"packages": {"test": "changed"}},
        )
    else:
        from models.analysis_record import AnalysisIdentity

        record = clip.analysis_records["cinematography"]
        identity = record.identity.to_dict()
        identity["prompt_sha256"] = "0" * 64
        clip.analysis_records["cinematography"] = replace(
            record, identity=AnalysisIdentity.from_dict(identity)
        )
    assert not complete(project)


def test_missing_source_context_cannot_prove_completion(analyzed):
    project, _ = analyzed
    assert "cinematography" not in compute_disabled_operations(
        project.clips, ["cinematography"]
    )


def test_completion_does_not_probe_local_runtime(analyzed, monkeypatch):
    project, settings = analyzed
    monkeypatch.setattr(
        "core.analysis.description.is_mlx_vlm_available",
        Mock(side_effect=AssertionError("UI must not probe MLX")),
    )
    assert complete(project)
    settings.cinematography_tier = "local"
    monkeypatch.setattr(
        "core.analysis_model_identity.known_mlx_vlm_availability", lambda: None
    )
    assert not complete(project)


def test_local_completion_defers_unprobed_backend(analyzed, monkeypatch):
    project, settings = analyzed
    settings.cinematography_tier = "local"
    settings.cinematography_local_model = "model"
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: True)
    task = cinematography_task(project.clips[0], project.sources[0])
    options = resolve_options()
    outcome = run_cinematography((task,), options)[0]
    assert CinematographyApplication(project, (task,), options).apply(project, outcome)
    monkeypatch.setattr(
        "core.analysis.description.is_mlx_vlm_available",
        Mock(side_effect=AssertionError("UI must not import MLX")),
    )
    monkeypatch.setattr(
        "core.analysis_model_identity.known_mlx_vlm_availability", lambda: True
    )
    assert complete(project)
    monkeypatch.setattr(
        "core.analysis_model_identity.known_mlx_vlm_availability", lambda: None
    )
    assert not complete(project)


def test_frame_completion_uses_actual_frame_mode(analyzed):
    from core.analysis_availability import operation_is_complete_for_clip
    from models.frame import Frame

    project, settings = analyzed
    settings.cinematography_input_mode = "video"
    frame = Frame(id="frame", file_path=project.clips[0].thumbnail_path)
    project.add_frames([frame])
    task = cinematography_task(frame)
    options = resolve_options()
    outcome = run_cinematography((task,), options)[0]
    assert CinematographyApplication(project, (task,), options).apply(project, outcome)
    assert operation_is_complete_for_clip("cinematography", frame)


@pytest.mark.asyncio
async def test_mcp_status_counts_verified_cinematography(analyzed, tmp_path):
    import json
    from scene_ripper_mcp.tools.analyze import get_analysis_status

    project, settings = analyzed
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    response = json.loads(await get_analysis_status(str(path)))
    assert response.get("success"), response
    assert response["analysis"]["cinematography"]["analyzed"] == 1
    settings.cinematography_model = "other"
    response = json.loads(await get_analysis_status(str(path)))
    assert response["analysis"]["cinematography"]["pending"] == 1
