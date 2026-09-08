"""Transcription completion verifies settings and media without running probes."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.analysis_availability import operation_is_complete_for_clip
from core.operations.transcription import (
    TranscriptionApplication,
    TranscriptionOptions,
    run_transcription,
)
from core.operations.transcription_records import transcription_task
from core.project import Project
from core.settings import Settings
from tests.test_spine_analyze import _build_project


@pytest.fixture
def analyzed(tmp_path, monkeypatch):
    settings = Settings(
        transcription_backend="faster-whisper",
        transcription_model="small.en",
        transcription_language="en",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription.transcribe_clip", Mock(return_value=[]))
    project = _build_project(tmp_path, 1)
    task = transcription_task(project.clips[0], project.sources[0])
    options = TranscriptionOptions(backend="faster-whisper")
    outcome = run_transcription((task,), options)[0]
    assert TranscriptionApplication(project, (task,), options).apply(project, outcome)
    return project, settings


def complete(project):
    return operation_is_complete_for_clip(
        "transcribe", project.clips[0], source=project.sources[0]
    )


def test_verified_silence_is_complete_after_save(analyzed, tmp_path):
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
        "language",
        "backend",
        "segmentation",
        "fps",
        "range",
        "path",
        "media",
        "value",
        "identity_range",
    ],
)
def test_invalidated_record_is_pending(analyzed, tmp_path, change):
    project, settings = analyzed
    clip = project.clips[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "failed":
        clip.analysis_records["transcribe"] = replace(
            clip.analysis_records["transcribe"], state="failed", error="offline"
        )
    elif change == "model":
        settings.transcription_model = "medium.en"
    elif change == "language":
        settings.transcription_language = "es"
    elif change == "backend":
        settings.transcription_backend = "groq"
    elif change == "segmentation":
        settings.transcription_segmentation_mode = "sentence"
    elif change == "fps":
        project.sources[0].fps = 24
    elif change == "range":
        clip.end_frame += 1
    elif change == "path":
        project.sources[0].file_path = tmp_path / "different.mp4"
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "identity_range":
        from models.analysis_record import AnalysisIdentity

        record = clip.analysis_records["transcribe"]
        data = record.identity.to_dict()
        data["source_range"]["fps"] += 1
        clip.analysis_records["transcribe"] = replace(
            record, identity=AnalysisIdentity.from_dict(data)
        )
    else:
        clip.transcript = None
    assert not complete(project)


def test_completion_requires_source_context(analyzed):
    project, _ = analyzed
    assert not operation_is_complete_for_clip("transcribe", project.clips[0])


def test_completion_does_not_probe_media_or_load_models(analyzed, monkeypatch):
    project, _ = analyzed
    forbidden = Mock(side_effect=AssertionError("completion must remain cheap"))
    for name in ("_has_audio_stream", "get_model", "get_mlx_model", "transcribe_clip"):
        monkeypatch.setattr("core.transcription." + name, forbidden)
    assert complete(project)
    forbidden.assert_not_called()


def test_no_audio_completion_does_not_repeat_probe(analyzed, monkeypatch):
    project, _ = analyzed
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: False)
    options = TranscriptionOptions(backend="faster-whisper")
    task = transcription_task(project.clips[0], project.sources[0], skip_existing=False)
    outcome = run_transcription((task,), options)[0]
    assert TranscriptionApplication(project, (task,), options).apply(project, outcome)
    forbidden = Mock(side_effect=AssertionError("do not probe on owner thread"))
    monkeypatch.setattr("core.transcription._has_audio_stream", forbidden)
    assert complete(project)
    forbidden.assert_not_called()


def test_runtime_change_requires_verification(analyzed, monkeypatch):
    project, _ = analyzed
    monkeypatch.setattr(
        "core.operations.transcription_records.model_runtime",
        lambda *args: {"packages": {"changed": "version"}},
    )
    assert not complete(project)


@pytest.mark.asyncio
async def test_mcp_status_counts_verified_silence_and_rejects_stale_settings(
    analyzed, tmp_path
):
    import json
    from scene_ripper_mcp.tools.analyze import get_analysis_status

    project, settings = analyzed
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    response = json.loads(await get_analysis_status(str(path)))
    assert response["analysis"]["transcripts"]["analyzed"] == 1
    settings.transcription_model = "medium.en"
    response = json.loads(await get_analysis_status(str(path)))
    assert response["analysis"]["transcripts"]["pending"] == 1
