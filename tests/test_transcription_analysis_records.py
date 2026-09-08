"""Transcript reuse and publication require matching execution and editorial inputs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.operations.transcription import (
    TranscriptionApplication,
    TranscriptionOptions,
    run_transcription,
)
from core.operations.transcription_records import transcription_task
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.analysis_record import AnalysisRecord
from tests.test_spine_analyze import _build_project


OPTIONS = TranscriptionOptions(backend="faster-whisper")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 1)
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    provider = Mock(
        return_value=[
            TranscriptSegment(
                0.1, 1.2, "hello", -0.6, [WordTimestamp(0.1, 1.2, "hello", 0.9)], "en"
            )
        ]
    )
    monkeypatch.setattr("core.transcription.transcribe_clip", provider)
    return project, provider


def run(project, options=OPTIONS, *, skip=True, apply=True):
    clip = project.clips[0]
    task = transcription_task(
        clip, project.sources_by_id[clip.source_id], skip_existing=skip
    )
    application = TranscriptionApplication(project, (task,), options)
    outcome = run_transcription((task,), options)[0]
    if apply:
        assert application.apply(project, outcome)
    return outcome, application


@pytest.mark.parametrize("empty", [False, True])
def test_verified_transcripts_reuse_and_round_trip(setup, tmp_path, empty):
    from core.project import Project

    project, provider = setup
    if empty:
        provider.return_value = []
    assert run(project)[0].status == "succeeded"
    project.save(tmp_path / "project.json")
    project = Project.load(project.path)
    assert run(project, replace(OPTIONS, parallelism=4))[0].status == "skipped"
    assert provider.call_count == 1


def test_legacy_value_recomputes(setup):
    project, provider = setup
    project.clips[0].transcript = []
    assert run(project)[0].status == "succeeded"
    assert provider.call_count == 1


@pytest.mark.parametrize(
    "change",
    ["model", "language", "segmentation", "media", "range", "fps", "projection"],
)
def test_changed_inputs_recompute(setup, change):
    project, provider = setup
    run(project)
    options = OPTIONS
    if change == "model":
        options = replace(options, model="base")
    elif change == "language":
        options = replace(options, language="es")
    elif change == "segmentation":
        options = replace(options, segmentation_mode="sentence")
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "range":
        project.clips[0].end_frame += 1
    elif change == "fps":
        project.sources[0].fps = 24
    else:
        project.clips[0].transcript[0].text = "edited"
    assert run(project, options)[0].status == "succeeded"
    assert provider.call_count == 2


def test_failure_preserves_display_and_invalidates_old_record(setup):
    project, provider = setup
    run(project)
    previous = project.clips[0].transcript
    provider.side_effect = RuntimeError("offline")
    outcome, _ = run(project, skip=False)
    assert outcome.status == "failed"
    assert project.clips[0].transcript == previous
    assert project.clips[0].analysis_records["transcribe"].state == "failed"


@pytest.mark.parametrize("change", ["record", "text", "media", "range", "source"])
def test_queued_delivery_rejects_newer_inputs(setup, change):
    project, _ = setup
    outcome, application = run(project, apply=False)
    if change == "record":
        project.clips[0].analysis_records["transcribe"] = AnalysisRecord.legacy(
            {"transcript": []}
        )
    elif change == "text":
        project.clips[0].transcript = []
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "range":
        project.clips[0].end_frame += 1
    else:
        project.clips[0].source_id = "different"
    assert not application.apply(project, outcome)


def test_raw_delivery_clears_prior_verification(setup):
    from core.operations.transcription import TranscriptionOutcome, snapshot_tasks

    project, _ = setup
    run(project)
    tasks = snapshot_tasks(project.clips, project.sources_by_id, skip_existing=False)
    assert TranscriptionApplication(project, tasks).apply(
        project, TranscriptionOutcome(project.clips[0].id, "succeeded", ())
    )
    assert project.clips[0].analysis_records["transcribe"].provenance == "unknown"


@pytest.mark.parametrize("invalid", ["time", "text", "word"])
def test_invalid_inference_is_recorded_as_failure(setup, invalid):
    project, provider = setup
    segment = provider.return_value[0]
    if invalid == "time":
        segment.start_time = float("nan")
    elif invalid == "text":
        segment.text = None
    else:
        segment.words[0].probability = 2
    outcome, _ = run(project)
    assert outcome.code == "invalid_result"
    assert project.clips[0].transcript is None
    assert project.clips[0].analysis_records["transcribe"].state == "failed"


def test_different_requested_model_cannot_publish_result(setup):
    project, _ = setup
    outcome, _ = run(project, apply=False)
    clip = project.clips[0]
    task = transcription_task(clip, project.sources[0])
    application = TranscriptionApplication(
        project, (task,), replace(OPTIONS, model="base")
    )
    assert not application.apply(project, outcome)


def test_no_audio_record_reuses_without_inference(setup, monkeypatch):
    project, provider = setup
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: False)
    provider.return_value = []
    run(project)
    assert run(project)[0].status == "skipped"
    assert provider.call_count == 1
    record = project.clips[0].analysis_records["transcribe"]
    assert record.identity.to_dict()["model"]["execution"]["backend"] == "audio-probe"


def test_media_changed_during_inference_does_not_publish_record(setup):
    project, provider = setup

    def compute(**kwargs):
        project.sources[0].file_path.write_bytes(b"changed during inference")
        return []

    provider.side_effect = compute
    outcome, application = run(project, apply=False)
    assert outcome.code == "stale_input"
    assert not application.apply(project, outcome)


def test_actual_execution_is_retained_and_not_reused_for_different_request(setup):
    project, provider = setup

    def compute(**kwargs):
        kwargs["on_execution"](
            {
                "backend": "faster-whisper",
                "model": "actual-model",
                "input_mode": "audio",
            }
        )
        return []

    provider.side_effect = compute
    run(project)
    record = project.clips[0].analysis_records["transcribe"]
    assert record.identity.to_dict()["model"]["execution"]["model"] == "actual-model"
    assert run(project)[0].status == "succeeded"
    assert provider.call_count == 2


def test_same_content_with_new_stamp_refreshes_binding_without_inference(setup):
    import os

    project, provider = setup
    run(project)
    prior = project.clips[0].analysis_records["transcribe"]
    path = project.sources[0].file_path
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    assert run(project)[0].status == "skipped"
    current = project.clips[0].analysis_records["transcribe"]
    assert prior.identity == current.identity
    assert prior.input_json != current.input_json
    assert provider.call_count == 1


def test_batch_hashes_shared_source_once(setup, monkeypatch):
    from pathlib import Path
    from models.clip import Clip

    project, provider = setup
    source = project.sources[0]
    project.add_clips(
        [Clip(id="second", source_id=source.id, start_frame=60, end_frame=120)]
    )
    original = Path.open
    reads = []

    def open_file(path, *args, **kwargs):
        if path == source.file_path:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    tasks = tuple(transcription_task(clip, source) for clip in project.clips)
    outcomes = run_transcription(tasks, replace(OPTIONS, parallelism=2))
    assert all(outcome.status == "succeeded" for outcome in outcomes)
    assert provider.call_count == 2
    assert len(reads) == 1
