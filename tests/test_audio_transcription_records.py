"""Whole-audio transcription verifies inputs, execution, and owner publication."""

from dataclasses import asdict, replace
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.operations.audio_transcription import (
    AudioTranscriptionApplication,
    AudioTranscriptionTask,
    AudioTranscriptionOutcome,
    run_audio_transcription,
)
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.audio_source import AudioSource

OPTIONS = TranscriptionOptions(backend="faster-whisper")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    audio = AudioSource(
        file_path=path, duration_seconds=2, sample_rate=48000, channels=2
    )
    project = Project.new()
    project.add_audio_source(audio)
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    provider = Mock(
        return_value=[
            TranscriptSegment(
                0, 1, "hello", -0.5, [WordTimestamp(0, 1, "hello", 0.8)], "en"
            )
        ]
    )
    monkeypatch.setattr("core.transcription.transcribe_video", provider)
    return project, provider


def run(project, options=OPTIONS, *, apply=True, skip=True, cancel=None):
    task = AudioTranscriptionTask.from_audio(
        project.audio_sources[0], verified=True, skip_existing=skip
    )
    application = AudioTranscriptionApplication(project, task, options)
    outcome = run_audio_transcription(task, options, cancel_event=cancel)
    outcome = AudioTranscriptionOutcome.from_dict(
        json.loads(json.dumps(asdict(outcome)))
    )
    if apply:
        assert application.apply(project, outcome)
    return outcome, application


@pytest.mark.parametrize("empty", [False, True])
def test_verified_audio_reuses_after_save(setup, tmp_path, empty):
    project, provider = setup
    if empty:
        provider.return_value = []
    assert run(project)[0].status == "succeeded"
    assert project.save(tmp_path / "project.json")
    project = Project.load(project.path)
    assert run(project, replace(OPTIONS, parallelism=4))[0].status == "skipped"
    assert provider.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "language",
        "media",
        "sample_rate",
        "duration",
        "channels",
        "legacy",
        "text",
    ],
)
def test_stale_audio_requires_new_inference(setup, change):
    project, provider = setup
    run(project)
    audio = project.audio_sources[0]
    options = OPTIONS
    if change == "model":
        options = replace(options, model="medium.en")
    elif change == "language":
        options = replace(options, language="es")
    elif change == "media":
        audio.file_path.write_bytes(b"replacement")
    elif change == "sample_rate":
        audio.sample_rate = 16000
    elif change == "duration":
        audio.duration_seconds += 1
    elif change == "channels":
        audio.channels = 1
    elif change == "legacy":
        audio.analysis_records.clear()
    else:
        audio.transcript[0].text = "edited"
    assert run(project, options)[0].status == "succeeded"
    assert provider.call_count == 2


def test_failed_refresh_preserves_display_and_saves_failure_record(setup):
    project, provider = setup
    run(project)
    previous = project.audio_sources[0].transcript
    provider.side_effect = RuntimeError("offline")
    assert run(project, skip=False)[0].status == "failed"
    assert project.audio_sources[0].transcript == previous
    assert project.audio_sources[0].analysis_records["transcribe"].state == "failed"


@pytest.mark.parametrize("change", ["record", "text", "metadata", "media", "session"])
def test_delivery_rejects_changed_owner_state(setup, change):
    from models.analysis_record import AnalysisRecord

    project, _ = setup
    outcome, application = run(project, apply=False)
    audio = project.audio_sources[0]
    if change == "record":
        audio.analysis_records["transcribe"] = AnalysisRecord.legacy({"transcript": []})
    elif change == "text":
        audio.transcript = []
    elif change == "metadata":
        audio.channels += 1
    elif change == "media":
        audio.file_path.write_bytes(b"replacement")
    else:
        project.clear()
    assert not application.apply(project, outcome)


def test_cancelled_inference_does_not_publish_verification(setup):
    project, provider = setup
    cancel = Event()

    def compute(*args, **kwargs):
        cancel.set()
        return []

    provider.side_effect = compute
    outcome, application = run(project, apply=False, cancel=cancel)
    assert outcome.status == "unprocessed" and outcome.record_json is None
    assert not application.apply(project, outcome)


@pytest.mark.parametrize(
    "backend,extraction",
    [
        ("faster-whisper", "pyav-direct/v1"),
        ("groq", "container-upload/v1"),
        ("mlx-whisper", "whole-file-pcm-s16le-16000-mono/v1"),
    ],
)
def test_runtime_describes_actual_whole_file_decoding(
    setup, monkeypatch, backend, extraction
):
    project, provider = setup
    monkeypatch.setattr("core.transcription._resolve_backend", lambda _: backend)
    options = replace(
        OPTIONS,
        backend=backend,
        cloud_model="cloud-model" if backend == "groq" else None,
    )
    run(project, options)
    runtime = (
        project.audio_sources[0]
        .analysis_records["transcribe"]
        .identity.to_dict()["model"]
    )
    assert runtime["extraction"] == extraction
    assert ("ffmpeg" in runtime["binaries"]) == (backend == "mlx-whisper")
    assert provider.call_count == 1


def test_observers_see_transcript_and_record_together(setup):
    project, _ = setup
    observed = []

    def observe(event, data):
        if event == "audio_sources_changed":
            audio = data[0]
            record = audio.analysis_records["transcribe"]
            observed.append(
                (
                    record.provenance,
                    record.value
                    == {"transcript": [s.to_dict() for s in audio.transcript]},
                )
            )

    project.add_observer(observe)
    run(project)
    assert observed == [("verified", True)]


def test_mismatched_record_is_rejected_before_mutation(setup):
    project, _ = setup
    run(project)
    audio = project.audio_sources[0]
    original = audio.to_dict()
    with pytest.raises(ValueError, match="do not match"):
        project.set_audio_transcript(
            audio.id, [], analysis_record=audio.analysis_records["transcribe"]
        )
    assert audio.to_dict() == original


def test_actual_execution_is_retained_for_failed_attempt(setup):
    project, provider = setup

    def compute(*args, **kwargs):
        kwargs["on_execution"](
            {
                "backend": "faster-whisper",
                "model": "actual-model",
                "input_mode": "audio",
            }
        )
        raise RuntimeError("inference failed")

    provider.side_effect = compute
    assert run(project)[0].status == "failed"
    record = project.audio_sources[0].analysis_records["transcribe"]
    assert record.identity.to_dict()["model"]["execution"]["model"] == "actual-model"


def test_invalid_word_probability_is_failure_not_cached_success(setup):
    project, provider = setup
    provider.return_value[0].words[0].probability = float("nan")
    assert run(project)[0].status == "failed"
    assert project.audio_sources[0].transcript is None


def test_raw_delivery_clears_previous_verification(setup):
    project, _ = setup
    run(project)
    audio = project.audio_sources[0]
    task = AudioTranscriptionTask.from_audio(audio)
    assert AudioTranscriptionApplication(project, task).apply(
        project, AudioTranscriptionOutcome(audio.id, "succeeded")
    )
    assert audio.analysis_records["transcribe"].provenance == "unknown"
