"""Explicit whole-audio transcript reuse preserves data and owner boundaries."""

from threading import Event
from unittest.mock import patch

import pytest

from core.operations.audio_transcription import AudioTranscriptionApplication, AudioTranscriptionTask, run_audio_transcription
from core.operations.legacy_reuse import accept_legacy_audio_transcript, legacy_transcription_options
from core.project import Project
from core.settings import Settings
from core.spine.analysis_reuse import accept_legacy_audio_transcripts
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.audio_source import AudioSource


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    project = Project.new()
    project.add_audio_source(AudioSource(id="audio", file_path=path, duration_seconds=2, sample_rate=48000, channels=2))
    return project


@pytest.mark.parametrize("empty", [False, True])
def test_audio_acceptance_preserves_words_and_reuses_without_inference(project, empty):
    from core.analysis_availability import audio_transcription_is_complete

    audio = project.audio_sources[0]
    audio.transcript = [] if empty else [TranscriptSegment(0, 1, "hello", -0.5, [WordTimestamp(0, 1, "hello", 0.8)])]
    original = audio.transcript
    settings = Settings(transcription_backend="groq", transcription_cloud_model="test")
    options = legacy_transcription_options(settings)
    result = accept_legacy_audio_transcripts(project, [audio.id], options=options)
    assert result["accepted"] == [audio.id]
    assert audio.transcript is original
    assert audio.analysis_records["transcribe"].legacy_reuse
    assert audio.analysis_records["transcribe"].provenance == "unknown"
    assert audio_transcription_is_complete(audio, settings=settings)
    with patch("core.operations.audio_transcription._run_audio_transcription", side_effect=AssertionError("no inference")):
        reused = run_audio_transcription(AudioTranscriptionTask.from_audio(audio, verified=True), options)
    assert reused.status == "skipped"
    settings.transcription_cloud_model = "changed"
    assert not audio_transcription_is_complete(audio, settings=settings)


@pytest.mark.parametrize("invalid", ["missing", "outside", "word", "duration", "future"])
def test_audio_acceptance_rejects_invalid_or_future_values(project, invalid):
    from models.analysis_record import UnreadableAnalysisRecord

    audio = project.audio_sources[0]
    audio.transcript = [TranscriptSegment(0, 1, "hello")]
    if invalid == "missing":
        audio.transcript = None
    elif invalid == "outside":
        audio.transcript[0].end_time = 3
    elif invalid == "word":
        audio.transcript[0].words = [WordTimestamp(0, 1.5, "hello")]
    elif invalid == "duration":
        audio.duration_seconds = 0
    else:
        audio.analysis_records["transcribe"] = UnreadableAnalysisRecord('{"version":999}')
    previous = audio.analysis_records.get("transcribe")
    result = accept_legacy_audio_transcripts(project, options=legacy_transcription_options(Settings(transcription_backend="groq")))
    assert not result["accepted"] and len(result["failed"]) == 1
    assert audio.analysis_records.get("transcribe") is previous


@pytest.mark.parametrize("change", ["duration", "path", "transcript", "session"])
def test_late_audio_acceptance_is_rejected(project, tmp_path, change):
    audio = project.audio_sources[0]
    audio.transcript = []
    options = legacy_transcription_options(Settings(transcription_backend="groq"))
    task = AudioTranscriptionTask.from_audio(audio, verified=True)
    application = AudioTranscriptionApplication(project, task, options)
    outcome = accept_legacy_audio_transcript(task, options)
    if change == "duration":
        audio.duration_seconds = 3
    elif change == "path":
        project.path = tmp_path / "changed.json"
    elif change == "transcript":
        audio.transcript = [TranscriptSegment(0, 1, "edited")]
    else:
        project = Project.new()
    assert not application.apply(project, outcome)
    assert "transcribe" not in audio.analysis_records


def test_cancelled_audio_acceptance_leaves_values_unprocessed(project):
    event = Event()
    event.set()
    result = accept_legacy_audio_transcripts(project, cancel_event=event, options=legacy_transcription_options(Settings(transcription_backend="groq")))
    assert result["unprocessed"] == ["audio"]
    assert not project.audio_sources[0].analysis_records


def test_cli_audio_acceptance_saves_unknown_provenance(project, tmp_path):
    from click.testing import CliRunner
    from cli.commands.analyze import analyze

    project.audio_sources[0].transcript = []
    path = tmp_path / "audio.sceneripper"
    assert project.save(path)
    result = CliRunner().invoke(analyze, ["accept-legacy-audio", str(path), "--audio-source-id", "audio"])
    assert result.exit_code == 0, result.output
    loaded = Project.load(path)
    record = loaded.audio_sources[0].analysis_records["transcribe"]
    assert record.legacy_reuse and record.provenance == "unknown"
    assert loaded.audio_sources[0].transcript == []
    loaded.close_writer()
