"""Shared audio computation cannot publish cancelled or changed inputs."""

from threading import Event
from unittest.mock import patch

import pytest

from core.operations.audio_transcription import (
    AudioTranscriptionApplication,
    AudioTranscriptionTask,
    run_audio_transcription,
)
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from core.transcription_models import TranscriptSegment, WordTimestamp
from models.audio_source import AudioSource


@pytest.mark.parametrize("change", ["none", "before", "during", "cancel", "failure"])
def test_audio_computation_checks_media_and_cancellation(tmp_path, change):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"audio")
    audio = AudioSource(file_path=path)
    task = AudioTranscriptionTask.from_audio(audio)
    cancelled = Event()
    if change == "before":
        path.write_bytes(b"replacement")

    def provider(*args, **kwargs):
        if change == "during":
            path.write_bytes(b"replacement")
        if change == "cancel":
            cancelled.set()
        if change == "failure":
            raise RuntimeError("provider failed")
        return []

    with patch("core.transcription.transcribe_video", side_effect=provider) as compute:
        outcome = run_audio_transcription(
            task, TranscriptionOptions(), cancel_event=cancelled
        )
    assert (
        outcome.status
        == {
            "none": "succeeded",
            "before": "failed",
            "during": "failed",
            "cancel": "unprocessed",
            "failure": "failed",
        }[change]
    )
    if change == "before":
        compute.assert_not_called()
    assert audio.transcript is None


def test_audio_application_preserves_words_and_detaches_provider_result(tmp_path):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"audio")
    audio = AudioSource(file_path=path)
    project = Project.new()
    project.add_audio_source(audio)
    task = AudioTranscriptionTask.from_audio(audio)
    application = AudioTranscriptionApplication(project, task)
    segments = [
        TranscriptSegment(
            0.123456789,
            1.23456789,
            "hello",
            0.987654321,
            words=[WordTimestamp(0.123456789, 1.23456789, "hello", 0.987654321)],
        )
    ]
    with patch("core.transcription.transcribe_video", return_value=segments):
        outcome = run_audio_transcription(task, TranscriptionOptions())
    assert application.apply(project, outcome)
    assert not application.apply(project, outcome)
    expected = audio.to_dict()
    segments[0].text = "mutated"
    outcome.segments[0].words[0].text = "mutated"
    assert audio.to_dict() == expected
    saved = tmp_path / "project.json"
    assert project.save(saved)
    project.close_writer()
    restored = Project.load(saved)
    try:
        assert restored.audio_sources[0].to_dict() == expected
    finally:
        restored.close_writer()
