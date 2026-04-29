"""Tests for the audio source transcription flow (U6)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from core.transcription import TranscriptSegment
from models.audio_source import AudioSource


def _capture_signals(worker):
    transcript_emissions: list[tuple] = []
    error_emissions: list[str] = []
    progress_emissions: list[tuple[int, int]] = []
    finished_emissions: list[bool] = []

    worker.transcript_ready.connect(
        lambda aid, segs: transcript_emissions.append((aid, segs))
    )
    worker.error.connect(lambda msg: error_emissions.append(msg))
    worker.progress.connect(lambda c, t: progress_emissions.append((c, t)))
    worker.finished_signal.connect(lambda: finished_emissions.append(True))

    return transcript_emissions, error_emissions, progress_emissions, finished_emissions


@pytest.fixture
def audio_with_file(tmp_path):
    audio_file = tmp_path / "podcast.mp3"
    audio_file.write_bytes(b"\x00" * 32)
    return AudioSource(
        id="aud-1",
        file_path=audio_file,
        duration_seconds=120.0,
        sample_rate=44100,
        channels=2,
    )


class TestAudioTranscribeWorker:
    def test_happy_path_emits_transcript(self, audio_with_file):
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        segments = [
            TranscriptSegment(start_time=0.0, end_time=2.0, text="hello", confidence=0.9),
            TranscriptSegment(start_time=2.0, end_time=4.0, text="world", confidence=0.95),
        ]

        worker = AudioTranscribeWorker(audio_with_file)
        transcripts, errors, progress, finished = _capture_signals(worker)

        with patch(
            "core.transcription.transcribe_video",
            return_value=segments,
        ):
            worker.run()

        assert errors == []
        assert len(transcripts) == 1
        emitted_id, emitted_segments = transcripts[0]
        assert emitted_id == "aud-1"
        assert emitted_segments == segments
        assert finished == [True]
        # Initial 0/100 + final 100/100 progress
        assert progress[0] == (0, 100)
        assert progress[-1] == (100, 100)

    def test_missing_file_emits_error(self, tmp_path):
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        ghost = AudioSource(
            id="ghost",
            file_path=tmp_path / "missing.wav",
            duration_seconds=10.0,
        )
        worker = AudioTranscribeWorker(ghost)
        transcripts, errors, _, finished = _capture_signals(worker)

        worker.run()

        assert transcripts == []
        assert any("missing on disk" in e for e in errors)
        assert finished == [True]

    def test_transcription_exception_surfaces_as_error(self, audio_with_file):
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        worker = AudioTranscribeWorker(audio_with_file)
        transcripts, errors, _, finished = _capture_signals(worker)

        with patch(
            "core.transcription.transcribe_video",
            side_effect=RuntimeError("Whisper backend exploded"),
        ):
            worker.run()

        assert transcripts == []
        assert any("Transcription failed" in e and "Whisper backend exploded" in e for e in errors)
        assert finished == [True]

    def test_cancellation_skips_transcript_emit(self, audio_with_file):
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        worker = AudioTranscribeWorker(audio_with_file)
        worker.cancel()
        transcripts, errors, _, _ = _capture_signals(worker)

        with patch("core.transcription.transcribe_video", return_value=[]):
            worker.run()

        # Cancellation isn't an error; just no transcript emitted
        assert transcripts == []
        assert errors == []

    def test_empty_segments_round_trip_to_audio_source(self, audio_with_file):
        """Even with no detected speech, transcript_ready should fire with []
        so the caller can mark "transcribed but silent" and won't re-run.
        """
        from ui.workers.audio_transcribe_worker import AudioTranscribeWorker

        worker = AudioTranscribeWorker(audio_with_file)
        transcripts, errors, _, _ = _capture_signals(worker)

        with patch("core.transcription.transcribe_video", return_value=[]):
            worker.run()

        assert errors == []
        assert len(transcripts) == 1
        assert transcripts[0] == ("aud-1", [])


class TestAudioSourceTranscriptIntegration:
    def test_setting_transcript_round_trips_through_save_load(self, tmp_path):
        """End-to-end: transcribe → set on AudioSource → save → load → segments survive."""
        from core.project import Project

        # Set up a project with an audio source
        audio_file = tmp_path / "voice.wav"
        audio_file.write_bytes(b"")
        audio = AudioSource(
            id="aud-1",
            file_path=audio_file,
            duration_seconds=30.0,
            sample_rate=16000,
            channels=1,
        )

        project_path = tmp_path / "p.sceneripper"
        project = Project(path=project_path)
        project.add_audio_source(audio)

        # Apply a "transcription result" to the audio source
        audio.transcript = [
            TranscriptSegment(start_time=0.0, end_time=2.5, text="hello world", confidence=0.9),
            TranscriptSegment(start_time=2.5, end_time=5.0, text="how are you", confidence=0.85),
        ]

        # Save and reload
        assert project.save(project_path)
        reloaded = Project.load(project_path)

        assert len(reloaded.audio_sources) == 1
        loaded_audio = reloaded.audio_sources[0]
        assert loaded_audio.transcript is not None
        assert len(loaded_audio.transcript) == 2
        assert loaded_audio.transcript[0].text == "hello world"
        assert loaded_audio.transcript[1].text == "how are you"
