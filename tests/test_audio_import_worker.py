"""Tests for AudioImportWorker."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.audio_source import AudioSource
from ui.workers.audio_import_worker import AudioImportWorker


def _capture_signals(worker):
    """Wire up plain-Python collectors for the worker's signals.

    QThread/Signals work fine without a running event loop when we drive
    run() synchronously: emitted values land in our list immediately.
    """
    audio_emissions: list[AudioSource] = []
    error_emissions: list[str] = []
    progress_emissions: list[tuple[int, int]] = []
    finished_emissions: list[bool] = []

    worker.audio_ready.connect(lambda a: audio_emissions.append(a))
    worker.error.connect(lambda msg: error_emissions.append(msg))
    worker.progress.connect(lambda c, t: progress_emissions.append((c, t)))
    worker.finished_signal.connect(lambda: finished_emissions.append(True))

    return audio_emissions, error_emissions, progress_emissions, finished_emissions


@pytest.fixture
def fake_processor():
    """Return a FakeProcessor class that simulates FFmpegProcessor for tests."""
    class FakeProcessor:
        def __init__(self, *, audio_info=None, raise_on_init=None,
                     raise_on_probe=None, ffprobe_available=True):
            self._audio_info = audio_info
            self._raise_on_probe = raise_on_probe
            self.ffprobe_available = ffprobe_available
            if raise_on_init is not None:
                raise raise_on_init

        def get_audio_info(self, path):
            if self._raise_on_probe is not None:
                raise self._raise_on_probe
            return self._audio_info

    return FakeProcessor


class TestAudioImportWorker:
    def test_happy_path_emits_audio_ready(self, tmp_path, fake_processor):
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")

        info = {"duration": 180.0, "sample_rate": 44100, "channels": 2, "codec": "wav"}
        worker = AudioImportWorker(audio_file)
        audio_emissions, errors, progress, finished = _capture_signals(worker)

        with patch(
            "ui.workers.audio_import_worker.FFmpegProcessor",
            lambda: fake_processor(audio_info=info),
        ):
            worker.run()

        assert errors == []
        assert len(audio_emissions) == 1
        emitted = audio_emissions[0]
        assert isinstance(emitted, AudioSource)
        assert emitted.file_path == audio_file
        assert emitted.duration_seconds == 180.0
        assert emitted.sample_rate == 44100
        assert emitted.channels == 2
        assert progress == [(0, 1), (1, 1)]
        assert finished == [True]

    def test_missing_file_emits_error(self, tmp_path, fake_processor):
        worker = AudioImportWorker(tmp_path / "does_not_exist.wav")
        audio_emissions, errors, _, finished = _capture_signals(worker)

        # No need to patch processor — error fires before it's constructed
        worker.run()

        assert audio_emissions == []
        assert any("File not found" in e for e in errors)
        assert finished == [True]

    def test_no_audio_stream_emits_error(self, tmp_path, fake_processor):
        not_audio = tmp_path / "wat.bin"
        not_audio.write_bytes(b"\x00" * 16)

        worker = AudioImportWorker(not_audio)
        audio_emissions, errors, _, _ = _capture_signals(worker)

        with patch(
            "ui.workers.audio_import_worker.FFmpegProcessor",
            lambda: fake_processor(raise_on_probe=ValueError("No audio stream found")),
        ):
            worker.run()

        assert audio_emissions == []
        assert any("Not an audio file" in e for e in errors)

    def test_ffprobe_failure_emits_error(self, tmp_path, fake_processor):
        bad_file = tmp_path / "broken.wav"
        bad_file.write_bytes(b"")

        worker = AudioImportWorker(bad_file)
        audio_emissions, errors, _, _ = _capture_signals(worker)

        with patch(
            "ui.workers.audio_import_worker.FFmpegProcessor",
            lambda: fake_processor(raise_on_probe=RuntimeError("FFprobe failed")),
        ):
            worker.run()

        assert audio_emissions == []
        assert any("Failed to probe audio" in e for e in errors)

    def test_zero_duration_is_rejected(self, tmp_path, fake_processor):
        audio_file = tmp_path / "empty.wav"
        audio_file.write_bytes(b"")

        info = {"duration": 0.0, "sample_rate": 44100, "channels": 2, "codec": "wav"}
        worker = AudioImportWorker(audio_file)
        audio_emissions, errors, _, _ = _capture_signals(worker)

        with patch(
            "ui.workers.audio_import_worker.FFmpegProcessor",
            lambda: fake_processor(audio_info=info),
        ):
            worker.run()

        assert audio_emissions == []
        assert any("zero duration" in e for e in errors)

    def test_cancellation_before_emission_skips_audio_ready(self, tmp_path, fake_processor):
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")

        info = {"duration": 60.0, "sample_rate": 44100, "channels": 2, "codec": "wav"}
        worker = AudioImportWorker(audio_file)
        worker.cancel()  # set cancellation before run()
        audio_emissions, errors, _, _ = _capture_signals(worker)

        with patch(
            "ui.workers.audio_import_worker.FFmpegProcessor",
            lambda: fake_processor(audio_info=info),
        ):
            worker.run()

        assert audio_emissions == []
        # Cancellation is not an error — no error message emitted
        assert errors == []

    def test_ffprobe_not_available_emits_error(self, tmp_path, fake_processor):
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"")

        worker = AudioImportWorker(audio_file)
        audio_emissions, errors, _, _ = _capture_signals(worker)

        with patch(
            "ui.workers.audio_import_worker.FFmpegProcessor",
            lambda: fake_processor(audio_info=None, ffprobe_available=False),
        ):
            worker.run()

        assert audio_emissions == []
        assert any("FFprobe is not available" in e for e in errors)
