"""Tests for the AudioSource model."""

from pathlib import Path

import pytest

from core.transcription import TranscriptSegment
from models.audio_source import AudioSource


class TestAudioSourceBasics:
    def test_default_id_is_uuid(self):
        a = AudioSource()
        b = AudioSource()
        assert a.id != b.id
        assert len(a.id) >= 32  # UUID-ish

    def test_filename_property(self):
        audio = AudioSource(file_path=Path("/tmp/song.mp3"))
        assert audio.filename == "song.mp3"

    def test_duration_str_minutes_seconds(self):
        audio = AudioSource(duration_seconds=125.5)
        assert audio.duration_str == "2:05"

    def test_duration_str_seconds_only(self):
        audio = AudioSource(duration_seconds=42.0)
        assert audio.duration_str == "0:42"

    def test_duration_str_hours(self):
        audio = AudioSource(duration_seconds=3725.0)  # 1h 02m 05s
        assert audio.duration_str == "1:02:05"

    def test_duration_str_zero(self):
        audio = AudioSource(duration_seconds=0.0)
        assert audio.duration_str == "0:00"


class TestAudioSourceSerialization:
    def test_round_trip_preserves_fields(self):
        original = AudioSource(
            id="aud-1",
            file_path=Path("/tmp/song.wav"),
            duration_seconds=180.5,
            sample_rate=44100,
            channels=2,
        )
        data = original.to_dict()
        restored = AudioSource.from_dict(data)

        assert restored.id == original.id
        assert restored.file_path == original.file_path
        assert restored.duration_seconds == original.duration_seconds
        assert restored.sample_rate == original.sample_rate
        assert restored.channels == original.channels

    def test_to_dict_with_base_path_stores_relative(self, tmp_path):
        audio_file = tmp_path / "audio" / "song.mp3"
        audio_file.parent.mkdir()
        audio_file.write_bytes(b"")

        audio = AudioSource(file_path=audio_file)
        data = audio.to_dict(base_path=tmp_path)

        assert data["file_path"] == "audio/song.mp3"
        assert data["_absolute_path"] == audio_file.as_posix()

    def test_to_dict_with_base_path_falls_back_to_absolute_when_not_relative(self, tmp_path):
        # File outside the base path
        outside = Path("/elsewhere/song.mp3")
        audio = AudioSource(file_path=outside)
        data = audio.to_dict(base_path=tmp_path)

        assert data["file_path"] == outside.as_posix()
        assert data["_absolute_path"] == outside.as_posix()

    def test_from_dict_resolves_relative_against_base_path(self, tmp_path):
        audio_file = tmp_path / "music" / "track.wav"
        audio_file.parent.mkdir()
        audio_file.write_bytes(b"")

        data = {
            "id": "aud-1",
            "file_path": "music/track.wav",
            "duration_seconds": 60.0,
        }
        restored = AudioSource.from_dict(data, base_path=tmp_path)
        assert restored.file_path == audio_file.resolve()

    def test_from_dict_uses_absolute_fallback_when_relative_missing(self, tmp_path):
        # File doesn't exist at the relative location, but absolute path does
        actual = tmp_path / "actual.mp3"
        actual.write_bytes(b"")

        data = {
            "id": "aud-1",
            "file_path": "missing/track.mp3",
            "_absolute_path": actual.as_posix(),
            "duration_seconds": 30.0,
        }
        # base_path that doesn't contain "missing/track.mp3"
        unrelated_base = tmp_path / "other"
        unrelated_base.mkdir()
        restored = AudioSource.from_dict(data, base_path=unrelated_base)
        assert restored.file_path == actual

    def test_from_dict_rejects_path_traversal(self, tmp_path):
        data = {
            "id": "aud-1",
            "file_path": "../escape.mp3",
            "duration_seconds": 10.0,
        }
        with pytest.raises(ValueError, match="Path traversal"):
            AudioSource.from_dict(data, base_path=tmp_path)

    def test_round_trip_with_transcript(self):
        segments = [
            TranscriptSegment(start_time=0.0, end_time=2.5, text="hello world", confidence=0.9),
            TranscriptSegment(start_time=2.5, end_time=5.0, text="how are you", confidence=0.85),
        ]
        original = AudioSource(
            id="aud-1",
            file_path=Path("/tmp/voice.wav"),
            duration_seconds=5.0,
            transcript=segments,
        )
        data = original.to_dict()
        restored = AudioSource.from_dict(data)

        assert restored.transcript is not None
        assert len(restored.transcript) == 2
        assert restored.transcript[0].text == "hello world"
        assert restored.transcript[0].start_time == 0.0
        assert restored.transcript[1].text == "how are you"
        assert restored.transcript[1].confidence == 0.85

    def test_transcript_none_round_trips_cleanly(self):
        original = AudioSource(file_path=Path("/tmp/song.mp3"), duration_seconds=10.0)
        data = original.to_dict()
        assert "transcript" not in data
        restored = AudioSource.from_dict(data)
        assert restored.transcript is None

    def test_from_dict_generates_id_when_missing(self):
        data = {"file_path": "/tmp/song.mp3", "duration_seconds": 10.0}
        restored = AudioSource.from_dict(data)
        assert restored.id  # non-empty UUID

    def test_from_dict_defaults_optional_fields(self):
        data = {"file_path": "/tmp/song.mp3"}
        restored = AudioSource.from_dict(data)
        assert restored.duration_seconds == 0.0
        assert restored.sample_rate == 0
        assert restored.channels == 0
        assert restored.transcript is None
