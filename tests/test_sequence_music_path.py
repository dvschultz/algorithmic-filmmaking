"""Tests for Sequence.music_path persistence."""

from pathlib import Path
import tempfile

from models.sequence import Sequence, Track


def test_music_path_default_none():
    """New sequences have no music_path by default."""
    seq = Sequence()
    assert seq.music_path is None


def test_music_path_to_dict_omitted_when_none():
    """music_path is not included in to_dict when None."""
    seq = Sequence()
    data = seq.to_dict()
    assert "music_path" not in data


def test_music_path_to_dict_included_when_set():
    """music_path is included in to_dict when set."""
    seq = Sequence(music_path="/test/music.mp3")
    data = seq.to_dict()
    assert data["music_path"] == "/test/music.mp3"


def test_music_path_roundtrip():
    """music_path survives to_dict/from_dict roundtrip."""
    seq = Sequence(music_path="/test/music.mp3", algorithm="staccato")
    data = seq.to_dict()
    restored = Sequence.from_dict(data)
    assert restored.music_path == "/test/music.mp3"
    assert restored.algorithm == "staccato"


def test_music_path_relative_to_base():
    """music_path is stored relative to base_path in to_dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        music = base / "audio" / "song.mp3"
        music.parent.mkdir(parents=True, exist_ok=True)
        music.touch()

        seq = Sequence(music_path=str(music))
        data = seq.to_dict(base_path=base)
        assert data["music_path"] == "audio/song.mp3"


def test_music_path_resolved_from_base():
    """music_path relative path is resolved from base_path in from_dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        music = base / "audio" / "song.mp3"
        music.parent.mkdir(parents=True, exist_ok=True)
        music.touch()

        data = {"music_path": "audio/song.mp3"}
        seq = Sequence.from_dict(data, base_path=base)
        assert seq.music_path == str(music.resolve())


def test_music_path_missing_file_cleared():
    """music_path is set to None if the file doesn't exist on load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        data = {"music_path": "audio/missing.mp3"}
        seq = Sequence.from_dict(data, base_path=base)
        assert seq.music_path is None


def test_backward_compatible_load():
    """Old project data without music_path loads without error."""
    data = {
        "name": "Old Sequence",
        "fps": 24.0,
        "algorithm": "duration",
        "tracks": [{"id": "t1", "name": "Video 1", "clips": []}],
    }
    seq = Sequence.from_dict(data)
    assert seq.music_path is None
    assert seq.algorithm == "duration"
