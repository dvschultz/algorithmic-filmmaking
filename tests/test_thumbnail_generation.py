"""Tests for thumbnail generation command construction."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from core.thumbnail import ThumbnailGenerator


def test_thumbnail_seek_time_is_decimal_for_fractional_timestamps(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr("core.thumbnail.find_binary", lambda _name: "/usr/bin/ffmpeg")

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("core.thumbnail.subprocess.run", fake_run)

    generator = ThumbnailGenerator(cache_dir=tmp_path)
    generator.generate_thumbnail(
        video_path=Path("/tmp/video.mp4"),
        timestamp_seconds=Fraction(1514513, 9000),
        output_path=tmp_path / "thumb.jpg",
    )

    seek_arg = commands[0][commands[0].index("-ss") + 1]
    assert seek_arg == "168.279222"
    assert "/" not in seek_arg


def test_clip_thumbnail_normalizes_fractional_clip_times(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr("core.thumbnail.find_binary", lambda _name: "/usr/bin/ffmpeg")

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("core.thumbnail.subprocess.run", fake_run)

    generator = ThumbnailGenerator(cache_dir=tmp_path)
    generator.generate_clip_thumbnail(
        video_path=Path("/tmp/video.mp4"),
        start_seconds=Fraction(100, 3),
        end_seconds=Fraction(130, 3),
        output_path=tmp_path / "thumb.jpg",
    )

    seek_arg = commands[0][commands[0].index("-ss") + 1]
    assert seek_arg == "36.666667"
    assert "/" not in seek_arg
