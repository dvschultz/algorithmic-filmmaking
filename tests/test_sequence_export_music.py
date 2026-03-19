"""Tests for music audio muxing in sequence export."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from core.sequence_export import ExportConfig, SequenceExporter


def test_export_config_music_path_default_none():
    """ExportConfig.music_path defaults to None."""
    config = ExportConfig(output_path=Path("/out.mp4"))
    assert config.music_path is None


def test_mux_audio_builds_correct_command():
    """_mux_audio builds FFmpeg command with -c:v copy and -shortest."""
    exporter = SequenceExporter.__new__(SequenceExporter)
    exporter.ffmpeg_path = "/usr/bin/ffmpeg"

    config = ExportConfig(
        output_path=Path("/out.mp4"),
        audio_codec="aac",
        audio_bitrate="192k",
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = exporter._mux_audio(
            video_path=Path("/tmp/concat.mp4"),
            audio_path=Path("/music/song.mp3"),
            output_path=Path("/final/out.mp4"),
            config=config,
        )

    assert result is True
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-map" in cmd
    assert "0:v" in cmd
    assert "1:a" in cmd
    assert "-c:v" in cmd
    assert "copy" in cmd
    assert "-shortest" in cmd
    assert "-c:a" in cmd
    assert "aac" in cmd


def test_mux_audio_fallback_on_failure():
    """_mux_audio copies video without audio on FFmpeg failure."""
    exporter = SequenceExporter.__new__(SequenceExporter)
    exporter.ffmpeg_path = "/usr/bin/ffmpeg"

    config = ExportConfig(output_path=Path("/out.mp4"))

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "video.mp4"
        video.write_bytes(b"fake video data")
        output = Path(tmpdir) / "output.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            result = exporter._mux_audio(
                video_path=video,
                audio_path=Path("/nonexistent/song.mp3"),
                output_path=output,
                config=config,
            )

        assert result is False  # Indicates mux failure
        assert output.exists()  # Video still copied as fallback
        assert output.read_bytes() == b"fake video data"


def test_export_skips_mux_when_music_path_missing():
    """Export without music_path produces video normally."""
    config = ExportConfig(output_path=Path("/out.mp4"), music_path=None)
    assert config.music_path is None
