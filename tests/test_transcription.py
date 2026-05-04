"""Regression tests for transcription edge cases."""

from pathlib import Path

import pytest

from core.transcription import FFmpegNotFoundError, transcribe_clip, transcribe_video


def test_transcribe_clip_skips_video_without_audio(monkeypatch):
    """Video-only clips should return no transcript without invoking FFmpeg."""
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _path: False)
    monkeypatch.setattr(
        "core.transcription.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ffmpeg should not run")),
    )

    result = transcribe_clip(
        Path("/tmp/video_only.mp4"),
        start_time=0.0,
        end_time=5.0,
        backend="mlx-whisper",
    )

    assert result == []


def test_transcribe_video_skips_video_without_audio(monkeypatch):
    """Whole-video transcription should short-circuit when no audio stream exists."""
    progress = []

    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _path: False)
    monkeypatch.setattr(
        "core.transcription._resolve_backend",
        lambda _backend: (_ for _ in ()).throw(AssertionError("backend resolution should not run")),
    )

    result = transcribe_video(
        Path("/tmp/video_only.mp4"),
        backend="auto",
        progress_callback=lambda pct, message: progress.append((pct, message)),
    )

    assert result == []
    assert progress == [(1.0, "No audio track found")]


def test_transcribe_clip_missing_ffmpeg_raises_clear_error(monkeypatch):
    """Missing FFmpeg should fail before subprocess tries the literal command."""
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _path: None)
    monkeypatch.setattr("core.transcription._resolve_backend", lambda _backend: "faster-whisper")
    monkeypatch.setattr("core.transcription.find_binary", lambda _name: None)
    monkeypatch.setattr(
        "core.transcription.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess should not run without ffmpeg")
        ),
    )

    with pytest.raises(FFmpegNotFoundError, match="FFmpeg is required for transcription"):
        transcribe_clip(
            Path("/tmp/video.mp4"),
            start_time=0.0,
            end_time=5.0,
            backend="faster-whisper",
        )
