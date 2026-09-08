"""Provider provenance describes the execution that actually produced a transcript."""

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.transcription import transcribe_clip, transcribe_video
from core.transcription_models import ModelDownloadError


@pytest.fixture
def audio(monkeypatch):
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription._require_ffmpeg", lambda: "/fake/ffmpeg")

    def extract(command, **kwargs):
        Path(command[-1]).write_bytes(b"audio")

    monkeypatch.setattr("core.transcription.subprocess.run", extract)
    monkeypatch.setattr("core.transcription._chdir_for_mlx", contextlib.nullcontext)


@pytest.mark.parametrize("route", ["clip", "video"])
@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("backend", ["auto", "mlx-whisper"])
def test_reports_resolved_backend_and_actual_model(
    audio, monkeypatch, route, available, backend
):
    monkeypatch.setattr(
        "core.transcription.is_mlx_whisper_available", lambda: available
    )
    events = []

    def load(name):
        assert events  # Report before loading, including a load failure.
        assert name == "large-v3-turbo"
        raise ModelDownloadError("offline")

    monkeypatch.setattr("core.transcription.get_model", load)
    monkeypatch.setattr("core.transcription.get_mlx_model", load)
    with pytest.raises(ModelDownloadError):
        if route == "clip":
            transcribe_clip(
                Path("video.mp4"),
                0,
                1,
                model_name="large-v3-turbo",
                backend=backend,
                on_execution=events.append,
            )
        else:
            transcribe_video(
                Path("video.mp4"),
                model_name="large-v3-turbo",
                backend=backend,
                on_execution=events.append,
            )
    assert events == [
        {
            "backend": "mlx-whisper" if available else "faster-whisper",
            "model": "large-v3" if available else "large-v3-turbo",
            "input_mode": "audio",
        }
    ]


@pytest.mark.parametrize("route", ["clip", "video"])
def test_no_audio_reports_probe_without_resolving_or_loading_models(monkeypatch, route):
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: False)
    resolve = Mock(side_effect=AssertionError("No model needed for video-only media"))
    monkeypatch.setattr("core.transcription._resolve_backend", resolve)
    events = []
    if route == "clip":
        result = transcribe_clip(Path("video.mp4"), 0, 1, on_execution=events.append)
    else:
        result = transcribe_video(Path("video.mp4"), on_execution=events.append)
    assert result == []
    assert events == [
        {"backend": "audio-probe", "model": None, "input_mode": "no-audio"}
    ]
    resolve.assert_not_called()


@pytest.mark.parametrize("route", ["clip", "video"])
@pytest.mark.parametrize("explicit", [False, True])
def test_groq_model_is_pinned_before_extraction_and_matches_report(
    audio, monkeypatch, tmp_path, route, explicit
):
    selected = SimpleNamespace(transcription_cloud_model="whisper-large-v3")
    monkeypatch.setattr("core.settings.load_settings", lambda: selected)
    monkeypatch.setattr("core.settings.get_groq_api_key", lambda: "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    events = []
    requested = "whisper-large-v3-turbo" if explicit else "whisper-large-v3"

    def report(execution):
        events.append(execution)
        selected.transcription_cloud_model = "changed-during-extraction"

    def infer(**kwargs):
        assert kwargs["model"] == f"groq/{requested}"
        assert events == [
            {"backend": "groq", "model": requested, "input_mode": "audio"}
        ]
        return SimpleNamespace(segments=[], text="")

    inference = Mock(side_effect=infer)
    monkeypatch.setattr("litellm.transcription", inference)
    kwargs = {"backend": "groq", "on_execution": report}
    if explicit:
        kwargs["cloud_model"] = requested
    if route == "clip":
        result = transcribe_clip(tmp_path / "video.mp4", 0, 1, **kwargs)
    else:
        result = transcribe_video(tmp_path / "video.mp4", **kwargs)
    assert result == []
    inference.assert_called_once()


def test_execution_is_reported_even_when_audio_extraction_fails(audio, monkeypatch):
    from core.transcription_models import TranscriptionError

    events = []

    def fail(command, **kwargs):
        assert events == [
            {"backend": "faster-whisper", "model": "small.en", "input_mode": "audio"}
        ]
        raise TranscriptionError("extraction unavailable")

    monkeypatch.setattr("core.transcription.subprocess.run", fail)
    with pytest.raises(TranscriptionError, match="extraction unavailable"):
        transcribe_clip(
            Path("video.mp4"),
            0,
            1,
            backend="faster-whisper",
            on_execution=events.append,
        )
