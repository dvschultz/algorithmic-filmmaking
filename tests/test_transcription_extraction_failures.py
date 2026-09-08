"""Extraction failures are not successful empty transcripts on any backend."""

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.transcription import transcribe_clip, transcribe_video
from core.transcription_models import TranscriptionError


@pytest.mark.parametrize("route", ["clip", "mlx-video", "groq-video"])
@pytest.mark.parametrize("failure", ["ffmpeg", "empty"])
def test_extraction_failure_raises_and_removes_temporary_audio(
    tmp_path, monkeypatch, route, failure
):
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription._resolve_backend", lambda backend: backend)
    monkeypatch.setattr("core.transcription._require_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr("core.settings.get_groq_api_key", lambda: "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    inference = Mock(
        side_effect=AssertionError("Must not transcribe failed extraction")
    )
    monkeypatch.setattr("core.transcription.get_model", inference)
    monkeypatch.setattr("core.transcription.get_mlx_model", inference)
    monkeypatch.setattr("litellm.transcription", inference)
    extracted = []

    def extract(command, **kwargs):
        extracted.append(Path(command[-1]))
        if failure == "ffmpeg":
            raise subprocess.CalledProcessError(1, command, stderr=b"decode failed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("core.transcription.subprocess.run", extract)
    with pytest.raises(TranscriptionError, match="audio"):
        if route == "clip":
            transcribe_clip(tmp_path / "video.mp4", 0.0, 2.0, backend="faster-whisper")
        else:
            transcribe_video(
                tmp_path / "video.mp4",
                backend="mlx-whisper" if route == "mlx-video" else "groq",
            )
    assert len(extracted) == 1
    assert not extracted[0].exists()
    inference.assert_not_called()


def test_confirmed_no_audio_stays_a_valid_empty_result(monkeypatch, tmp_path):
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: False)
    extraction = Mock(side_effect=AssertionError("No extraction for video-only media"))
    monkeypatch.setattr("core.transcription.subprocess.run", extraction)
    assert transcribe_clip(tmp_path / "video.mp4", 0, 2) == []
    assert transcribe_video(tmp_path / "video.mp4") == []
    extraction.assert_not_called()


def test_failed_extraction_does_not_create_saved_transcript_or_receipt(
    tmp_path, monkeypatch
):
    from threading import Event
    from core.jobs.store import JobStore
    from core.jobs.transcription import run_transcription_job
    from core.operations.transcription import TranscriptionOptions
    from core.project import Project
    from tests.test_description_operations import project_with_thumbnails

    project = project_with_thumbnails(tmp_path, 1)
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr(
        "core.transcription._resolve_backend", lambda _: "faster-whisper"
    )
    monkeypatch.setattr("core.transcription._require_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(
        "core.transcription.subprocess.run",
        Mock(
            side_effect=subprocess.CalledProcessError(
                1, "ffmpeg", stderr=b"decode failed"
            )
        ),
    )
    result = run_transcription_job(
        JobStore(tmp_path / "jobs.db"),
        project.path,
        None,
        TranscriptionOptions(backend="faster-whisper"),
        lambda *_: None,
        Event(),
    )
    assert len(result["result"]["failed"]) == 1
    saved = Project.load(project.path)
    assert saved.clips[0].transcript is None
    assert not saved.metadata.job_results


def test_groq_preflight_failure_removes_temporary_audio(tmp_path, monkeypatch):
    import tempfile
    from core.transcription_models import FFmpegNotFoundError

    original = tempfile.NamedTemporaryFile
    created = []

    def temporary(**kwargs):
        result = original(dir=tmp_path, **kwargs)
        created.append(Path(result.name))
        return result

    monkeypatch.setattr("core.transcription.tempfile.NamedTemporaryFile", temporary)
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription._resolve_backend", lambda _: "groq")
    monkeypatch.setattr("core.settings.get_groq_api_key", lambda: "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        "core.transcription._require_ffmpeg", Mock(side_effect=FFmpegNotFoundError())
    )
    with pytest.raises(FFmpegNotFoundError):
        transcribe_video(tmp_path / "video.mp4", backend="groq")
    assert len(created) == 1
    assert not created[0].exists()
