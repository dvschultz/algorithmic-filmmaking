"""Headless audio transcription shares computation and recovers interrupted saves."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.audio_transcription import (
    audio_transcription_job_spec,
    run_audio_transcription_job,
)
from core.jobs.store import JobStore
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from core.transcription_models import TranscriptSegment
from models.audio_source import AudioSource


@pytest.fixture
def setup(tmp_path, monkeypatch):
    media = tmp_path / "audio.wav"
    media.write_bytes(b"audio")
    project = Project.new()
    project.add_audio_source(AudioSource(id="audio", file_path=media))
    path = tmp_path / "project.json"
    project.save(path)
    project.close_writer()
    monkeypatch.setattr(
        "core.jobs.audio_transcription.audio_transcription_runtime",
        lambda: {"runtime": 1},
    )
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    provider = Mock(
        return_value=[TranscriptSegment(0.123456789, 1.23456789, "hello", -0.123456789)]
    )
    monkeypatch.setattr("core.transcription.transcribe_video", provider)
    store = JobStore(tmp_path / "jobs.db")
    yield path, provider, store
    store.close()


def run(setup, **kwargs):
    path, _, store = setup
    return run_audio_transcription_job(
        store,
        path,
        "audio",
        TranscriptionOptions(backend="faster-whisper"),
        lambda *_: None,
        Event(),
        **kwargs,
    )


@pytest.mark.parametrize("empty", [False, True])
def test_saved_audio_skip_and_force(setup, empty):
    path, provider, _ = setup
    if empty:
        provider.return_value = []
    assert run(setup)["result"]["status"] == "succeeded"
    assert run(setup)["result"]["status"] == "skipped"
    assert provider.call_count == 1
    assert json.loads(path.read_text())["audio_sources"][0]["transcript"] == [
        s.to_dict() for s in provider.return_value
    ]
    assert run(setup, force=True)["success"]
    assert provider.call_count == 2


def test_failed_save_reuses_computation(setup, monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup)
    assert run(setup)["success"]
    assert setup[1].call_count == 1


def test_failed_checkpoint_reconciles_before_new_force(setup, monkeypatch):
    path, provider, store = setup
    with monkeypatch.context() as patcher:
        patcher.setattr(
            store, "checkpoint_results", Mock(side_effect=OSError("checkpoint failed"))
        )
        with pytest.raises(OSError):
            run(setup, force=True)
    assert run(setup, force=True)["result"]["status"] == "recovered"
    assert provider.call_count == 1
    assert run(setup, force=True)["success"]
    assert provider.call_count == 2


@pytest.mark.parametrize("change", ["media", "runtime", "project"])
def test_queued_inputs_cannot_drift(setup, monkeypatch, change):
    path, provider, store = setup
    project = Project.load(path)
    operation = audio_transcription_job_spec(
        project, "audio", TranscriptionOptions(backend="faster-whisper")
    )
    if change == "media":
        project.audio_sources[0].file_path.write_bytes(b"new media")
    elif change == "runtime":
        monkeypatch.setattr(
            "core.jobs.audio_transcription.audio_transcription_runtime",
            lambda: {"runtime": 2},
        )
    else:
        project.set_audio_transcript("audio", [])
        project.save()
    project.close_writer()
    with pytest.raises(RuntimeError, match="changed while queued"):
        run(setup, operation=operation)
    provider.assert_not_called()


def test_cancel_during_compute_does_not_save(setup):
    path, provider, store = setup
    before = path.read_bytes()
    cancel = Event()
    provider.side_effect = lambda *a, **kw: cancel.set() or []
    result = run_audio_transcription_job(
        store,
        path,
        "audio",
        TranscriptionOptions(backend="faster-whisper"),
        lambda *_: None,
        cancel,
    )
    assert result["success"] is False
    assert path.read_bytes() == before


def test_cli_audio_command(setup):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    register_commands()
    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "transcribe-audio",
            str(setup[0]),
            "audio",
            "--backend",
            "faster-whisper",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["success"]
    assert setup[1].call_count == 1


def test_agent_audio_metadata_preserves_word_timestamps(setup):
    from core.spine.audio_sources import get_audio_source
    from core.transcription_models import WordTimestamp

    path, provider, _ = setup
    provider.return_value[0].words = [WordTimestamp(.123456789, 1.23456789, "hello", .9)]
    provider.return_value[0].language = "en"
    assert run(setup)["success"]
    project = Project.load(path)
    try:
        transcript = get_audio_source(project, "audio")["audio_source"]["transcript"]
        assert transcript == [s.to_dict() for s in provider.return_value]
    finally:
        project.close_writer()
