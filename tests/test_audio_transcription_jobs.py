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
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    provider = Mock(
        return_value=[TranscriptSegment(0.123456789, 1.23456789, "hello", -0.123456789)]
    )
    monkeypatch.setattr("core.transcription.transcribe_video", provider)
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda path: True)
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


def test_saved_audio_contains_verified_record(setup):
    assert run(setup)["success"]
    saved = json.loads(setup[0].read_text())["audio_sources"][0]
    assert saved["analysis_records"]["transcribe"]["identity"] is not None


def test_legacy_transcript_is_recomputed(setup):
    project = Project.load(setup[0])
    project.set_audio_transcript("audio", [TranscriptSegment(0, 1, "legacy")])
    project.save()
    project.close_writer()
    assert run(setup)["result"]["status"] == "succeeded"
    assert setup[1].call_count == 1


def test_failure_record_preserves_displayed_transcript(setup):
    assert run(setup)["success"]
    setup[1].side_effect = RuntimeError("provider unavailable")
    assert not run(setup, force=True)["success"]
    saved = json.loads(setup[0].read_text())["audio_sources"][0]
    assert saved["transcript"][0]["text"] == "hello"
    assert saved["analysis_records"]["transcribe"]["state"] == "failed"


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


@pytest.mark.parametrize("change", ["model", "media", "metadata", "record"])
def test_verified_audio_recomputes_for_changed_semantics(setup, change):
    path, provider, store = setup
    assert run(setup)["success"]
    options = TranscriptionOptions(backend="faster-whisper")
    if change == "model":
        from dataclasses import replace

        options = replace(options, model="medium.en")
    else:
        project = Project.load(path)
        audio = project.audio_sources[0]
        if change == "media":
            audio.file_path.write_bytes(b"different audio")
        elif change == "metadata":
            audio.duration_seconds = 20
        else:
            audio.analysis_records.clear()
        project.save()
        project.close_writer()
    result = run_audio_transcription_job(
        store, path, "audio", options, lambda *_: None, Event()
    )
    assert result["result"]["status"] == "succeeded"
    assert provider.call_count == 2


def test_missing_old_receipt_does_not_invalidate_verified_audio(setup):
    import sqlite3

    path, provider, _ = setup
    assert run(setup)["success"]
    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute("DELETE FROM job_results")
    assert run(setup)["result"]["status"] == "skipped"
    assert provider.call_count == 1


def test_cancel_during_verified_reuse_does_not_report_success(setup, monkeypatch):
    path, provider, store = setup
    assert run(setup)["success"]
    before = path.read_bytes()
    cancel = Event()
    monkeypatch.setattr(
        "core.transcription._has_audio_stream", lambda path: cancel.set() or True
    )
    result = run_audio_transcription_job(
        store,
        path,
        "audio",
        TranscriptionOptions(backend="faster-whisper"),
        lambda *_: None,
        cancel,
    )
    assert result["success"] is False
    assert result["error"] == "cancelled"
    assert path.read_bytes() == before
    assert provider.call_count == 1


def test_failed_forced_refresh_save_reuses_computation(setup, monkeypatch):
    assert run(setup)["success"]
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup, force=True)
    assert setup[1].call_count == 2
    assert run(setup, force=True)["success"]
    assert setup[1].call_count == 2


def test_manual_edit_of_managed_audio_requires_force(setup):
    assert run(setup)["success"]
    project = Project.load(setup[0])
    project.set_audio_transcript("audio", [TranscriptSegment(0, 1, "manual")])
    project.save()
    project.close_writer()
    with pytest.raises(RuntimeError, match="use force"):
        run(setup)
    assert setup[1].call_count == 1
    assert run(setup, force=True)["success"]
    assert setup[1].call_count == 2


def test_record_change_after_failed_checkpoint_cannot_acknowledge_receipt(
    setup, monkeypatch
):
    path, provider, store = setup
    with monkeypatch.context() as patcher:
        patcher.setattr(
            store, "checkpoint_results", Mock(side_effect=OSError("checkpoint failed"))
        )
        with pytest.raises(OSError):
            run(setup, force=True)
    project = Project.load(path)
    project.audio_sources[0].analysis_records.clear()
    project.save()
    project.close_writer()
    assert run(setup, force=True)["result"]["status"] == "succeeded"
    assert provider.call_count == 2


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
            lambda *args: {"runtime": 2},
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
    provider.return_value[0].words = [
        WordTimestamp(0.123456789, 1.23456789, "hello", 0.9)
    ]
    provider.return_value[0].language = "en"
    assert run(setup)["success"]
    project = Project.load(path)
    try:
        transcript = get_audio_source(project, "audio")["audio_source"]["transcript"]
        assert transcript == [s.to_dict() for s in provider.return_value]
    finally:
        project.close_writer()
