"""Headless audio import saves once and recovers interrupted publication."""

import json
from threading import Event
from unittest.mock import Mock

import pytest

from core.jobs.audio_import import audio_import_job_spec, run_audio_import_job
from core.jobs.store import JobStore
from core.project import Project
from core.settings import Settings


@pytest.fixture
def setup(tmp_path, monkeypatch):
    media = tmp_path / "voice.wav"
    media.write_bytes(b"audio")
    project = Project.new()
    path = tmp_path / "project.sceneripper"
    project.save(path)
    project.close_writer()
    monkeypatch.setattr(
        "core.jobs.audio_import.audio_import_runtime", lambda: {"runtime": 1}
    )
    settings = Settings(cache_dir=tmp_path)
    # Import the compatibility module before patching its source module so its
    # copied function binding cannot retain another test's temporary loader.
    monkeypatch.setattr("cli.utils.config.load_settings", lambda: settings)
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    processor = Mock(ffprobe_available=True)
    processor.get_audio_info.return_value = dict(
        duration=1.23456789, sample_rate=48000, channels=2
    )
    monkeypatch.setattr("core.ffmpeg.FFmpegProcessor", lambda: processor)
    store = JobStore(tmp_path / "jobs.db")
    yield path, media, processor.get_audio_info, store
    store.close()


def run(setup, **kwargs):
    path, _, _, store = setup
    return run_audio_import_job(
        store, path, "voice.wav", lambda *_: None, Event(), **kwargs
    )


def test_save_and_repeat_preserve_audio_id(setup):
    path, media, probe, store = setup
    first = run(setup)
    assert first["success"] and first["result"]["status"] == "succeeded"
    saved = json.loads(path.read_text())
    assert saved["audio_sources"][0]["id"] == first["result"]["audio_source_id"]
    assert saved["audio_sources"][0]["duration_seconds"] == 1.23456789
    assert all(store.get_result(rid)["committed"] for rid in saved["job_results"])
    alias = media.with_name("alias.wav")
    alias.symlink_to(media)
    second = run_audio_import_job(store, path, str(alias), lambda *_: None, Event())
    assert second["result"]["audio_source_id"] == first["result"]["audio_source_id"]
    assert second["result"]["status"] == "skipped"
    probe.assert_called_once()
    assert len(json.loads(path.read_text())["audio_sources"]) == 1
    media.unlink()
    assert run(setup)["result"]["audio_source_id"] == first["result"]["audio_source_id"]


def test_failed_save_reuses_original_probe_and_id(setup, monkeypatch):
    path, _, probe, store = setup
    before = path.read_bytes()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup)
    assert path.read_bytes() == before
    result = run(setup)
    assert result["success"]
    probe.assert_called_once()
    saved = json.loads(path.read_text())
    assert len(saved["audio_sources"]) == 1
    receipt = store.get_result(next(iter(saved["job_results"])))
    assert (
        json.loads(receipt["payload_json"])["outcome"]["audio_source_id"]
        == result["result"]["audio_source_id"]
    )


def test_failed_checkpoint_is_reconciled_without_duplicate(setup, monkeypatch):
    path, _, probe, store = setup
    with monkeypatch.context() as patcher:
        patcher.setattr(
            store, "checkpoint_results", Mock(side_effect=OSError("checkpoint failed"))
        )
        with pytest.raises(OSError):
            run(setup)
    before = json.loads(path.read_text())["audio_sources"]
    assert run(setup)["result"]["status"] == "recovered"
    assert json.loads(path.read_text())["audio_sources"] == before
    assert run(setup)["result"]["status"] == "skipped"
    probe.assert_called_once()
    assert all(
        store.get_result(rid)["committed"]
        for rid in json.loads(path.read_text())["job_results"]
    )


@pytest.mark.parametrize("change", ["media", "runtime", "project"])
def test_queued_inputs_are_frozen(setup, monkeypatch, change):
    path, media, probe, _ = setup
    project = Project.load(path)
    operation = audio_import_job_spec(project, "voice.wav")
    if change == "media":
        media.write_bytes(b"changed audio")
    if change == "runtime":
        monkeypatch.setattr(
            "core.jobs.audio_import.audio_import_runtime", lambda: {"runtime": 2}
        )
    if change == "project":
        project.name = "Edited"
        project.save()
    project.close_writer()
    with pytest.raises(RuntimeError, match="changed while queued"):
        run(setup, operation=operation)
    probe.assert_not_called()


def test_cancellation_during_probe_does_not_save(setup):
    path, _, probe, store = setup
    before = path.read_bytes()
    cancel = Event()

    def cancelled(*_):
        cancel.set()
        return dict(duration=10, sample_rate=48000, channels=2)

    probe.side_effect = cancelled
    assert not run_audio_import_job(store, path, "voice.wav", lambda *_: None, cancel)[
        "success"
    ]
    assert path.read_bytes() == before


def test_failed_probe_does_not_publish(setup):
    path, _, probe, _ = setup
    before = path.read_bytes()
    probe.side_effect = RuntimeError("probe failed")
    result = run(setup)
    assert not result["success"] and "probe failed" in result["error"]
    assert path.read_bytes() == before


def test_removed_audio_reimports_as_new_generation(setup):
    path, _, probe, _ = setup
    first = run(setup)
    project = Project.load(path)
    project.remove_audio_source(first["result"]["audio_source_id"])
    project.save()
    project.close_writer()
    second = run(setup)
    assert second["result"]["audio_source_id"] != first["result"]["audio_source_id"]
    assert probe.call_count == 2


def test_cli_imports_and_saves_project_relative_audio(setup):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    register_commands()
    result = CliRunner().invoke(
        cli, ["--json", "import-audio", str(setup[0]), "voice.wav"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"]["filename"] == "voice.wav"
    assert len(json.loads(setup[0].read_text())["audio_sources"]) == 1
