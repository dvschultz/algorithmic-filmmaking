"""Durable transcript receipts retain verification through saves and recovery."""

from dataclasses import replace
import json
import os
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.jobs.transcription import run_transcription_job
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from tests.test_spine_analyze import _build_project

OPTIONS = TranscriptionOptions(backend="faster-whisper")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Register before patching settings: the legacy CLI config imports its loader
    # by value, so importing it under a patch would leak that patch to later tests.
    from cli.main import register_commands
    register_commands()
    project = _build_project(tmp_path, 2)
    project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    provider = Mock(return_value=[])
    monkeypatch.setattr("core.transcription.transcribe_clip", provider)
    store = JobStore(tmp_path / "jobs.db")
    yield project.path, store, provider
    store.close()


def run(setup, options=OPTIONS, **kwargs):
    path, store, _ = setup
    return run_transcription_job(
        store, path, None, options, lambda *_: None, Event(), **kwargs
    )["result"]


def test_records_survive_save_and_parallelism_does_not_repeat_inference(setup):
    path, _, provider = setup
    assert len(run(setup)["succeeded"]) == 2
    project = Project.load(path)
    assert all(
        clip.analysis_records["transcribe"].provenance == "verified"
        for clip in project.clips
    )
    assert len(run(setup, replace(OPTIONS, parallelism=4))["skipped"]) == 2
    assert provider.call_count == 2


def test_valid_records_reuse_without_old_receipt_rows(setup, monkeypatch):
    run(setup)
    path, store, provider = setup
    monkeypatch.setattr(store, "get_result", lambda _: None)
    assert len(run(setup)["skipped"]) == 2
    assert provider.call_count == 2
    assert (
        Project.load(path).clips[0].analysis_records["transcribe"].state == "succeeded"
    )


def test_legacy_presence_is_not_verified_even_with_skip_existing(setup):
    path, _, provider = setup
    project = Project.load(path)
    project.clips[0].transcript = []
    project.save()
    assert len(run(setup, skip_existing=True)["succeeded"]) == 2
    assert provider.call_count == 2


def test_failed_refresh_saves_failure_record_and_preserves_transcript(setup):
    run(setup)
    path, _, provider = setup
    provider.side_effect = RuntimeError("offline")
    assert len(run(setup, replace(OPTIONS, model="base"))["failed"]) == 2
    project = Project.load(path)
    assert all(
        clip.transcript == [] and clip.analysis_records["transcribe"].state == "failed"
        for clip in project.clips
    )
    assert len(project.metadata.job_results) == 2


@pytest.mark.parametrize("failure", ["save", "checkpoint"])
def test_forced_batch_recovers_exact_record_without_repeating_inference(
    setup, monkeypatch, failure
):
    path, store, provider = setup
    run(setup)
    with monkeypatch.context() as patcher:
        if failure == "save":
            patcher.setattr(
                "core.jobs.commits.save_with_mtime_check",
                Mock(side_effect=OSError("interrupted")),
            )
        else:
            patcher.setattr(
                store, "checkpoint_results", Mock(side_effect=OSError("interrupted"))
            )
        with pytest.raises(OSError, match="interrupted"):
            run(setup, force=True)
    assert provider.call_count == 4
    recovered = run(setup, force=True)
    assert len(recovered["succeeded"] + recovered["skipped"]) == 2
    assert provider.call_count == 4
    assert all(
        clip.analysis_records["transcribe"].state == "succeeded"
        for clip in Project.load(path).clips
    )
    run(setup, force=True)
    assert provider.call_count == 6


def test_no_audio_record_is_reusable(setup, monkeypatch):
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: False)
    run(setup)
    assert len(run(setup)["skipped"]) == 2
    assert setup[2].call_count == 2


def test_new_stamp_refreshes_binding_without_inference(setup):
    path, _, provider = setup
    run(setup)
    project = Project.load(path)
    old = project.clips[0].analysis_records["transcribe"]
    media = project.sources[0].file_path
    stamp = media.stat()
    os.utime(media, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    assert len(run(setup)["skipped"]) == 2
    current = Project.load(path).clips[0].analysis_records["transcribe"]
    assert current.identity == old.identity and current.input_json != old.input_json
    assert provider.call_count == 2


def test_corrupt_present_receipt_is_not_treated_as_missing(setup, monkeypatch):
    from core.jobs.commits import StaleJobResult

    run(setup)
    _, store, _ = setup
    original = store.get_result

    def corrupt(result_id):
        row = original(result_id)
        if row is not None:
            row = dict(row)
            row["payload_json"] = json.dumps({"segments": []})
        return row

    monkeypatch.setattr(store, "get_result", corrupt)
    with pytest.raises(StaleJobResult, match="corrupt"):
        run(setup)


def test_cli_checks_populated_transcripts_against_requested_model(setup):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    path, _, provider = setup
    register_commands()
    runner = CliRunner()
    for model, expected in (("small.en", 2), ("medium.en", 4), ("medium.en", 4)):
        result = runner.invoke(
            cli,
            ["--json", "transcribe", str(path), "--model", model, "--language", "en"],
        )
        assert result.exit_code == 0, result.output
        assert provider.call_count == expected


def test_durable_records_are_reusable_by_direct_headless_operation(setup, monkeypatch):
    from core.spine.analyze import transcribe

    path, _, provider = setup
    run(setup)
    monkeypatch.setattr(
        "core.transcription._resolve_backend", lambda _: "faster-whisper"
    )
    result = transcribe(Project.load(path), model="small.en", language="en")
    assert len(result["result"]["skipped"]) == 2
    assert provider.call_count == 2
