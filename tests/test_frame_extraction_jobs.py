"""Headless extraction publishes complete batches and reuses interrupted work."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.frame_extraction import (
    frame_extraction_job_spec,
    run_frame_extraction_job,
)
from core.jobs.store import JobStore
from core.project import Project
from models.clip import Clip, Source
from tests.test_frame_extraction_operations import extract


@pytest.fixture
def setup(tmp_path, monkeypatch):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    project = Project.new()
    project.add_source(Source(id="source", file_path=media, width=20, height=12))
    project.add_clips(
        [Clip(id="clip", source_id="source", start_frame=0, end_frame=15)]
    )
    path = tmp_path / "project.sceneripper"
    project.save(path)
    project.close_writer()
    monkeypatch.setattr(
        "core.jobs.frame_extraction.frame_extraction_runtime", lambda: {"runtime": 1}
    )
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    provider = Mock(side_effect=extract)
    monkeypatch.setattr("core.ffmpeg.extract_frames_batch", provider)
    store = JobStore(tmp_path / "jobs.db")
    yield path, provider, store
    store.close()


def run(setup, **kwargs):
    path, _, store = setup
    return run_frame_extraction_job(
        store,
        path,
        "source",
        lambda *_: None,
        Event(),
        interval=5,
        clip_id="clip",
        **kwargs,
    )


@pytest.mark.parametrize("empty", [False, True])
def test_saved_extraction_and_deliberate_repeat(setup, empty):
    path, provider, store = setup
    if empty:
        provider.side_effect = lambda *a, **kw: []
    first = run(setup)
    assert first["success"]
    assert first["result"]["frame_count"] == (0 if empty else 3)
    snapshot = json.loads(path.read_text())
    assert [frame["id"] for frame in snapshot.get("frames", [])] == first["result"][
        "frame_ids"
    ]
    assert all(store.get_result(rid)["committed"] for rid in snapshot["job_results"])
    second = run(setup)
    assert second["success"]
    assert provider.call_count == 2
    assert not set(first["result"]["frame_ids"]) & set(second["result"]["frame_ids"])
    assert len(json.loads(path.read_text()).get("frames", [])) == (0 if empty else 6)


def test_failed_save_reuses_original_artifacts(setup, monkeypatch):
    path, provider, store = setup
    before = path.read_bytes()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup)
    assert path.read_bytes() == before
    files = set(path.parent.glob("frames/*/frames/*.png"))
    assert len(files) == 3
    assert run(setup)["success"]
    assert provider.call_count == 1
    assert set(path.parent.glob("frames/*/frames/*.png")) == files


def test_failed_checkpoint_reconciles_before_new_batch(setup, monkeypatch):
    path, provider, store = setup
    with monkeypatch.context() as patcher:
        patcher.setattr(
            store, "checkpoint_results", Mock(side_effect=OSError("checkpoint failed"))
        )
        with pytest.raises(OSError):
            run(setup)
    before = json.loads(path.read_text())["frames"]
    assert run(setup)["result"]["status"] == "recovered"
    assert provider.call_count == 1
    assert json.loads(path.read_text())["frames"] == before
    assert run(setup)["success"]
    assert provider.call_count == 2


def test_modified_artifact_cannot_replay_after_failed_save(setup, monkeypatch):
    path, provider, _ = setup
    before = path.read_bytes()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup)
    next(path.parent.glob("frames/*/frames/*.png")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        run(setup)
    assert path.read_bytes() == before
    assert provider.call_count == 1


def test_provider_failure_does_not_publish_empty_success(setup):
    path, provider, _ = setup
    before = path.read_bytes()
    provider.side_effect = RuntimeError("decoder failed")
    result = run(setup)
    assert result["success"] is False
    assert "decoder failed" in result["error"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["media", "runtime", "project", "clip"])
def test_queued_inputs_cannot_drift(setup, monkeypatch, change):
    path, provider, _ = setup
    project = Project.load(path)
    operation = frame_extraction_job_spec(project, "source", interval=5, clip_id="clip")
    if change == "media":
        project.sources[0].file_path.write_bytes(b"changed video")
    if change == "runtime":
        monkeypatch.setattr(
            "core.jobs.frame_extraction.frame_extraction_runtime",
            lambda: {"runtime": 2},
        )
    if change == "project":
        project.name = "Edited"
        project.save()
    if change == "clip":
        project.clips[0].start_frame = 1
        project.save()
    project.close_writer()
    with pytest.raises(RuntimeError, match="changed while queued"):
        run(setup, operation=operation)
    provider.assert_not_called()


def test_cancel_during_computation_does_not_save(setup):
    path, provider, store = setup
    before = path.read_bytes()
    cancel = Event()

    def cancelled(*args, **kwargs):
        paths = extract(*args, **kwargs)
        cancel.set()
        return paths

    provider.side_effect = cancelled
    result = run_frame_extraction_job(
        store, path, "source", lambda *_: None, cancel, interval=5
    )
    assert result["success"] is False
    assert path.read_bytes() == before
    assert not list(path.parent.glob("frames/*/frames/*.png"))


def test_cli_extracts_and_saves(setup):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    register_commands()
    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "extract-frames",
            str(setup[0]),
            "source",
            "--interval",
            "5",
            "--clip-id",
            "clip",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"]["frame_count"] == 3
    assert len(json.loads(setup[0].read_text())["frames"]) == 3
