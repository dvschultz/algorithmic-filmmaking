"""Headless still-image import shares computation and durable batch commits."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from core.jobs.image_import import image_import_job_spec, run_image_import_job
from core.jobs.store import JobStore
from core.project import Project
from tests.test_image_import_operations import make_image


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = [
        make_image(tmp_path / "one" / "same.png"),
        make_image(tmp_path / "two" / "same.png", (17, 23)),
    ]
    project = Project.new()
    path = tmp_path / "project.sceneripper"
    project.save(path)
    project.close_writer()
    monkeypatch.setattr(
        "core.jobs.image_import.image_import_runtime", lambda: {"runtime": 1}
    )
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    from core.thumbnail import generate_image_thumbnail

    provider = Mock(wraps=generate_image_thumbnail)
    monkeypatch.setattr("core.thumbnail.generate_image_thumbnail", provider)
    store = JobStore(tmp_path / "jobs.db")
    yield path, paths, provider, store
    store.close()


def run(setup, **kwargs):
    path, paths, _, store = setup
    return run_image_import_job(
        store, path, [str(p) for p in paths], lambda *_: None, Event(), **kwargs
    )


@pytest.mark.parametrize("copy_files", [True, False])
def test_saved_images_and_deliberate_repeat(setup, copy_files):
    path, paths, provider, store = setup
    first = run(setup, copy_files=copy_files)
    assert first["success"] and first["result"]["imported_count"] == 2
    saved = Project.load(path)
    assert [f.id for f in saved.frames] == first["result"]["frame_ids"]
    assert [(f.width, f.height) for f in saved.frames] == [(400, 300), (17, 23)]
    assert [f.file_path == p for f, p in zip(saved.frames, paths)] == [
        not copy_files
    ] * 2
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    saved.close_writer()
    second = run(setup, copy_files=copy_files)
    assert second["success"]
    assert not set(first["result"]["frame_ids"]) & set(second["result"]["frame_ids"])
    assert len(json.loads(path.read_text())["frames"]) == 4 and provider.call_count == 4


def test_failed_save_reuses_original_files_and_ids(setup, monkeypatch):
    path, _, provider, _ = setup
    before = path.read_bytes()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup)
    assert path.read_bytes() == before
    files = set((path.parent / "frames").rglob("*.png"))
    assert len(files) == 2
    assert run(setup)["success"]
    assert provider.call_count == 2
    assert set((path.parent / "frames").rglob("*.png")) == files


def test_failed_checkpoint_reconciles_before_new_batch(setup, monkeypatch):
    path, _, provider, store = setup
    with monkeypatch.context() as patcher:
        patcher.setattr(
            store, "checkpoint_results", Mock(side_effect=OSError("checkpoint failed"))
        )
        with pytest.raises(OSError):
            run(setup)
    before = json.loads(path.read_text())["frames"]
    assert run(setup)["result"]["status"] == "recovered"
    assert json.loads(path.read_text())["frames"] == before and provider.call_count == 2
    assert all(
        store.get_result(rid)["committed"]
        for rid in json.loads(path.read_text())["job_results"]
    )


def test_partial_errors_do_not_lose_valid_images(setup):
    path, paths, provider, store = setup
    bad = path.parent / "bad.png"
    bad.write_bytes(b"invalid")
    result = run_image_import_job(
        store,
        path,
        [str(paths[0]), str(bad), str(path.parent)],
        lambda *_: None,
        Event(),
    )
    assert result["success"] and result["result"]["imported_count"] == 1
    assert len(result["result"]["errors"]) == 2
    assert len(json.loads(path.read_text())["frames"]) == 1
    provider.assert_called_once()


def test_all_invalid_inputs_leave_project_unchanged(setup):
    path, _, provider, store = setup
    before = path.read_bytes()
    bad = path.parent / "bad.png"
    bad.write_bytes(b"invalid")
    result = run_image_import_job(
        store, path, [str(bad), "missing.png"], lambda *_: None, Event()
    )
    assert result["success"] is False
    assert path.read_bytes() == before
    assert not list((path.parent / "frames").rglob("*.png"))
    provider.assert_not_called()


@pytest.mark.parametrize("change", ["media", "runtime", "project", "order", "policy"])
def test_queued_inputs_cannot_drift(setup, monkeypatch, change):
    path, paths, provider, _ = setup
    project = Project.load(path)
    operation = image_import_job_spec(project, [str(p) for p in paths])
    if change == "media":
        paths[1].write_bytes(b"changed")
    if change == "runtime":
        monkeypatch.setattr(
            "core.jobs.image_import.image_import_runtime", lambda: {"runtime": 2}
        )
    if change == "project":
        project.name = "Edited"
        project.save()
    if change == "order":
        paths.reverse()
    project.close_writer()
    with pytest.raises(RuntimeError, match="changed while queued"):
        run(setup, operation=operation, copy_files=change != "policy")
    provider.assert_not_called()


def test_cancel_during_computation_does_not_save(setup):
    path, paths, provider, store = setup
    before = path.read_bytes()
    cancel = Event()
    result = run_image_import_job(
        store, path, [str(p) for p in paths], lambda *_: cancel.set(), cancel
    )
    assert not result["success"]
    assert path.read_bytes() == before
    assert not list((path.parent / "frames").rglob("*.png"))


def test_changed_copied_artifact_cannot_replay(setup, monkeypatch):
    path, _, provider, _ = setup
    before = path.read_bytes()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(setup)
    next((path.parent / "frames").rglob("*.png")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before publication"):
        run(setup)
    assert path.read_bytes() == before and provider.call_count == 2


@pytest.mark.parametrize("reference", [False, True])
def test_cli_imports_relative_paths_and_saves(setup, reference):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    register_commands()
    args = ["--json", "import-images", str(setup[0]), "one/same.png", "two/same.png"]
    if reference:
        args.append("--reference")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result"]["imported_count"] == 2
    project = Project.load(setup[0])
    assert (project.frames[0].file_path == setup[1][0]) == reference
    project.close_writer()
