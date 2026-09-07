"""Saved boundary pairs survive failed project saves and checkpoint writes."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.store import JobStore
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 3)
    project.sources[0].file_path.write_bytes(b"video")
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(return_value=([0.123456789] * 768, [0.987654321] * 768))
    monkeypatch.setattr("core.analysis.embeddings.extract_boundary_embeddings", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    from core.jobs.boundary_embeddings import run_boundary_embedding_job

    path, store, _ = setup
    return run_boundary_embedding_job(
        store, path, None, lambda *_: None, Event(), **kwargs
    )["result"]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_pairs_after_store_reopen(setup, force):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup, force=force)
    assert Project.load(path).clips[0].first_frame_embedding is None
    assert compute.call_count == 3
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        compute.side_effect = AssertionError("must reuse computation")
        assert len(run((path, reopened, compute), force=force)["succeeded"]) == 3
    finally:
        reopened.close()
    clip = Project.load(path).clips[0]
    assert clip.first_frame_embedding == [0.123456789] * 768
    assert clip.last_frame_embedding == [0.987654321] * 768


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_reconciles_full_precision(setup, force):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert len(run(setup, force=force)["skipped"]) == 3
    assert compute.call_count == 3
    assert all(
        store.get_result(rid)["committed"]
        for rid in Project.load(path).metadata.job_results
    )


def test_invalid_pair_is_not_journaled(setup):
    path, _, compute = setup
    compute.return_value = ([0.1] * 768, [0.0] * 768)
    assert len(run(setup)["failed"]) == 3
    assert Project.load(path).metadata.job_results == {}


def test_cancel_keeps_accepted_prefix(setup):
    from core.jobs.boundary_embeddings import run_boundary_embedding_job

    path, store, compute = setup
    cancel = Event()
    result = run_boundary_embedding_job(
        store, path, None, lambda *_: cancel.set(), cancel
    )["result"]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 2
    assert len(Project.load(path).metadata.job_results) == 1
    assert len(run(setup)["succeeded"]) == 2
    assert compute.call_count == 3


def test_force_refresh_recovery_uses_new_generation(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = ([0.3] * 768, [0.4] * 768)
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    compute.side_effect = AssertionError("must reuse forced computation")
    assert len(run(setup, force=True)["succeeded"]) == 3
    assert Project.load(path).clips[0].last_frame_embedding == [0.4] * 768


@pytest.mark.parametrize("change", ["source", "runtime"])
def test_queued_job_rejects_changed_inputs_before_compute(setup, monkeypatch, change):
    from core.jobs.boundary_embeddings import boundary_embedding_job_spec
    from core.jobs.commits import StaleJobResult

    path, _, compute = setup
    project = Project.load(path)
    operation = boundary_embedding_job_spec(project, None, arguments={})
    if change == "source":
        project.sources[0].file_path.write_bytes(b"replacement source")
    else:
        monkeypatch.setattr(
            "core.jobs.boundary_embeddings._runtime", lambda: {"changed": True}
        )
    with pytest.raises(StaleJobResult):
        run(setup, operation=operation)
    compute.assert_not_called()


def test_model_retained_between_pairs_and_cache_does_not_unload(setup, monkeypatch):
    unload = Mock()
    monkeypatch.setattr("core.analysis.embeddings.unload_model", unload)
    run(setup)
    run(setup)
    unload.assert_called_once()


def test_corrupt_committed_payload_is_not_recomputed(setup):
    import sqlite3
    from core.jobs.commits import StaleJobResult

    path, _, compute = setup
    run(setup)
    compute.reset_mock()
    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute("UPDATE job_results SET payload_json = '{}' ")
    with pytest.raises(StaleJobResult):
        run(setup)
    compute.assert_not_called()


def test_cli_boundary_embeddings_persists_pairs(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, _ = setup
    register_commands()
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent)
    )
    response = CliRunner().invoke(
        cli, ["--json", "analyze", "boundary-embeddings", str(path), "--clip-id", "c-0"]
    )
    assert response.exit_code == 0, response.output
    saved = Project.load(path)
    assert saved.clips[0].last_frame_embedding == [0.987654321] * 768
    assert saved.clips[1].last_frame_embedding is None


def test_model_failure_stops_later_pairs(setup):
    from core.errors import ModelDownloadError

    path, _, compute = setup
    compute.side_effect = ModelDownloadError("model unavailable")
    result = run(setup)
    assert len(result["failed"]) == 1
    assert len(result["unprocessed"]) == 2
    compute.assert_called_once()
    assert not Project.load(path).metadata.job_results


def test_saved_pair_edit_is_preserved_without_force(setup):
    path, _, compute = setup
    run(setup)
    project = Project.load(path)
    project.clips[0].last_frame_embedding = [0.7] * 768
    assert project.save()
    compute.reset_mock()
    assert len(run(setup)["skipped"]) == 3
    compute.assert_not_called()
    assert Project.load(path).clips[0].last_frame_embedding == [0.7] * 768


def test_partial_pair_is_completed_together(setup):
    path, _, _ = setup
    project = Project.load(path)
    project.clips[0].first_frame_embedding = [0.7] * 768
    assert project.save()
    run(setup)
    assert Project.load(path).clips[0].first_frame_embedding == [0.123456789] * 768


def test_cancel_during_inference_does_not_record_late_pair(setup):
    from core.jobs.boundary_embeddings import run_boundary_embedding_job

    path, store, compute = setup
    cancel = Event()

    def infer(**kwargs):
        cancel.set()
        return [0.1] * 768, [0.2] * 768

    compute.side_effect = infer
    result = run_boundary_embedding_job(store, path, None, lambda *_: None, cancel)[
        "result"
    ]
    assert len(result["unprocessed"]) == 3
    assert not Project.load(path).metadata.job_results
    compute.assert_called_once()
