"""Durable face receipts reconcile full and serialized embedding precision."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.faces import face_job_spec, run_face_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.faces import FaceOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(
        return_value=[
            {
                "bbox": [0, 0, 10, 10],
                "embedding": [0.123456789] * 512,
                "confidence": 0.9,
                "frame_number": 0,
            }
        ]
    )
    monkeypatch.setattr("core.analysis.faces.extract_faces_from_clip", compute)
    monkeypatch.setattr("core.analysis.faces._load_insightface", Mock())
    monkeypatch.setattr("core.analysis.faces.unload_model", Mock())
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_face_job(store, path, None, lambda *_: None, Event(), **kwargs)["result"]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_full_precision_computation(setup, force):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert Project.load(path).clips[0].face_embeddings is None
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        compute.side_effect = AssertionError("No recomputation")
        assert len(run((path, reopened, compute), force=force)["succeeded"]) == 2
    finally:
        reopened.close()
    assert compute.call_count == 2
    assert Project.load(path).clips[0].face_embeddings[0]["embedding"][0] == 0.12346


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_reconciles_saved_rounding(setup, force):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert len(run(setup, force=force)["skipped"]) == 2
    assert compute.call_count == 2
    assert all(
        store.get_result(rid)["committed"]
        for rid in Project.load(path).metadata.job_results
    )


def test_manual_embedding_edit_is_not_overwritten(setup):
    path, _, compute = setup
    run(setup)
    saved = Project.load(path)
    saved.clips[0].face_embeddings[0]["embedding"][0] = 0.5
    assert saved.save()
    run(setup)
    assert compute.call_count == 2
    assert Project.load(path).clips[0].face_embeddings[0]["embedding"][0] == 0.5


def test_source_change_during_inference_is_rejected(setup):
    path, _, compute = setup
    source = Project.load(path).sources[0].file_path

    def changed(**kwargs):
        source.write_bytes(b"changed")
        return []

    compute.side_effect = changed
    with pytest.raises(StaleJobResult):
        run(setup)
    assert not Project.load(path).metadata.job_results


def test_queued_interval_and_runtime_are_frozen(setup, monkeypatch):
    path, _, compute = setup
    operation = face_job_spec(Project.load(path), None, FaceOptions(0.4), arguments={})
    run(setup, operation=operation)
    assert compute.call_args.kwargs["sample_interval"] == 0.4
    operation = face_job_spec(Project.load(path), None, FaceOptions(), arguments={})
    monkeypatch.setattr("core.jobs.faces._runtime", lambda: {"changed": True})
    with pytest.raises(StaleJobResult):
        run(setup, operation=operation)


def test_generic_plan_retains_faces_after_later_failure(setup, monkeypatch):
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, compute = setup
    monkeypatch.setitem(
        ANALYZE_CLIP_OPERATION_MAP,
        "shots",
        Mock(side_effect=RuntimeError("later failure")),
    )
    for _ in range(2):
        op = analysis_job_spec(
            Project.load(path), arguments={"operations": ["face_embeddings", "shots"]}
        )
        with pytest.raises(RuntimeError):
            run_analysis_job(store, path, op, lambda *_: None, Event())
    assert len(Project.load(path).metadata.job_results) == 2
    assert compute.call_count == 2


def test_empty_and_failed_faces_are_distinct(setup):
    path, _, compute = setup
    compute.side_effect = [ValueError("decode failed"), []]
    result = run(setup)
    assert len(result["failed"]) == len(result["succeeded"]) == 1
    assert Project.load(path).clips[0].face_embeddings is None
    assert Project.load(path).clips[1].face_embeddings == []


def test_cancel_preserves_accepted_prefix(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_face_job(store, path, None, lambda *_: cancel.set(), cancel)["result"]
    assert len(result["succeeded"]) == len(result["unprocessed"]) == 1
    assert compute.call_count == 1
    assert len(Project.load(path).metadata.job_results) == 1


def test_force_refresh_reuses_failed_save_generation(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = []
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    compute.side_effect = AssertionError("No recomputation")
    assert len(run(setup, force=True)["succeeded"]) == 2
    assert compute.call_count == 4
    assert len(Project.load(path).metadata.job_results) == 4


def test_job_unloads_model_once_for_multiple_results(setup, monkeypatch):
    unload = Mock()
    monkeypatch.setattr("core.analysis.faces.unload_model", unload)
    run(setup)
    unload.assert_called_once()
    monkeypatch.setattr(
        "core.analysis.faces._load_insightface",
        Mock(side_effect=AssertionError("Cache hit must not load model")),
    )
    run(setup)
    unload.assert_called_once()


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
def test_corrupt_face_receipt_does_not_trigger_recomputation(setup, column):
    import sqlite3

    path, _, compute = setup
    run(setup)
    rid = next(iter(Project.load(path).metadata.job_results))
    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute(
            f"UPDATE job_results SET {column} = ? WHERE result_id = ?", ("{}", rid)
        )
    with pytest.raises(StaleJobResult):
        run(setup)
    assert compute.call_count == 2


def test_cli_faces_reconciles_checkpoint_retry(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, compute = setup
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent)
    )
    register_commands()
    args = ["--json", "analyze", "faces", str(path), "--sample-interval", ".4"]
    with patch.object(
        JobStore, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        assert CliRunner().invoke(cli, args).exit_code != 0
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert compute.call_count == 2
    assert compute.call_args.kwargs["sample_interval"] == 0.4
    store = JobStore(path.parent / "jobs.db")
    try:
        assert all(
            store.get_result(rid)["committed"]
            for rid in Project.load(path).metadata.job_results
        )
    finally:
        store.close()


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_sampling(setup, interval):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    path, _, compute = setup
    register_commands()
    result = CliRunner().invoke(
        cli, ["analyze", "faces", str(path), "--sample-interval", interval]
    )
    assert result.exit_code == 7
    compute.assert_not_called()
