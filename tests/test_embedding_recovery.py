"""Batched embedding computation survives project save/checkpoint interruptions."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.embeddings import embedding_job_spec, run_embedding_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.embeddings import EmbeddingOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 3)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(side_effect=lambda paths: [[0.123456789] * 768 for _ in paths])
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", compute
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_embedding_job(store, path, None, lambda *_: None, Event(), **kwargs)[
        "result"
    ]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_batch_after_store_reopen(setup, force):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert Project.load(path).clips[0].embedding is None
    compute.assert_called_once()
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        compute.side_effect = AssertionError("No recomputation")
        assert len(run((path, reopened, compute), force=force)["succeeded"]) == 3
    finally:
        reopened.close()
    compute.assert_called_once()
    assert Project.load(path).clips[0].embedding == [0.123456789] * 768


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_reconciles_saved_results(setup, force):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert len(run(setup, force=force)["skipped"]) == 3
    compute.assert_called_once()
    assert all(
        store.get_result(rid)["committed"]
        for rid in Project.load(path).metadata.job_results
    )


def test_force_refresh_recovers_after_failed_save(setup):
    path, _, compute = setup
    run(setup)
    compute.side_effect = lambda paths: [[0.2] * 768 for _ in paths]
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    compute.side_effect = AssertionError("No recomputation")
    assert len(run(setup, force=True)["succeeded"]) == 3
    assert compute.call_count == 2
    assert Project.load(path).clips[0].embedding == [0.2] * 768


def test_invalid_vector_is_not_a_successful_receipt(setup):
    path, _, compute = setup
    compute.side_effect = None
    compute.return_value = [[0.0] * 768, [0.1] * 768, [0.2] * 768]
    assert len(run(setup)["failed"]) == 1
    assert len(Project.load(path).metadata.job_results) == 2
    compute.side_effect = lambda paths: [[0.3] * 768 for _ in paths]
    assert len(run(setup)["succeeded"]) == 1
    assert len(compute.call_args.args[0]) == 1


def test_cancellation_preserves_prefix_and_recorded_neighbors(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_embedding_job(store, path, None, lambda *_: cancel.set(), cancel)[
        "result"
    ]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 2
    assert len(Project.load(path).metadata.job_results) == 1
    assert len(run(setup)["succeeded"]) == 2
    compute.assert_called_once()


def test_chunks_share_model_ownership_and_cache_does_not_load(setup, monkeypatch):
    _, _, compute = setup
    unload = Mock()
    monkeypatch.setattr("core.analysis.embeddings.unload_model", unload)
    run(setup, options=EmbeddingOptions(2))
    run(setup, options=EmbeddingOptions(2))
    assert [len(c.args[0]) for c in compute.call_args_list] == [2, 1]
    unload.assert_called_once()


@pytest.mark.parametrize("change", ["image", "source", "runtime"])
def test_inputs_changed_during_compute_are_rejected(setup, monkeypatch, change):
    path, _, compute = setup
    project = Project.load(path)

    def changed(paths):
        if change == "runtime":
            monkeypatch.setattr(
                "core.jobs.embeddings._runtime", lambda: {"changed": True}
            )
        else:
            target = (
                project.clips[0].thumbnail_path
                if change == "image"
                else project.sources[0].file_path
            )
            target.write_bytes(b"changed")
        return [[0.1] * 768 for _ in paths]

    compute.side_effect = changed
    with pytest.raises(StaleJobResult):
        run(setup)
    assert not Project.load(path).metadata.job_results


def test_queued_options_and_media_are_frozen(setup):
    path, _, compute = setup
    project = Project.load(path)
    operation = embedding_job_spec(project, None, EmbeddingOptions(2), arguments={})
    project.clips[0].thumbnail_path.write_bytes(b"changed")
    with pytest.raises(StaleJobResult):
        run(setup, operation=operation)
    compute.assert_not_called()


@pytest.mark.parametrize("column", ["payload_json", "spec_json"])
def test_corrupt_receipt_does_not_recompute(setup, column):
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
    compute.assert_called_once()


def test_changed_embedding_projection_requires_recomputation(setup):
    path, _, compute = setup
    run(setup)
    project = Project.load(path)
    project.clips[0].embedding[0] = 0.99
    assert project.save()
    run(setup)
    assert Project.load(path).clips[0].embedding[0] == 0.123456789
    assert compute.call_count == 2


def test_generic_plan_retains_embeddings_after_later_failure(setup, monkeypatch):
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, compute = setup
    monkeypatch.setitem(
        ANALYZE_CLIP_OPERATION_MAP,
        "colors",
        Mock(side_effect=RuntimeError("later failure")),
    )
    for _ in range(2):
        op = analysis_job_spec(
            Project.load(path), arguments={"operations": ["embeddings", "colors"]}
        )
        with pytest.raises(RuntimeError):
            run_analysis_job(store, path, op, lambda *_: None, Event())
    compute.assert_called_once()
    assert len(Project.load(path).metadata.job_results) == 3


def test_cli_reconciles_checkpoint_retry(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, compute = setup
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent)
    )
    register_commands()
    args = ["--json", "analyze", "embeddings", str(path)]
    with patch.object(
        JobStore, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        assert CliRunner().invoke(cli, args).exit_code != 0
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    compute.assert_called_once()


def test_model_failure_does_not_retry_every_later_batch(setup):
    path, _, compute = setup
    compute.side_effect = RuntimeError("model unavailable")
    result = run(setup, options=EmbeddingOptions(1))
    compute.assert_called_once()
    assert len(result["failed"]) == 1
    assert len(result["unprocessed"]) == 2
    assert not Project.load(path).metadata.job_results


def test_failed_save_keeps_every_vector_in_large_batch(setup):
    path, _, compute = setup
    project = project_with_thumbnails(path.parent, 20)
    assert project.save(path)
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, options=EmbeddingOptions(20))
    compute.assert_called_once()
    compute.side_effect = AssertionError("No recomputation of batch tail")
    assert len(run(setup, options=EmbeddingOptions(20))["succeeded"]) == 20
    compute.assert_called_once()


@pytest.mark.parametrize("change", ["image", "runtime", "range"])
def test_retry_invalidates_recorded_computation_when_inputs_change(
    setup, monkeypatch, change
):
    path, _, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup)
    project = Project.load(path)
    if change == "image":
        project.clips[0].thumbnail_path.write_bytes(b"changed")
    elif change == "runtime":
        monkeypatch.setattr(
            "core.jobs.embeddings._runtime", lambda: {"model": "changed"}
        )
    else:
        for clip in project.clips:
            clip.start_frame += 1
        assert project.save()
    assert len(run(setup)["succeeded"]) == 3
    assert compute.call_count == 2


def test_precancelled_job_does_not_load_or_publish(setup):
    path, store, compute = setup
    cancel = Event()
    cancel.set()
    result = run_embedding_job(store, path, None, lambda *_: None, cancel)["result"]
    assert len(result["unprocessed"]) == 3
    assert not Project.load(path).metadata.job_results
    compute.assert_not_called()
