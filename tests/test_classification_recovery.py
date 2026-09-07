"""Classification recovery preserves computation and user-owned metadata."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.classification import classification_job_spec, run_classification_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.classification import ClassificationOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(return_value=[("person", 0.8)])
    monkeypatch.setattr("core.analysis.classification.classify_frame", compute)
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_classification_job(
        store, path, None, lambda *_: None, Event(), **kwargs
    )["result"]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_computation_after_reopening_store(setup, force):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup, force=force)
    assert Project.load(path).clips[0].object_labels is None
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        compute.side_effect = AssertionError("Must reuse computation")
        assert len(run((path, reopened, compute), force=force)["succeeded"]) == 2
        assert compute.call_count == 2
    finally:
        reopened.close()


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_reconciles_without_new_inference(setup, force):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            run(setup, force=force)
    assert len(run(setup, force=force)["skipped"]) == 2
    saved = Project.load(path)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    assert compute.call_count == 2


def test_external_analysis_thumbnail_checkpoint_retry_preserves_display_image(setup):
    path, store, compute = setup
    original = Project.load(path).clips[0].thumbnail_path
    analysis_image = path.parent / "analysis.jpg"
    analysis_image.write_bytes(b"analysis image")
    cid = Project.load(path).clips[0].id
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError):
            run(setup, thumbnail_paths={cid: analysis_image})
    # CLI skips thumbnail generation for already-populated targets on retry.
    run(setup)
    saved = Project.load(path)
    assert saved.clips[0].thumbnail_path == original
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    assert compute.call_count == 2


def test_existing_user_labels_are_preserved(setup):
    path, _, compute = setup
    run(setup)
    saved = Project.load(path)
    saved.clips[0].object_labels = ["manual"]
    saved.update_clips([saved.clips[0]])
    assert saved.save()
    run(setup)
    assert Project.load(path).clips[0].object_labels == ["manual"]
    assert compute.call_count == 2


@pytest.mark.parametrize("media", ["image", "source"])
def test_media_changed_during_inference_is_not_published(setup, media):
    path, _, compute = setup
    project = Project.load(path)
    target = (
        project.clips[0].thumbnail_path
        if media == "image"
        else project.sources[0].file_path
    )

    def change(*args, **kwargs):
        target.write_bytes(b"changed media")
        return [("person", 0.8)]

    compute.side_effect = change
    with pytest.raises(StaleJobResult, match="inputs changed"):
        run(setup)
    assert not Project.load(path).metadata.job_results


def test_queued_options_are_frozen_and_runtime_change_is_rejected(setup, monkeypatch):
    path, store, compute = setup
    operation = classification_job_spec(
        Project.load(path), None, ClassificationOptions(2, 0.5), arguments={}
    )
    run(setup, operation=operation)
    assert compute.call_args.kwargs["top_k"] == 2
    assert compute.call_args.kwargs["threshold"] == 0.5
    operation = classification_job_spec(
        Project.load(path), None, ClassificationOptions(), arguments={}
    )
    monkeypatch.setattr("core.jobs.classification._runtime", lambda: {"changed": True})
    with pytest.raises(StaleJobResult, match="queued"):
        run(setup, operation=operation)
    assert compute.call_count == 2


def test_generic_plan_keeps_classification_after_later_failure(setup, monkeypatch):
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, compute = setup
    monkeypatch.setitem(
        ANALYZE_CLIP_OPERATION_MAP,
        "colors",
        Mock(side_effect=RuntimeError("later failure")),
    )
    for _ in range(2):
        operation = analysis_job_spec(
            Project.load(path), arguments={"operations": ["classify", "colors"]}
        )
        with pytest.raises(RuntimeError, match="later failure"):
            run_analysis_job(store, path, operation, lambda *_: None, Event())
    assert len(Project.load(path).metadata.job_results) == 2
    assert compute.call_count == 2


def test_partial_failure_and_empty_labels_are_saved_correctly(setup):
    path, _, compute = setup
    compute.side_effect = [ValueError("provider failed"), []]
    result = run(setup)
    assert len(result["failed"]) == 1
    assert len(result["succeeded"]) == 1
    saved = Project.load(path)
    assert saved.clips[0].object_labels is None
    assert saved.clips[1].object_labels == []


def test_cancel_retains_finished_targets(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_classification_job(store, path, None, lambda *_: cancel.set(), cancel)[
        "result"
    ]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 1
    assert compute.call_count == 1
    assert len(Project.load(path).metadata.job_results) == 1


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
def test_corrupt_receipt_refuses_recomputation(setup, column):
    import sqlite3

    path, _, compute = setup
    run(setup)
    rid = next(iter(Project.load(path).metadata.job_results))
    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute(
            f"UPDATE job_results SET {column} = ? WHERE result_id = ?", ("{}", rid)
        )
    with pytest.raises(StaleJobResult, match="corrupt"):
        run(setup)
    assert compute.call_count == 2


def test_missing_receipt_refuses_recomputation(setup):
    path, _, compute = setup
    run(setup)
    empty = JobStore(path.parent / "empty.db")
    try:
        with pytest.raises(StaleJobResult, match="missing"):
            run((path, empty, compute))
    finally:
        empty.close()
    assert compute.call_count == 2


def test_force_refresh_starts_new_generation_and_preserves_failed_save(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = [("car", 0.9)]
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    assert Project.load(path).clips[0].object_labels == ["person"]
    compute.side_effect = AssertionError("Must reuse refresh")
    assert len(run(setup, force=True)["succeeded"]) == 2
    assert compute.call_count == 4
    assert len(Project.load(path).metadata.job_results) == 4


def test_cli_reconciles_saved_receipts_without_regenerating_images(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, compute = setup
    settings = Settings(cache_dir=path.parent)
    monkeypatch.setattr("cli.commands.analyze.CLIConfig.load", lambda: settings)
    image = path.parent / "analysis.jpg"
    image.write_bytes(b"analysis")
    generator = Mock()
    generator.generate_clip_thumbnail.return_value = image
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", lambda **_: generator)
    register_commands()
    args = ["--json", "analyze", "classify", str(path)]
    with patch.object(
        JobStore, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        assert CliRunner().invoke(cli, args).exit_code != 0
    generator.generate_clip_thumbnail.side_effect = AssertionError("Already analyzed")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    store = JobStore(path.parent / "jobs.db")
    try:
        assert all(
            store.get_result(rid)["committed"]
            for rid in Project.load(path).metadata.job_results
        )
    finally:
        store.close()
    assert compute.call_count == 2
