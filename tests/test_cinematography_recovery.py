"""Cinematography results survive save/checkpoint failures without extra inference."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.commits import StaleJobResult
from core.jobs.cinematography import cinematography_job_spec, run_cinematography_job
from core.jobs.store import JobStore
from core.operations.cinematography import CinematographyOptions
from models.cinematography import CinematographyAnalysis
from core.project import Project
from tests.test_description_operations import project_with_thumbnails

OPTIONS = CinematographyOptions("cloud", "frame", "gpt-test", "local-model")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(return_value=CinematographyAnalysis(shot_size="CU"))
    monkeypatch.setattr("core.analysis.cinematography.analyze_cinematography", compute)
    return path, store, compute


def run(setup, **kwargs):
    path, store, _ = setup
    return run_cinematography_job(
        store, path, None, lambda *_: None, Event(), options=OPTIONS, **kwargs
    )["result"]


def test_failed_save_reuses_recorded_computation(setup):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup)
    assert Project.load(path).clips[0].cinematography is None
    store.close()
    store = JobStore(path.parent / "jobs.db")
    compute.side_effect = AssertionError("Must reuse result")
    assert len(run((path, store, compute))["succeeded"]) == 2
    assert compute.call_count == 2
    assert len(Project.load(path).metadata.job_results) == 2


def test_checkpoint_failure_reconciles_and_preserves_user_edits(setup):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            run(setup)
    assert len(run(setup)["skipped"]) == 2
    saved = Project.load(path)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    saved.clips[0].shot_type = "wide"
    assert saved.save()
    run(setup)
    assert Project.load(path).clips[0].shot_type == "wide"
    assert compute.call_count == 2


def test_changed_media_during_compute_is_not_saved(setup):
    path, _, compute = setup
    image = Project.load(path).clips[0].thumbnail_path

    def change(*args, **kwargs):
        image.write_bytes(b"changed")
        return CinematographyAnalysis(shot_size="CU")

    compute.side_effect = change
    with pytest.raises(StaleJobResult, match="inputs changed"):
        run(setup)
    assert Project.load(path).clips[0].cinematography is None


def test_queued_settings_are_frozen_and_media_changes_rejected(setup, monkeypatch):
    path, store, compute = setup
    project = Project.load(path)
    operation = cinematography_job_spec(project, None, OPTIONS, arguments={})
    monkeypatch.setattr(
        "core.jobs.cinematography.resolve_options",
        lambda: pytest.fail("Must use captured options"),
    )
    result = run_cinematography_job(
        store, path, None, lambda *_: None, Event(), operation=operation
    )
    assert len(result["result"]["succeeded"]) == 2
    assert compute.call_args.kwargs["model"] == "gpt-test"
    project = Project.load(path)
    operation = cinematography_job_spec(project, None, OPTIONS, arguments={})
    project.clips[0].thumbnail_path.write_bytes(b"different")
    with pytest.raises(StaleJobResult, match="queued"):
        run_cinematography_job(
            store, path, None, lambda *_: None, Event(), operation=operation
        )


def test_model_change_does_not_reuse_unsaved_result(setup):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup)
    alternate = CinematographyOptions("cloud", "frame", "gpt-other", "local-model")
    run_cinematography_job(
        store, path, None, lambda *_: None, Event(), options=alternate
    )
    assert compute.call_count == 4


def test_cancel_preserves_prior_results(setup):
    path, store, compute = setup
    cancel = Event()

    def progress(*args):
        cancel.set()

    result = run_cinematography_job(
        store, path, None, progress, cancel, options=OPTIONS
    )["result"]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 1
    assert Project.load(path).clips[0].cinematography.shot_size == "CU"
    assert compute.call_count == 1


def test_missing_receipt_payload_refuses_paid_recomputation(setup):
    path, _, compute = setup
    run(setup)
    empty = JobStore(path.parent / "empty-jobs.db")
    with pytest.raises(StaleJobResult, match="missing"):
        run_cinematography_job(
            empty, path, None, lambda *_: None, Event(), options=OPTIONS
        )
    assert compute.call_count == 2


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
def test_corrupt_receipt_refuses_recomputation(setup, column):
    path, store, compute = setup
    run(setup)
    rid = next(iter(Project.load(path).metadata.job_results))
    # Simulate on-disk corruption without altering the receipt stored in the project.
    import sqlite3

    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute(
            f"UPDATE job_results SET {column} = ? WHERE result_id = ?", ("{}", rid)
        )
    with pytest.raises(StaleJobResult, match="corrupt"):
        run(setup)
    assert compute.call_count == 2


def test_generic_analysis_reuses_result_after_later_failure(setup, monkeypatch):
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.settings import Settings
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, compute = setup
    settings = Settings(cinematography_tier="cloud", cinematography_model="gpt-test")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setitem(
        ANALYZE_CLIP_OPERATION_MAP,
        "colors",
        Mock(side_effect=RuntimeError("later failure")),
    )
    for _ in range(2):
        operation = analysis_job_spec(
            Project.load(path), arguments={"operations": ["cinematography", "colors"]}
        )
        with pytest.raises(RuntimeError, match="later failure"):
            run_analysis_job(store, path, operation, lambda *_: None, Event())
    assert len(Project.load(path).metadata.job_results) == 2
    assert compute.call_count == 2


def test_failed_item_does_not_discard_other_results(setup):
    path, store, compute = setup
    compute.side_effect = [
        ValueError("invalid model"),
        CinematographyAnalysis(shot_size="MS"),
    ]
    result = run(setup)
    assert len(result["failed"]) == 1
    assert len(result["succeeded"]) == 1
    saved = Project.load(path)
    assert saved.clips[0].cinematography is None
    assert saved.clips[1].cinematography.shot_size == "MS"
    assert len(saved.metadata.job_results) == 1


def test_missing_thumbnail_is_per_item(setup):
    path, _, compute = setup
    project = Project.load(path)
    project.clips[0].thumbnail_path = path.parent / "missing.jpg"
    assert project.save()
    result = run(setup)
    assert result["failed"][0]["code"] == "thumbnail_missing"
    assert len(result["succeeded"]) == 1
    assert compute.call_count == 1


def test_local_runtime_change_rejects_queued_job(setup, monkeypatch):
    path, store, compute = setup
    options = CinematographyOptions("local", "frame", "cloud-model", "local-model")
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: True)
    operation = cinematography_job_spec(Project.load(path), None, options, arguments={})
    monkeypatch.setattr("core.analysis.description.is_mlx_vlm_available", lambda: False)
    with pytest.raises(StaleJobResult, match="queued"):
        run_cinematography_job(
            store, path, None, lambda *_: None, Event(), operation=operation
        )
    compute.assert_not_called()


def test_changed_source_media_during_compute_is_not_saved(setup):
    path, _, compute = setup
    source = Project.load(path).sources[0].file_path

    def change(**_):
        source.write_bytes(b"changed source content")
        return CinematographyAnalysis()

    compute.side_effect = change
    with pytest.raises(StaleJobResult, match="inputs changed"):
        run(setup)
    assert not Project.load(path).metadata.job_results
