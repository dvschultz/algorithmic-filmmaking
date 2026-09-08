"""ObjectDetection recovery preserves computation and user-owned metadata."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.object_detection import (
    object_detection_job_spec,
    run_object_detection_job,
)
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.object_detection import ObjectDetectionOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(
        return_value=[{"label": "person", "confidence": 0.8, "bbox": [0, 0, 1, 1]}]
    )
    monkeypatch.setattr("core.analysis.detection.detect_objects", compute)
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_object_detection_job(
        store, path, None, lambda *_: None, Event(), **kwargs
    )["result"]


def test_failed_attempts_are_saved_without_success_receipts(setup):
    path, _, compute = setup
    compute.side_effect = RuntimeError("provider failed")
    assert len(run(setup)["failed"]) == 2
    loaded = Project.load(path)
    assert not loaded.metadata.job_results
    assert all(clip.analysis_records["detect_objects"].state == "failed" for clip in loaded.clips)
    compute.side_effect = None
    assert len(run(setup)["succeeded"]) == 2
    assert compute.call_count == 4


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_computation_after_reopening_store(setup, force):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup, force=force)
    assert Project.load(path).clips[0].detected_objects is None
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


def test_edited_projection_requires_recomputation(setup):
    path, _, compute = setup
    run(setup)
    saved = Project.load(path)
    saved.clips[0].detected_objects = ["manual"]
    saved.update_clips([saved.clips[0]])
    assert saved.save()
    run(setup)
    assert Project.load(path).clips[0].detected_objects == compute.return_value
    assert compute.call_count == 3


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
        return [{"label": "person", "confidence": 0.8, "bbox": [0, 0, 1, 1]}]

    compute.side_effect = change
    with pytest.raises(StaleJobResult, match="inputs changed"):
        run(setup)
    assert not Project.load(path).metadata.job_results


def test_queued_options_are_frozen_and_runtime_change_is_rejected(setup, monkeypatch):
    path, store, compute = setup
    operation = object_detection_job_spec(
        Project.load(path), None, ObjectDetectionOptions(0.4, True), arguments={}
    )
    run(setup, operation=operation)
    assert compute.call_args.kwargs["confidence_threshold"] == 0.4
    operation = object_detection_job_spec(
        Project.load(path), None, ObjectDetectionOptions(), arguments={}
    )
    monkeypatch.setattr(
        "core.jobs.object_detection._runtime", lambda: {"changed": True}
    )
    with pytest.raises(StaleJobResult, match="queued"):
        run(setup, operation=operation)
    assert compute.call_count == 2


def test_generic_plan_keeps_object_detection_after_later_failure(setup, monkeypatch):
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
            Project.load(path), arguments={"operations": ["detect_objects", "colors"]}
        )
        with pytest.raises(RuntimeError, match="later failure"):
            run_analysis_job(store, path, operation, lambda *_: None, Event())
    assert len(Project.load(path).metadata.job_results) == 2
    assert compute.call_count == 2


def test_partial_failure_and_empty_objects_are_saved_correctly(setup):
    path, _, compute = setup
    compute.side_effect = [ValueError("provider failed"), []]
    result = run(setup)
    assert len(result["failed"]) == 1
    assert len(result["succeeded"]) == 1
    saved = Project.load(path)
    assert saved.clips[0].detected_objects is None
    assert saved.clips[1].detected_objects == []


def test_cancel_retains_finished_targets(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_object_detection_job(
        store, path, None, lambda *_: cancel.set(), cancel
    )["result"]
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


def test_verified_record_outlives_job_receipts(setup):
    path, _, compute = setup
    run(setup)
    empty = JobStore(path.parent / "empty.db")
    try:
        assert len(run((path, empty, compute))["skipped"]) == 2
    finally:
        empty.close()
    assert compute.call_count == 2


def test_force_refresh_starts_new_generation_and_preserves_failed_save(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = [{"label": "car", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    assert Project.load(path).clips[0].detected_objects[0]["label"] == "person"
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
    args = ["--json", "analyze", "objects", str(path)]
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


@pytest.mark.parametrize("force", [False, True])
def test_people_only_recovery_preserves_objects(setup, monkeypatch, force):
    path, store, compute = setup
    saved = Project.load(path)
    for clip in saved.clips:
        clip.detected_objects = [{"label": "cat"}]
    assert saved.save()
    people = Mock(return_value=0)
    monkeypatch.setattr("core.analysis.detection.count_people", people)
    options = ObjectDetectionOptions(detect_all=False)
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, options=options, force=force)
    people.side_effect = AssertionError("Must reuse")
    assert len(run(setup, options=options, force=force)["succeeded"]) == 2
    assert people.call_count == 2
    assert compute.call_count == 0
    assert all(
        c.person_count == 0 and c.detected_objects == [{"label": "cat"}]
        for c in Project.load(path).clips
    )


def test_people_and_objects_do_not_reuse_each_others_results(setup, monkeypatch):
    path, _, compute = setup
    people = Mock(return_value=2)
    monkeypatch.setattr("core.analysis.detection.count_people", people)
    run(setup, options=ObjectDetectionOptions(detect_all=False))
    assert all(c.detected_objects is None for c in Project.load(path).clips)
    run(setup)
    assert people.call_count == compute.call_count == 2
    assert all(c.person_count == 1 for c in Project.load(path).clips)
    assert len(Project.load(path).metadata.job_results) == 4


def test_model_load_failure_stops_request(setup):
    from core.errors import ModelDownloadError

    path, _, compute = setup
    compute.side_effect = ModelDownloadError("model unavailable")
    result = run(setup)
    assert compute.call_count == 1
    assert result["failed"][0]["code"] == "model_load_failed"
    assert result["unprocessed"][0]["code"] == "model_unavailable"
    assert not Project.load(path).metadata.job_results


@pytest.mark.parametrize("command", ["objects", "people"])
def test_cli_preserves_thumbnail_dependency_exit_code(setup, monkeypatch, command):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from cli.utils.errors import ExitCode
    from core.settings import Settings

    path, _, _ = setup
    monkeypatch.setattr("cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent))
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", Mock(side_effect=RuntimeError("ffmpeg missing")))
    register_commands()
    result = CliRunner().invoke(cli, ["analyze", command, str(path)])
    assert result.exit_code == ExitCode.DEPENDENCY_MISSING
    assert "ffmpeg missing" in result.output
