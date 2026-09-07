"""Shot jobs recover computation without overwriting newer saved metadata."""

import json
import sqlite3
from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.shots import shot_job_spec, run_shot_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.shots import ShotTypeOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_shot_job(store, path, None, lambda *_: None, Event(), **kwargs)["result"]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_computation_after_store_reopen(setup, force):
    path, store, compute = setup
    before = path.read_bytes()
    with patch(
        "core.jobs.commits.save_with_mtime_check", side_effect=OSError("save failed")
    ):
        with pytest.raises(OSError, match="save failed"):
            run(setup, force=force)
    assert path.read_bytes() == before
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        compute.side_effect = AssertionError("Must reuse recorded inference")
        assert len(run((path, reopened, compute), force=force)["succeeded"]) == 2
        assert compute.call_count == 2
    finally:
        reopened.close()


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_reconciles_without_inference(setup, force):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=OSError("checkpoint failed")
    ):
        with pytest.raises(OSError):
            run(setup, force=force)
    assert len(run(setup, force=force)["skipped"]) == 2
    saved = Project.load(path)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    assert compute.call_count == 2


def test_force_generation_reuses_failed_refresh(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = ("close-up", 0.8)
    with patch(
        "core.jobs.commits.save_with_mtime_check", side_effect=OSError("disk full")
    ):
        with pytest.raises(OSError):
            run(setup, force=True)
    assert Project.load(path).clips[0].shot_type == "wide"
    compute.side_effect = AssertionError("Must reuse refresh")
    assert len(run(setup, force=True)["succeeded"]) == 2
    saved = Project.load(path)
    assert saved.clips[0].shot_type == "close-up"
    assert len(saved.metadata.job_results) == 4
    assert compute.call_count == 4


def test_external_analysis_image_retry_preserves_display_thumbnail(setup):
    path, store, compute = setup
    project = Project.load(path)
    cid = project.clips[0].id
    original = project.clips[0].thumbnail_path
    image = path.parent / "analysis.jpg"
    image.write_bytes(b"analysis image")
    with patch.object(
        store, "checkpoint_results", side_effect=OSError("checkpoint failed")
    ):
        with pytest.raises(OSError):
            run(setup, thumbnail_paths={cid: image})
    run(setup)
    saved = Project.load(path)
    assert saved.clips[0].thumbnail_path == original
    assert compute.call_count == 2
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


def test_manual_shot_value_is_preserved(setup):
    path, _, compute = setup
    run(setup)
    project = Project.load(path)
    project.clips[0].shot_type = "manual"
    project.update_clips([project.clips[0]])
    assert project.save()
    result = run(setup)
    assert result["skipped"][0]["reason"] == "already_populated"
    assert Project.load(path).clips[0].shot_type == "manual"
    assert compute.call_count == 2


@pytest.mark.parametrize("change", ["image", "source", "runtime", "project"])
def test_queued_drift_rejected_before_inference(setup, monkeypatch, change):
    from core.project_revision import ProjectRevisionConflict

    path, _, compute = setup
    project = Project.load(path)
    operation = shot_job_spec(project, None, ShotTypeOptions(), arguments={})
    if change == "runtime":
        monkeypatch.setattr("core.jobs.shots._runtime", lambda: {"changed": True})
    elif change == "project":
        project.metadata.name = "changed"
        assert project.save()
    else:
        media = (
            project.clips[0].thumbnail_path
            if change == "image"
            else project.sources[0].file_path
        )
        media.write_bytes(b"changed")
    with pytest.raises((StaleJobResult, ProjectRevisionConflict)):
        run(setup, operation=operation)
    compute.assert_not_called()


def test_captured_cloud_options_are_used(setup, monkeypatch):
    path, _, local = setup
    cloud = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type_tiered", cloud)
    operation = shot_job_spec(
        Project.load(path), None, ShotTypeOptions("cloud", "captured"), arguments={}
    )
    run(setup, operation=operation, options=ShotTypeOptions("cloud", "changed"))
    assert cloud.call_count == 2
    assert cloud.call_args.kwargs["cloud_model"] == "captured"
    local.assert_not_called()


def test_prompt_change_during_inference_rejects_publication(setup, monkeypatch):
    from core.analysis.shots import SHOT_TYPE_PROMPTS

    path, _, compute = setup
    before = path.read_bytes()
    key = next(iter(SHOT_TYPE_PROMPTS))
    prompts = list(SHOT_TYPE_PROMPTS[key])
    monkeypatch.setitem(SHOT_TYPE_PROMPTS, key, prompts)

    def classify(_):
        prompts.append("changed while computing")
        return "wide", 0.9

    compute.side_effect = classify
    with pytest.raises(StaleJobResult):
        run(setup)
    assert path.read_bytes() == before
    compute.assert_called_once()


def test_partial_provider_failure_and_unknown_labels_are_not_saved(setup):
    path, _, compute = setup
    compute.side_effect = [RuntimeError("provider failed"), ("unknown", 0.1)]
    result = run(setup)
    assert len(result["failed"]) == 2
    assert not Project.load(path).metadata.job_results
    assert all(clip.shot_type is None for clip in Project.load(path).clips)


def test_cancel_retains_finished_items(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_shot_job(store, path, None, lambda *_: cancel.set(), cancel)["result"]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 1
    assert compute.call_count == 1
    assert len(Project.load(path).metadata.job_results) == 1


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
def test_corrupt_receipt_refuses_recomputation(setup, column):
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


def test_invalid_pending_payload_is_not_accepted(setup):
    from hashlib import sha256

    path, _, compute = setup
    before = path.read_bytes()
    with patch(
        "core.jobs.commits.save_with_mtime_check", side_effect=OSError("disk full")
    ):
        with pytest.raises(OSError):
            run(setup)
    # A structurally invalid payload must fail even when its checksum is valid.
    payload = json.dumps({"shot_type": "wide"}, sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(path.parent / "jobs.db") as connection:
        connection.execute(
            "UPDATE job_results SET payload_json = ?, payload_digest = ?",
            (payload, sha256(payload.encode()).hexdigest()),
        )
    with pytest.raises(StaleJobResult, match="Invalid recorded"):
        run(setup)
    assert path.read_bytes() == before
    assert compute.call_count == 2


def test_generic_plan_preserves_shots_after_later_failure(setup, monkeypatch):
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
            Project.load(path), arguments={"operations": ["shots", "colors"]}
        )
        with pytest.raises(RuntimeError, match="later failure"):
            run_analysis_job(store, path, operation, lambda *_: None, Event())
    assert len(Project.load(path).metadata.job_results) == 2
    assert compute.call_count == 2


def test_cli_checkpoint_retry_avoids_thumbnail_and_provider_work(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, compute = setup
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent)
    )
    image = path.parent / "analysis.jpg"
    image.write_bytes(b"analysis")
    generator = Mock()
    generator.generate_clip_thumbnail.return_value = image
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", lambda **_: generator)
    register_commands()
    args = ["--json", "analyze", "shots", str(path)]
    with patch.object(
        JobStore, "checkpoint_results", side_effect=OSError("checkpoint failed")
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
