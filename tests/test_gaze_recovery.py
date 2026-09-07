"""Durable gaze results retain computation precision and recover explicit retries."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.gaze import gaze_job_spec, run_gaze_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.operations.gaze import GazeOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(
        return_value={
            "gaze_yaw": 2.123456,
            "gaze_pitch": -1.23456,
            "gaze_category": "at_camera",
        }
    )
    monkeypatch.setattr("core.analysis.gaze.extract_gaze_from_clip", compute)
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", Mock())
    monkeypatch.setattr("core.analysis.gaze.unload_model", Mock())
    yield path, store, compute
    store.close()


def run(setup, **kwargs):
    path, store, _ = setup
    return run_gaze_job(store, path, None, lambda *_: None, Event(), **kwargs)["result"]


@pytest.mark.parametrize("force", [False, True])
def test_failed_save_reuses_computation_after_store_reopen(setup, force):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=force)
    assert Project.load(path).clips[0].gaze_category is None
    store.close()
    reopened = JobStore(path.parent / "jobs.db")
    try:
        compute.side_effect = AssertionError("No recomputation")
        assert len(run((path, reopened, compute), force=force)["succeeded"]) == 2
    finally:
        reopened.close()
    assert compute.call_count == 2
    assert Project.load(path).clips[0].gaze_yaw == 2.12


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_failure_recognizes_saved_angle_precision(setup, force):
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


def test_manual_angle_edit_is_preserved(setup):
    path, _, compute = setup
    run(setup)
    project = Project.load(path)
    project.clips[0].gaze_yaw = 9.99
    assert project.save()
    run(setup)
    assert compute.call_count == 2
    assert Project.load(path).clips[0].gaze_yaw == 9.99


def test_no_gaze_is_durable_but_decode_error_is_not(setup):
    path, _, compute = setup
    compute.side_effect = [ValueError("decode failed"), None]
    assert [row["code"] for row in run(setup)["failed"]] == [
        "gaze_failed",
        "no_gaze_detected",
    ]
    assert len(Project.load(path).metadata.job_results) == 1
    compute.side_effect = None
    assert len(run(setup)["succeeded"]) == 1
    assert compute.call_count == 3
    assert Project.load(path).clips[1].gaze_category is None


def test_force_empty_result_reconciles_default_retry(setup):
    path, store, compute = setup
    compute.return_value = None
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    run(setup)
    assert compute.call_count == 2
    assert all(
        store.get_result(rid)["committed"]
        for rid in Project.load(path).metadata.job_results
    )


def test_force_empty_result_clears_old_angles(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = None
    run(setup, force=True)
    clip = Project.load(path).clips[0]
    assert (clip.gaze_yaw, clip.gaze_pitch, clip.gaze_category) == (None, None, None)


def test_force_refresh_recovers_after_failed_save(setup):
    path, _, compute = setup
    run(setup)
    compute.return_value = {
        "gaze_yaw": 18.125,
        "gaze_pitch": 0.123,
        "gaze_category": "looking_right",
    }
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    compute.side_effect = AssertionError("No recomputation")
    assert len(run(setup, force=True)["succeeded"]) == 2
    assert compute.call_count == 4
    assert Project.load(path).clips[0].gaze_category == "looking_right"


def test_model_load_failure_stops_remaining_items(setup, monkeypatch):
    path, _, compute = setup
    load = Mock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", load)
    result = run(setup)
    assert result["failed"][0]["code"] == "model_load_failed"
    assert len(result["unprocessed"]) == 1
    load.assert_called_once()
    compute.assert_not_called()
    assert not Project.load(path).metadata.job_results


def test_source_change_during_inference_rejected(setup):
    path, _, compute = setup
    source = Project.load(path).sources[0].file_path

    def changed(**kwargs):
        source.write_bytes(b"changed")
        return None

    compute.side_effect = changed
    with pytest.raises(StaleJobResult):
        run(setup)


def test_submission_freezes_interval_and_runtime(setup, monkeypatch):
    path, _, compute = setup
    operation = gaze_job_spec(Project.load(path), None, GazeOptions(0.4), arguments={})
    run(setup, operation=operation)
    assert compute.call_args.kwargs["sample_interval"] == 0.4
    operation = gaze_job_spec(Project.load(path), None, GazeOptions(), arguments={})
    monkeypatch.setattr("core.jobs.gaze._runtime", lambda: {"changed": True})
    with pytest.raises(StaleJobResult):
        run(setup, operation=operation)


def test_cancel_preserves_accepted_prefix(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_gaze_job(store, path, None, lambda *_: cancel.set(), cancel)["result"]
    assert len(result["succeeded"]) == len(result["unprocessed"]) == 1
    compute.assert_called_once()
    assert len(Project.load(path).metadata.job_results) == 1


def test_model_session_retained_across_commits(setup, monkeypatch):
    load, unload = Mock(), Mock()
    monkeypatch.setattr("core.analysis.gaze.load_face_mesh", load)
    monkeypatch.setattr("core.analysis.gaze.unload_model", unload)
    run(setup)
    run(setup)
    load.assert_called_once()
    unload.assert_called_once()


@pytest.mark.parametrize("column", ["spec_json", "payload_json"])
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
    assert compute.call_count == 2


def test_generic_plan_retains_results_after_later_failure(setup, monkeypatch):
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
            Project.load(path), arguments={"operations": ["gaze", "shots"]}
        )
        with pytest.raises(RuntimeError):
            run_analysis_job(store, path, op, lambda *_: None, Event())
    assert compute.call_count == 2
    assert len(Project.load(path).metadata.job_results) == 2


def test_cli_reconciles_checkpoint_retry(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, _, compute = setup
    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load", lambda: Settings(cache_dir=path.parent)
    )
    register_commands()
    args = ["--json", "analyze", "gaze", str(path), "--sample-interval", ".4"]
    with patch.object(
        JobStore, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        assert CliRunner().invoke(cli, args).exit_code != 0
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert compute.call_count == 2
    assert compute.call_args.kwargs["sample_interval"] == 0.4


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_sampling(setup, interval):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    path, _, compute = setup
    register_commands()
    assert (
        CliRunner()
        .invoke(cli, ["analyze", "gaze", str(path), "--sample-interval", interval])
        .exit_code
        == 7
    )
    compute.assert_not_called()
