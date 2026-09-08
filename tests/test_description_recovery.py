"""Description results survive save/checkpoint failures without extra inference."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.commits import StaleJobResult
from core.jobs.description import description_job_spec, run_description_job
from core.jobs.store import JobStore
from core.operations.description import DescriptionOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails

OPTIONS = DescriptionOptions("cloud", model="gpt-test", input_mode="frame")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    compute = Mock(return_value=("Generated", "gpt-test"))
    monkeypatch.setattr("core.analysis.description.describe_frame", compute)
    return path, store, compute


def run(setup, **kwargs):
    path, store, _ = setup
    return run_description_job(
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
    assert Project.load(path).clips[0].description is None
    compute.side_effect = AssertionError("Must reuse result")
    assert len(run(setup)["succeeded"]) == 2
    assert compute.call_count == 2
    assert len(Project.load(path).metadata.job_results) == 2


def test_checkpoint_failure_reconciles_and_edited_projection_requires_recompute(setup):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            run(setup)
    assert len(run(setup)["skipped"]) == 2
    saved = Project.load(path)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    saved.clips[0].description = "User edit"
    assert saved.save()
    run(setup)
    assert Project.load(path).clips[0].description == "Generated"
    assert compute.call_count == 3


def test_forced_refresh_reuses_failed_save_then_advances_generation(setup):
    run(setup)
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup, force=True)
    run(setup, force=True)
    assert setup[2].call_count == 4
    run(setup, force=True)
    assert setup[2].call_count == 6


def test_changed_media_during_compute_is_not_saved(setup):
    path, _, compute = setup
    image = Project.load(path).clips[0].thumbnail_path

    def change(*args, **kwargs):
        image.write_bytes(b"changed")
        return "Wrong input", "gpt-test"

    compute.side_effect = change
    with pytest.raises(StaleJobResult, match="inputs changed"):
        run(setup)
    assert Project.load(path).clips[0].description is None


def test_queued_settings_are_frozen_and_media_changes_rejected(setup, monkeypatch):
    path, store, compute = setup
    project = Project.load(path)
    operation = description_job_spec(project, None, OPTIONS, arguments={})
    monkeypatch.setattr(
        "core.jobs.description.resolve_options",
        lambda: pytest.fail("Must use captured options"),
    )
    result = run_description_job(
        store, path, None, lambda *_: None, Event(), operation=operation
    )
    assert len(result["result"]["succeeded"]) == 2
    assert compute.call_args.kwargs["model_name"] == "gpt-test"
    project = Project.load(path)
    operation = description_job_spec(project, None, OPTIONS, arguments={})
    project.clips[0].thumbnail_path.write_bytes(b"different")
    with pytest.raises(StaleJobResult, match="queued"):
        run_description_job(
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
    alternate = DescriptionOptions("cloud", model="gpt-other", input_mode="frame")
    run_description_job(store, path, None, lambda *_: None, Event(), options=alternate)
    assert compute.call_count == 4


def test_cancel_preserves_prior_results_and_missing_thumbnail_is_per_item(setup):
    path, store, compute = setup
    cancel = Event()

    def progress(*args):
        cancel.set()

    result = run_description_job(store, path, None, progress, cancel, options=OPTIONS)[
        "result"
    ]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 1
    assert Project.load(path).clips[0].description == "Generated"
    assert compute.call_count == 1


def test_missing_receipt_payload_reuses_verified_project_record(setup):
    path, _, compute = setup
    run(setup)
    empty = JobStore(path.parent / "empty-jobs.db")
    result = run_description_job(
        empty, path, None, lambda *_: None, Event(), options=OPTIONS
    )
    assert len(result["result"]["skipped"]) == 2
    assert compute.call_count == 2


def test_saved_prompt_change_requires_new_description(setup):
    from dataclasses import replace

    path, store, compute = setup
    run(setup)
    result = run_description_job(
        store,
        path,
        None,
        lambda *_: None,
        Event(),
        options=replace(OPTIONS, prompt="Different prompt"),
    )
    assert len(result["result"]["succeeded"]) == 2
    assert compute.call_count == 4


def test_job_failures_save_failed_records_without_new_receipts(setup):
    from dataclasses import replace

    path, store, compute = setup
    run(setup)
    receipts = Project.load(path).metadata.job_results.copy()
    compute.side_effect = RuntimeError("Invalid provider response")
    result = run_description_job(
        store,
        path,
        None,
        lambda *_: None,
        Event(),
        options=replace(OPTIONS, prompt="Different prompt"),
    )
    assert len(result["result"]["failed"]) == 2
    saved = Project.load(path)
    assert saved.metadata.job_results == receipts
    assert all(
        c.description == "Generated"
        and c.analysis_records["describe"].state == "failed"
        for c in saved.clips
    )


def test_committed_image_fallback_does_not_satisfy_video_request(setup):
    from dataclasses import replace

    path, store, compute = setup
    options = replace(OPTIONS, model="gemini-test", input_mode="video")

    def fallback(*args, **kwargs):
        kwargs["on_execution"](
            {"backend": "cloud", "model": "gemini-test", "input_mode": "frame"}
        )
        return "Image fallback", "gemini-test"

    compute.side_effect = fallback
    for _ in range(2):
        result = run_description_job(
            store, path, None, lambda *_: None, Event(), options=options
        )
        assert len(result["result"]["succeeded"]) == 2
    assert compute.call_count == 4


def test_high_resolution_job_image_does_not_replace_display_thumbnail(setup):
    path, store, compute = setup
    project = Project.load(path)
    display = project.clips[0].thumbnail_path
    image = path.parent / "high-resolution.jpg"
    image.write_bytes(b"high resolution image")
    result = run_description_job(
        store,
        path,
        [project.clips[0].id],
        lambda *_: None,
        Event(),
        options=OPTIONS,
        thumbnail_paths={project.clips[0].id: image},
    )
    assert len(result["result"]["succeeded"]) == 1
    assert Project.load(path).clips[0].thumbnail_path == display
    assert compute.call_args.args[0] == image
    result = run_description_job(
        store, path, [project.clips[0].id], lambda *_: None, Event(), options=OPTIONS
    )
    assert len(result["result"]["skipped"]) == 1
    assert compute.call_count == 1

    saved = Project.load(path)
    saved.sources[0].fps += 1
    assert saved.save()
    result = run_description_job(
        store, path, [saved.clips[0].id], lambda *_: None, Event(), options=OPTIONS
    )
    assert len(result["result"]["succeeded"]) == 1
    assert compute.call_args.args[0] == display
    assert compute.call_count == 2


def test_failed_save_recovery_ignores_parallelism(setup):
    from dataclasses import replace

    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError):
            run(setup)
    compute.side_effect = AssertionError(
        "Scheduling must not invalidate completed computation"
    )
    result = run_description_job(
        store,
        path,
        None,
        lambda *_: None,
        Event(),
        options=replace(OPTIONS, parallelism=4),
    )
    assert len(result["result"]["succeeded"]) == 2
    assert compute.call_count == 2


def test_cli_recovery_and_headless_reuse_share_receipts(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.settings import Settings

    path, store, compute = setup
    settings = Settings(
        cache_dir=path.parent,
        description_model_cloud="gpt-test",
        description_input_mode="frame",
    )
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    monkeypatch.setattr("cli.commands.analyze.CLIConfig.load", lambda: settings)
    thumbnail = Project.load(path).clips[0].thumbnail_path
    generator = Mock()
    generator.generate_clip_thumbnail.return_value = thumbnail
    monkeypatch.setattr("core.thumbnail.ThumbnailGenerator", lambda **kw: generator)
    register_commands()
    runner = CliRunner()
    command = ["--json", "analyze", "describe", str(path), "--tier", "cloud"]
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        assert runner.invoke(cli, command).exit_code != 0
    result = runner.invoke(cli, command)
    assert result.exit_code == 0, result.output
    assert '"analyzed_clips": 2' in result.output
    assert compute.call_count == 2
    assert generator.generate_clip_thumbnail.call_args.kwargs["width"] == 640
    assert compute.call_args.kwargs["input_mode"] == "frame"
    assert len(run(setup)["skipped"]) == 2
    assert compute.call_count == 2


def test_generic_analysis_preserves_description_before_later_failure(
    setup, monkeypatch
):
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP
    from core.settings import Settings

    path, store, compute = setup
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: Settings(
            cache_dir=path.parent,
            description_model_tier="cloud",
            description_model_cloud="gpt-test",
            description_input_mode="frame",
        ),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("later failure")

    monkeypatch.setitem(ANALYZE_CLIP_OPERATION_MAP, "colors", fail)
    for _ in range(2):
        project = Project.load(path)
        operation = analysis_job_spec(
            project, arguments={"operations": ["describe", "colors"]}
        )
        with pytest.raises(RuntimeError, match="later failure"):
            run_analysis_job(store, path, operation, lambda *_: None, Event())
    assert compute.call_count == 2
    assert len(Project.load(path).metadata.job_results) == 2


def test_cli_thumbnail_dependency_failure_preserves_exit_code(setup, monkeypatch):
    from click.testing import CliRunner
    from cli.main import cli, register_commands

    path, _, compute = setup
    monkeypatch.setattr(
        "core.thumbnail.ThumbnailGenerator",
        Mock(side_effect=RuntimeError("ffmpeg missing")),
    )
    register_commands()
    result = CliRunner().invoke(
        cli, ["analyze", "describe", str(path), "--tier", "cloud"]
    )
    assert result.exit_code == 4
    compute.assert_not_called()
