"""Shared shot execution, cancellation, and owner publication contracts."""

from dataclasses import replace
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from core.operations.shots import (
    ShotTypeApplication,
    ShotTypeOptions,
    ShotTypeOutcome,
    ShotTypeTask,
    run_shot_types,
)
from core.project import Project
from models.clip import Clip, Source
from models.frame import Frame


@pytest.fixture
def sample(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    project = Project.new()
    source = Source(id="source", file_path=video, fps=24.0, duration_seconds=10.0)
    project.add_source(source)
    clip = Clip(
        id="shared",
        source_id=source.id,
        start_frame=0,
        end_frame=24,
        thumbnail_path=image,
    )
    project.add_clips([clip])
    frame = Frame(id="shared", file_path=image)
    project.add_frames([frame])
    return project, ShotTypeTask(clip.id, image, video, 0, 24, 24.0), frame


def test_partial_errors_skip_and_missing_media(sample, monkeypatch):
    _, task, _ = sample
    compute = Mock(side_effect=[("wide", 0.9), RuntimeError("unavailable")])
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    tasks = (
        task,
        replace(task, clip_id="skip", skip=True),
        replace(task, clip_id="missing", thumbnail_path=None),
        replace(task, clip_id="failed"),
    )
    outcomes = run_shot_types(tasks, ShotTypeOptions())
    assert [item.status for item in outcomes] == [
        "succeeded",
        "skipped",
        "failed",
        "failed",
    ]
    assert outcomes[2].code == "thumbnail_missing"
    assert outcomes[3].message == "unavailable"
    assert compute.call_count == 2


@pytest.mark.parametrize(
    "raw",
    [
        ("unknown", 0.9),
        ("", 0.9),
        ("wide", float("nan")),
        ("wide", 1.1),
        ("wide", True),
        (None, 0.8),
    ],
)
def test_invalid_provider_output_never_publishes(sample, monkeypatch, raw):
    project, task, _ = sample
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", lambda _: raw)
    (outcome,) = run_shot_types((task,), ShotTypeOptions())
    assert outcome.status == "failed"
    assert not ShotTypeApplication(project, (task,)).apply(project, outcome)
    assert project.clips[0].shot_type is None


def test_cancel_during_provider_and_before_next_task(sample, monkeypatch):
    _, task, _ = sample
    cancel = Event()

    def compute(_):
        cancel.set()
        return "wide", 0.9

    provider = Mock(side_effect=compute)
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", provider)
    delivered = []
    outcomes = run_shot_types(
        (task, replace(task, clip_id="second")),
        ShotTypeOptions(),
        cancel_event=cancel,
        on_outcome=delivered.append,
    )
    assert [x.status for x in outcomes] == ["unprocessed", "unprocessed"]
    assert not delivered
    assert provider.call_count == 1


@pytest.mark.parametrize(
    "change", ["image", "video", "trim", "existing", "replacement", "path"]
)
def test_owner_rejects_changed_input(sample, tmp_path, change):
    project, task, _ = sample
    application = ShotTypeApplication(project, (task,))
    if change == "image":
        task.thumbnail_path.write_bytes(b"changed")
    elif change == "video":
        task.source_path.write_bytes(b"changed")
    elif change == "trim":
        project.clips[0].end_frame += 1
    elif change == "existing":
        project.clips[0].shot_type = "close-up"
    elif change == "replacement":
        project.clips_by_id[task.clip_id] = replace(project.clips[0])
    else:
        project.path = tmp_path / "new.json"
    assert not application.apply(
        project, ShotTypeOutcome(task.clip_id, "succeeded", "wide", 0.9)
    )


def test_clip_frame_id_collision_and_duplicate_delivery(sample):
    project, clip_task, frame = sample
    frame_task = ShotTypeTask(frame.id, frame.file_path, target_type="frame")
    application = ShotTypeApplication(project, (clip_task, frame_task))
    result = ShotTypeOutcome(frame.id, "succeeded", "wide", 0.9, target_type="frame")
    assert application.apply(project, result)
    assert frame.shot_type == "wide"
    assert project.clips[0].shot_type is None
    assert not application.apply(project, result)
    assert application.apply(
        project, ShotTypeOutcome(clip_task.clip_id, "succeeded", "close-up", 0.9)
    )
    assert project.clips[0].shot_type == "close-up"


def test_queued_media_change_does_not_call_provider(sample, monkeypatch):
    _, task, _ = sample
    compute = Mock()
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", compute)
    task.thumbnail_path.write_bytes(b"new image")
    (result,) = run_shot_types((task,), ShotTypeOptions())
    assert result.code == "stale_input"
    compute.assert_not_called()


def test_serial_model_admission_is_cancellable(sample, monkeypatch):
    _, task, _ = sample
    admitted, release, cancel = Event(), Event(), Event()

    def compute(_):
        admitted.set()
        assert release.wait(5)
        return "wide", 0.9

    provider = Mock(side_effect=compute)
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", provider)
    running = Thread(target=lambda: run_shot_types((task,), ShotTypeOptions()))
    running.start()
    try:
        assert admitted.wait(5)
        cancel.set()
        (result,) = run_shot_types((task,), ShotTypeOptions(), cancel_event=cancel)
        assert result.status == "unprocessed"
        assert provider.call_count == 1
    finally:
        release.set()
        running.join(5)
    assert not running.is_alive()


def test_cloud_uses_captured_settings_without_local_preload(sample, monkeypatch):
    _, task, _ = sample
    cloud = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots_cloud.classify_shot_cloud", cloud)
    monkeypatch.setattr(
        "core.settings.load_settings", Mock(side_effect=AssertionError("live settings"))
    )
    monkeypatch.setattr(
        "core.analysis.shots.load_classification_model",
        Mock(side_effect=AssertionError("local preload")),
    )
    (result,) = run_shot_types((task,), ShotTypeOptions("cloud", "captured-model"))
    assert result.status == "succeeded"
    assert cloud.call_args.kwargs["model"] == "captured-model"


def test_cli_uses_shared_operation_and_preserves_saved_results(
    sample, tmp_path, monkeypatch
):
    import json
    import inspect
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from types import SimpleNamespace

    monkeypatch.setattr(
        "cli.commands.analyze.CLIConfig.load",
        lambda: SimpleNamespace(cache_dir=tmp_path / "cache"),
    )

    register_commands()

    project, task, _ = sample
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    project.close_writer()
    thumbnail = Mock(return_value=task.thumbnail_path)
    monkeypatch.setattr(
        "core.thumbnail.ThumbnailGenerator.generate_clip_thumbnail", thumbnail
    )
    provider = Mock(return_value=("wide", 0.9))
    monkeypatch.setattr("core.analysis.shots.classify_shot_type", provider)
    runner = CliRunner(
        **(
            {"mix_stderr": False}
            if "mix_stderr" in inspect.signature(CliRunner).parameters
            else {}
        )
    )
    result = runner.invoke(cli, ["--json", "analyze", "shots", str(path)])
    assert result.exit_code == 0, (result.stdout, result.stderr, result.exception)
    assert json.loads(result.stdout)["shot_types"] == {"wide": 1}
    assert json.loads(path.read_text())["clips"][0]["shot_type"] == "wide"
    provider.assert_called_once_with(task.thumbnail_path)
    result = runner.invoke(cli, ["--json", "analyze", "shots", str(path)])
    assert result.exit_code == 0, result.output
    assert provider.call_count == 1
