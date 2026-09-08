from threading import Event
from unittest.mock import patch

from core.operations.transcription import (
    TranscriptionOptions,
    TranscriptionOutcome,
    TranscriptionTask,
    run_transcription,
)


def test_cancel_stops_unscheduled_tasks_and_preserves_order():
    tasks = tuple(TranscriptionTask(str(i), None, 0, 1, 30) for i in range(4))
    cancel = Event()
    seen = []

    def completed(outcome):
        seen.append(outcome.clip_id)
        cancel.set()

    with patch(
        "core.operations.transcription.compute_task",
        side_effect=lambda task, _, **kwargs: TranscriptionOutcome(task.clip_id, "succeeded"),
    ):
        outcomes = run_transcription(
            tasks,
            TranscriptionOptions(backend="faster-whisper"),
            cancel_event=cancel,
            on_outcome=completed,
        )
    assert seen == ["0"]
    assert [outcome.status for outcome in outcomes] == [
        "succeeded",
        "unprocessed",
        "unprocessed",
        "unprocessed",
    ]


def test_critical_failure_aborts_remaining_tasks():
    tasks = tuple(TranscriptionTask(str(i), None, 0, 1, 30) for i in range(3))
    with patch(
        "core.operations.transcription.compute_task",
        return_value=TranscriptionOutcome("0", "failed", critical=True),
    ) as compute:
        result = run_transcription(
            tasks, TranscriptionOptions(backend="faster-whisper")
        )
    assert compute.call_count == 1
    assert [outcome.status for outcome in result] == [
        "failed",
        "unprocessed",
        "unprocessed",
    ]


def test_snapshot_and_empty_transcript_semantics(tmp_path):
    from core.operations.transcription import snapshot_tasks
    from tests.test_spine_analyze import _build_project
    from core.spine.analyze import transcribe

    project = _build_project(tmp_path, 2)
    tasks = snapshot_tasks(project.clips, project.sources_by_id)
    project.clips[0].start_frame = 33
    assert tasks[0].start_time == 0
    with patch("core.transcription.transcribe_clip", return_value=[]) as compute:
        result = transcribe(project)
        assert len(result["result"]["succeeded"]) == 2
        assert all(clip.transcript == [] for clip in project.clips)
        result = transcribe(project)
        assert len(result["result"]["skipped"]) == 2
        assert compute.call_count == 2


def test_mlx_parallelism_is_serial_and_callbacks_stay_on_caller():
    import threading
    import time

    caller = threading.get_ident()
    active = 0
    maximum = 0
    lock = threading.Lock()

    def compute(task, options, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return TranscriptionOutcome(task.clip_id, "succeeded")

    tasks = tuple(TranscriptionTask(str(i), None, 0, 1, 30) for i in range(6))
    threads = []
    with (
        patch("core.transcription._resolve_backend", return_value="mlx-whisper"),
        patch("core.operations.transcription.compute_task", side_effect=compute),
    ):
        run_transcription(
            tasks,
            TranscriptionOptions(parallelism=4),
            on_outcome=lambda _: threads.append(threading.get_ident()),
        )
    assert maximum == 1
    assert threads == [caller] * 6


def test_cli_and_sync_mcp_use_shared_batch_and_persist_silence(tmp_path):
    import json
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.project import Project
    from scene_ripper_mcp.tools.analyze import _transcribe_sync
    from tests.test_spine_analyze import _build_project
    from types import SimpleNamespace

    project = _build_project(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    register_commands()
    with (
        patch(
            "core.settings.load_settings",
            return_value=SimpleNamespace(cache_dir=tmp_path),
        ),
        patch("core.transcription.transcribe_clip", return_value=[]) as compute,
    ):
        result = CliRunner().invoke(cli, ["--json", "transcribe", str(path)])
        assert result.exit_code == 0, result.output
        assert compute.call_count == 2
        assert all(clip.transcript == [] for clip in Project.load(path).clips)
        assert len(Project.load(path).metadata.job_results) == 2
        response = json.loads(_transcribe_sync(path, "small.en", "en"))
        assert response["success"] and response["transcribed_clips"] == 2
        assert response["skipped_clips"] == 0
        assert compute.call_count == 4
        assert len(Project.load(path).metadata.job_results) == 4


def test_headless_results_reject_project_replacement(tmp_path):
    from tests.test_spine_analyze import _build_project
    from core.spine.analyze import transcribe

    project = _build_project(tmp_path, 1)
    original = project.clips[0]
    with patch("core.transcription.transcribe_clip", return_value=[]):
        result = transcribe(project, progress_callback=lambda *_: project.clear())
    assert original.transcript is None
    assert result["result"]["failed"][0]["code"] == "stale_target"


def test_critical_failure_preserves_running_success():
    from threading import Barrier

    gate = Barrier(2)

    def compute(task, options, **kwargs):
        gate.wait(timeout=3)
        return TranscriptionOutcome(
            task.clip_id,
            "failed" if task.clip_id == "0" else "succeeded",
            critical=task.clip_id == "0",
        )

    tasks = tuple(TranscriptionTask(str(i), None, 0, 1, 30) for i in range(2))
    with patch("core.operations.transcription.compute_task", side_effect=compute):
        result = run_transcription(
            tasks, TranscriptionOptions(backend="faster-whisper", parallelism=2)
        )
    assert result[0].status == "failed"
    assert result[1].status == "succeeded"


def test_invalid_frame_rate_is_per_clip_failure(tmp_path):
    from tests.test_spine_analyze import _build_project
    from core.spine.analyze import transcribe

    project = _build_project(tmp_path, 2)
    project.sources[0].fps = 0
    result = transcribe(project)
    assert len(result["result"]["failed"]) == 2
    assert all(item["code"] == "invalid_target" for item in result["result"]["failed"])


def test_cli_and_mcp_preserve_dependency_failure_envelopes(tmp_path):
    import json
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.transcription_models import FasterWhisperNotInstalledError
    from scene_ripper_mcp.tools.analyze import _transcribe_sync
    from tests.test_spine_analyze import _build_project
    from types import SimpleNamespace

    project = _build_project(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    register_commands()
    with (
        patch(
            "core.settings.load_settings",
            return_value=SimpleNamespace(cache_dir=tmp_path),
        ),
        patch(
            "core.transcription.transcribe_clip",
            side_effect=FasterWhisperNotInstalledError(),
        ),
    ):
        result = CliRunner().invoke(cli, ["transcribe", str(path)])
        assert result.exit_code == 4, result.output
        result = json.loads(_transcribe_sync(path, "small.en", "en"))
        assert result["success"] is False
        assert "faster-whisper" in result["error"]
