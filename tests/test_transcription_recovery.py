from threading import Event
from unittest.mock import patch

from core.jobs.store import JobStore
from core.jobs.transcription import run_transcription_job
from core.operations.transcription import TranscriptionOptions
from tests.test_spine_analyze import _build_project


def test_cli_retry_then_sync_mcp_refresh_share_result_cache(tmp_path):
    import json
    from types import SimpleNamespace
    from click.testing import CliRunner
    from cli.main import cli, register_commands
    from core.project import Project
    from scene_ripper_mcp.tools.analyze import _transcribe_sync

    project = _build_project(tmp_path, 1)
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
        with patch(
            "core.jobs.commits.save_with_mtime_check",
            side_effect=RuntimeError("save failed"),
        ):
            failed = CliRunner().invoke(cli, ["transcribe", str(path)])
            assert failed.exit_code == 1
        result = CliRunner().invoke(cli, ["--json", "transcribe", str(path)])
        assert result.exit_code == 0, result.output
        assert compute.call_count == 1
        assert len(Project.load(path).metadata.job_results) == 1
        with patch(
            "core.jobs.commits.save_with_mtime_check",
            side_effect=RuntimeError("save failed"),
        ):
            assert (
                json.loads(_transcribe_sync(path, "small.en", "en"))["success"] is False
            )
        result = json.loads(_transcribe_sync(path, "small.en", "en"))
        assert result["transcribed_clips"] == 1
        assert compute.call_count == 2
        assert len(Project.load(path).metadata.job_results) == 2


def test_forced_runs_recover_failed_save_and_allow_subsequent_refresh(tmp_path):
    import pytest
    from core.project import Project
    from core.transcription_models import TranscriptSegment

    project = _build_project(tmp_path, 1)
    project.clips[0].transcript = [TranscriptSegment(0, 1, "manual")]
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")

    def run():
        return run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(backend="faster-whisper"),
            lambda *_: None,
            Event(),
            force=True,
        )

    with patch("core.transcription.transcribe_clip", return_value=[]) as compute:
        with patch(
            "core.jobs.commits.save_with_mtime_check",
            side_effect=RuntimeError("save failed"),
        ):
            with pytest.raises(RuntimeError, match="save failed"):
                run()
        assert Project.load(path).clips[0].transcript[0].text == "manual"
        assert len(run()["result"]["succeeded"]) == 1
        assert compute.call_count == 1
        assert len(run()["result"]["succeeded"]) == 1
        assert compute.call_count == 2
    assert len(Project.load(path).metadata.job_results) == 2


def test_submission_is_checked_after_acquiring_writer_lease(tmp_path):
    import pytest
    from contextlib import contextmanager
    from core.jobs.commits import result_batch
    from core.jobs.transcription import transcription_job_spec
    from core.project import Project
    from core.project_revision import ProjectRevisionConflict
    from core.spine.project_io import load_with_mtime

    project = _build_project(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    options = TranscriptionOptions(backend="faster-whisper")
    submitted, _ = load_with_mtime(path)
    operation = transcription_job_spec(submitted, None, options, arguments={})

    @contextmanager
    def changed_before_lease(*args, **kwargs):
        changed = Project.load(path)
        changed.clips[0].end_frame += 1
        assert changed.save()
        with result_batch(*args, **kwargs) as batch:
            yield batch

    with patch("core.jobs.transcription.result_batch", changed_before_lease):
        with patch("core.transcription.transcribe_clip") as compute:
            with pytest.raises(ProjectRevisionConflict):
                run_transcription_job(
                    JobStore(tmp_path / "jobs.db"),
                    path,
                    None,
                    options,
                    lambda *_: None,
                    Event(),
                    operation=operation,
                )
            compute.assert_not_called()


def test_recovers_after_project_saved_before_checkpoint(tmp_path):
    import pytest

    project = _build_project(tmp_path, 2)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")

    def run():
        return run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(backend="faster-whisper"),
            lambda *_: None,
            Event(),
        )

    with patch("core.transcription.transcribe_clip", return_value=[]) as compute:
        with patch.object(
            store, "checkpoint_results", side_effect=RuntimeError("crash")
        ):
            with pytest.raises(RuntimeError, match="crash"):
                run()
        result = run()
        assert compute.call_count == 2
        assert len(result["result"]["skipped"]) == 2


def test_recovers_computation_after_failed_save_and_rejects_edited_output(tmp_path):
    import pytest
    from core.project import Project
    from core.jobs.commits import StaleJobResult
    from core.transcription_models import TranscriptSegment

    project = _build_project(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")

    def run():
        return run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(backend="faster-whisper"),
            lambda *_: None,
            Event(),
        )

    with patch("core.transcription.transcribe_clip", return_value=[]) as compute:
        with patch(
            "core.jobs.commits.save_with_mtime_check",
            side_effect=RuntimeError("save failed"),
        ):
            with pytest.raises(RuntimeError, match="save failed"):
                run()
        assert Project.load(path).clips[0].transcript is None
        assert len(run()["result"]["succeeded"]) == 1
        assert compute.call_count == 1
        edited = Project.load(path)
        edited.clips[0].transcript = [TranscriptSegment(0, 1, "manual")]
        assert edited.save()
        with pytest.raises(StaleJobResult, match="output changed"):
            run()
        with pytest.raises(StaleJobResult, match="output changed"):
            run_transcription_job(
                store,
                path,
                None,
                TranscriptionOptions(model="large-v3", backend="faster-whisper"),
                lambda *_: None,
                Event(),
            )
        assert compute.call_count == 1


def test_cancellation_saves_completed_targets_for_retry(tmp_path):
    project = _build_project(tmp_path, 3)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    cancel = Event()
    with patch("core.transcription.transcribe_clip", return_value=[]) as compute:
        result = run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(backend="faster-whisper"),
            lambda *_: cancel.set(),
            cancel,
        )
        assert len(result["result"]["succeeded"]) == 1
        assert len(result["result"]["unprocessed"]) == 2
        result = run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(backend="faster-whisper"),
            lambda *_: None,
            Event(),
        )
        assert len(result["result"]["skipped"]) == 1
        assert len(result["result"]["succeeded"]) == 2
        assert compute.call_count == 3


def test_media_replacement_during_compute_is_not_published(tmp_path):
    import os
    import pytest
    from core.project import Project
    from core.jobs.commits import StaleJobResult

    project = _build_project(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    media = project.sources[0].file_path
    stamp = media.stat()

    def compute(**kwargs):
        media.write_bytes(
            b"edit"
        )  # Same size and restored mtime still change ctime/hash.
        os.utime(media, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return []

    with patch("core.transcription.transcribe_clip", side_effect=compute):
        with pytest.raises(StaleJobResult, match="inputs changed"):
            run_transcription_job(
                JobStore(tmp_path / "jobs.db"),
                path,
                None,
                TranscriptionOptions(backend="faster-whisper"),
                lambda *_: None,
                Event(),
            )
    assert Project.load(path).clips[0].transcript is None


def test_retry_new_model_after_save_failure_reuses_pending_output(tmp_path):
    import pytest
    from core.project import Project
    from core.transcription_models import TranscriptSegment, WordTimestamp

    project = _build_project(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")

    def run(model):
        return run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(model=model, backend="faster-whisper"),
            lambda *_: None,
            Event(),
        )

    with patch("core.transcription.transcribe_clip", return_value=[]):
        run("small.en")
    segment = TranscriptSegment(
        0, 1, "hello", 0.9, [WordTimestamp(0, 1, "hello", 0.8)], "en"
    )
    with patch("core.transcription.transcribe_clip", return_value=[segment]) as compute:
        with patch(
            "core.jobs.commits.save_with_mtime_check",
            side_effect=RuntimeError("save failed"),
        ):
            with pytest.raises(RuntimeError, match="save failed"):
                run("base")
        assert len(run("base")["result"]["succeeded"]) == 1
        assert len(run("base")["result"]["skipped"]) == 1
        assert compute.call_count == 1
    assert Project.load(path).clips[0].transcript[0].to_dict() == segment.to_dict()
