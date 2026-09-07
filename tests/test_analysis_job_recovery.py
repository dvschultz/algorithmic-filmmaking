"""Cross-step persistence and submission ownership for analysis jobs."""

from threading import Event
from unittest.mock import patch

import pytest

from core.jobs.analysis import analysis_job_spec, run_analysis_job
from core.jobs.store import JobStore
from core.project import Project
from core.spine.project_io import load_with_mtime
from tests.test_spine_analyze import _build_project


def setup_job(tmp_path, operations):
    project = _build_project(tmp_path, 1)
    path = tmp_path / "project.json"
    assert project.save(path)
    project, _ = load_with_mtime(path)
    spec = analysis_job_spec(
        project, arguments={"operations": operations, "clip_ids": None}
    )
    return path, JobStore(tmp_path / "jobs.db"), spec


def test_completed_transcription_survives_later_step_failure(tmp_path):
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, spec = setup_job(tmp_path, ["transcribe", "shots"])
    with (
        patch("core.transcription.transcribe_clip", return_value=[]) as compute,
        patch.dict(
            ANALYZE_CLIP_OPERATION_MAP,
            {
                "shots": lambda *a, **k: (_ for _ in ()).throw(
                    RuntimeError("later failure")
                )
            },
        ),
    ):
        with pytest.raises(RuntimeError, match="later failure"):
            run_analysis_job(store, path, spec, lambda *_: None, Event())
        saved = Project.load(path)
        assert saved.clips[0].transcript == []
        assert len(saved.metadata.job_results) == 1
        current, _ = load_with_mtime(path)
        retry = analysis_job_spec(
            current, arguments={"operations": ["transcribe"], "clip_ids": None}
        )
        result = run_analysis_job(store, path, retry, lambda *_: None, Event())
        assert (
            len(result["result"]["operations"]["transcribe"]["result"]["skipped"]) == 1
        )
        assert compute.call_count == 1


def test_later_step_reloads_transcription_and_preserves_receipts(tmp_path):
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, spec = setup_job(tmp_path, ["transcribe", "shots"])

    def shots(project, *args, **kwargs):
        assert project.clips[0].transcript == []
        assert len(project.metadata.job_results) == 1
        project.clips[0].shot_type = "wide"
        return {"success": True}

    with (
        patch("core.transcription.transcribe_clip", return_value=[]),
        patch.dict(ANALYZE_CLIP_OPERATION_MAP, {"shots": shots}),
    ):
        run_analysis_job(store, path, spec, lambda *_: None, Event())
    saved = Project.load(path)
    assert saved.clips[0].transcript == []
    assert saved.clips[0].shot_type == "wide"
    assert len(saved.metadata.job_results) == 1


def test_media_changed_by_preceding_step_rejects_transcription(tmp_path):
    from core.jobs.commits import StaleJobResult
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, spec = setup_job(tmp_path, ["shots", "transcribe"])

    def shots(project, *args, **kwargs):
        project.sources[0].file_path.write_bytes(b"replacement")
        return {"success": True}

    with (
        patch("core.transcription.transcribe_clip") as compute,
        patch.dict(ANALYZE_CLIP_OPERATION_MAP, {"shots": shots}),
    ):
        with pytest.raises(StaleJobResult, match="inputs changed"):
            run_analysis_job(store, path, spec, lambda *_: None, Event())
        compute.assert_not_called()


def test_progress_cancellation_stops_next_step():
    from core.operations.analysis_plan import run_analysis_plan

    cancel = Event()
    with patch("builtins.print") as execute:
        result = run_analysis_plan(
            ["shots"], execute, progress=lambda *_: cancel.set(), cancel=cancel
        )
    execute.assert_not_called()
    assert result["result"]["operations"] == {}


def test_generic_analysis_preserves_existing_transcript_from_other_model(tmp_path):
    from core.jobs.transcription import run_transcription_job
    from core.operations.transcription import TranscriptionOptions

    path, store, _ = setup_job(tmp_path, ["transcribe"])
    with patch("core.transcription.transcribe_clip", return_value=[]) as compute:
        run_transcription_job(
            store,
            path,
            None,
            TranscriptionOptions(model="small.en"),
            lambda *_: None,
            Event(),
        )
        current, _ = load_with_mtime(path)
        spec = analysis_job_spec(current, arguments={"operations": ["transcribe"]})
        result = run_analysis_job(store, path, spec, lambda *_: None, Event())
        assert compute.call_count == 1
        assert (
            len(result["result"]["operations"]["transcribe"]["result"]["skipped"]) == 1
        )
