"""Durable alignment validates inputs and recovers computed/saved results."""

from threading import Event
from unittest.mock import Mock, patch

import pytest

from core.jobs.alignment import alignment_job_spec, run_alignment_job
from core.jobs.commits import StaleJobResult
from core.jobs.store import JobStore
from core.project import Project
from core.spine.project_io import load_with_mtime
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 2)
    for clip in project.clips:
        clip.transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    path = tmp_path / "project.json"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda *_: (True, [])
    )

    def extract(*args, **kwargs):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake")
        return wav

    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", extract)
    compute = Mock(return_value=[WordTimestamp(0, 1, "hello", 0.9)])
    from core.analysis.alignment import ALIGNMENT_MODEL

    monkeypatch.setattr(
        "core.operations.alignment_records.alignment_model_revision", lambda: "r1"
    )

    def provider(*args, **kwargs):
        kwargs["on_execution"](
            {"backend": "ctc", "model": ALIGNMENT_MODEL, "revision": "r1"}
        )
        return compute(*args, **kwargs)

    monkeypatch.setattr("core.analysis.alignment.align_words", provider)
    return path, store, compute


def run(setup, **kwargs):
    path, store, _ = setup
    return run_alignment_job(store, path, None, lambda *_: None, Event(), **kwargs)[
        "result"
    ]


def test_failed_save_reuses_computation_without_runtime(setup, monkeypatch):
    path, store, compute = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup)
    assert Project.load(path).clips[0].transcript[0].words is None
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda *_: (False, ["missing"])
    )
    assert len(run(setup)["succeeded"]) == 2
    assert compute.call_count == 2
    assert len(Project.load(path).metadata.job_results) == 2


def test_failed_checkpoint_reconciles_and_preserves_edited_words(setup):
    path, store, compute = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            run(setup)
    assert len(run(setup)["skipped"]) == 2
    assert compute.call_count == 2
    saved = Project.load(path)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
    saved.clips[0].transcript[0].words[0].text = "manual"
    assert saved.save()
    with pytest.raises(StaleJobResult, match="use force"):
        run(setup)
    assert Project.load(path).clips[0].transcript[0].words[0].text == "manual"


def test_force_retry_reuses_failed_save_but_next_refresh_computes(setup):
    path, store, compute = setup
    run(setup)
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup, force=True)
    run(setup, force=True)
    assert compute.call_count == 4
    run(setup, force=True)
    assert compute.call_count == 6


def test_media_replacement_during_compute_is_not_saved(setup):
    path, store, compute = setup
    media = Project.load(path).sources[0].file_path

    def replace(*args, **kwargs):
        media.write_bytes(b"changed")
        return []

    compute.side_effect = replace
    with pytest.raises(StaleJobResult, match="inputs changed"):
        run(setup)
    assert Project.load(path).clips[0].transcript[0].words is None


def test_queued_transcript_changes_are_rejected(setup):
    path, store, compute = setup
    project, _ = load_with_mtime(path)
    spec = alignment_job_spec(project, None, force=False, arguments={})
    project.clips[0].transcript[0].text = "changed"
    assert project.save()
    from core.project_revision import ProjectRevisionConflict

    with pytest.raises(ProjectRevisionConflict):
        run(setup, operation=spec)
    compute.assert_not_called()


def test_cancellation_saves_completed_targets_for_retry(setup):
    path, store, compute = setup
    cancel = Event()
    result = run_alignment_job(store, path, None, lambda *_: cancel.set(), cancel)[
        "result"
    ]
    assert len(result["succeeded"]) == 1
    assert len(result["unprocessed"]) == 1
    saved = Project.load(path)
    assert saved.clips[0].transcript[0].words is not None
    assert saved.clips[1].transcript[0].words is None
    result = run(setup)
    assert len(result["skipped"]) == 1
    assert len(result["succeeded"]) == 1
    assert compute.call_count == 2


def test_empty_alignment_result_is_reused(setup):
    path, store, compute = setup
    compute.return_value = []
    assert len(run(setup)["succeeded"]) == 2
    assert len(run(setup)["skipped"]) == 2
    assert compute.call_count == 2
    assert Project.load(path).clips[0].transcript[0].words == []


def test_reconciles_all_identical_refresh_receipts_after_checkpoint_failures(setup):
    path, store, compute = setup
    run(setup)
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint failed")
    ):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="checkpoint failed"):
                run(setup, force=True)
    run(setup)
    assert compute.call_count == 4
    saved = Project.load(path)
    assert len(saved.metadata.job_results) == 4
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)
