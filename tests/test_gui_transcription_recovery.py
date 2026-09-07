"""Saved GUI transcription records results without saving unrelated edits."""

from dataclasses import replace
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.gui_transcription import GuiTranscriptionCache
from core.jobs.commits import StaleJobResult
from core.jobs.media import media_stamp
from core.operations.transcription import (
    TranscriptionApplication,
    TranscriptionOptions,
    snapshot_tasks,
)
from core.project import Project
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project


OPTIONS = TranscriptionOptions(backend="faster-whisper", parallelism=2)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 2)
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    compute = Mock(
        return_value=[
            TranscriptSegment(
                0, 1, "hello", 0.8, [WordTimestamp(0, 1, "hello", 0.9)], "en"
            )
        ]
    )
    monkeypatch.setattr("core.transcription.transcribe_clip", compute)
    return project, compute


def run(
    project,
    *,
    options=OPTIONS,
    prepare=lambda: True,
    clips=None,
    apply=False,
    cancel=None,
):
    clips = project.clips if clips is None else clips
    tasks = snapshot_tasks(clips, project.sources_by_id, skip_existing=False)
    cache = GuiTranscriptionCache(
        project.path,
        project.metadata.id,
        {c.id: c.source_id for c in clips},
        project.metadata.job_results,
        options=options,
        previous_transcripts={
            c.id: [s.to_dict() for s in c.transcript]
            if c.transcript is not None
            else None
            for c in clips
        },
        media_stamps={s.file_path: media_stamp(s.file_path) for s in project.sources},
    )
    application = TranscriptionApplication(project, tasks)

    def deliver(outcome):
        if apply and outcome.status == "succeeded":
            assert application.apply(project, outcome)
            receipt = cache.results[outcome.clip_id]
            assert receipt.matches(outcome)
            project.record_job_result(receipt.result_id, receipt.digest)

    return cache.run(tasks, cancel or Event(), prepare, deliver, lambda *_: None)


def test_restart_reuses_complete_transcripts_without_preflight(setup):
    project, compute = setup
    first = run(project)
    prepare = Mock(side_effect=AssertionError("cache hit must skip preparation"))
    reopened = Project.load(project.path)
    assert run(reopened, prepare=prepare, apply=True) == first
    assert compute.call_count == 2
    assert len(reopened.metadata.job_results) == 2
    assert Project.load(project.path).clips[0].transcript is None
    assert reopened.save()
    assert Project.load(project.path).clips[0].transcript[0].words[0].text == "hello"


def test_failed_save_recovers_recorded_results(setup, monkeypatch):
    project, compute = setup
    run(project, apply=True)
    with monkeypatch.context() as patch:
        patch.setattr("core.project.save_project", lambda **_: False)
        assert not project.save()
    reopened = Project.load(project.path)
    run(reopened, apply=True)
    assert compute.call_count == 2
    assert reopened.save()


def test_mixed_cache_and_missing_work_computes_only_missing_target(setup):
    project, compute = setup
    run(project, clips=project.clips[:1])
    prepare = Mock(return_value=True)
    outcomes = run(project, prepare=prepare)
    assert [o.clip_id for o in outcomes] == [c.id for c in project.clips]
    assert all(o.status == "succeeded" for o in outcomes)
    assert compute.call_count == 2
    prepare.assert_called_once()


def test_cache_misses_retain_parallel_inference(setup):
    project, compute = setup
    barrier = Barrier(2)

    def concurrent(**kwargs):
        barrier.wait(timeout=5)
        return []

    compute.side_effect = concurrent
    assert all(o.status == "succeeded" for o in run(project))
    assert compute.call_count == 2


def test_identical_successful_refreshes_advance_receipt_generation(setup):
    project, compute = setup
    compute.return_value = []
    for _ in range(3):
        run(project, apply=True)
        assert project.save()
    assert compute.call_count == 6
    assert len(project.metadata.job_results) == 6


def test_model_change_does_not_reuse_other_model_results(setup):
    project, compute = setup
    run(project)
    run(project, options=replace(OPTIONS, model="medium"))
    assert compute.call_count == 4


def test_cancelled_cache_run_does_not_prepare_or_publish(setup):
    project, compute = setup
    run(project)
    cancel = Event()
    cancel.set()
    prepare = Mock()
    outcomes = run(project, cancel=cancel, prepare=prepare, apply=True)
    assert all(o.status == "unprocessed" for o in outcomes)
    assert not project.metadata.job_results
    assert compute.call_count == 2
    prepare.assert_not_called()


def test_media_changed_during_preflight_prevents_inference(setup):
    project, compute = setup

    def prepare():
        project.sources[0].file_path.write_bytes(b"changed")
        return True

    with pytest.raises(StaleJobResult, match="media changed"):
        run(project, prepare=prepare)
    compute.assert_not_called()


def test_journal_rejects_failed_output_before_recording(setup):
    from core.jobs.gui_results import GuiResultJournal
    from core.operations.transcription import TranscriptionOutcome

    project, _ = setup
    journal = GuiResultJournal(
        project.path,
        project.metadata.id,
        {c.id: c.source_id for c in project.clips},
        {},
        kind="gui_transcribe",
        arguments={},
        media_stamps={s.file_path: media_stamp(s.file_path) for s in project.sources},
    )
    journal.start(Event())
    request, _ = journal.prepare(project.clips[0].id, {}, project.sources[0].file_path)
    with pytest.raises(StaleJobResult, match="target"):
        journal.record(request, TranscriptionOutcome(project.clips[0].id, "failed"))
    assert journal.store.get_result(request.spec.result_id) is None
