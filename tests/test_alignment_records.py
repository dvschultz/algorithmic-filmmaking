"""Verified alignment preserves editorial input and authenticates reuse."""

from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import pytest

from core.analysis.alignment import ALIGNMENT_MODEL
from core.operations.alignment import (
    AlignmentApplication,
    run_alignment,
    snapshot_alignment_tasks,
)
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 1)
    clip = project.clips[0]
    clip.transcript = [TranscriptSegment(0, 1, "hello world", language="en")]
    monkeypatch.setattr(
        "core.operations.alignment_records.alignment_model_revision", lambda: "r1"
    )

    def extract(*a, **kw):
        path = tmp_path / "alignment.wav"
        path.write_bytes(b"audio")
        return path

    def align(*a, **kw):
        kw["on_execution"](
            {
                "backend": "ctc",
                "model": ALIGNMENT_MODEL,
                "revision": "r1",
                "scope": "whole_clip",
            }
        )
        return [
            WordTimestamp(0, 0.5, "hello", 0.9),
            WordTimestamp(0.5, 1, "world", 0.9),
        ]

    provider = Mock(side_effect=align)
    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", extract)
    monkeypatch.setattr("core.analysis.alignment.align_words", provider)
    return project, provider


def tasks(project, **kw):
    return snapshot_alignment_tasks(
        project.clips, project.sources_by_id, verified=True, **kw
    )


def execute(project, **kw):
    submitted = tasks(project, **kw)
    application = AlignmentApplication(project, submitted)
    outcome = run_alignment(submitted)[0]
    assert application.apply(project, outcome), outcome
    return outcome


def test_verified_alignment_reuses_after_save(setup, tmp_path):
    from core.project import Project

    project, provider = setup
    assert execute(project).status == "succeeded"
    project.save(tmp_path / "project.json")
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        assert execute(reopened).status == "skipped"
        assert provider.call_count == 1
        assert reopened.clips[0].transcript[0].words[0].text == "hello"
    finally:
        reopened.close_writer()


@pytest.mark.parametrize(
    "change", ["legacy", "text", "language", "media", "revision", "words"]
)
def test_changed_semantics_recompute(setup, monkeypatch, change):
    project, provider = setup
    execute(project)
    clip = project.clips[0]
    if change == "legacy":
        clip.analysis_records.pop("align_words")
    elif change == "text":
        clip.transcript[0].text = "edited"
    elif change == "language":
        clip.transcript[0].language = "es"
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed media")
    elif change == "revision":
        monkeypatch.setattr(
            "core.operations.alignment_records.alignment_model_revision", lambda: "r2"
        )
        provider.side_effect = (
            lambda *a, **kw: kw["on_execution"](
                {"backend": "ctc", "model": ALIGNMENT_MODEL, "revision": "r2"}
            )
            or []
        )
    else:
        clip.transcript[0].words = []
    assert execute(project).status == "succeeded"
    assert provider.call_count == 2


def test_failure_preserves_prior_words(setup):
    project, provider = setup
    execute(project)
    previous = project.clips[0].transcript[0].to_dict()
    provider.side_effect = RuntimeError("provider unavailable")
    assert execute(project, skip_existing=False).status == "failed"
    assert project.clips[0].transcript[0].to_dict() == previous
    assert project.clips[0].analysis_records["align_words"].state == "failed"


@pytest.mark.parametrize("change", ["transcript", "record", "media", "session"])
def test_late_delivery_is_rejected(setup, change):
    project, _ = setup
    submitted = tasks(project)
    application = AlignmentApplication(project, submitted)
    outcome = run_alignment(submitted)[0]
    if change == "transcript":
        project.clips[0].transcript[0].text = "edited"
    elif change == "record":
        from models.analysis_record import AnalysisRecord

        project.clips[0].analysis_records["align_words"] = AnalysisRecord.legacy({})
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    else:
        project.clear()
    assert not application.apply(project, outcome)


def test_cancelled_alignment_never_publishes_record(setup):
    project, provider = setup
    cancel = Event()
    provider.side_effect = lambda *a, **kw: cancel.set() or []
    submitted = tasks(project)
    outcome = run_alignment(submitted, cancel_event=cancel)[0]
    assert outcome.status == "unprocessed"
    assert outcome.record_json is None
    assert not AlignmentApplication(project, submitted).apply(project, outcome)


def test_invalid_word_probability_is_failed_record(setup):
    project, provider = setup
    original = provider.side_effect
    provider.side_effect = lambda *a, **kw: [
        replace(original(*a, **kw)[0], probability=2)
    ]
    assert execute(project).status == "failed"
    assert project.clips[0].transcript[0].words is None


def test_verified_empty_word_result_reuses(setup):
    project, provider = setup
    original = provider.side_effect
    provider.side_effect = lambda *a, **kw: original(*a, **kw) and []
    assert execute(project).status == "succeeded"
    assert project.clips[0].transcript[0].words == []
    assert execute(project).status == "skipped"
    assert provider.call_count == 1


def test_unknown_model_revision_cannot_claim_verified_success(setup, monkeypatch):
    project, provider = setup
    monkeypatch.setattr(
        "core.operations.alignment_records.alignment_model_revision", lambda: None
    )
    provider.side_effect = (
        lambda *a, **kw: kw["on_execution"](
            {"backend": "ctc", "model": ALIGNMENT_MODEL, "revision": None}
        )
        or []
    )
    assert execute(project).status == "failed"
    assert project.clips[0].transcript[0].words is None


def test_observers_receive_matching_alignment_record(setup, monkeypatch):
    project, _ = setup
    seen = []
    original = project.update_clips

    def observe(clips):
        clip = clips[0]
        seen.append(
            clip.analysis_records["align_words"].value
            == {"transcript": [s.to_dict() for s in clip.transcript]}
        )
        return original(clips)

    monkeypatch.setattr(project, "update_clips", observe)
    execute(project)
    assert seen == [True]


def test_altered_reuse_words_are_rejected(setup):
    project, _ = setup
    execute(project)
    submitted = tasks(project)
    application = AlignmentApplication(project, submitted)
    outcome = run_alignment(submitted)[0]
    assert outcome.status == "skipped"
    assert not application.apply(project, replace(outcome, words=()))


def test_raw_publication_clears_prior_alignment_verification(setup, monkeypatch):
    from core.operations.alignment import AlignmentOutcome

    project, _ = setup
    execute(project)
    submitted = snapshot_alignment_tasks(
        project.clips, project.sources_by_id, skip_existing=False
    )
    words = tuple(project.clips[0].transcript[0].words)
    seen = []
    original = project.update_clips

    def observe(clips):
        record = clips[0].analysis_records["align_words"]
        seen.append((record.provenance, record.value))
        return original(clips)

    monkeypatch.setattr(project, "update_clips", observe)
    assert AlignmentApplication(project, submitted).apply(
        project, AlignmentOutcome(project.clips[0].id, "succeeded", words)
    )
    assert project.clips[0].analysis_records["align_words"].provenance == "unknown"
    assert seen == [("unknown", {"transcript": [s.to_dict() for s in project.clips[0].transcript]})]


@pytest.mark.parametrize("change", ["record", "save_as"])
def test_raw_publication_rejects_changed_owner_binding(setup, tmp_path, change):
    from core.operations.alignment import AlignmentOutcome

    project, _ = setup
    execute(project)
    submitted = snapshot_alignment_tasks(project.clips, project.sources_by_id, skip_existing=False)
    application = AlignmentApplication(project, submitted)
    clip = project.clips[0]
    words = tuple(clip.transcript[0].words)
    if change == "record":
        clip.analysis_records["align_words"] = replace(clip.analysis_records["align_words"], state="failed")
    else:
        project.save(tmp_path / "different.sceneripper")
    assert not application.apply(project, AlignmentOutcome(clip.id, "succeeded", words))


@pytest.mark.parametrize("failed", [False, True])
def test_transcription_replacement_invalidates_alignment_before_notification(setup, monkeypatch, failed):
    from core.operations.transcription import TranscriptionApplication, TranscriptionOptions, run_transcription
    from core.operations.transcription_records import transcription_task

    project, _ = setup
    execute(project)
    clip, source = project.clips[0], project.sources[0]
    prior_record = clip.analysis_records["align_words"]
    monkeypatch.setattr("core.transcription._has_audio_stream", lambda _: True)
    monkeypatch.setattr("core.transcription.transcribe_clip", Mock(side_effect=RuntimeError("provider failed")) if failed else Mock(return_value=[]))
    task = transcription_task(clip, source, skip_existing=False)
    options = TranscriptionOptions(backend="faster-whisper")
    application = TranscriptionApplication(project, (task,), options)
    seen = []
    original = project.update_clips

    def observe(clips):
        seen.append(clips[0].analysis_records["align_words"])
        return original(clips)

    monkeypatch.setattr(project, "update_clips", observe)
    outcome = run_transcription((task,), options)[0]
    assert application.apply(project, outcome)
    if failed:
        assert clip.analysis_records["align_words"] == prior_record
        assert not seen
    else:
        assert clip.transcript == []
        assert len(seen) == 1
        assert seen[0].provenance == "unknown"
        assert seen[0].value == {"transcript": []}
