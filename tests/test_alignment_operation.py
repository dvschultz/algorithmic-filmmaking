"""Alignment snapshots and owner-thread application protect editable transcripts."""

from threading import Event
from unittest.mock import patch

from core.operations.alignment import (
    AlignmentApplication,
    run_alignment,
    snapshot_alignment_tasks,
)
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project


def test_alignment_uses_snapshot_and_rejects_edited_transcript(tmp_path):
    project = _build_project(tmp_path, 1)
    clip = project.clips[0]
    clip.transcript = [TranscriptSegment(0, 1, "original", language="en")]
    tasks = snapshot_alignment_tasks(project.clips, project.sources_by_id)
    application = AlignmentApplication(project, tasks)
    clip.transcript[0].text = "edited"
    clip.start_frame = 300
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")

    def compute(path, segments, **kwargs):
        assert segments[0].text == "original"
        segments[0].text = "provider mutation"
        return [WordTimestamp(0, 1, "original")]

    with (
        patch(
            "core.analysis.alignment.extract_audio_to_wav", return_value=wav
        ) as extract,
        patch("core.analysis.alignment.align_words", side_effect=compute),
    ):
        outcome = run_alignment(tasks)[0]
    assert extract.call_args.kwargs["start_time"] == 0
    assert not wav.exists()
    assert not application.apply(project, outcome)
    assert clip.transcript[0].text == "edited"
    assert clip.transcript[0].words is None


def test_alignment_applies_empty_words_once_through_model(tmp_path):
    project = _build_project(tmp_path, 1)
    clip = project.clips[0]
    clip.transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    tasks = snapshot_alignment_tasks(project.clips, project.sources_by_id)
    application = AlignmentApplication(project, tasks)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    with (
        patch("core.analysis.alignment.extract_audio_to_wav", return_value=wav),
        patch("core.analysis.alignment.align_words", return_value=[]),
    ):
        outcome = run_alignment(tasks)[0]
    generation = project.mutation_generation
    assert application.apply(project, outcome)
    applied_generation = project.mutation_generation
    assert applied_generation > generation
    assert not application.apply(project, outcome)
    assert project.mutation_generation == applied_generation
    assert clip.transcript[0].words == []
    assert (
        snapshot_alignment_tasks(project.clips, project.sources_by_id)[0].skip_reason
        == "already_aligned"
    )


def test_cancel_during_alignment_cleans_audio_and_marks_remaining(tmp_path):
    project = _build_project(tmp_path, 2)
    for clip in project.clips:
        clip.transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    tasks = snapshot_alignment_tasks(project.clips, project.sources_by_id)
    cancel = Event()
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fake")
    delivered = []

    def compute(*args, **kwargs):
        cancel.set()
        return [WordTimestamp(0, 1, "hello")]

    with (
        patch("core.analysis.alignment.extract_audio_to_wav", return_value=wav),
        patch("core.analysis.alignment.align_words", side_effect=compute) as align,
    ):
        outcomes = run_alignment(
            tasks, cancel_event=cancel, on_outcome=delivered.append
        )
    assert align.call_count == 1
    assert not delivered and not wav.exists()
    assert [o.code for o in outcomes] == ["cancelled", "cancelled"]
