"""Detached, serial word alignment and owner-thread transcript application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from threading import Event
from typing import Callable, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus
from core.operations.transcription import (
    TranscriptionApplication,
    TranscriptionOutcome,
    TranscriptionTask,
    snapshot_tasks,
)
from core.transcription_models import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source


def needs_alignment(clip: Clip) -> bool:
    return any(segment.words is None for segment in (clip.transcript or ()))


@dataclass(frozen=True)
class AlignmentTask:
    target: TranscriptionTask
    transcript_json: str
    skip_reason: str | None = None

    @property
    def clip_id(self) -> str:
        return self.target.clip_id


@dataclass(frozen=True)
class AlignmentOutcome:
    clip_id: str
    status: OutcomeStatus
    words: tuple[WordTimestamp, ...] = ()
    code: str | None = None
    message: str | None = None


def snapshot_alignment_tasks(
    clips: list[Clip], sources_by_id: dict[str, Source], *, skip_existing: bool = True
) -> tuple[AlignmentTask, ...]:
    targets = snapshot_tasks(clips, sources_by_id, skip_existing=False)
    return tuple(
        AlignmentTask(
            target,
            json.dumps(
                [s.to_dict() for s in (clip.transcript or [])],
                sort_keys=True,
                allow_nan=False,
            ),
            "no_transcript"
            if not clip.transcript
            else (
                "already_aligned"
                if skip_existing and not needs_alignment(clip)
                else None
            ),
        )
        for clip, target in zip(clips, targets)
    )


def _compute(task: AlignmentTask, cancel: Event) -> AlignmentOutcome:
    if task.skip_reason:
        return AlignmentOutcome(task.clip_id, "skipped", code=task.skip_reason)
    if task.target.error:
        return AlignmentOutcome(
            task.clip_id, "failed", code="invalid_target", message=task.target.error
        )
    if task.target.source_path is None:
        return AlignmentOutcome(
            task.clip_id, "failed", code="source_missing", message="Source unavailable"
        )
    from core.analysis import alignment

    wav_path = None
    try:
        wav_path = alignment.extract_audio_to_wav(
            task.target.source_path,
            start_time=task.target.start_time,
            end_time=task.target.end_time,
        )
        if cancel.is_set():
            return AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
        segments = [
            TranscriptSegment.from_dict(value)
            for value in json.loads(task.transcript_json)
        ]
        words = alignment.align_words(str(wav_path), segments, extract_audio=False)
        return AlignmentOutcome(task.clip_id, "succeeded", tuple(deepcopy(words)))
    except Exception as exc:
        return AlignmentOutcome(
            task.clip_id, "failed", code="alignment_failed", message=str(exc)
        )
    finally:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove alignment WAV %s: %s", wav_path, exc)


def run_alignment(
    tasks: tuple[AlignmentTask, ...],
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[AlignmentOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[AlignmentOutcome, ...]:
    """Run serially; cancellation suppresses in-flight publication, not cleanup."""
    cancel = cancel_event if cancel_event is not None else Event()
    outcomes: list[AlignmentOutcome] = []
    if progress is not None:
        progress(0, len(tasks))
    for index, task in enumerate(tasks):
        if cancel.is_set():
            outcomes.extend(
                AlignmentOutcome(t.clip_id, "unprocessed", code="cancelled")
                for t in tasks[index:]
            )
            break
        outcome = _compute(task, cancel)
        if cancel.is_set():
            outcomes.extend(
                AlignmentOutcome(t.clip_id, "unprocessed", code="cancelled")
                for t in tasks[index:]
            )
            break
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(deepcopy(outcome))
        if progress is not None:
            progress(index + 1, len(tasks))
    return tuple(outcomes)


def aligned_segments(
    task: AlignmentTask, words: tuple[WordTimestamp, ...]
) -> tuple[TranscriptSegment, ...]:
    """Distribute words onto a detached copy of the submitted transcript."""
    from core.analysis.alignment import distribute_words_to_segments

    segments = [
        TranscriptSegment.from_dict(value) for value in json.loads(task.transcript_json)
    ]
    distribute_words_to_segments(segments, list(deepcopy(words)))
    return tuple(segments)


class AlignmentApplication:
    """Distribute words onto a detached transcript before guarded publication."""

    def __init__(self, project: Project, tasks: tuple[AlignmentTask, ...]) -> None:
        self.tasks = {task.clip_id: task for task in tasks}
        self.application = TranscriptionApplication(
            project, tuple(t.target for t in tasks)
        )

    def apply(self, project: Project, outcome: AlignmentOutcome) -> bool:
        task = self.tasks.get(outcome.clip_id)
        if (
            task is None
            or task.skip_reason is not None
            or outcome.status != "succeeded"
        ):
            return False
        clip = project.clips_by_id.get(outcome.clip_id)
        if (
            clip is None
            or json.dumps(
                [s.to_dict() for s in (clip.transcript or [])],
                sort_keys=True,
                allow_nan=False,
            )
            != task.transcript_json
        ):
            return False
        return self.application.apply(
            project,
            TranscriptionOutcome(
                outcome.clip_id, "succeeded", aligned_segments(task, outcome.words)
            ),
        )
