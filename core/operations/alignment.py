"""Detached, serial word alignment and owner-thread transcript application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
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
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from models.analysis_record import AnalysisRecord
from core.operations.alignment_records import (
    alignment_snapshot,
    alignment_runtime,
    alignment_identity,
    alignment_parameters,
    execution_is_current,
    alignment_execution_reusable,
    prior_execution,
)
from core.operations.transcription_records import (
    transcription_segments_value,
    transcription_value,
)

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
    analysis_json: str | None = None
    skip_existing: bool = True

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
    record_json: str | None = None

    @property
    def has_result(self) -> bool:
        return self.status == "succeeded" or (
            self.status == "skipped" and self.record_json is not None
        )

    @property
    def can_apply(self) -> bool:
        return self.has_result or (
            self.status == "failed" and self.record_json is not None
        )


def snapshot_alignment_tasks(
    clips: list[Clip],
    sources_by_id: dict[str, Source],
    *,
    skip_existing: bool = True,
    verified: bool = False,
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
                if not verified and skip_existing and not needs_alignment(clip)
                else None
            ),
            alignment_snapshot(clip, sources_by_id[clip.source_id]).to_json()
            if verified and not target.error and clip.source_id in sources_by_id
            else None,
            skip_existing,
        )
        for clip, target in zip(clips, targets)
    )


def _compute_raw(
    task: AlignmentTask,
    cancel: Event,
    on_execution: Callable[[dict], None] | None = None,
) -> AlignmentOutcome:
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
        kwargs = {"on_execution": on_execution} if on_execution is not None else {}
        words = alignment.align_words(
            str(wav_path), segments, extract_audio=False, **kwargs
        )
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


def _compute(
    task: AlignmentTask, cancel: Event, fingerprints: AnalysisFingerprints | None = None,
    prepare: Callable[[], bool] | None = None,
) -> AlignmentOutcome:
    if task.analysis_json is None or task.skip_reason:
        if not task.skip_reason and prepare is not None and not prepare():
            cancel.set()
            return AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
        return _compute_raw(task, cancel)
    try:
        snapshot = AnalysisSnapshot.from_json(task.analysis_json)
        if json.loads(snapshot.value_json).get("transcript") != json.loads(
            task.transcript_json
        ):
            raise ValueError("Alignment transcript differs from its input snapshot")
        if not snapshot.inputs.unchanged():
            raise ValueError("Alignment media changed before computation")
        fingerprints = fingerprints or AnalysisFingerprints(cancel)
        runtime = alignment_runtime(execution=prior_execution(snapshot))
        identity = alignment_identity(
            snapshot, task.transcript_json, fingerprints, runtime
        )
        reused = (
            snapshot.reusable_record(identity)
            if task.skip_existing and alignment_execution_reusable(snapshot.record, runtime)
            else None
        )
        if reused is not None:
            if (
                cancel.is_set()
                or not snapshot.inputs.unchanged()
                or alignment_runtime(execution=runtime["execution"]) != runtime
            ):
                raise ValueError("Alignment inputs changed during verification")
            words = tuple(
                WordTimestamp.from_dict(word)
                for segment in reused.value["transcript"]
                for word in (segment.get("words") or [])
            )
            return AlignmentOutcome(
                task.clip_id,
                "skipped",
                words,
                code="valid_analysis",
                record_json=json.dumps(reused.to_dict(), sort_keys=True),
            )
        if prepare is not None and not prepare():
            cancel.set()
            return AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
        if cancel.is_set() or not snapshot.inputs.unchanged():
            raise ValueError("Alignment inputs changed during preparation")
        before = alignment_runtime()
        events: list[dict] = []
        outcome = _compute_raw(
            task, cancel, lambda event: events.append(deepcopy(event))
        )
        if cancel.is_set():
            return AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
        actual = alignment_runtime(execution=events)
        # Loading the model may populate a previously absent cache revision.
        if not snapshot.inputs.unchanged() or {
            k: v for k, v in before.items() if k not in ("revision", "execution")
        } != {k: v for k, v in actual.items() if k not in ("revision", "execution")}:
            raise ValueError("Alignment media or runtime changed during computation")
        if (
            before.get("revision") is not None
            and before["revision"] != actual["revision"]
        ):
            raise ValueError("Alignment model revision changed during computation")
        if outcome.status == "succeeded":
            try:
                if not events:
                    raise ValueError("Alignment engine did not report its execution")
                if any(
                    word.end > task.target.end_time - task.target.start_time + 1e-6
                    for word in outcome.words
                ):
                    raise ValueError("Alignment word lies outside the clip")
                value = transcription_segments_value(
                    aligned_segments(task, outcome.words)
                )
                if not execution_is_current(actual):
                    raise ValueError("Alignment model revision could not be verified")
            except (ValueError, TypeError, AttributeError) as exc:
                outcome = replace(
                    outcome,
                    status="failed",
                    words=(),
                    code="invalid_alignment",
                    message=str(exc),
                )
        identity = alignment_identity(
            snapshot, task.transcript_json, fingerprints, actual
        )
        record = (
            AnalysisRecord.success(
                identity, value, input_snapshot=snapshot.inputs.to_dict()
            )
            if outcome.status == "succeeded"
            else replace(
                AnalysisRecord.failure(identity, outcome.message or "Alignment failed"),
                input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
            )
        )
        return replace(
            outcome, record_json=json.dumps(record.to_dict(), sort_keys=True)
        )
    except Exception as exc:
        return AlignmentOutcome(
            task.clip_id,
            "unprocessed" if cancel.is_set() else "failed",
            message=str(exc),
        )


def run_alignment(
    tasks: tuple[AlignmentTask, ...],
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[AlignmentOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
    prepare: Callable[[], bool] | None = None,
) -> tuple[AlignmentOutcome, ...]:
    """Run serially; cancellation suppresses in-flight publication, not cleanup."""
    cancel = cancel_event if cancel_event is not None else Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
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
        outcome = _compute(task, cancel, fingerprints, prepare)
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
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path
        self.consumed: set[str] = set()
        self.tasks = {task.clip_id: task for task in tasks}
        self.targets = {
            task.clip_id: (
                project.clips_by_id.get(task.clip_id),
                project.sources_by_id.get(project.clips_by_id[task.clip_id].source_id)
                if task.clip_id in project.clips_by_id
                else None,
                deepcopy(project.clips_by_id[task.clip_id].analysis_records)
                if task.clip_id in project.clips_by_id
                else {},
            )
            for task in tasks
        }
        self.application = TranscriptionApplication(
            project, tuple(t.target for t in tasks)
        )

    def apply(self, project: Project, outcome: AlignmentOutcome) -> bool:
        task = self.tasks.get(outcome.clip_id)
        if task is not None and task.analysis_json is not None:
            return self._apply_verified(project, task, outcome)
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or project.path != self.path
            or task is None
            or task.skip_reason is not None
            or outcome.status != "succeeded"
        ):
            return False
        clip = project.clips_by_id.get(outcome.clip_id)
        if (
            clip is None
            or clip.analysis_records != self.targets[outcome.clip_id][2]
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

    def _apply_verified(
        self, project: Project, task: AlignmentTask, outcome: AlignmentOutcome
    ) -> bool:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or project.path != self.path
            or outcome.clip_id in self.consumed
            or not outcome.can_apply
            or outcome.record_json is None
        ):
            return False

        def publish() -> bool:
            self.consumed.add(outcome.clip_id)
            clip, source, prior_records = self.targets[outcome.clip_id]
            if (
                clip is None
                or source is None
                or project.clips_by_id.get(outcome.clip_id) is not clip
                or project.sources_by_id.get(clip.source_id) is not source
                or clip.analysis_records != prior_records
            ):
                return False
            try:
                snapshot = AnalysisSnapshot.from_json(task.analysis_json or "")
                record = AnalysisRecord.from_dict(json.loads(outcome.record_json or ""))
                current = alignment_snapshot(clip, source)
                if (
                    current != snapshot
                    or not snapshot.inputs.unchanged()
                    or record.identity is None
                ):
                    return False
                identity = record.identity.to_dict()
                if (
                    identity["operation"] != "align_words"
                    or identity["operation_version"] != 2
                    or identity["schema_version"] != 1
                    or identity["parameters"]
                    != alignment_parameters(task.transcript_json)
                    or identity["source_range"]
                    != json.loads(snapshot.inputs.range_json)
                    or identity["sampling"] != {"policy": "half-open-clip-audio/v1"}
                    or json.loads(record.input_json or "null")
                    != snapshot.inputs.to_dict()
                ):
                    return False
                if outcome.has_result:
                    # Reuse preserves the saved distribution exactly; success
                    # distributes new flat words onto detached editorial segments.
                    segments = (
                        tuple(
                            TranscriptSegment.from_dict(s)
                            for s in record.value["transcript"]
                        )
                        if outcome.status == "skipped"
                        else aligned_segments(task, outcome.words)
                    )
                    value = transcription_segments_value(segments)
                    if record.state != "succeeded" or record.value != value:
                        return False
                    if outcome.status == "skipped" and [
                        word.to_dict() for word in outcome.words
                    ] != [
                        word.to_dict()
                        for segment in segments
                        for word in (segment.words or [])
                    ]:
                        return False
                    if outcome.status == "skipped" and value != transcription_value(
                        clip
                    ):
                        return False
                elif record.state != "failed":
                    return False
            except (ValueError, TypeError, KeyError, AttributeError):
                return False
            project.record_analysis("clip", outcome.clip_id, "align_words", record)
            if outcome.status == "succeeded":
                # Alignment changes the transcript projection; its former
                # transcription verification must not survive that change.
                project.record_analysis(
                    "clip", outcome.clip_id, "transcribe", AnalysisRecord.legacy(value)
                )
                clip.transcript = list(deepcopy(segments))
                project.update_clips([clip])
            return True

        return project.session.apply_external(publish)
