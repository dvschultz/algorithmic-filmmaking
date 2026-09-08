"""Shared, GUI-free transcription scheduling over detached clip tasks."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING, Any, Callable

from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project
    from core.analysis_records import AnalysisFingerprints


@dataclass(frozen=True)
class TranscriptionTask:
    clip_id: str
    source_path: Path | None
    start_time: float
    end_time: float
    fps: float
    skip: bool = False
    error: str | None = None
    analysis_json: str | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    model: str = "small.en"
    language: str | None = "en"
    backend: str = "auto"
    segmentation_mode: str = "backend"
    segment_max_seconds: float = 12.0
    parallelism: int = 1
    cloud_model: str | None = None


def resolve_transcription_options(
    options: TranscriptionOptions,
) -> TranscriptionOptions:
    """Freeze backend and cloud settings before queueing or identifying a batch."""
    from core.transcription import _resolve_backend, transcription_model

    backend = _resolve_backend(options.backend)
    return replace(
        options,
        backend=backend,
        cloud_model=transcription_model(backend, options.model, options.cloud_model)
        if backend == "groq"
        else None,
    )


@dataclass(frozen=True)
class TranscriptionOutcome:
    clip_id: str
    status: OutcomeStatus
    segments: tuple[Any, ...] = ()
    code: str | None = None
    message: str | None = None
    critical: bool = False
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


def snapshot_tasks(
    clips: list, sources: dict, *, skip_existing: bool = True
) -> tuple[TranscriptionTask, ...]:
    tasks = []
    for clip in clips:
        source = sources.get(clip.source_id)
        start_time = end_time = 0.0
        error = None
        try:
            if source:
                if not isfinite(source.fps) or source.fps <= 0:
                    raise ValueError("Source frame rate must be positive and finite")
                start_time = clip.start_time(source.fps)
                end_time = clip.end_time(source.fps)
                if (
                    not isfinite(start_time)
                    or not isfinite(end_time)
                    or end_time <= start_time
                ):
                    raise ValueError("Clip range must have positive finite duration")
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            error = str(exc)
        tasks.append(
            TranscriptionTask(
                clip.id,
                Path(source.file_path) if source else None,
                start_time,
                end_time,
                source.fps if source else 0,
                skip_existing and clip.transcript is not None,
                error,
            )
        )
    return tuple(tasks)


def _compute_task(
    task: TranscriptionTask,
    options: TranscriptionOptions,
    on_execution: Callable[[dict[str, str | None]], None] | None = None,
    prepare: Callable[[], bool] | None = None,
) -> TranscriptionOutcome:
    from core.transcription_models import (
        FFmpegNotFoundError,
        FasterWhisperNotInstalledError,
        ModelDownloadError,
    )

    if task.skip:
        return TranscriptionOutcome(task.clip_id, "skipped", code="already_populated")
    if task.error is not None:
        return TranscriptionOutcome(
            task.clip_id, "failed", code="invalid_target", message=task.error
        )
    if task.source_path is None or not task.source_path.exists():
        return TranscriptionOutcome(
            task.clip_id,
            "failed",
            code="source_file_missing",
            message="source not found",
        )
    try:
        from core.transcription import transcribe_clip

        if prepare is not None and not prepare():
            return TranscriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
        if task.analysis_json is not None:
            from core.analysis_records import AnalysisSnapshot

            if not AnalysisSnapshot.from_json(task.analysis_json).inputs.unchanged():
                return TranscriptionOutcome(task.clip_id, "failed", code="stale_input")
        options = resolve_transcription_options(options)
        segments = transcribe_clip(
            source_path=task.source_path,
            start_time=task.start_time,
            end_time=task.end_time,
            model_name=options.model,
            language=options.language,
            backend=options.backend,
            segmentation_mode=options.segmentation_mode,
            segment_max_seconds=options.segment_max_seconds,
            cloud_model=options.cloud_model,
            on_execution=on_execution,
        )
        return TranscriptionOutcome(
            task.clip_id, "succeeded", tuple(deepcopy(segments))
        )
    except Exception as exc:
        return TranscriptionOutcome(
            task.clip_id,
            "failed",
            code="dependency_missing"
            if isinstance(exc, (FFmpegNotFoundError, FasterWhisperNotInstalledError))
            else "transcription_failed",
            message=str(exc),
            critical=isinstance(
                exc,
                (
                    FFmpegNotFoundError,
                    FasterWhisperNotInstalledError,
                    ModelDownloadError,
                ),
            ),
        )


def compute_task(
    task: TranscriptionTask,
    options: TranscriptionOptions,
    *,
    fingerprints: AnalysisFingerprints | None = None,
    prepare: Callable[[], bool] | None = None,
) -> TranscriptionOutcome:
    """Verify detached records before inference and retain failed execution state."""
    if task.analysis_json is None:
        return _compute_task(task, options, prepare=prepare)
    if (
        task.error is not None
        or task.source_path is None
        or not task.source_path.is_file()
    ):
        return _compute_task(replace(task, skip=False), options)
    from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
    from core.operations.transcription_records import (
        transcription_identity,
        transcription_runtime,
        transcription_segments_value,
    )
    from core.transcription import _has_audio_stream
    from core.transcription_models import TranscriptSegment
    from models.analysis_record import AnalysisRecord

    try:
        options = resolve_transcription_options(options)
        snapshot = AnalysisSnapshot.from_json(task.analysis_json)
        fingerprints = fingerprints or AnalysisFingerprints()
        runtime = transcription_runtime(options)
        if (
            task.source_path is not None
            and _has_audio_stream(task.source_path) is False
        ):
            runtime = transcription_runtime(
                options,
                execution={
                    "backend": "audio-probe",
                    "model": None,
                    "input_mode": "no-audio",
                },
            )
        identity = transcription_identity(snapshot, options, fingerprints, runtime)
        reused = snapshot.reusable_record(identity) if task.skip else None
        if reused is not None:
            if (
                not snapshot.inputs.unchanged()
                or transcription_runtime(options, execution=runtime["execution"])
                != runtime
            ):
                return TranscriptionOutcome(task.clip_id, "failed", code="stale_input")
            return TranscriptionOutcome(
                task.clip_id,
                "skipped",
                tuple(
                    TranscriptSegment.from_dict(s) for s in reused.value["transcript"]
                ),
                code="valid_analysis",
                record_json=json.dumps(reused.to_dict(), sort_keys=True),
            )
        execution = None
        execution_runtime = None

        def report(value: dict[str, str | None]) -> None:
            nonlocal execution, execution_runtime
            execution = dict(value)
            execution_runtime = transcription_runtime(options, execution=execution)

        outcome = _compute_task(
            replace(task, skip=False),
            options,
            report,
            prepare if runtime["execution"]["backend"] != "audio-probe" else None,
        )
        if outcome.status == "unprocessed":
            return outcome
        if outcome.status == "succeeded":
            try:
                transcription_segments_value(outcome.segments)
            except (ValueError, AttributeError, TypeError) as exc:
                outcome = replace(
                    outcome,
                    status="failed",
                    segments=(),
                    code="invalid_result",
                    message=str(exc),
                )
        actual = execution_runtime if execution_runtime is not None else runtime
        if (
            not snapshot.inputs.unchanged()
            or transcription_runtime(options, execution=runtime["execution"]) != runtime
            or transcription_runtime(options, execution=actual["execution"]) != actual
        ):
            return TranscriptionOutcome(task.clip_id, "failed", code="stale_input")
        identity = transcription_identity(snapshot, options, fingerprints, actual)
        record = (
            AnalysisRecord.success(
                identity,
                {"transcript": [s.to_dict() for s in outcome.segments]},
                input_snapshot=snapshot.inputs.to_dict(),
            )
            if outcome.status == "succeeded"
            else replace(
                AnalysisRecord.failure(
                    identity, outcome.message or outcome.code or "Transcription failed"
                ),
                input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
            )
        )
        return replace(
            outcome, record_json=json.dumps(record.to_dict(), sort_keys=True)
        )
    except Exception as exc:
        return TranscriptionOutcome(
            task.clip_id, "failed", code="stale_input", message=str(exc)
        )


def run_transcription(
    tasks: tuple[TranscriptionTask, ...],
    options: TranscriptionOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[TranscriptionOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    prepare: Callable[[], bool] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
) -> tuple[TranscriptionOutcome, ...]:
    """Run a bounded batch; callbacks run on the caller, results keep input order."""
    if not tasks:
        return ()
    options = resolve_transcription_options(options)
    parallelism = (
        1 if options.backend == "mlx-whisper" else min(max(1, options.parallelism), 4)
    )
    cancelled = cancel_event or Event()
    from core.analysis_records import AnalysisFingerprints

    fingerprints = fingerprints or AnalysisFingerprints(cancelled)
    preparation_lock = Lock()
    prepared: bool | None = None
    preparation_error: Exception | None = None

    def prepare_once() -> bool:
        nonlocal prepared, preparation_error
        with preparation_lock:
            if preparation_error is not None:
                raise preparation_error
            if prepared is None:
                try:
                    prepared = not cancelled.is_set() and (prepare is None or prepare())
                except Exception as exc:
                    preparation_error = exc
                    raise
            return prepared

    outcomes: dict[int, TranscriptionOutcome] = {}
    next_index = 0
    halted = False
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        pending: dict[Future[TranscriptionOutcome], int] = {}
        while pending or next_index < len(tasks):
            while (
                not halted
                and not cancelled.is_set()
                and len(pending) < parallelism
                and next_index < len(tasks)
            ):
                pending[
                    pool.submit(
                        compute_task,
                        tasks[next_index],
                        options,
                        fingerprints=fingerprints,
                        prepare=prepare_once,
                    )
                ] = next_index
                next_index += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                if future.cancelled():
                    continue
                outcome = future.result()
                if cancelled.is_set():
                    continue
                outcomes[index] = outcome
                if on_outcome:
                    on_outcome(deepcopy(outcome))
                if progress:
                    progress(len(outcomes), len(tasks))
                if outcome.critical:
                    halted = True
            if cancelled.is_set() or halted:
                for future in pending:
                    future.cancel()
                # Preserve successful in-flight results after critical failure;
                # cancellation closes publication but still waits for native work.
                if cancelled.is_set():
                    break
    return tuple(
        outcomes.get(
            i,
            TranscriptionOutcome(
                task.clip_id,
                "unprocessed",
                code="cancelled" if cancelled.is_set() else "batch_aborted",
            ),
        )
        for i, task in enumerate(tasks)
    )


def _media_stamp(path: Path | None) -> tuple[int, int, int, int, int] | None:
    try:
        stat = path.stat() if path is not None else None
        return (
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if stat
            else None
        )
    except OSError:
        return None


class TranscriptionApplication:
    """Owner-thread, one-use application of results to unchanged clip targets."""

    def __init__(
        self,
        project: Project,
        tasks: tuple[TranscriptionTask, ...],
        options: TranscriptionOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.options = (
            resolve_transcription_options(options) if options is not None else None
        )
        self._consumed: set[str] = set()
        self._targets: dict[str, tuple] = {}
        stamps = {
            path: _media_stamp(path) for path in {task.source_path for task in tasks}
        }
        for task in tasks:
            clip = project.clips_by_id.get(task.clip_id)
            source = project.sources_by_id.get(clip.source_id) if clip else None
            self._targets[task.clip_id] = (
                task,
                clip,
                source,
                deepcopy(clip.transcript) if clip else None,
                stamps[task.source_path],
                (clip.start_frame, clip.end_frame) if clip else None,
            )

    def apply(self, project: Project, outcome: TranscriptionOutcome) -> bool:
        """Publish one successful result without accepting duplicate delivery."""
        return self.apply_batch(project, (outcome,))[0]

    def apply_batch(
        self,
        project: Project,
        outcomes: tuple[TranscriptionOutcome, ...],
    ) -> tuple[bool, ...]:
        """Validate one project revision and notify once for a headless batch."""
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or not any(
                o.can_apply and o.clip_id not in self._consumed for o in outcomes
            )
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = tuple(self._apply(project, outcome) for outcome in outcomes)
            updated = [
                project.clips_by_id[o.clip_id]
                for o, valid in zip(outcomes, accepted)
                if valid and o.status == "succeeded"
            ]
            if updated:
                project.update_clips(updated)
            return accepted

        return project.session.apply_external(publish)

    def _apply(self, project: Project, outcome: TranscriptionOutcome) -> bool:
        if not outcome.can_apply or outcome.clip_id in self._consumed:
            return False
        self._consumed.add(outcome.clip_id)
        binding = self._targets.get(outcome.clip_id)
        if binding is None:
            return False
        task, clip, source, expected, stamp, frames = binding
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or clip is None
            or project.clips_by_id.get(outcome.clip_id) is not clip
            or (clip.start_frame, clip.end_frame) != frames
            or clip.transcript != expected
            or source is None
            or project.sources_by_id.get(clip.source_id) is not source
            or source.file_path != task.source_path
            or source.fps != task.fps
            or task.error is not None
            or clip.start_time(source.fps) != task.start_time
            or clip.end_time(source.fps) != task.end_time
            or stamp is None
            or _media_stamp(task.source_path) != stamp
        ):
            return False
        from core.analysis_records import AnalysisSnapshot
        from core.operations.transcription_records import (
            transcription_parameters,
            transcription_value,
            transcription_task,
            transcription_segments_value,
        )
        from models.analysis_record import AnalysisRecord

        try:
            value = transcription_segments_value(outcome.segments)
        except (ValueError, AttributeError, TypeError):
            return False
        if outcome.record_json is not None:
            try:
                record = AnalysisRecord.from_dict(json.loads(outcome.record_json))
                snapshot = (
                    AnalysisSnapshot.from_json(task.analysis_json)
                    if task.analysis_json
                    else None
                )
                previous = clip.analysis_records.get("transcribe")
                if not isinstance(previous, AnalysisRecord):
                    previous = None
                current_json = transcription_task(clip, source).analysis_json
                if (
                    snapshot is None
                    or record.identity is None
                    or current_json is None
                    or snapshot.inputs
                    != AnalysisSnapshot.from_json(current_json).inputs
                    or record.identity.operation != "transcribe"
                    or record.identity.to_dict()["operation_version"] != 2
                    or record.identity.to_dict()["schema_version"] != 1
                    or previous != snapshot.record
                    or transcription_value(clip) != json.loads(snapshot.value_json)
                    or not snapshot.inputs.unchanged()
                    or json.loads(record.input_json or "null")
                    != snapshot.inputs.to_dict()
                    or (
                        self.options is not None
                        and record.identity.to_dict()["parameters"]
                        != transcription_parameters(self.options)
                    )
                    or (
                        outcome.has_result
                        and (record.state != "succeeded" or record.value != value)
                    )
                    or (outcome.status == "failed" and record.state != "failed")
                ):
                    return False
            except (ValueError, TypeError, KeyError):
                return False
        else:
            record = AnalysisRecord.legacy(value)
        project.record_analysis("clip", outcome.clip_id, "transcribe", record)
        if outcome.status == "succeeded":
            # Replacing word timings invalidates their previous alignment
            # verification before update_clips notifies project observers.
            if "align_words" in clip.analysis_records:
                project.record_analysis(
                    "clip", outcome.clip_id, "align_words", AnalysisRecord.legacy(value)
                )
            clip.transcript = list(deepcopy(outcome.segments))
        return True
