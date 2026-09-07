"""Shared, GUI-free transcription scheduling over detached clip tasks."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from threading import Event
from typing import Any, Callable

from core.operations.contracts import OutcomeStatus


@dataclass(frozen=True)
class TranscriptionTask:
    clip_id: str
    source_path: Path | None
    start_time: float
    end_time: float
    fps: float
    skip: bool = False
    error: str | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    model: str = "small.en"
    language: str | None = "en"
    backend: str = "auto"
    segmentation_mode: str = "backend"
    segment_max_seconds: float = 12.0
    parallelism: int = 1


@dataclass(frozen=True)
class TranscriptionOutcome:
    clip_id: str
    status: OutcomeStatus
    segments: tuple[Any, ...] = ()
    code: str | None = None
    message: str | None = None
    critical: bool = False


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


def compute_task(
    task: TranscriptionTask, options: TranscriptionOptions
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

        segments = transcribe_clip(
            source_path=task.source_path,
            start_time=task.start_time,
            end_time=task.end_time,
            model_name=options.model,
            language=options.language,
            backend=options.backend,
            segmentation_mode=options.segmentation_mode,
            segment_max_seconds=options.segment_max_seconds,
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


def run_transcription(
    tasks: tuple[TranscriptionTask, ...],
    options: TranscriptionOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[TranscriptionOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[TranscriptionOutcome, ...]:
    """Run a bounded batch; callbacks run on the caller, results keep input order."""
    from core.transcription import _resolve_backend

    if not tasks:
        return ()
    options = replace(options, backend=_resolve_backend(options.backend))
    parallelism = (
        1 if options.backend == "mlx-whisper" else min(max(1, options.parallelism), 4)
    )
    cancelled = cancel_event or Event()
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
                pending[pool.submit(compute_task, tasks[next_index], options)] = (
                    next_index
                )
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
