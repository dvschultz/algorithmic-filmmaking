"""Detached ImageNet classification shared by GUI and headless workflows."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Literal

from core.operations.contracts import OutcomeStatus

# MobileNet inference shares a process-wide model. Serialize across jobs, too.
_inference_lock = Lock()


@dataclass(frozen=True)
class ClassificationTask:
    clip_id: str
    thumbnail_path: Path | None
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"


@dataclass(frozen=True)
class ClassificationOptions:
    top_k: int = 5
    threshold: float = 0.1


@dataclass(frozen=True)
class ClassificationOutcome:
    clip_id: str
    status: OutcomeStatus
    labels: tuple[tuple[str, float], ...] = ()
    code: str | None = None
    message: str | None = None


def compute_classification(
    task: ClassificationTask, options: ClassificationOptions, cancel: Event
) -> ClassificationOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> ClassificationOutcome:
        return ClassificationOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    while not _inference_lock.acquire(timeout=0.05):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
    try:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        from core.analysis.classification import classify_frame

        raw = classify_frame(
            task.thumbnail_path, top_k=options.top_k, threshold=options.threshold
        )
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        labels = tuple((label, float(confidence)) for label, confidence in raw)
        if any(
            not isinstance(label, str)
            or not label
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            for label, confidence in labels
        ):
            raise ValueError("Invalid classification result")
        return outcome("succeeded", labels=labels)
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("failed", code="classification_failed", message=str(exc))
    finally:
        _inference_lock.release()


def run_classification(
    tasks: tuple[ClassificationTask, ...],
    options: ClassificationOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[ClassificationOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[ClassificationOutcome, ...]:
    """Run serially on the caller thread with cancellable model admission."""
    cancel = cancel_event or Event()
    outcomes = []
    for task in tasks:
        result = compute_classification(task, options, cancel)
        outcomes.append(result)
        if not cancel.is_set():
            if on_outcome:
                on_outcome(result)
            if progress:
                progress(len(outcomes), len(tasks))
    return tuple(outcomes)
