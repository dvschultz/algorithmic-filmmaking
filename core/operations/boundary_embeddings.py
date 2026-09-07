"""Detached first/last-frame embeddings with shared DINOv2 model ownership."""

from dataclasses import dataclass
from contextlib import nullcontext
from math import isfinite
from pathlib import Path
from threading import Event

from core.operations.contracts import OutcomeStatus
from core.operations.embeddings import (
    EmbeddingOutcome,
    _EmbeddingModelSession,
    embedding_model_session,
)


@dataclass(frozen=True)
class BoundaryEmbeddingTask:
    clip_id: str
    source_path: Path
    start_frame: int
    end_frame: int
    fps: float
    skip: bool = False


@dataclass(frozen=True)
class BoundaryEmbeddingOutcome:
    clip_id: str
    status: OutcomeStatus
    first: tuple[float, ...] = ()
    last: tuple[float, ...] = ()
    model: str | None = None
    code: str | None = None
    message: str | None = None


def run_boundary_embeddings(
    tasks: tuple[BoundaryEmbeddingTask, ...],
    *,
    cancel_event: Event | None = None,
    model_session: _EmbeddingModelSession | None = None,
) -> tuple[BoundaryEmbeddingOutcome, ...]:
    """Validate each pair atomically and reject changed media or late cancellation."""
    from core.errors import ModelDownloadError
    from core.jobs.media import media_stamp

    cancel = cancel_event or Event()
    stamps = [
        media_stamp(task.source_path) if not task.skip else None for task in tasks
    ]
    outcomes: list[BoundaryEmbeddingOutcome] = []
    with (
        nullcontext(model_session)
        if model_session is not None
        else embedding_model_session()
    ) as session:
        for task, stamp in zip(tasks, stamps):
            if cancel.is_set() or session.failed:
                break
            if task.skip:
                outcomes.append(
                    BoundaryEmbeddingOutcome(
                        task.clip_id, "skipped", code="already_populated"
                    )
                )
                continue
            try:
                if (
                    isinstance(task.fps, bool)
                    or not isfinite(task.fps)
                    or task.fps <= 0
                    or isinstance(task.start_frame, bool)
                    or not isinstance(task.start_frame, int)
                    or isinstance(task.end_frame, bool)
                    or not isinstance(task.end_frame, int)
                    or task.start_frame < 0
                    or task.end_frame <= task.start_frame
                ):
                    raise ValueError(
                        "Boundary embeddings require a valid frame range and positive finite fps"
                    )
                if stamp is None or media_stamp(task.source_path) != stamp:
                    raise ValueError("Boundary embedding source is missing or changed")
                if not session.acquire(cancel):
                    break
                if media_stamp(task.source_path) != stamp:
                    raise ValueError(
                        "Boundary embedding source changed while waiting for model"
                    )
                from core.analysis.embeddings import extract_boundary_embeddings

                session.attempted = True
                first, last = extract_boundary_embeddings(
                    source_path=task.source_path,
                    start_frame=task.start_frame,
                    end_frame=task.end_frame,
                    fps=task.fps,
                    cancel_event=cancel,
                )
                if cancel.is_set():
                    break
                if media_stamp(task.source_path) != stamp:
                    raise ValueError(
                        "Boundary embedding source changed during inference"
                    )
                first_result = EmbeddingOutcome.from_vector(task.clip_id, first)
                last_result = EmbeddingOutcome.from_vector(task.clip_id, last)
                outcome = BoundaryEmbeddingOutcome(
                    task.clip_id,
                    "succeeded",
                    first_result.vector,
                    last_result.vector,
                    first_result.model,
                )
            except Exception as exc:
                if cancel.is_set():
                    break
                if isinstance(exc, ModelDownloadError):
                    session.failed = True
                outcome = BoundaryEmbeddingOutcome(
                    task.clip_id, "failed", code="embedding_failed", message=str(exc)
                )
            outcomes.append(outcome)
        outcomes.extend(
            BoundaryEmbeddingOutcome(
                task.clip_id,
                "unprocessed",
                code="cancelled" if cancel.is_set() else "embedding_failed",
            )
            for task in tasks[len(outcomes) :]
        )
    return tuple(
        BoundaryEmbeddingOutcome(
            task.clip_id,
            "failed",
            code="source_changed",
            message="Boundary embedding source changed before delivery",
        )
        if outcome.status == "succeeded" and media_stamp(task.source_path) != stamp
        else outcome
        for task, stamp, outcome in zip(tasks, stamps, outcomes)
    )
