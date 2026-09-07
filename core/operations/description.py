"""Detached description computation shared by GUI and headless callers."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from core.operations.contracts import OutcomeStatus
from core.provider_errors import is_transient_provider_error

DEFAULT_PROMPT = (
    "Describe this video frame in 3 sentences or less. "
    "Focus on the main subjects, action, and setting."
)
RETRY_DELAYS = (2, 5, 10)


@dataclass(frozen=True)
class DescriptionTask:
    clip_id: str
    thumbnail_path: Path | None
    source_path: Path | None
    start_frame: int
    end_frame: int
    fps: float | None
    skip: bool = False


@dataclass(frozen=True)
class DescriptionOptions:
    tier: str
    prompt: str = DEFAULT_PROMPT
    parallelism: int = 1


@dataclass(frozen=True)
class DescriptionOutcome:
    clip_id: str
    status: OutcomeStatus
    description: str | None = None
    model: str | None = None
    code: str | None = None
    message: str | None = None


def resolve_tier(tier: str | None) -> str:
    if not tier:
        from core.settings import load_settings

        tier = load_settings().description_model_tier
    return "local" if tier in ("cpu", "gpu") else tier


def compute_description(
    task: DescriptionTask, options: DescriptionOptions, cancel: Event
) -> DescriptionOutcome:
    if cancel.is_set():
        return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
    if task.skip:
        return DescriptionOutcome(task.clip_id, "skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return DescriptionOutcome(task.clip_id, "failed", code="thumbnail_missing")

    from core.analysis.description import describe_frame

    for attempt in range(len(RETRY_DELAYS) + 1):
        if cancel.is_set():
            return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
        try:
            description, model = describe_frame(
                task.thumbnail_path,
                tier=options.tier,
                prompt=options.prompt,
                source_path=task.source_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                fps=task.fps,
            )
            if cancel.is_set():
                return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
            if not description or description.startswith("Error"):
                return DescriptionOutcome(
                    task.clip_id,
                    "failed",
                    code="description_failed",
                    message=description or "Provider returned an empty description",
                )
            return DescriptionOutcome(task.clip_id, "succeeded", description, model)
        except Exception as exc:
            if cancel.is_set():
                return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
            if attempt < len(RETRY_DELAYS) and is_transient_provider_error(str(exc)):
                cancel.wait(RETRY_DELAYS[attempt])
                continue
            return DescriptionOutcome(
                task.clip_id, "failed", code="description_failed", message=str(exc)
            )
    raise AssertionError("Retry loop must return an outcome")


def run_description(
    tasks: tuple[DescriptionTask, ...],
    options: DescriptionOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[DescriptionOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[DescriptionOutcome, ...]:
    """Bound admission, serialize local inference, and suppress cancelled results."""
    cancel = cancel_event or Event()
    parallelism = (
        1
        if options.tier in ("local", "cpu", "gpu")
        else min(max(1, options.parallelism), 5)
    )
    outcomes: dict[int, DescriptionOutcome] = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        pending: dict[Future[DescriptionOutcome], int] = {}
        while pending or next_index < len(tasks):
            while (
                not cancel.is_set()
                and len(pending) < parallelism
                and next_index < len(tasks)
            ):
                pending[
                    pool.submit(compute_description, tasks[next_index], options, cancel)
                ] = next_index
                next_index += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = DescriptionOutcome(
                        tasks[index].clip_id,
                        "failed",
                        code="description_failed",
                        message=str(exc),
                    )
                if cancel.is_set():
                    continue
                outcomes[index] = outcome
                if on_outcome:
                    on_outcome(outcome)
                if progress:
                    progress(len(outcomes), len(tasks))
            if cancel.is_set():
                for future in pending:
                    future.cancel()
                break
    return tuple(
        outcomes.get(
            i, DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
        )
        for i, task in enumerate(tasks)
    )
