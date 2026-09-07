"""Detached custom-query computation shared by GUI and headless callers."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Callable, Literal, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus
from core.provider_errors import is_transient_provider_error

if TYPE_CHECKING:
    from core.project import Project


@dataclass(frozen=True)
class CustomQueryTask:
    clip_id: str
    thumbnail_path: Path | None
    query: str
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"


@dataclass(frozen=True)
class CustomQueryOptions:
    tier: str
    model: str
    parallelism: int = 1


@dataclass(frozen=True)
class CustomQueryOutcome:
    clip_id: str
    query: str
    status: OutcomeStatus
    match: bool | None = None
    confidence: float | None = None
    model: str | None = None
    code: str | None = None
    message: str | None = None


def resolve_options(
    tier: str | None = None, parallelism: int = 1
) -> CustomQueryOptions:
    from core.settings import load_settings

    settings = load_settings()
    tier = tier or settings.description_model_tier
    # Preserve the custom-query legacy routing contract.
    tier = {"cpu": "local", "gpu": "cloud"}.get(tier, tier)
    return CustomQueryOptions(
        tier,
        settings.description_model_local
        if tier == "local"
        else settings.description_model_cloud,
        parallelism,
    )


def compute_custom_query(
    task: CustomQueryTask, options: CustomQueryOptions, cancel: Event
) -> CustomQueryOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> CustomQueryOutcome:
        return CustomQueryOutcome(task.clip_id, task.query, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if not task.query.strip():
        return outcome("failed", code="missing_query", message="query is required")
    if task.skip:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    from core.analysis.custom_query import evaluate_custom_query

    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        try:
            match, confidence, model = evaluate_custom_query(
                image_path=task.thumbnail_path,
                query=task.query,
                tier=options.tier,
                model_name=options.model,
            )
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            return outcome("succeeded", match=match, confidence=confidence, model=model)
        except Exception as exc:
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            if attempt < len(delays) and is_transient_provider_error(str(exc)):
                cancel.wait(delays[attempt])
                continue
            return outcome("failed", code="custom_query_failed", message=str(exc))
    raise AssertionError("Retry loop must return")


def run_custom_query(
    tasks: tuple[CustomQueryTask, ...],
    options: CustomQueryOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[CustomQueryOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CustomQueryOutcome, ...]:
    """Bound cloud admission; keep local inference on the caller's worker thread."""
    options = replace(
        options, tier={"cpu": "local", "gpu": "cloud"}.get(options.tier, options.tier)
    )
    cancel = cancel_event or Event()
    outcomes: dict[int, CustomQueryOutcome] = {}

    def publish(index: int, outcome: CustomQueryOutcome) -> None:
        if cancel.is_set():
            return
        outcomes[index] = outcome
        if on_outcome:
            on_outcome(outcome)
        if progress:
            progress(len(outcomes), len(tasks))

    if options.tier == "local":
        for index, task in enumerate(tasks):
            if cancel.is_set():
                break
            publish(index, compute_custom_query(task, options, cancel))
    else:
        parallelism = min(max(1, options.parallelism), 5)
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            pending: dict[Future[CustomQueryOutcome], int] = {}
            next_index = 0
            while pending or next_index < len(tasks):
                while (
                    not cancel.is_set()
                    and len(pending) < parallelism
                    and next_index < len(tasks)
                ):
                    pending[
                        pool.submit(
                            compute_custom_query, tasks[next_index], options, cancel
                        )
                    ] = next_index
                    next_index += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = CustomQueryOutcome(
                            tasks[index].clip_id,
                            tasks[index].query,
                            "failed",
                            code="custom_query_failed",
                            message=str(exc),
                        )
                    publish(index, result)
                if cancel.is_set():
                    for future in pending:
                        future.cancel()
                    break
    return tuple(
        outcomes.get(
            i,
            CustomQueryOutcome(
                task.clip_id, task.query, "unprocessed", code="cancelled"
            ),
        )
        for i, task in enumerate(tasks)
    )


class CustomQueryApplication:
    """Append once on the owner thread, only to unchanged clip inputs."""

    def __init__(self, project: "Project", tasks: tuple[CustomQueryTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: CustomQueryTask) -> tuple | None:
        from core.jobs.media import media_stamp

        # Frame query storage is not part of the Frame model. Never resolve a
        # frame task through a colliding clip ID.
        if task.target_type != "clip":
            return None
        clip = project.clips_by_id.get(task.clip_id)
        if clip is None or clip.thumbnail_path != task.thumbnail_path:
            return None
        stamp = media_stamp(task.thumbnail_path) if task.thumbnail_path else None
        if stamp is None:
            return None
        source = project.sources_by_id.get(clip.source_id)
        source_path = source.file_path if source else None
        return (
            clip,
            source,
            (
                clip.source_id,
                clip.start_frame,
                clip.end_frame,
                stamp,
                source_path,
                source.fps if source else None,
                media_stamp(source_path) if source_path else None,
                deepcopy(clip.custom_queries),
            ),
        )

    def apply(self, project: "Project", outcome: CustomQueryOutcome) -> bool:
        return self.apply_batch(project, (outcome,))[0]

    def apply_batch(
        self, project: "Project", outcomes: tuple[CustomQueryOutcome, ...]
    ) -> tuple[bool, ...]:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or not any(outcome.status == "succeeded" for outcome in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            updated = []
            for outcome in outcomes:
                task = self.tasks.get(outcome.clip_id)
                expected = self.bindings.get(outcome.clip_id)
                valid = False
                if (
                    outcome.status == "succeeded"
                    and outcome.clip_id not in self.consumed
                ):
                    self.consumed.add(outcome.clip_id)
                    current = self._binding(project, task) if task else None
                    if (
                        task is not None
                        and outcome.query == task.query
                        and expected is not None
                        and current is not None
                        and current[0] is expected[0]
                        and current[1] is expected[1]
                        and current[2] == expected[2]
                    ):
                        clip = current[0]
                        clip.custom_queries = [
                            *(clip.custom_queries or []),
                            {
                                "query": outcome.query,
                                "match": outcome.match,
                                "confidence": round(outcome.confidence or 0.0, 4),
                                "model": outcome.model,
                            },
                        ]
                        updated.append(clip)
                        valid = True
                accepted.append(valid)
            if updated:
                project.update_clips(updated)
            return tuple(accepted)

        return project.session.apply_external(publish)
