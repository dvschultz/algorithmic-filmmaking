"""Detached description computation shared by GUI and headless callers."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Callable, TYPE_CHECKING, Literal

from core.operations.contracts import OutcomeStatus
from core.provider_errors import is_transient_provider_error

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip
    from models.frame import Frame

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
    target_type: Literal["clip", "frame"] = "clip"


@dataclass(frozen=True)
class DescriptionOptions:
    tier: str
    prompt: str = DEFAULT_PROMPT
    parallelism: int = 1
    model: str | None = None
    input_mode: str | None = None


def resolve_options(
    tier: str | None = None, prompt: str | None = None, parallelism: int = 1
) -> DescriptionOptions:
    """Snapshot non-secret provider settings before work is queued."""
    from core.settings import load_settings

    settings = load_settings()
    tier = resolve_tier(tier or settings.description_model_tier)
    return DescriptionOptions(
        tier,
        prompt or DEFAULT_PROMPT,
        parallelism,
        settings.description_model_local
        if tier == "local"
        else settings.description_model_cloud,
        settings.description_input_mode,
    )


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
                model_name=options.model,
                input_mode=options.input_mode,
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
    if options.model is None or options.input_mode is None:
        resolved = resolve_options(options.tier, options.prompt, options.parallelism)
        options = replace(
            resolved,
            model=options.model or resolved.model,
            input_mode=options.input_mode or resolved.input_mode,
        )
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


class DescriptionApplication:
    """Apply results once, on the owner thread, to unchanged clip/frame inputs."""

    def __init__(self, project: "Project", tasks: tuple[DescriptionTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self._tasks = {task.clip_id: task for task in tasks}
        self._bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self._consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: DescriptionTask) -> tuple | None:
        from core.jobs.media import media_stamp

        source = None
        target: Clip | Frame | None
        identity: tuple
        if task.target_type == "frame":
            target = project.frames_by_id.get(task.clip_id)
            if target is None or target.file_path != task.thumbnail_path:
                return None
            identity = (target.source_id, target.clip_id, target.frame_number)
        else:
            target = project.clips_by_id.get(task.clip_id)
            if target is None:
                return None
            source = project.sources_by_id.get(target.source_id)
            if (
                target.thumbnail_path != task.thumbnail_path
                or (target.start_frame, target.end_frame)
                != (task.start_frame, task.end_frame)
                or (source.file_path if source else None) != task.source_path
                or (source.fps if source else None) != task.fps
            ):
                return None
            identity = (target.source_id, target.start_frame, target.end_frame)
        image_stamp = media_stamp(task.thumbnail_path) if task.thumbnail_path else None
        source_stamp = media_stamp(task.source_path) if task.source_path else None
        if image_stamp is None:
            return None
        return (
            target,
            source,
            (
                identity,
                target.description,
                target.description_model,
                getattr(target, "description_frames", None),
                image_stamp,
                source_stamp,
            ),
        )

    def apply(self, project: "Project", outcome: DescriptionOutcome) -> bool:
        return self.apply_batch(project, (outcome,))[0]

    def apply_batch(
        self, project: "Project", outcomes: tuple[DescriptionOutcome, ...]
    ) -> tuple[bool, ...]:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or not any(o.status == "succeeded" for o in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            clips = []
            for outcome in outcomes:
                valid = False
                task = self._tasks.get(outcome.clip_id)
                expected = self._bindings.get(outcome.clip_id)
                if (
                    outcome.status == "succeeded"
                    and outcome.clip_id not in self._consumed
                ):
                    self._consumed.add(outcome.clip_id)
                    current = self._binding(project, task) if task else None
                    if (
                        task is not None
                        and current is not None
                        and expected is not None
                        and current[0] is expected[0]
                        and current[1] is expected[1]
                        and current[2] == expected[2]
                    ):
                        target = current[0]
                        if task.target_type == "frame":
                            project.update_frame(
                                outcome.clip_id,
                                description=outcome.description,
                                description_model=outcome.model,
                            )
                        else:
                            target.description = outcome.description
                            target.description_model = outcome.model
                            target.description_frames = 1
                            clips.append(target)
                        valid = True
                accepted.append(valid)
            if clips:
                project.update_clips(clips)
            return tuple(accepted)

        return project.session.apply_external(publish)
