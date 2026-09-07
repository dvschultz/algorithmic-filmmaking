"""Detached cinematography computation shared by GUI and headless adapters."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Event
from typing import Callable, Literal, TYPE_CHECKING

from core.operations.contracts import OutcomeStatus
from core.provider_errors import is_transient_provider_error
from models.cinematography import CinematographyAnalysis

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip
    from models.frame import Frame


@dataclass(frozen=True)
class CinematographyTask:
    clip_id: str
    thumbnail_path: Path | None
    source_path: Path | None
    start_frame: int
    end_frame: int
    fps: float | None
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"


@dataclass(frozen=True)
class CinematographyOptions:
    tier: str
    mode: str
    model: str
    local_model: str
    parallelism: int = 1


def resolve_options(
    mode: str | None = None, model: str | None = None, parallelism: int = 1
) -> CinematographyOptions:
    from core.settings import load_settings

    settings = load_settings()
    return CinematographyOptions(
        settings.cinematography_tier,
        mode or settings.cinematography_input_mode,
        model or settings.cinematography_model,
        settings.cinematography_local_model,
        parallelism,
    )


@dataclass(frozen=True)
class CinematographyOutcome:
    clip_id: str
    status: OutcomeStatus
    analysis_json: str | None = None
    code: str | None = None
    message: str | None = None

    @property
    def analysis(self) -> CinematographyAnalysis | None:
        return (
            CinematographyAnalysis.from_dict(json.loads(self.analysis_json))
            if self.analysis_json is not None
            else None
        )


def compute_cinematography(
    task: CinematographyTask, options: CinematographyOptions, cancel: Event
) -> CinematographyOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> CinematographyOutcome:
        return CinematographyOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    from core.analysis.cinematography import analyze_cinematography

    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        try:
            analysis = analyze_cinematography(
                thumbnail_path=task.thumbnail_path,
                source_path=task.source_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                fps=task.fps,
                mode="frame" if task.target_type == "frame" else options.mode,
                model=options.model,
                tier=options.tier,
                local_model=options.local_model,
            )
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            return outcome(
                "succeeded",
                analysis_json=json.dumps(
                    analysis.to_dict(), sort_keys=True, allow_nan=False
                ),
            )
        except Exception as exc:
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            if attempt < len(delays) and is_transient_provider_error(str(exc)):
                cancel.wait(delays[attempt])
                continue
            return outcome("failed", code="cinematography_failed", message=str(exc))
    raise AssertionError("Retry loop must return")


def run_cinematography(
    tasks: tuple[CinematographyTask, ...],
    options: CinematographyOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[CinematographyOutcome], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[CinematographyOutcome, ...]:
    """Bound cloud admission and run local inference serially on the caller thread."""
    cancel = cancel_event or Event()
    outcomes: dict[int, CinematographyOutcome] = {}

    def compute(task: CinematographyTask) -> CinematographyOutcome:
        try:
            return compute_cinematography(task, options, cancel)
        except Exception as exc:
            return CinematographyOutcome(
                task.clip_id, "failed", code="cinematography_failed", message=str(exc)
            )

    def publish(index: int, outcome: CinematographyOutcome) -> None:
        if cancel.is_set():
            return
        outcomes[index] = outcome
        if on_outcome:
            on_outcome(outcome)
        if progress:
            progress(len(outcomes), len(tasks), outcome.clip_id)

    if options.tier == "local":
        for index, task in enumerate(tasks):
            if cancel.is_set():
                break
            publish(index, compute(task))
    else:
        parallelism = min(max(1, options.parallelism), 5)
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            pending: dict[Future[CinematographyOutcome], int] = {}
            next_index = 0
            while pending or next_index < len(tasks):
                while (
                    not cancel.is_set()
                    and len(pending) < parallelism
                    and next_index < len(tasks)
                ):
                    pending[pool.submit(compute, tasks[next_index])] = next_index
                    next_index += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    publish(pending.pop(future), future.result())
                if cancel.is_set():
                    for future in pending:
                        future.cancel()
                    break
    return tuple(
        outcomes.get(
            i, CinematographyOutcome(task.clip_id, "unprocessed", code="cancelled")
        )
        for i, task in enumerate(tasks)
    )


class CinematographyApplication:
    """Apply analysis and derived shot type once to unchanged clip/frame inputs."""

    def __init__(
        self, project: "Project", tasks: tuple[CinematographyTask, ...]
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.clip_id: task for task in tasks}
        self.bindings = {task.clip_id: self._binding(project, task) for task in tasks}
        self.consumed: set[str] = set()

    @staticmethod
    def _binding(project: "Project", task: CinematographyTask) -> tuple | None:
        from core.jobs.media import media_stamp

        source = None
        source_path = None
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
            source_path = source.file_path if source else None
            if (
                target.thumbnail_path != task.thumbnail_path
                or (target.start_frame, target.end_frame)
                != (task.start_frame, task.end_frame)
                or (source_path if source_path and source_path.exists() else None)
                != task.source_path
                or (source.fps if source else None) != task.fps
            ):
                return None
            identity = (
                target.source_id,
                target.start_frame,
                target.end_frame,
                source_path,
            )
        image_stamp = media_stamp(task.thumbnail_path) if task.thumbnail_path else None
        if image_stamp is None:
            return None
        return (
            target,
            source,
            (
                identity,
                image_stamp,
                media_stamp(source_path) if source_path else None,
                target.cinematography.to_dict()
                if target.cinematography is not None
                else None,
                target.shot_type,
            ),
        )

    def apply(self, project: "Project", outcome: CinematographyOutcome) -> bool:
        return self.apply_batch(project, (outcome,))[0]

    def apply_batch(
        self, project: "Project", outcomes: tuple[CinematographyOutcome, ...]
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
                task = self.tasks.get(outcome.clip_id)
                expected = self.bindings.get(outcome.clip_id)
                if (
                    outcome.status == "succeeded"
                    and outcome.clip_id not in self.consumed
                ):
                    self.consumed.add(outcome.clip_id)
                    current = self._binding(project, task) if task else None
                    if (
                        task is not None
                        and expected is not None
                        and current is not None
                        and current[0] is expected[0]
                        and current[1] is expected[1]
                        and current[2] == expected[2]
                    ):
                        analysis = outcome.analysis
                        if analysis is not None:
                            shot_type = analysis.get_simple_shot_type()
                            if task.target_type == "frame":
                                project.update_frame(
                                    outcome.clip_id,
                                    cinematography=analysis,
                                    shot_type=shot_type,
                                )
                            else:
                                current[0].cinematography = analysis
                                current[0].shot_type = shot_type
                                clips.append(current[0])
                            valid = True
                accepted.append(valid)
            if clips:
                project.update_clips(clips)
            return tuple(accepted)

        return project.session.apply_external(publish)
