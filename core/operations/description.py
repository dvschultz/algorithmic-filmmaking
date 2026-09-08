"""Detached description computation shared by GUI and headless callers."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import Event
from typing import Any, Callable, TYPE_CHECKING, Literal, cast

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot, model_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord
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
    analysis_json: str | None = None


@dataclass(frozen=True)
class DescriptionOptions:
    tier: str
    prompt: str = DEFAULT_PROMPT
    parallelism: int = 1
    model: str | None = None
    input_mode: str | None = None


def description_task(
    target: Any,
    source: Any = None,
    *,
    image_path: Path | None = None,
    skip_existing: bool = True,
) -> DescriptionTask:
    """Detach media, editorial bindings, and previous analysis before dispatch."""
    kind = getattr(
        target, "target_type", "clip" if hasattr(target, "start_frame") else "frame"
    )
    if kind not in ("clip", "frame"):
        raise ValueError("Invalid description target type")
    image = (
        image_path
        or getattr(target, "image_path", None)
        or (
            getattr(target, "file_path", None)
            if kind == "frame"
            else getattr(target, "thumbnail_path", None)
        )
    )
    video = (
        (source.file_path if source else getattr(target, "video_path", None))
        if kind == "clip"
        else None
    )
    image = Path(image) if image is not None else None
    video = Path(video) if video is not None else None
    fps = source.fps if source else getattr(target, "fps", None)
    source_range = (
        {"start_frame": target.start_frame, "end_frame": target.end_frame, "fps": fps}
        if kind == "clip"
        else {"frame_number": getattr(target, "frame_number", None)}
    )
    files = {"image": image} if image else {}
    if video is not None:
        files["video"] = video
    snapshot = AnalysisSnapshot.capture(
        target,
        "describe",
        files,
        source_range,
        {
            "description": target.description,
            "description_model": getattr(target, "description_model", None),
            "description_frames": getattr(target, "description_frames", None),
        },
    )
    return DescriptionTask(
        target.id,
        image,
        video,
        getattr(target, "start_frame", None) or 0,
        getattr(target, "end_frame", None) or 0,
        fps,
        skip_existing,
        cast(Literal["clip", "frame"], kind),
        snapshot.to_json(),
    )


def description_runtime(task: DescriptionTask, options: DescriptionOptions) -> dict:
    """Resolve expected execution on the worker, without loading model weights."""
    from core.analysis.description import is_mlx_vlm_available, is_video_capable_model
    from core.analysis_model_identity import local_description_runtime

    if options.model is None or options.input_mode is None:
        raise ValueError("Description options must be resolved before computation")
    packages: tuple[str, ...]
    if resolve_tier(options.tier) == "local":
        execution = {
            **local_description_runtime(options.model, mlx=is_mlx_vlm_available()),
            "input_mode": "frame",
        }
        packages = (
            ("mlx-vlm", "mlx", "Pillow")
            if execution["backend"] == "mlx"
            else ("torch", "transformers", "Pillow")
        )
    else:
        video = (
            options.input_mode == "video"
            and is_video_capable_model(options.model)
            and task.source_path is not None
            and task.fps is not None
        )
        execution = {
            "backend": "cloud",
            "model": options.model,
            "input_mode": "video" if video else "frame",
        }
        packages = ("litellm", "Pillow")
    runtime = {
        "execution": execution,
        "packages": model_runtime("describe", packages)["packages"],
    }
    if execution["input_mode"] == "video":
        from core.binary_resolver import find_binary
        from core.jobs.media import media_stamp

        binary = find_binary("ffmpeg")
        runtime["video_extraction"] = {
            "algorithm": "description-segment/v1",
            "ffmpeg": str(binary) if binary else None,
            "ffmpeg_stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
        }
    return runtime


def description_identity(
    snapshot: AnalysisSnapshot,
    options: DescriptionOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
    *,
    execution: dict | None = None,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="describe",
        operation_version=2,
        model={**runtime, "execution": execution or runtime["execution"]},
        parameters={
            "tier": resolve_tier(options.tier),
            "model": options.model,
            "input_mode": options.input_mode,
        },
        sampling={"policy": "description-input/v1"},
        prompt=options.prompt,
    )


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


def resolve_tier(tier: str | None) -> str:
    if not tier:
        from core.settings import load_settings

        tier = load_settings().description_model_tier
    return "local" if tier in ("cpu", "gpu") else tier


def compute_description(
    task: DescriptionTask,
    options: DescriptionOptions,
    cancel: Event,
    *,
    fingerprints: AnalysisFingerprints | None = None,
) -> DescriptionOutcome:
    if cancel.is_set():
        return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
    if task.skip and task.analysis_json is None:
        return DescriptionOutcome(task.clip_id, "skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return DescriptionOutcome(task.clip_id, "failed", code="thumbnail_missing")

    from core.analysis.description import describe_frame

    snapshot = None
    identity = None
    runtime = None
    execution = None
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    try:
        if task.analysis_json is not None:
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            runtime = description_runtime(task, options)
            execution = runtime["execution"]
            identity = description_identity(snapshot, options, fingerprints, runtime)
            reused = snapshot.reusable_record(identity) if task.skip else None
            if reused is not None:
                return DescriptionOutcome(
                    task.clip_id,
                    "skipped",
                    reused.value["description"],
                    reused.value["description_model"],
                    code="valid_analysis",
                    record_json=json.dumps(reused.to_dict(), sort_keys=True),
                )
    except Exception as exc:
        if cancel.is_set():
            return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
        return DescriptionOutcome(
            task.clip_id, "failed", code="stale_input", message=str(exc)
        )

    def record_execution(actual: dict) -> None:
        nonlocal execution
        execution = dict(actual)

    for attempt in range(len(RETRY_DELAYS) + 1):
        if cancel.is_set():
            return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
        try:
            if snapshot is not None and not snapshot.inputs.unchanged():
                raise ValueError("Description input media changed")
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
                **({"on_execution": record_execution} if snapshot is not None else {}),
            )
            if cancel.is_set():
                return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
            if (
                not description
                or not description.strip()
                or description.startswith("Error")
            ):
                raise ValueError(
                    description or "Provider returned an empty description"
                )
            record = None
            if snapshot is not None and runtime is not None:
                if description_runtime(task, options) != runtime:
                    raise ValueError("Description runtime changed during inference")
                identity = description_identity(
                    snapshot,
                    options,
                    fingerprints,
                    runtime,
                    execution=execution,
                )
                record = AnalysisRecord.success(
                    identity,
                    {
                        "description": description,
                        "description_model": model,
                        "description_frames": 1 if task.target_type == "clip" else None,
                    },
                    input_snapshot=snapshot.inputs.to_dict(),
                )
            return DescriptionOutcome(
                task.clip_id,
                "succeeded",
                description,
                model,
                record_json=json.dumps(record.to_dict(), sort_keys=True)
                if record
                else None,
            )
        except Exception as exc:
            if cancel.is_set():
                return DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
            if attempt < len(RETRY_DELAYS) and is_transient_provider_error(str(exc)):
                cancel.wait(RETRY_DELAYS[attempt])
                continue
            record = None
            if (
                snapshot is not None
                and runtime is not None
                and snapshot.inputs.unchanged()
            ):
                identity = description_identity(
                    snapshot, options, fingerprints, runtime, execution=execution
                )
                record = replace(
                    AnalysisRecord.failure(identity, str(exc)),
                    input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
                )
            return DescriptionOutcome(
                task.clip_id,
                "failed",
                code="description_failed",
                message=str(exc),
                record_json=json.dumps(record.to_dict(), sort_keys=True)
                if record
                else None,
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
    fingerprints = AnalysisFingerprints(cancel)
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
                    pool.submit(
                        compute_description,
                        tasks[next_index],
                        options,
                        cancel,
                        fingerprints=fingerprints,
                    )
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

    def __init__(
        self,
        project: "Project",
        tasks: tuple[DescriptionTask, ...],
        options: DescriptionOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.options = options
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
                target.analysis_records.get("describe"),
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
            or not any(o.can_apply for o in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            clips = []
            for outcome in outcomes:
                valid = False
                task = self._tasks.get(outcome.clip_id)
                expected = self._bindings.get(outcome.clip_id)
                if outcome.can_apply and outcome.clip_id not in self._consumed:
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
                        value = {
                            "description": outcome.description,
                            "description_model": outcome.model,
                            "description_frames": 1
                            if task.target_type == "clip"
                            else None,
                        }
                        record = (
                            AnalysisRecord.from_dict(json.loads(outcome.record_json))
                            if outcome.record_json is not None
                            else AnalysisRecord.legacy(value)
                        )
                        if outcome.record_json is not None:
                            snapshot = (
                                AnalysisSnapshot.from_json(task.analysis_json)
                                if task.analysis_json
                                else None
                            )
                            if (
                                snapshot is None
                                or record.identity is None
                                or record.identity.operation != "describe"
                                or json.loads(record.input_json or "null")
                                != snapshot.inputs.to_dict()
                                or not snapshot.inputs.unchanged()
                                or snapshot.record
                                != target.analysis_records.get("describe")
                                or json.loads(snapshot.value_json)
                                != {
                                    "description": target.description,
                                    "description_model": target.description_model,
                                    "description_frames": getattr(
                                        target, "description_frames", None
                                    ),
                                }
                                or (
                                    outcome.has_result
                                    and (
                                        record.state != "succeeded"
                                        or record.value != value
                                    )
                                )
                                or (
                                    outcome.status == "failed"
                                    and record.state != "failed"
                                )
                            ):
                                accepted.append(False)
                                continue
                            if self.options is not None:
                                from hashlib import sha256

                                identity = record.identity.to_dict()
                                if (
                                    identity["parameters"]
                                    != {
                                        "tier": resolve_tier(self.options.tier),
                                        "model": self.options.model,
                                        "input_mode": self.options.input_mode,
                                    }
                                    or identity["prompt_sha256"]
                                    != sha256(self.options.prompt.encode()).hexdigest()
                                ):
                                    accepted.append(False)
                                    continue
                        project.record_analysis(
                            task.target_type, outcome.clip_id, "describe", record
                        )
                        if outcome.status == "failed":
                            accepted.append(True)
                            continue
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
