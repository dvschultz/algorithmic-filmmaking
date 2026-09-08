"""Detached cinematography computation shared by GUI and headless adapters."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Any, Callable, Literal, TYPE_CHECKING, cast

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot, model_runtime
from models.analysis_record import AnalysisIdentity, AnalysisRecord

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
    snapshot_json: str | None = None


@dataclass(frozen=True)
class CinematographyOptions:
    tier: str
    mode: str
    model: str
    local_model: str
    parallelism: int = 1


def cinematography_value(target: Any) -> dict:
    return {
        "cinematography": target.cinematography.to_dict()
        if target.cinematography
        else None,
        "shot_type": target.shot_type,
    }


def cinematography_task(
    target: Any,
    source: Any = None,
    *,
    image_path: Path | None = None,
    skip_existing: bool = True,
) -> CinematographyTask:
    kind = getattr(
        target, "target_type", "clip" if hasattr(target, "start_frame") else "frame"
    )
    if kind not in ("clip", "frame"):
        raise ValueError("Invalid cinematography target type")
    image = (
        image_path
        or getattr(target, "image_path", None)
        or getattr(target, "thumbnail_path", None)
        or getattr(target, "file_path", None)
    )
    video = (
        (source.file_path if source else getattr(target, "video_path", None))
        if kind == "clip"
        else None
    )
    video = video if video is not None and video.exists() else None
    fps = source.fps if source else getattr(target, "fps", None)
    start, end = (
        getattr(target, "start_frame", 0) or 0,
        getattr(target, "end_frame", 0) or 0,
    )
    source_range = (
        {"start_frame": start, "end_frame": end, "fps": fps}
        if kind == "clip"
        else {"frame_number": target.frame_number}
    )
    files = {"image": image} if image else {}
    if video is not None:
        files["video"] = video
    snapshot = AnalysisSnapshot.capture(
        target, "cinematography", files, source_range, cinematography_value(target)
    )
    return CinematographyTask(
        target.id,
        image,
        video,
        start,
        end,
        fps,
        skip_existing,
        cast(Literal["clip", "frame"], kind),
        snapshot.to_json(),
    )


def cinematography_parameters(options: CinematographyOptions) -> dict:
    return {
        "tier": options.tier,
        "mode": options.mode,
        "model": options.local_model if options.tier == "local" else options.model,
    }


def cinematography_prompt(execution: dict) -> str:
    from core.analysis.cinematography import (
        CINEMATOGRAPHY_PROMPT_FRAME,
        CINEMATOGRAPHY_PROMPT_VIDEO,
    )

    if execution["input_mode"] == "video":
        return CINEMATOGRAPHY_PROMPT_VIDEO
    return CINEMATOGRAPHY_PROMPT_FRAME + (
        "\n\nRespond with ONLY a JSON object, no other text."
        if execution["backend"] == "mlx"
        else ""
    )


def cinematography_runtime(
    task: CinematographyTask,
    options: CinematographyOptions,
    *,
    execution: dict | None = None,
) -> dict:
    from core.analysis.cinematography import CINEMATOGRAPHY_SCHEMA

    if execution is None:
        if options.tier == "local":
            from core.analysis.description import is_mlx_vlm_available

            execution = {
                "backend": "mlx" if is_mlx_vlm_available() else "unavailable",
                "model": options.local_model,
                "input_mode": "frame",
            }
        else:
            video = (
                task.target_type == "clip"
                and options.mode == "video"
                and task.source_path is not None
                and task.source_path.exists()
                and task.fps is not None
            )
            execution = {
                "backend": "cloud",
                "model": options.model,
                "input_mode": "video" if video else "frame",
            }
    runtime = {
        "execution": execution,
        "packages": model_runtime(
            "cinematography",
            ("mlx", "mlx-vlm", "Pillow")
            if options.tier == "local"
            else ("litellm", "Pillow"),
        )["packages"],
        "normalization": "cinematography/v1",
        "schema_sha256": sha256(
            json.dumps(CINEMATOGRAPHY_SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
        "max_tokens": 256 if options.tier == "local" else None,
    }
    if execution["input_mode"] == "video":
        from core.binary_resolver import find_binary
        from core.jobs.media import media_stamp

        binary = find_binary("ffmpeg")
        runtime["extraction"] = {
            "algorithm": "half-open-segment/v1",
            "max_video_size_mb": 20.0,
            "ffmpeg": str(binary) if binary else None,
            "stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
        }
    return runtime


def cinematography_identity(
    snapshot: AnalysisSnapshot,
    options: CinematographyOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="cinematography",
        operation_version=2,
        model=runtime,
        parameters=cinematography_parameters(options),
        sampling={"policy": runtime["execution"]["input_mode"] + "/v1"},
        prompt=cinematography_prompt(runtime["execution"]),
    )


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

    @property
    def analysis(self) -> CinematographyAnalysis | None:
        return (
            CinematographyAnalysis.from_dict(json.loads(self.analysis_json))
            if self.analysis_json is not None
            else None
        )


def compute_cinematography(
    task: CinematographyTask,
    options: CinematographyOptions,
    cancel: Event,
    *,
    fingerprints: AnalysisFingerprints | None = None,
) -> CinematographyOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> CinematographyOutcome:
        return CinematographyOutcome(task.clip_id, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if task.skip and task.snapshot_json is None:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    from core.analysis.cinematography import analyze_cinematography

    snapshot = None
    runtime = None
    execution = None
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    try:
        if task.snapshot_json is not None:
            snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
            runtime = cinematography_runtime(task, options)
            identity = cinematography_identity(snapshot, options, fingerprints, runtime)
            reused = snapshot.reusable_record(identity) if task.skip else None
            if reused is not None:
                return outcome(
                    "skipped",
                    code="valid_analysis",
                    analysis_json=json.dumps(
                        reused.value["cinematography"], sort_keys=True
                    ),
                    record_json=json.dumps(reused.to_dict(), sort_keys=True),
                )
    except Exception as exc:
        return (
            outcome("unprocessed", code="cancelled")
            if cancel.is_set()
            else outcome("failed", code="stale_input", message=str(exc))
        )

    def record_execution(actual: dict) -> None:
        nonlocal execution
        execution = dict(actual)

    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        try:
            if snapshot is not None and not snapshot.inputs.unchanged():
                raise ValueError("Cinematography input media changed")
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
                **({"on_execution": record_execution} if snapshot is not None else {}),
            )
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            result = outcome(
                "succeeded",
                analysis_json=json.dumps(
                    analysis.to_dict(), sort_keys=True, allow_nan=False
                ),
            )
            if snapshot is not None and runtime is not None:
                if cinematography_runtime(task, options) != runtime:
                    raise ValueError("Cinematography runtime changed")
                actual = execution or {
                    **runtime["execution"],
                    "model": analysis.analysis_model or runtime["execution"]["model"],
                    "input_mode": analysis.analysis_mode,
                }
                actual_runtime = cinematography_runtime(task, options, execution=actual)
                identity = cinematography_identity(
                    snapshot, options, fingerprints, actual_runtime
                )
                record = AnalysisRecord.success(
                    identity,
                    {
                        "cinematography": analysis.to_dict(),
                        "shot_type": analysis.get_simple_shot_type(),
                    },
                    input_snapshot=snapshot.inputs.to_dict(),
                )
                result = replace(
                    result, record_json=json.dumps(record.to_dict(), sort_keys=True)
                )
            return result
        except Exception as exc:
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            if attempt < len(delays) and is_transient_provider_error(str(exc)):
                cancel.wait(delays[attempt])
                continue
            record = None
            if (
                snapshot is not None
                and runtime is not None
                and snapshot.inputs.unchanged()
            ):
                actual_runtime = (
                    cinematography_runtime(task, options, execution=execution)
                    if execution is not None
                    else runtime
                )
                identity = cinematography_identity(
                    snapshot, options, fingerprints, actual_runtime
                )
                record = replace(
                    AnalysisRecord.failure(identity, str(exc)),
                    input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
                )
            return outcome(
                "failed",
                code="cinematography_failed",
                message=str(exc),
                record_json=json.dumps(record.to_dict(), sort_keys=True)
                if record
                else None,
            )
    raise AssertionError("Retry loop must return")


def run_cinematography(
    tasks: tuple[CinematographyTask, ...],
    options: CinematographyOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[CinematographyOutcome], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
) -> tuple[CinematographyOutcome, ...]:
    """Bound cloud admission and run local inference serially on the caller thread."""
    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    outcomes: dict[int, CinematographyOutcome] = {}

    def compute(task: CinematographyTask) -> CinematographyOutcome:
        try:
            return compute_cinematography(
                task, options, cancel, fingerprints=fingerprints
            )
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
        self,
        project: "Project",
        tasks: tuple[CinematographyTask, ...],
        options: CinematographyOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.options = options
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
                target.analysis_records.get("cinematography"),
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
            or not any(o.can_apply for o in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            clips = []
            for outcome in outcomes:
                valid = False
                task = self.tasks.get(outcome.clip_id)
                expected = self.bindings.get(outcome.clip_id)
                if outcome.can_apply and outcome.clip_id not in self.consumed:
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
                        record = (
                            AnalysisRecord.from_dict(json.loads(outcome.record_json))
                            if outcome.record_json is not None
                            else None
                        )
                        if record is not None:
                            snapshot = (
                                AnalysisSnapshot.from_json(task.snapshot_json)
                                if task.snapshot_json
                                else None
                            )
                            prior = current[0].analysis_records.get("cinematography")
                            if (
                                snapshot is None
                                or record.identity is None
                                or record.identity.operation != "cinematography"
                                or record.identity.to_dict()["operation_version"] != 2
                                or record.identity.to_dict()["prompt_sha256"]
                                != sha256(
                                    cinematography_prompt(
                                        record.identity.to_dict()["model"]["execution"]
                                    ).encode()
                                ).hexdigest()
                                or not snapshot.inputs.unchanged()
                                or json.loads(record.input_json or "null")
                                != snapshot.inputs.to_dict()
                                or snapshot.record
                                != (
                                    prior if isinstance(prior, AnalysisRecord) else None
                                )
                                or json.loads(snapshot.value_json)
                                != cinematography_value(current[0])
                                or (
                                    self.options is not None
                                    and record.identity.to_dict()["parameters"]
                                    != cinematography_parameters(self.options)
                                )
                                or (
                                    outcome.has_result
                                    and (
                                        analysis is None
                                        or record.state != "succeeded"
                                        or record.value
                                        != {
                                            "cinematography": analysis.to_dict(),
                                            "shot_type": analysis.get_simple_shot_type(),
                                        }
                                    )
                                )
                                or (
                                    outcome.status == "failed"
                                    and record.state != "failed"
                                )
                            ):
                                accepted.append(False)
                                continue
                            project.record_analysis(
                                task.target_type,
                                outcome.clip_id,
                                "cinematography",
                                record,
                            )
                            if outcome.status != "succeeded":
                                accepted.append(True)
                                continue
                        if analysis is not None:
                            shot_type = analysis.get_simple_shot_type()
                            if record is None:
                                project.record_analysis(
                                    task.target_type,
                                    outcome.clip_id,
                                    "cinematography",
                                    AnalysisRecord.legacy(
                                        {
                                            "cinematography": analysis.to_dict(),
                                            "shot_type": shot_type,
                                        }
                                    ),
                                )
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
