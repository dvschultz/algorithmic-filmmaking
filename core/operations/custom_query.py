"""Detached custom-query computation shared by GUI and headless callers."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from copy import deepcopy
from pathlib import Path
from math import isfinite
from threading import Event
from typing import Any, Callable, Literal, TYPE_CHECKING
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import (
    CUSTOM_QUERY_RESPONSE_SCHEMA,
    custom_query_prompt,
)
from models.analysis_record import AnalysisIdentity, AnalysisRecord

from core.operations.contracts import OutcomeStatus
from core.provider_errors import is_transient_provider_error

if TYPE_CHECKING:
    from core.settings import Settings
    from core.project import Project


@dataclass(frozen=True)
class CustomQueryTask:
    clip_id: str
    thumbnail_path: Path | None
    query: str
    skip: bool = False
    target_type: Literal["clip", "frame"] = "clip"
    analysis_json: str | None = None


@dataclass(frozen=True)
class CustomQueryOptions:
    tier: str
    model: str
    parallelism: int = 1


def custom_query_record_key(query: str) -> str:
    """Keep independent verification state for each exact, trimmed query."""
    return "custom_query:" + sha256(query.strip().encode()).hexdigest()


def latest_query_result(target: Any, query: str) -> dict | None:
    return next(
        (
            deepcopy(item)
            for item in reversed(target.custom_queries or [])
            if item.get("query") == query.strip()
        ),
        None,
    )


def custom_query_task(
    target: Any,
    source: Any,
    query: str,
    *,
    image_path: Path | None = None,
    skip_existing: bool = False,
) -> CustomQueryTask:
    if getattr(target, "target_type", "clip") != "clip":
        raise ValueError("Frame custom-query storage is not supported")
    query = query.strip()
    image = (
        image_path
        or getattr(target, "image_path", None)
        or getattr(target, "thumbnail_path", None)
    )
    video = source.file_path if source else getattr(target, "video_path", None)
    files = {"image": image} if image is not None else {}
    if video is not None:
        files["video"] = video
    snapshot = AnalysisSnapshot.capture(
        target,
        custom_query_record_key(query),
        files,
        {
            "start_frame": target.start_frame,
            "end_frame": target.end_frame,
            "fps": source.fps if source else getattr(target, "fps", None),
        },
        {"result": latest_query_result(target, query)},
    )
    return CustomQueryTask(
        target.id, image, query, skip_existing, analysis_json=snapshot.to_json()
    )


def custom_query_runtime(options: CustomQueryOptions, *, allow_imports: bool = True) -> dict:
    from core.operations.description import (
        DescriptionOptions,
        DescriptionTask,
        description_runtime,
    )

    tier = {"cpu": "local", "gpu": "cloud"}.get(options.tier, options.tier)
    runtime = description_runtime(
        DescriptionTask("", None, None, 0, 0, None),
        DescriptionOptions(tier, model=options.model, input_mode="frame"),
        allow_imports=allow_imports,
    )
    return {
        **runtime,
        "response_schema": CUSTOM_QUERY_RESPONSE_SCHEMA,
        "max_tokens": 50
        if tier == "cloud"
        else 256
        if runtime["execution"]["backend"] == "mlx"
        else None,
    }


def custom_query_identity(
    snapshot: AnalysisSnapshot,
    query: str,
    options: CustomQueryOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation=custom_query_record_key(query),
        operation_version=2,
        model=runtime,
        parameters={
            "tier": {"cpu": "local", "gpu": "cloud"}.get(options.tier, options.tier),
            "model": options.model,
            "query": query,
        },
        sampling={"policy": "single-image/v1"},
        prompt=custom_query_prompt(query),
    )


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
    def value(self) -> dict:
        return {
            "query": self.query,
            "match": self.match,
            "confidence": round(self.confidence or 0.0, 4),
            "model": self.model,
        }


def resolve_options(
    tier: str | None = None, parallelism: int = 1,
    *, settings: "Settings | None" = None,
) -> CustomQueryOptions:
    from core.settings import load_settings

    settings = settings if settings is not None else load_settings()
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
    task: CustomQueryTask,
    options: CustomQueryOptions,
    cancel: Event,
    *,
    fingerprints: AnalysisFingerprints | None = None,
) -> CustomQueryOutcome:
    def outcome(status: OutcomeStatus, **kwargs) -> CustomQueryOutcome:
        return CustomQueryOutcome(task.clip_id, task.query, status, **kwargs)

    if cancel.is_set():
        return outcome("unprocessed", code="cancelled")
    if not task.query.strip():
        return outcome("failed", code="missing_query", message="query is required")
    if task.skip and task.analysis_json is None:
        return outcome("skipped", code="already_populated")
    if task.thumbnail_path is None or not task.thumbnail_path.exists():
        return outcome("failed", code="thumbnail_missing")
    from core.analysis.custom_query import evaluate_custom_query

    snapshot = None
    runtime = None
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
    try:
        if task.analysis_json is not None:
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            runtime = custom_query_runtime(options)
            identity = custom_query_identity(
                snapshot, task.query, options, fingerprints, runtime
            )
            reused = snapshot.reusable_record(identity) if task.skip else None
            if reused is not None:
                value = reused.value["result"]
                return outcome(
                    "skipped",
                    match=value["match"],
                    confidence=value["confidence"],
                    model=value["model"],
                    code="valid_analysis",
                    record_json=json.dumps(reused.to_dict(), sort_keys=True),
                )
    except Exception as exc:
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        return outcome("failed", code="stale_input", message=str(exc))

    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        if cancel.is_set():
            return outcome("unprocessed", code="cancelled")
        try:
            if snapshot is not None and not snapshot.inputs.unchanged():
                raise ValueError("Custom-query input media changed")
            if snapshot is not None and options.tier in ("local", "cpu"):
                from core.analysis.description import _load_local_model

                _load_local_model(options.model)
                if cancel.is_set():
                    return outcome("unprocessed", code="cancelled")
                if not snapshot.inputs.unchanged():
                    raise ValueError(
                        "Custom-query input media changed during model loading"
                    )
            match, confidence, model = evaluate_custom_query(
                image_path=task.thumbnail_path,
                query=task.query,
                tier=options.tier,
                model_name=options.model,
            )
            if cancel.is_set():
                return outcome("unprocessed", code="cancelled")
            if type(match) is not bool:
                raise ValueError("Custom query match must be a boolean")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("Custom query confidence must be between 0 and 1")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("Custom query model must be a nonempty string")
            result = outcome(
                "succeeded", match=match, confidence=confidence, model=model
            )
            if snapshot is not None and runtime is not None:
                if custom_query_runtime(options) != runtime:
                    raise ValueError("Custom-query runtime changed")
                actual_runtime = {
                    **runtime,
                    "execution": {**runtime["execution"], "model": model},
                }
                identity = custom_query_identity(
                    snapshot, task.query, options, fingerprints, actual_runtime
                )
                record = AnalysisRecord.success(
                    identity,
                    {"result": result.value},
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
                identity = custom_query_identity(
                    snapshot, task.query, options, fingerprints, runtime
                )
                record = replace(
                    AnalysisRecord.failure(identity, str(exc)),
                    input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
                )
            return outcome(
                "failed",
                code="custom_query_failed",
                message=str(exc),
                record_json=json.dumps(record.to_dict(), sort_keys=True)
                if record
                else None,
            )
    raise AssertionError("Retry loop must return")


def run_custom_query(
    tasks: tuple[CustomQueryTask, ...],
    options: CustomQueryOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[CustomQueryOutcome], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
) -> tuple[CustomQueryOutcome, ...]:
    """Bound cloud admission; keep local inference on the caller's worker thread."""
    options = replace(
        options, tier={"cpu": "local", "gpu": "cloud"}.get(options.tier, options.tier)
    )
    cancel = cancel_event or Event()
    fingerprints = fingerprints or AnalysisFingerprints(cancel)
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
            try:
                result = compute_custom_query(
                    task, options, cancel, fingerprints=fingerprints
                )
            except Exception as exc:
                result = CustomQueryOutcome(
                    task.clip_id,
                    task.query,
                    "failed",
                    code="custom_query_failed",
                    message=str(exc),
                )
            publish(index, result)
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
                            compute_custom_query,
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

    def __init__(
        self,
        project: "Project",
        tasks: tuple[CustomQueryTask, ...],
        options: CustomQueryOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.options = options
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
                clip.analysis_records.get(custom_query_record_key(task.query)),
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
            or not any(outcome.can_apply for outcome in outcomes)
        ):
            return tuple(False for _ in outcomes)

        def publish() -> tuple[bool, ...]:
            accepted = []
            updated = []
            for outcome in outcomes:
                task = self.tasks.get(outcome.clip_id)
                expected = self.bindings.get(outcome.clip_id)
                valid = False
                if outcome.can_apply and outcome.clip_id not in self.consumed:
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
                        key = custom_query_record_key(task.query)
                        record = (
                            AnalysisRecord.from_dict(json.loads(outcome.record_json))
                            if outcome.record_json is not None
                            else AnalysisRecord.legacy({"result": outcome.value})
                        )
                        if outcome.record_json is not None:
                            snapshot = (
                                AnalysisSnapshot.from_json(task.analysis_json)
                                if task.analysis_json
                                else None
                            )
                            prior = clip.analysis_records.get(key)
                            if (
                                snapshot is None
                                or record.identity is None
                                or record.identity.operation != key
                                or record.identity.to_dict()["operation_version"] != 2
                                or record.identity.to_dict()["parameters"].get("query")
                                != task.query
                                or record.identity.to_dict()["prompt_sha256"]
                                != sha256(
                                    custom_query_prompt(task.query).encode()
                                ).hexdigest()
                                or not snapshot.inputs.unchanged()
                                or json.loads(record.input_json or "null")
                                != snapshot.inputs.to_dict()
                                or snapshot.record
                                != (
                                    prior if isinstance(prior, AnalysisRecord) else None
                                )
                                or json.loads(snapshot.value_json)
                                != {"result": latest_query_result(clip, task.query)}
                                or (
                                    outcome.has_result
                                    and (
                                        record.state != "succeeded"
                                        or record.value != {"result": outcome.value}
                                    )
                                )
                                or (
                                    outcome.status == "failed"
                                    and record.state != "failed"
                                )
                            ):
                                accepted.append(False)
                                continue
                            if self.options is not None and record.identity.to_dict()[
                                "parameters"
                            ] != {
                                "tier": {"cpu": "local", "gpu": "cloud"}.get(
                                    self.options.tier, self.options.tier
                                ),
                                "model": self.options.model,
                                "query": task.query,
                            }:
                                accepted.append(False)
                                continue
                        project.record_analysis("clip", outcome.clip_id, key, record)
                        if outcome.status != "succeeded":
                            accepted.append(True)
                            continue
                        clip.custom_queries = [
                            *(clip.custom_queries or []),
                            outcome.value,
                        ]
                        updated.append(clip)
                        valid = True
                accepted.append(valid)
            if updated:
                project.update_clips(updated)
            return tuple(accepted)

        return project.session.apply_external(publish)
