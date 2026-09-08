"""Generic job-management MCP tools.

These five tools sit on top of the jobs framework in
``core/jobs/`` (with MCP compatibility exports) and serve every long-running op (R5 — existing
synchronous tools keep working alongside).

Field names use ``snake_case`` consistent with the rest of the codebase
(R26); they will be mechanically renamed if/when the SEP-1686 Tasks spec
stabilises.

Information-disclosure discipline (R28):
- ``list_jobs`` and ``get_job_status`` return the safe projection only —
  no ``args_json``, no ``result_json``, no ``error`` payload.
- ``get_job_result`` is the single tool that surfaces the sensitive
  payload, and only when the row is in a terminal state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from core.jobs.spec import OperationSpec
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.jobs.store import (
    JobNotFoundError,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_CRASHED,
    STATUS_FAILED,
    TERMINAL_STATUSES,
)
from scene_ripper_mcp.server import mcp

logger = logging.getLogger(__name__)


def _lifespan(ctx: Context) -> dict:
    """Return the lifespan context dict from a request context.

    FastMCP exposes the dict yielded by ``lifespan`` as
    ``ctx.request_context.lifespan_context``.
    """
    return ctx.request_context.lifespan_context


def _wrap_error(error: BaseException, *, code: str = "tool_error") -> dict:
    """Translate a raised exception into a structured error response.

    ``asyncio.CancelledError`` becomes a structured error rather than
    propagating — defends FastMCP SDK #1152 (R25).
    """
    if isinstance(error, asyncio.CancelledError):
        return {"success": False, "error": {"code": "cancelled", "message": "client cancelled request"}}
    return {"success": False, "error": {"code": code, "message": str(error)}}


@mcp.tool()
async def get_job_status(
    task_id: Annotated[str, "Job task_id returned by start_*"],
    ctx: Context = None,
) -> str:
    """Return the current status, progress, and queue position of a job.

    Excludes sensitive payload (args, result, error traceback). Use
    ``get_job_result`` after status reaches a terminal value to fetch the
    result.
    """
    try:
        store = _lifespan(ctx)["job_store"]
        try:
            row = store.get(task_id)
        except JobNotFoundError:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "job_not_found",
                        "message": f"No job with task_id={task_id!r}",
                    },
                }
            )
        return json.dumps(
            {
                "success": True,
                **row.to_safe_projection(),
                "poll_interval": 5,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        logger.exception("get_job_status failed")
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def get_job_result(
    task_id: Annotated[str, "Job task_id returned by start_*"],
    ctx: Context = None,
) -> str:
    """Return terminal output and, for failed/cancelled/crashed jobs, an error.

    Error responses include ``result`` when the runner recorded available output.
    The terminal status still describes the overall job, not each item's outcome.

    Returns a ``not_terminal`` error when the job is still running or
    queued — never returns a partial result.
    """
    try:
        store = _lifespan(ctx)["job_store"]
        try:
            row = store.get(task_id)
        except JobNotFoundError:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "job_not_found",
                        "message": f"No job with task_id={task_id!r}",
                    },
                }
            )

        if row.status not in TERMINAL_STATUSES:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "not_terminal",
                        "message": (
                            f"Job is in status {row.status!r}; "
                            "use get_job_status to poll, then re-call "
                            "get_job_result once it reaches a terminal state."
                        ),
                        "status": row.status,
                        "progress": row.progress,
                    },
                }
            )

        payload = row.result
        if row.status == STATUS_COMPLETED:
            return json.dumps(
                {
                    "success": True,
                    "task_id": row.id,
                    "status": row.status,
                    "result": payload,
                }
            )

        # Failed / cancelled / crashed — surface the sanitized error.
        code_map = {
            STATUS_FAILED: "job_failed",
            STATUS_CANCELLED: "job_cancelled",
            STATUS_CRASHED: "job_crashed",
        }
        error = {"code": code_map[row.status], "message": row.error or row.status}
        if row.status == STATUS_FAILED and isinstance(payload, dict):
            conflict = payload.get("error")
            if isinstance(conflict, dict) and conflict.get("code") == "project_busy":
                error = conflict
        return json.dumps(
            {
                "success": False,
                "task_id": row.id,
                "status": row.status,
                "error": error,
                **({"result": payload} if payload is not None else {}),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        logger.exception("get_job_result failed")
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def cancel_job(
    task_id: Annotated[str, "Job task_id to cancel"],
    ctx: Context = None,
) -> str:
    """Signal cancellation of a running or queued job.

    Returns ``ok=true`` when the cancel event was set; ``ok=false`` when
    the job is already terminal or unknown.
    """
    try:
        runtime = _lifespan(ctx)["job_runtime"]
        store = _lifespan(ctx)["job_store"]
        try:
            row = store.get(task_id)
        except JobNotFoundError:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "job_not_found",
                        "message": f"No job with task_id={task_id!r}",
                    },
                }
            )

        if row.status in TERMINAL_STATUSES:
            return json.dumps(
                {
                    "success": False,
                    "ok": False,
                    "task_id": task_id,
                    "status": row.status,
                    "error": {
                        "code": "already_terminal",
                        "message": (
                            f"Job is already in terminal status {row.status!r}"
                        ),
                    },
                }
            )

        ok = runtime.cancel(task_id)
        return json.dumps(
            {
                "success": True,
                "ok": ok,
                "task_id": task_id,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        logger.exception("cancel_job failed")
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def list_jobs(
    status_filter: Annotated[
        Optional[list[str]],
        "Optional list of statuses to include (e.g. ['queued', 'running']).",
    ] = None,
    kind_filter: Annotated[
        Optional[str],
        "Optional kind (op type) to filter by.",
    ] = None,
    project_filter: Annotated[
        Optional[str],
        "Optional project_path to filter by (canonical absolute path).",
    ] = None,
    ctx: Context = None,
) -> str:
    """List jobs with the safe-projection shape (no payload columns).

    Use ``get_job_result`` to fetch the result/error of any specific job.
    """
    try:
        store = _lifespan(ctx)["job_store"]
        rows = store.list(
            status_filter=status_filter,
            kind_filter=kind_filter,
            project_filter=project_filter,
        )
        return json.dumps(
            {
                "success": True,
                "count": len(rows),
                "jobs": [r.to_safe_projection() for r in rows],
            }
        )
    except BaseException as exc:  # noqa: BLE001
        logger.exception("list_jobs failed")
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def purge_old_jobs(
    days: Annotated[int, "Delete terminal jobs older than this many days"] = 30,
    ctx: Context = None,
    include_results: Annotated[
        bool, "Also delete old committed computation receipts released by all tracked projects. Legacy and recoverable receipts are retained."
    ] = False,
) -> str:
    """Delete terminal-status job rows older than ``days``.

    Running and queued rows are never purged. There is no automatic TTL on
    ``start_*`` calls (R22) — pruning is explicit.
    ``include_results`` additionally releases eligible receipt payload owners;
    managed-artifact collection can then reclaim their unreferenced files.
    """
    try:
        if type(days) is not int or days < 0:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "invalid_days",
                        "message": "days must be a nonnegative integer",
                    },
                }
            )
        store = _lifespan(ctx)["job_store"]
        deleted_results = (
            await asyncio.to_thread(store.purge_old_results, days=days)
            if include_results else None
        )
        deleted = store.purge_old_jobs(days=days)
        return json.dumps(
            {
                "success": True,
                "deleted_count": deleted,
                "days": days,
                **({"deleted_result_count": deleted_results} if include_results else {}),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        logger.exception("purge_old_jobs failed")
        return json.dumps(_wrap_error(exc))


# =============================================================================
# start_* tools — long-running ops dispatched through the jobs framework.
# =============================================================================


def _start_job(
    ctx: Context,
    *,
    kind: str,
    args: dict,
    project_path: Optional[str],
    project_mtime_at_start: Optional[float],
    idempotency_key: Optional[str],
    run,
    operation: OperationSpec | None = None,
) -> str:
    """Common path for start_* tools: validate, submit, wrap errors."""
    from scene_ripper_mcp.jobs.runtime import InvalidIdempotencyKeyError

    try:
        runtime = _lifespan(ctx)["job_runtime"]
        try:
            result = runtime.submit(
                kind=kind,
                args=args,
                run=run,
                project_path=project_path,
                project_mtime_at_start=project_mtime_at_start,
                idempotency_key=idempotency_key,
                operation=operation,
            )
        except InvalidIdempotencyKeyError as exc:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "invalid_idempotency_key",
                        "message": str(exc),
                    },
                }
            )
        return json.dumps({"success": True, **result})
    except BaseException as exc:  # noqa: BLE001
        logger.exception("start_%s failed", kind)
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def start_detect_scenes_bulk(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    source_ids: Annotated[
        list[str],
        "List of source IDs to detect scenes on (use list_sources to discover)",
    ],
    sensitivity: Annotated[
        float, "Detection sensitivity (1.0=more scenes, 10.0=fewer)"
    ] = 3.0,
    idempotency_key: Annotated[
        Optional[str],
        "Optional idempotency key (max 255 chars) — same key + same project = "
        "same job",
    ] = None,
    ctx: Context = None,
) -> str:
    """Start a bulk scene-detection job over an existing project's sources.

    Returns ``{task_id, status: "queued", poll_interval}`` immediately.
    Poll with ``get_job_status``; fetch the result with ``get_job_result``
    after status reaches a terminal value.

    Each source is saved with a result receipt before checkpointing. Retries
    reuse recorded clips. Cancellation stops later sources; ordinary per-source
    failures are aggregated while persistence failures stop the job.
    """
    from core.project import MissingSourceError
    from scene_ripper_mcp.security import validate_project_path
    from core.spine.project_io import load_with_mtime

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    # Validate at submission; the worker reloads under writer ownership.
    try:
        project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    from core.jobs.detection import saved_detection_spec, run_saved_detection
    from core.jobs.commits import StaleJobResult
    from core.project_revision import ProjectFileRevision

    store = _lifespan(ctx)["job_store"]
    try:
        operation = saved_detection_spec(project, path, source_ids, sensitivity)
    except ValueError as exc:
        return json.dumps(_wrap_error(exc))

    def run(progress_callback, cancel_event):
        frozen = operation.arguments
        if operation.input_revision is not None:
            ProjectFileRevision(path, operation.input_revision).verify()
        current, _ = load_with_mtime(path)
        live = saved_detection_spec(current, path, frozen["source_ids"], frozen["sensitivity"])
        if live.inputs_json != operation.inputs_json:
            raise StaleJobResult("Detection inputs changed while the job was queued")
        return run_saved_detection(store, path, frozen["source_ids"], frozen["sensitivity"], progress_callback, cancel_event)

    return _start_job(
        ctx,
        kind="detect_scenes_bulk",
        args=operation.arguments,
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
        operation=operation,
    )


def _make_analyze_runner(spine_fn_name: str, *, save_after: bool = True, **op_kwargs):
    """Build the closure for an analyze-style start_* tool.

    The closure loads the project, runs the spine fn with the supplied
    ``op_kwargs``, and (when ``save_after``) saves with the mtime guard.
    Returns the spine result dict, or — when the save aborts —
    annotates it with ``project_modified_externally``.
    """

    def runner(path, mtime, clip_ids):
        from core.project import MissingSourceError
        from core.spine import analyze as analyze_module
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )

        def run(progress_callback, cancel_event):
            try:
                project, captured_mtime = load_with_mtime(path)
            except MissingSourceError as exc:
                return {
                    "success": False,
                    "error": {
                        "code": "source_files_missing",
                        "message": str(exc),
                    },
                }

            spine_fn = getattr(analyze_module, spine_fn_name)
            result = spine_fn(
                project,
                clip_ids,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                **op_kwargs,
            )
            if not save_after:
                return result
            try:
                save_with_mtime_check(project, path, captured_mtime)
            except ProjectModifiedExternally as exc:
                return {
                    "success": False,
                    "error": {
                        "code": "project_modified_externally",
                        "path": str(exc.path),
                        "expected_mtime": exc.expected_mtime,
                        "current_mtime": exc.current_mtime,
                    },
                    "result": result.get("result"),
                }
            return result

        return run

    return runner


async def _start_spine_analyze_job(
    *,
    ctx: Context,
    project_path: str,
    kind: str,
    spine_fn_name: str,
    clip_ids: Optional[list[str]],
    idempotency_key: Optional[str],
    args: Optional[dict] = None,
    op_kwargs: Optional[dict] = None,
) -> str:
    """Validate project and enqueue a standard spine analysis job."""
    from scene_ripper_mcp.security import validate_project_path
    from core.project import MissingSourceError
    from core.spine.project_io import load_with_mtime

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    try:
        _project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    op_kwargs = op_kwargs or {}
    payload = {"project_path": canonical, "clip_ids": clip_ids}
    if args:
        payload.update(args)

    operation = None
    if spine_fn_name == "analyze_clips":
        from core.jobs.analysis import analysis_job_spec, run_analysis_job

        try:
            operation = analysis_job_spec(_project, arguments=payload)
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_analysis_job(store, path, operation, progress_callback, cancel_event)
    elif spine_fn_name == "extract_text":
        from core.jobs.ocr import ocr_job_spec, run_ocr_job
        from core.operations.ocr import OcrOptions

        try:
            operation = ocr_job_spec(_project, clip_ids, OcrOptions(
                num_keyframes=payload.get("num_keyframes", 3),
                use_vlm_fallback=payload.get("use_vlm_fallback", True),
                vlm_model=payload.get("vlm_model"), vlm_only=payload.get("vlm_only", False),
            ), arguments=payload)
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_ocr_job(store, path, operation.arguments["clip_ids"],
                               progress_callback, cancel_event, operation=operation)
    elif spine_fn_name == "boundary_embeddings":
        from core.jobs.boundary_embeddings import boundary_embedding_job_spec, run_boundary_embedding_job

        try:
            operation = boundary_embedding_job_spec(_project, clip_ids, arguments=payload)
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_boundary_embedding_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "describe":
        from core.jobs.description import description_job_spec, run_description_job
        from core.operations.description import resolve_options

        try:
            operation = description_job_spec(
                _project, clip_ids,
                resolve_options(payload.get("tier"), payload.get("prompt")),
                arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_description_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
                force=operation.arguments.get("force", False),
            )
    elif spine_fn_name == "gaze":
        from core.jobs.gaze import gaze_job_spec, run_gaze_job
        from core.operations.gaze import GazeOptions

        try:
            operation = gaze_job_spec(
                _project, clip_ids, GazeOptions(payload["sample_interval"]),
                arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_gaze_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "embeddings":
        from core.jobs.embeddings import embedding_job_spec, run_embedding_job
        from core.operations.embeddings import EmbeddingOptions

        try:
            operation = embedding_job_spec(_project, clip_ids, EmbeddingOptions(), arguments=payload)
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_embedding_job(store, path, operation.arguments["clip_ids"], progress_callback, cancel_event, operation=operation)
    elif spine_fn_name == "analyze_scalars":
        from core.jobs.scalars import scalar_job_spec, run_scalar_job

        scalar_kind = "brightness" if payload["operation"] == "brightness" else "volume"
        try:
            operation = scalar_job_spec(
                _project, clip_ids, scalar_kind,
                num_samples=payload["num_samples"], arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_scalar_job(
                store, path, operation.arguments["clip_ids"], progress_callback,
                cancel_event, kind=scalar_kind, operation=operation,
            )
    elif spine_fn_name == "face_embeddings":
        from core.jobs.faces import face_job_spec, run_face_job
        from core.operations.faces import FaceOptions

        try:
            operation = face_job_spec(
                _project, clip_ids, FaceOptions(payload["sample_interval"]),
                arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_face_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "detect_objects":
        from core.jobs.object_detection import object_detection_job_spec, run_object_detection_job
        from core.operations.object_detection import ObjectDetectionOptions

        try:
            operation = object_detection_job_spec(
                _project, clip_ids, ObjectDetectionOptions(payload["confidence"], payload["detect_all"]),
                arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_object_detection_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "classify_content":
        from core.jobs.classification import classification_job_spec, run_classification_job
        from core.operations.classification import ClassificationOptions

        try:
            operation = classification_job_spec(
                _project, clip_ids, ClassificationOptions(payload["top_k"], payload["threshold"]),
                arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_classification_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "cinematography":
        from core.jobs.cinematography import cinematography_job_spec, run_cinematography_job
        from core.operations.cinematography import resolve_options

        try:
            operation = cinematography_job_spec(
                _project, clip_ids, resolve_options(payload.get("mode"), payload.get("model")),
                arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_cinematography_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "custom_query":
        from core.jobs.custom_query import custom_query_job_spec, run_custom_query_job
        from core.operations.custom_query import resolve_options

        try:
            operation = custom_query_job_spec(
                _project, clip_ids, resolve_options(payload.get("tier")), arguments=payload,
            )
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            return run_custom_query_job(
                store, path, operation.arguments["clip_ids"],
                progress_callback, cancel_event, operation=operation,
            )
    elif spine_fn_name == "align_words":
        from core.jobs.alignment import alignment_job_spec, run_alignment_job

        try:
            operation = alignment_job_spec(_project, clip_ids, force=payload["force"], arguments=payload)
        except ValueError as exc:
            return json.dumps(_wrap_error(exc))
        store = _lifespan(ctx)["job_store"]

        def run(progress_callback, cancel_event):
            frozen = operation.arguments
            return run_alignment_job(
                store, path, frozen["clip_ids"], progress_callback, cancel_event,
                force=frozen["force"], operation=operation,
            )
    else:
        runner_factory = _make_analyze_runner(spine_fn_name, **op_kwargs)
        run = runner_factory(path, mtime, clip_ids)

    # Public MCP job names are compatibility aliases for shared operations.
    # Project/result identity stays internal; submitted job metadata must use
    # the public name that callers will poll and find in their job history.
    from dataclasses import replace

    return _start_job(
        ctx,
        kind=kind,
        args=payload,
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
        operation=replace(operation, kind=kind) if operation is not None else None,
    )


@mcp.tool()
async def start_align_words(
    project_path: Annotated[str, "Absolute path to saved project file"],
    clip_ids: Annotated[Optional[list[str]], "Exact clip IDs (default: all)"] = None,
    force: Annotated[bool, "Replace existing word timestamps"] = False,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Align existing transcripts using the installed word-alignment runtime.

    Does not install dependencies. Poll/cancel through the standard job tools.
    """
    return await _start_spine_analyze_job(
        ctx=ctx, project_path=project_path, kind="align_words",
        spine_fn_name="align_words",
        clip_ids=list(clip_ids) if clip_ids is not None else None,
        idempotency_key=idempotency_key,
        args={"force": force}, op_kwargs={"skip_existing": not force},
    )


@mcp.tool()
async def start_analyze_colors(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[
        Optional[list[str]],
        "Optional list of clip IDs to analyze (default: all clips)",
    ] = None,
    num_colors: Annotated[
        int, "Number of dominant colors to extract per clip (1-10)"
    ] = 5,
    idempotency_key: Annotated[
        Optional[str], "Optional idempotency key (max 255 chars)"
    ] = None,
    ctx: Context = None,
) -> str:
    """Start a job that extracts dominant colors for the given clips.

    Per-clip granularity: cancellable between clips. Skip-existing is on
    by default — clips with ``dominant_colors`` already populated are
    untouched, which makes re-issuing after a crashed/cancelled run resume
    where it left off.
    """
    from scene_ripper_mcp.security import validate_project_path
    from core.project import MissingSourceError
    from core.spine.project_io import load_with_mtime

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    try:
        _project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    from core.jobs.colors import color_job_spec, run_colors
    from core.jobs.commits import StaleJobResult
    from core.operations.colors import color_request
    from core.project_revision import ProjectFileRevision

    store = _lifespan(ctx)["job_store"]
    arguments = {"project_path": canonical, "clip_ids": clip_ids, "num_colors": num_colors}
    revision = _project.session.file_revision
    try:
        operation = color_job_spec(
            color_request(_project, clip_ids, num_colors, skip_existing=False),
            arguments=arguments, persistence="job_history",
            session_id=_project.session.session_id,
            input_revision=revision.digest if revision is not None else None,
        )
    except ValueError as exc:
        return json.dumps(_wrap_error(exc))

    def run(progress_callback, cancel_event):
        frozen = operation.arguments
        if operation.input_revision is not None:
            ProjectFileRevision(path, operation.input_revision).verify()
        current, _ = load_with_mtime(path)
        live = color_job_spec(
            color_request(current, frozen["clip_ids"], frozen["num_colors"], skip_existing=False),
            arguments=frozen, persistence="job_history", session_id=operation.session_id,
            input_revision=operation.input_revision,
        )
        if live.inputs_json != operation.inputs_json:
            raise StaleJobResult("Color inputs changed while the job was queued")
        return run_colors(store, path, frozen["clip_ids"], frozen["num_colors"], progress_callback, cancel_event)

    return _start_job(
        ctx,
        kind="analyze_colors",
        args=operation.arguments,
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
        operation=operation,
    )


@mcp.tool()
async def start_analyze_shots(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[
        Optional[list[str]],
        "Optional list of clip IDs to classify (default: all clips)",
    ] = None,
    idempotency_key: Annotated[
        Optional[str], "Optional idempotency key (max 255 chars)"
    ] = None,
    ctx: Context = None,
) -> str:
    """Start a job that classifies shot type per clip.

    Requires thumbnails for each clip on disk; clips without thumbnails
    surface as ``thumbnail_missing`` failures (this op does not generate
    thumbnails — that's a separate concern).
    """
    from dataclasses import replace
    from scene_ripper_mcp.security import validate_project_path
    from core.project import MissingSourceError
    from core.spine.project_io import load_with_mtime
    from core.jobs.shots import shot_job_spec, run_shot_job
    from core.operations.shots import ShotTypeOptions

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    try:
        _project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    try:
        operation = shot_job_spec(
            _project, clip_ids, ShotTypeOptions(),
            arguments={"project_path": canonical, "clip_ids": clip_ids},
        )
    except ValueError as exc:
        return json.dumps(_wrap_error(exc))
    store = _lifespan(ctx)["job_store"]

    def run(progress_callback, cancel_event):
        return run_shot_job(
            store, path, operation.arguments["clip_ids"],
            progress_callback, cancel_event, operation=operation,
        )

    return _start_job(
        ctx,
        kind="analyze_shots",
        args=operation.arguments,
        operation=replace(operation, kind="analyze_shots"),
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
    )


@mcp.tool()
async def start_generate_thumbnails(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[
        Optional[list[str]],
        "Optional list of clip IDs to process (default: all clips)",
    ] = None,
    force: Annotated[
        bool,
        "Regenerate thumbnails even when clip.thumbnail_path already exists",
    ] = False,
    idempotency_key: Annotated[
        Optional[str], "Optional idempotency key (max 255 chars)"
    ] = None,
    ctx: Context = None,
) -> str:
    """Start a job that generates/backfills thumbnails for project clips.

    Existing on-disk thumbnails are skipped unless ``force`` is true. This
    is useful as a repair step before thumbnail-dependent analysis tools
    such as ``start_analyze_shots``.
    """
    from scene_ripper_mcp.security import validate_project_path
    from core.project import MissingSourceError
    from core.spine.project_io import load_with_mtime

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    try:
        _project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    def run(progress_callback, cancel_event):
        from core.project import MissingSourceError
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )
        from core.spine.thumbnails import generate_thumbnails

        try:
            project, captured_mtime = load_with_mtime(path)
        except MissingSourceError as exc:
            return {
                "success": False,
                "error": {
                    "code": "source_files_missing",
                    "message": str(exc),
                },
            }

        result = generate_thumbnails(
            project,
            clip_ids,
            force=force,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        try:
            save_with_mtime_check(project, path, captured_mtime)
        except ProjectModifiedExternally as exc:
            return {
                "success": False,
                "error": {
                    "code": "project_modified_externally",
                    "path": str(exc.path),
                    "expected_mtime": exc.expected_mtime,
                    "current_mtime": exc.current_mtime,
                },
                "result": result.get("result"),
            }
        return result

    return _start_job(
        ctx,
        kind="generate_thumbnails",
        args={"project_path": canonical, "clip_ids": clip_ids, "force": force},
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
    )


@mcp.tool()
async def start_import_images(
    project_path: Annotated[str, "Absolute path to a saved .sceneripper project"],
    file_paths: Annotated[list[str], "Image paths; relative paths use the project directory"],
    copy_files: bool = True,
    idempotency_key: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Import a batch and save; poll job status/result for completion.

    Valid images are saved with per-item errors for rejected inputs. Copying is
    the default; copy_files=false references originals. A new invocation appends
    new frames; use idempotency_key to retry the same submission.
    """
    from scene_ripper_mcp.security import validate_project_path
    from core.spine.project_io import load_with_mtime
    from core.jobs.image_import import image_import_job_spec, run_image_import_job

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        project, mtime = load_with_mtime(path)
        operation = image_import_job_spec(project, file_paths, copy_files=copy_files, validate_paths=True)
        store = _lifespan(ctx)["job_store"]

        def run(progress, cancel):
            args = operation.arguments
            return run_image_import_job(store, path, args["file_paths"], progress, cancel,
                copy_files=args["copy_files"], validate_paths=True, operation=operation)

        return _start_job(ctx, kind=operation.kind, args=operation.arguments,
            project_path=str(path), project_mtime_at_start=mtime,
            idempotency_key=idempotency_key, run=run, operation=operation)
    except Exception as exc:
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def start_import_audio(
    project_path: Annotated[str, "Absolute path to a saved .sceneripper project"],
    file_path: Annotated[str, "Absolute or project-relative audio file path"],
    idempotency_key: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Import audio and save; poll job status/result for completion.

    Repeated canonical paths return the existing audio ID. Interrupted saves reuse
    recorded probing; an idempotency key retries the same job submission.
    """
    from scene_ripper_mcp.security import validate_project_path
    from core.spine.project_io import load_with_mtime
    from core.jobs.audio_import import audio_import_job_spec, run_audio_import_job
    from scene_ripper_mcp.security import validate_path
    from core.audio_formats import is_audio_file

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        project, mtime = load_with_mtime(path)
        if not file_path:
            raise ValueError("Audio path cannot be empty")
        media = Path(file_path).expanduser()
        if not media.is_absolute():
            media = path.parent / media
        valid, error, media = validate_path(str(media), must_be_file=True)
        if not valid:
            raise ValueError(error)
        if not is_audio_file(media):
            raise ValueError(f"Unsupported audio format: {media.suffix or '<no extension>'}")
        existing = any(audio.file_path.expanduser().resolve() == media for audio in project.audio_sources)
        if not existing and not media.is_file():
            raise ValueError(f"File not found: {media}")
        operation = audio_import_job_spec(project, str(media))
        store = _lifespan(ctx)["job_store"]

        def run(progress, cancel):
            return run_audio_import_job(store, path, operation.arguments["file_path"],
                progress, cancel, operation=operation)

        return _start_job(ctx, kind=operation.kind, args=operation.arguments,
            project_path=str(path), project_mtime_at_start=mtime,
            idempotency_key=idempotency_key, run=run, operation=operation)
    except Exception as exc:
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def start_extract_frames(
    project_path: Annotated[str, "Absolute path to a saved .sceneripper project"],
    source_id: Annotated[str, "Exact video source ID"],
    mode: str = "interval",
    interval: int = 10,
    clip_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Append extracted frames and save; poll job status/result for completion.

    Modes are interval, all, and smart. clip_id restricts the source range.
    Interrupted saves reuse artifacts. A fresh invocation adds a new batch;
    use an idempotency key to retry the same job submission.
    """
    from scene_ripper_mcp.security import validate_project_path
    from core.spine.project_io import load_with_mtime
    from core.jobs.frame_extraction import frame_extraction_job_spec, run_frame_extraction_job

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        project, mtime = load_with_mtime(path)
        operation = frame_extraction_job_spec(project, source_id, mode=mode, interval=interval, clip_id=clip_id)
        store = _lifespan(ctx)["job_store"]

        def run(progress, cancel):
            args = operation.arguments
            return run_frame_extraction_job(store, path, args["source_id"], progress, cancel,
                mode=args["mode"], interval=args["interval"], clip_id=args["clip_id"], operation=operation)

        return _start_job(ctx, kind=operation.kind, args=operation.arguments,
            project_path=str(path), project_mtime_at_start=mtime,
            idempotency_key=idempotency_key, run=run, operation=operation)
    except Exception as exc:
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def start_transcribe_audio(
    project_path: Annotated[str, "Absolute path to a saved project"],
    audio_source_id: Annotated[str, "Exact imported audio-source ID from list_audio_sources"],
    model: str = "small.en",
    language: str = "en",
    backend: str = "auto",
    force: bool = False,
    segmentation_mode: str = "backend",
    segment_max_seconds: float = 12.0,
    idempotency_key: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Start standalone audio transcription; poll job status/result for completion.

    Saves the transcript. Existing transcripts, including silence, are preserved
    unless force is true. Interrupted saves reuse recorded computation.
    """
    from scene_ripper_mcp.security import validate_project_path
    from core.spine.project_io import load_with_mtime
    from core.jobs.audio_transcription import audio_transcription_job_spec, run_audio_transcription_job
    from core.operations.transcription import TranscriptionOptions

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        project, mtime = load_with_mtime(path)
        operation = audio_transcription_job_spec(project, audio_source_id,
            TranscriptionOptions(model=model, language=language, backend=backend,
                segmentation_mode=segmentation_mode, segment_max_seconds=segment_max_seconds), force=force)
        store = _lifespan(ctx)["job_store"]

        def run(progress, cancel):
            args = operation.arguments
            return run_audio_transcription_job(store, path, args["audio_source_id"],
                TranscriptionOptions(**args["options"]), progress, cancel,
                force=args["force"], operation=operation)

        return _start_job(ctx, kind=operation.kind, args=operation.arguments,
            project_path=str(path), project_mtime_at_start=mtime,
            idempotency_key=idempotency_key, run=run, operation=operation)
    except Exception as exc:
        return json.dumps(_wrap_error(exc))


@mcp.tool()
async def start_transcribe(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[
        Optional[list[str]],
        "Optional list of clip IDs to transcribe (default: all clips)",
    ] = None,
    model: Annotated[
        str, "Whisper model size: tiny / base / small / medium / large"
    ] = "base",
    language: Annotated[
        Optional[str], "ISO language code (default: auto-detect)"
    ] = None,
    idempotency_key: Annotated[
        Optional[str], "Optional idempotency key (max 255 chars)"
    ] = None,
    ctx: Context = None,
) -> str:
    """Start a per-clip transcription job."""
    from scene_ripper_mcp.security import validate_project_path
    from core.project import MissingSourceError
    from core.spine.project_io import load_with_mtime

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    try:
        _project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    from core.jobs.transcription import run_transcription_job, transcription_job_spec
    from core.operations.transcription import TranscriptionOptions

    store = _lifespan(ctx)["job_store"]
    arguments = {
        "project_path": canonical,
        "clip_ids": clip_ids,
        "model": model,
        "language": language,
    }
    try:
        operation = transcription_job_spec(
            _project,
            clip_ids,
            TranscriptionOptions(model=model, language=language),
            arguments=arguments,
        )
    except ValueError as exc:
        return json.dumps(_wrap_error(exc))

    def run(progress_callback, cancel_event):
        frozen = operation.arguments
        options = TranscriptionOptions(**json.loads(operation.inputs_json)["options"])
        return run_transcription_job(
            store,
            path,
            frozen["clip_ids"],
            options,
            progress_callback,
            cancel_event,
            operation=operation,
        )

    return _start_job(
        ctx,
        kind="transcribe",
        args=operation.arguments,
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
        operation=operation,
    )


@mcp.tool()
async def start_analyze_classify(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    top_k: Annotated[int, "Maximum labels per clip"] = 5,
    threshold: Annotated[float, "Minimum label confidence"] = 0.1,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that classifies thumbnail content with ImageNet labels."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="analyze_classify",
        spine_fn_name="classify_content",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"top_k": top_k, "threshold": threshold},
        op_kwargs={"top_k": top_k, "threshold": threshold},
    )


@mcp.tool()
async def start_detect_objects(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    confidence: Annotated[float, "Minimum detection confidence"] = 0.5,
    detect_all: Annotated[bool, "Detect all objects, not only people"] = True,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that detects objects and person counts on clip thumbnails."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="detect_objects",
        spine_fn_name="detect_objects",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"confidence": confidence, "detect_all": detect_all},
        op_kwargs={"confidence": confidence, "detect_all": detect_all},
    )


@mcp.tool()
async def start_extract_text(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    num_keyframes: Annotated[int, "Keyframes to sample per clip (1-5)"] = 3,
    use_vlm_fallback: Annotated[bool, "Use VLM fallback for weak OCR results"] = True,
    vlm_model: Annotated[Optional[str], "Optional VLM model override"] = None,
    vlm_only: Annotated[bool, "Skip OCR and use only VLM extraction"] = False,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
    force: Annotated[bool, "Replace existing OCR results"] = False,
) -> str:
    """Start a job that extracts visible text from clips."""
    op_kwargs = {
        "num_keyframes": num_keyframes,
        "use_vlm_fallback": use_vlm_fallback,
        "vlm_model": vlm_model,
        "vlm_only": vlm_only,
        "force": force,
    }
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="extract_text",
        spine_fn_name="extract_text",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args=op_kwargs,
        op_kwargs=op_kwargs,
    )


@mcp.tool()
async def start_describe(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    tier: Annotated[Optional[str], "Model tier override: local or cloud"] = None,
    prompt: Annotated[Optional[str], "Optional description prompt override"] = None,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
    force: Annotated[bool, "Replace existing descriptions"] = False,
) -> str:
    """Start a job that generates VLM clip descriptions."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="describe",
        spine_fn_name="describe",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"tier": tier, "prompt": prompt, "force": force},
        op_kwargs={"tier": tier, "prompt": prompt, "skip_existing": not force},
    )


@mcp.tool()
async def start_analyze_cinematography(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    mode: Annotated[Optional[str], "Input mode override: frame or video"] = None,
    model: Annotated[Optional[str], "Optional VLM model override"] = None,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that runs rich cinematography analysis."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="analyze_cinematography",
        spine_fn_name="cinematography",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"mode": mode, "model": model},
        op_kwargs={"mode": mode, "model": model},
    )


@mcp.tool()
async def start_analyze_scalars(
    project_path: Annotated[str, "Absolute path to saved project file"],
    operation: Annotated[str, "Scalar measurement: brightness or volume"],
    clip_ids: Annotated[Optional[list[str]], "Exact clip IDs (default: all)"] = None,
    num_samples: Annotated[int, "Positive brightness sample count; volume uses the full clip"] = 5,
    force: Annotated[bool, "Recompute existing results"] = False,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Measure brightness or volume with verified reuse and durable recovery.

    Poll and cancel through the standard job tools. Does not install dependencies.
    """
    if operation not in ("brightness", "volume") or type(num_samples) is not int or num_samples < 1:
        return json.dumps({"success": False, "error": {"code": "validation_error", "message": "Choose brightness or volume and a positive integer sample count"}})
    return await _start_spine_analyze_job(
        ctx=ctx, project_path=project_path, kind="analyze_scalars",
        spine_fn_name="analyze_scalars", clip_ids=list(clip_ids) if clip_ids is not None else None,
        idempotency_key=idempotency_key,
        args={"operation": operation, "num_samples": num_samples, "force": force},
    )


@mcp.tool()
async def start_detect_faces(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    sample_interval: Annotated[float, "Seconds between sampled frames"] = 1.0,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that extracts face embeddings."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="detect_faces",
        spine_fn_name="face_embeddings",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"sample_interval": sample_interval},
        op_kwargs={"sample_interval": sample_interval},
    )


@mcp.tool()
async def start_analyze_gaze(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    sample_interval: Annotated[float, "Seconds between sampled frames"] = 1.0,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that estimates gaze direction."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="analyze_gaze",
        spine_fn_name="gaze",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"sample_interval": sample_interval},
        op_kwargs={"sample_interval": sample_interval},
    )


@mcp.tool()
async def start_generate_boundary_embeddings(
    project_path: Annotated[str, "Absolute path to saved project file"],
    clip_ids: Annotated[Optional[list[str]], "Exact clip IDs (default: all)"] = None,
    force: Annotated[bool, "Replace existing boundary pairs"] = False,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Extract first/last-frame DINOv2 pairs with durable recovery.

    Requires an installed embedding runtime; does not install dependencies.
    Use the standard job status/result/cancel tools after submission.
    """
    return await _start_spine_analyze_job(
        ctx=ctx, project_path=project_path, kind="generate_boundary_embeddings",
        spine_fn_name="boundary_embeddings", clip_ids=clip_ids,
        idempotency_key=idempotency_key, args={"force": force},
    )


@mcp.tool()
async def start_generate_embeddings(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that extracts DINOv2 visual embeddings."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="generate_embeddings",
        spine_fn_name="embeddings",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
    )


@mcp.tool()
async def start_custom_query(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    query: Annotated[str, "Natural-language visual query to evaluate"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    tier: Annotated[Optional[str], "Model tier override: local or cloud"] = None,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start a job that evaluates a custom visual query against clips."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="custom_query",
        spine_fn_name="custom_query",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"query": query, "tier": tier},
        op_kwargs={"query": query, "tier": tier},
    )


@mcp.tool()
async def start_analyze_clips(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    operations: Annotated[list[str], "UI analysis operation keys to run"],
    clip_ids: Annotated[Optional[list[str]], "Optional clip IDs (default: all clips)"] = None,
    query: Annotated[Optional[str], "Required when operations includes custom_query"] = None,
    idempotency_key: Annotated[Optional[str], "Optional idempotency key (max 255 chars)"] = None,
    ctx: Context = None,
) -> str:
    """Start one canonical job for any UI analysis operation list."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="analyze_clips",
        spine_fn_name="analyze_clips",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"operations": operations, "query": query},
        op_kwargs={"operations": operations, "query": query},
    )


@mcp.tool()
async def start_download_videos(
    urls: Annotated[list[str], "List of video URLs to download (max 10)"],
    output_dir: Annotated[
        Optional[str],
        "Output directory (defaults to settings.download_dir)",
    ] = None,
    idempotency_key: Annotated[
        Optional[str], "Optional idempotency key (max 255 chars)"
    ] = None,
    ctx: Context = None,
) -> str:
    """Download URLs to ``output_dir``, recording verified receipts for retries.

    Per-URL granularity: cancellation observed between URLs. Per-URL
    failures (geo-block, DRM, deleted) are aggregated into the result;
    one failure does not poison the batch.
    """
    from scene_ripper_mcp.security import validate_path

    if not urls:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "no_urls", "message": "No URLs provided"},
            }
        )
    if len(urls) > 10:
        return json.dumps(
            {
                "success": False,
                "error": {
                    "code": "too_many_urls",
                    "message": "Maximum 10 URLs per batch",
                },
            }
        )

    if output_dir:
        valid, error, target = validate_path(output_dir, must_be_dir=True)
        if not valid:
            return json.dumps(
                {
                    "success": False,
                    "error": {"code": "invalid_output_dir", "message": error},
                }
            )
    else:
        from core.settings import load_settings

        target = load_settings().download_dir

    target = Path(target).expanduser().resolve()
    canonical_target = str(target)
    download_urls = tuple(urls)
    store = _lifespan(ctx)["job_store"]
    target_identity = (target.stat().st_dev, target.stat().st_ino) if target.exists() else None
    operation = OperationSpec.build(
        kind="download_videos", version=1,
        arguments={"urls": list(download_urls), "output_dir": canonical_target},
        inputs={"output_directory_identity": target_identity},
        persistence="job_history",
    )

    def run(progress_callback, cancel_event):
        from core.jobs.downloads import run_saved_downloads

        if target.resolve() != target or (
            target_identity is not None and (
                not target.exists() or (target.stat().st_dev, target.stat().st_ino) != target_identity
            )
        ):
            raise RuntimeError("Download directory changed before execution")

        return run_saved_downloads(
            store,
            list(download_urls),
            target,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    return _start_job(
        ctx,
        kind="download_videos",
        args={"urls": list(download_urls), "output_dir": canonical_target},
        # Downloads do not target a specific project; the per-project
        # mutex is bypassed (project_path=None).
        project_path=None,
        project_mtime_at_start=None,
        idempotency_key=idempotency_key,
        run=run,
        operation=operation,
    )


@mcp.tool()
async def start_detect_scenes_new_project(
    video_path: Annotated[str, "Absolute path to video file"],
    output_project_path: Annotated[
        str, "Path for the output .sceneripper project file (will be created)"
    ],
    sensitivity: Annotated[
        float, "Detection sensitivity (1.0=more scenes, 10.0=fewer)"
    ] = 3.0,
    idempotency_key: Annotated[
        Optional[str],
        "Optional idempotency key (max 255 chars)",
    ] = None,
    ctx: Context = None,
) -> str:
    """Start a job that creates a fresh project from a video file and runs
    scene detection on it.

    Returns ``{task_id, status, poll_interval}`` immediately. Poll
    ``get_job_status`` and fetch with ``get_job_result``.
    """
    from scene_ripper_mcp.security import validate_project_path, validate_video_path

    valid, error, video = validate_video_path(video_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Video: {error}"})

    valid, error, output = validate_project_path(
        output_project_path, must_exist=False
    )
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    canonical_output = str(output)

    def run(progress_callback, cancel_event):
        from core.spine.detect import detect_scenes_new_project

        return detect_scenes_new_project(
            video,
            output,
            sensitivity=sensitivity,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    return _start_job(
        ctx,
        kind="detect_scenes_new_project",
        args={
            "video_path": str(video),
            "output_project_path": canonical_output,
            "sensitivity": sensitivity,
        },
        project_path=canonical_output,
        project_mtime_at_start=None,
        idempotency_key=idempotency_key,
        run=run,
    )
