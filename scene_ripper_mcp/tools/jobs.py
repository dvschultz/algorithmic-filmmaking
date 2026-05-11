"""Generic job-management MCP tools.

These five tools sit on top of the jobs framework in
``scene_ripper_mcp/jobs/`` and serve every long-running op (R5 — existing
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
    """Return the final result for a completed job, or the structured error
    for a failed/cancelled/crashed job.

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

        if row.status == STATUS_COMPLETED:
            return json.dumps(
                {
                    "success": True,
                    "task_id": row.id,
                    "status": row.status,
                    "result": row.result,
                }
            )

        # Failed / cancelled / crashed — surface the sanitized error.
        code_map = {
            STATUS_FAILED: "job_failed",
            STATUS_CANCELLED: "job_cancelled",
            STATUS_CRASHED: "job_crashed",
        }
        return json.dumps(
            {
                "success": False,
                "task_id": row.id,
                "status": row.status,
                "error": {
                    "code": code_map[row.status],
                    "message": row.error or row.status,
                },
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
) -> str:
    """Delete terminal-status job rows older than ``days``.

    Running and queued rows are never purged. There is no automatic TTL on
    ``start_*`` calls (R22) — pruning is explicit.
    """
    try:
        if days < 0:
            return json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "invalid_days",
                        "message": "days must be >= 0",
                    },
                }
            )
        store = _lifespan(ctx)["job_store"]
        deleted = store.purge_old_jobs(days=days)
        return json.dumps(
            {
                "success": True,
                "deleted_count": deleted,
                "days": days,
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

    Per-source granularity: cancellation is observed between sources;
    per-source failures are aggregated into the result, never raised
    mid-batch.
    """
    from core.project import MissingSourceError
    from scene_ripper_mcp.security import validate_project_path
    from core.spine.project_io import load_with_mtime

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    canonical = str(path)

    # Load project once at submit time so the closure captures it (the worker
    # mutates project.sources clips, then save_with_mtime_check writes).
    try:
        project, mtime = load_with_mtime(path)
    except MissingSourceError as exc:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "source_files_missing", "message": str(exc)},
            }
        )

    def run(progress_callback, cancel_event):
        from core.spine.detect import detect_scenes_bulk
        from core.spine.project_io import (
            ProjectModifiedExternally,
            save_with_mtime_check,
        )

        result = detect_scenes_bulk(
            project,
            source_ids,
            sensitivity=sensitivity,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        try:
            save_with_mtime_check(project, path, mtime)
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
        kind="detect_scenes_bulk",
        args={
            "project_path": canonical,
            "source_ids": source_ids,
            "sensitivity": sensitivity,
        },
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
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
    runner_factory = _make_analyze_runner(spine_fn_name, **op_kwargs)
    run = runner_factory(path, mtime, clip_ids)

    payload = {"project_path": canonical, "clip_ids": clip_ids}
    if args:
        payload.update(args)

    return _start_job(
        ctx,
        kind=kind,
        args=payload,
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
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

    runner_factory = _make_analyze_runner("analyze_colors", num_colors=num_colors)
    run = runner_factory(path, mtime, clip_ids)

    return _start_job(
        ctx,
        kind="analyze_colors",
        args={
            "project_path": canonical,
            "clip_ids": clip_ids,
            "num_colors": num_colors,
        },
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
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

    runner_factory = _make_analyze_runner("analyze_shots")
    run = runner_factory(path, mtime, clip_ids)

    return _start_job(
        ctx,
        kind="analyze_shots",
        args={"project_path": canonical, "clip_ids": clip_ids},
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

    runner_factory = _make_analyze_runner(
        "transcribe", model=model, language=language
    )
    run = runner_factory(path, mtime, clip_ids)

    return _start_job(
        ctx,
        kind="transcribe",
        args={
            "project_path": canonical,
            "clip_ids": clip_ids,
            "model": model,
            "language": language,
        },
        project_path=canonical,
        project_mtime_at_start=mtime,
        idempotency_key=idempotency_key,
        run=run,
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
) -> str:
    """Start a job that extracts visible text from clips."""
    op_kwargs = {
        "num_keyframes": num_keyframes,
        "use_vlm_fallback": use_vlm_fallback,
        "vlm_model": vlm_model,
        "vlm_only": vlm_only,
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
) -> str:
    """Start a job that generates VLM clip descriptions."""
    return await _start_spine_analyze_job(
        ctx=ctx,
        project_path=project_path,
        kind="describe",
        spine_fn_name="describe",
        clip_ids=clip_ids,
        idempotency_key=idempotency_key,
        args={"tier": tier, "prompt": prompt},
        op_kwargs={"tier": tier, "prompt": prompt},
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
    """Start a job that downloads each URL in ``urls`` to ``output_dir``.

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

    canonical_target = str(target)

    def run(progress_callback, cancel_event):
        from core.spine.downloads import download_videos

        return download_videos(
            urls,
            target,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    return _start_job(
        ctx,
        kind="download_videos",
        args={"urls": urls, "output_dir": canonical_target},
        # Downloads do not target a specific project; the per-project
        # mutex is bypassed (project_path=None).
        project_path=None,
        project_mtime_at_start=None,
        idempotency_key=idempotency_key,
        run=run,
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
