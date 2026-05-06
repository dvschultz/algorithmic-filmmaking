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
