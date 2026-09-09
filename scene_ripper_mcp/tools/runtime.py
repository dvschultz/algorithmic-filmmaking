"""Native runtime profile tools: status, staged install job, rollback (plan U14)."""

from __future__ import annotations

import json
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp


@mcp.tool()
async def list_runtime_profiles(ctx: Context = None) -> str:
    """List native runtime profiles with install status, missing packages and overlays.

    Returns:
        JSON with ``profiles`` (each: ``profile`` id, ``family``, ``features``,
        ``task_kinds``, ``installed``, ``missing``, ``overlays``, ``description``)
        and ``native_worker_isolation``. Read-only.
    """
    from core.spine.runtime import list_runtime_profiles as _impl

    return json.dumps(_impl())


@mcp.tool()
async def get_runtime_profile_status(
    profile: Annotated[str, "Runtime profile id, e.g. transcription-whisper"],
    probe: Annotated[bool, "Also import the runtime in a fresh worker as a health check"] = False,
    ctx: Context = None,
) -> str:
    """Status of one runtime profile, optionally health-checked in an isolated worker."""
    from core.spine.runtime import get_runtime_profile_status as _impl

    return json.dumps(_impl(profile, probe=probe))


@mcp.tool()
async def rollback_runtime_profile(
    profile: Annotated[str, "Runtime profile id"],
    ctx: Context = None,
) -> str:
    """Remove a profile's newest promoted overlay so the previous runtime is active again."""
    from core.spine.runtime import rollback_runtime_profile as _impl

    return json.dumps(_impl(profile))


@mcp.tool()
async def start_install_runtime_profile(
    profile: Annotated[str, "Runtime profile id (allowlisted; never a package name)"],
    idempotency_key: Annotated[Optional[str], "Optional idempotency key"] = None,
    ctx: Context = None,
) -> str:
    """Install or repair a runtime profile as a durable job.

    Packages are staged, health-checked in a fresh worker and only then
    promoted; cancelling the job kills pip and keeps the previous runtime.
    Poll with ``get_job_status``; the result is the profile status with
    ``promoted_dir`` or an ``error``.
    """
    from core.jobs.runtime_install import run_runtime_install_job, runtime_install_job_spec
    from scene_ripper_mcp.tools.jobs import start_job

    try:
        operation = runtime_install_job_spec(profile)
    except ValueError as exc:
        return json.dumps({"success": False, "error": {"code": "unknown_profile", "message": str(exc)}})

    def run(progress_callback, cancel_event):
        return run_runtime_install_job(profile, progress_callback, cancel_event)

    return start_job(
        ctx, kind="install_runtime_profile", args={"profile": profile},
        project_path=None, project_mtime_at_start=None, idempotency_key=idempotency_key,
        run=run, operation=operation,
    )
