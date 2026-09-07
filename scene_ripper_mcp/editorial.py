"""Shared transport adapter for retained path-based editorial operations."""

import asyncio
import json
from typing import Callable

from mcp.server.fastmcp import Context
from core.project import Project
from core.spine.project_io import project_error, project_writer
from scene_ripper_mcp.security import validate_project_path


async def editorial_call(
    project_path: str,
    ctx: Context | None,
    operation: Callable[[Project], dict],
) -> str:
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        if ctx is not None:
            runtime = ctx.request_context.lifespan_context["project_sessions"]
            result = await runtime.call(
                lambda sessions: sessions.edit_path(path, operation)
            )
        else:

            def standalone():
                from core.spine.project_io import load_with_mtime, save_with_mtime_check

                with project_writer(path):
                    project, mtime = load_with_mtime(path)
                    generation = project.mutation_generation
                    result = operation(project)
                    if (
                        result.get("success")
                        and project.mutation_generation != generation
                    ):
                        save_with_mtime_check(project, path, mtime)
                    return result

            result = await asyncio.to_thread(standalone)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"success": False, "error": project_error(exc)})
