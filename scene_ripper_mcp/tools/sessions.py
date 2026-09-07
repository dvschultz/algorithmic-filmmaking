"""Durable edits and session-local history through shared headless sessions."""

from __future__ import annotations

import json
from typing import Callable

from mcp.server.fastmcp import Context

from core.spine import history, sequences
from core.spine.project_io import project_error
from scene_ripper_mcp.security import validate_project_path
from scene_ripper_mcp.server import mcp


async def _call(ctx: Context, operation: Callable) -> str:
    try:
        runtime = ctx.request_context.lifespan_context["project_sessions"]
        return json.dumps(await runtime.call(operation))
    except Exception as exc:
        return json.dumps({"success": False, "error": project_error(exc)})


@mcp.tool()
async def open_project_session(project_path: str, ctx: Context) -> str:
    """Retain an editing session and undo history until close or server shutdown.

    Edits autosave. Other path-based tools remain usable; their saved changes
    cause this session to reload and clear history on its next call.
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    return await _call(ctx, lambda sessions: sessions.open(path))


@mcp.tool()
async def get_project_session(session_id: str, ctx: Context) -> str:
    """Read sequences and undo/redo availability; reload externally changed files."""
    return await _call(ctx, lambda sessions: sessions.inspect(session_id))


@mcp.tool()
async def close_project_session(session_id: str, ctx: Context) -> str:
    """Discard session-local history. Successful edits are already saved."""
    return await _call(ctx, lambda sessions: sessions.close(session_id))


@mcp.tool()
async def create_session_sequence(
    session_id: str, name: str, ctx: Context, fps: float = 30.0
) -> str:
    """Create and save a sequence as one undoable edit in a retained session."""
    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id, lambda project: sequences.create_sequence(project, name, fps)
        ),
    )


@mcp.tool()
async def rename_session_sequence(
    session_id: str, sequence_id: str, name: str, ctx: Context
) -> str:
    """Rename and save a sequence as one undoable edit."""
    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id,
            lambda project: sequences.update_sequence(project, sequence_id, name=name),
        ),
    )


@mcp.tool()
async def delete_session_sequence(
    session_id: str, sequence_id: str, ctx: Context
) -> str:
    """Delete and save a sequence; undo restores its model references."""
    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id, lambda project: sequences.delete_sequence(project, sequence_id)
        ),
    )


@mcp.tool()
async def undo_project_session(session_id: str, ctx: Context) -> str:
    """Undo and save the last edit. External changes invalidate previous history."""
    return await _call(ctx, lambda sessions: sessions.edit(session_id, history.undo))


@mcp.tool()
async def redo_project_session(session_id: str, ctx: Context) -> str:
    """Redo and save the last undone edit without repeating media computation."""
    return await _call(ctx, lambda sessions: sessions.edit(session_id, history.redo))
