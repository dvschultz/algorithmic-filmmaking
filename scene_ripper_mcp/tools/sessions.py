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


@mcp.tool()
async def get_session_timeline(session_id: str, sequence_id: str, ctx: Context) -> str:
    """Inspect a retained sequence, including stable timeline clip IDs and tracks."""
    from core.spine import timeline

    return await _call(
        ctx,
        lambda sessions: sessions.read(
            session_id, lambda project: timeline.get_timeline(project, sequence_id)
        ),
    )


@mcp.tool()
async def insert_session_clips(
    session_id: str,
    sequence_id: str,
    clip_ids: list[str],
    ctx: Context,
    track_index: int = 0,
    start_frame: int | None = None,
) -> str:
    """Insert library clip IDs as one saved edit. Uses existing tracks.

    Omit start_frame to append; explicit placement does not ripple existing clips.
    Disabled or unknown clips reject the entire batch. Repeated IDs insert copies.
    """
    from core.spine import timeline

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id,
            lambda project: timeline.insert_clips(
                project,
                sequence_id,
                clip_ids,
                track_index=track_index,
                start_frame=start_frame,
            ),
        ),
    )


@mcp.tool()
async def remove_session_clips(
    session_id: str,
    sequence_id: str,
    clip_ids: list[str],
    ctx: Context,
    ripple: bool = False,
) -> str:
    """Remove timeline clip IDs as one saved edit; optionally close affected track gaps."""
    from core.spine import timeline

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id,
            lambda project: timeline.remove_clips(
                project, sequence_id, clip_ids, ripple=ripple
            ),
        ),
    )


@mcp.tool()
async def reorder_session_clips(
    session_id: str,
    sequence_id: str,
    clip_ids: list[str],
    ctx: Context,
    track_index: int = 0,
) -> str:
    """Pack a track in timeline-ID order; omitted clips follow in their existing order."""
    from core.spine import timeline

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id,
            lambda project: timeline.reorder_clips(
                project, sequence_id, clip_ids, track_index=track_index
            ),
        ),
    )


@mcp.tool()
async def edit_session_timeline_clip(
    session_id: str,
    sequence_id: str,
    clip_id: str,
    changes: dict,
    ctx: Context,
) -> str:
    """Save timing or transform changes as one undoable edit.

    Fields: start_frame, in_point, out_point, hold_frames, track_index,
    hflip, vflip, reverse. In/out points are absolute source frames (out exclusive).
    Frame clips use hold_frames; edits preserve other sequences and tracks.
    """
    from core.spine import timeline

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id,
            lambda project: timeline.edit_clip(project, sequence_id, clip_id, changes),
        ),
    )


@mcp.tool()
async def clear_session_timeline(
    session_id: str, sequence_id: str, ctx: Context
) -> str:
    """Clear all tracks as one saved edit; undo restores clips and their references."""
    from core.spine import timeline

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id, lambda project: timeline.clear_timeline(project, sequence_id)
        ),
    )


@mcp.tool()
async def set_session_clips_disabled(
    session_id: str,
    clip_ids: list[str],
    disabled: bool,
    ctx: Context,
) -> str:
    """Enable or disable library clips as one saved undoable edit."""
    from core.spine.clips import set_clips_disabled

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id, lambda project: set_clips_disabled(project, clip_ids, disabled)
        ),
    )


@mcp.tool()
async def update_session_clip(
    session_id: str,
    clip_id: str,
    fields: dict,
    ctx: Context,
) -> str:
    """Save editable clip metadata as one undoable edit.

    Fields include name, notes, tags, shot_type, description, object_labels,
    and transcript. Unsupported fields are rejected by the shared model.
    """
    from core.spine.metadata import update_clip_from_json

    return await _call(
        ctx,
        lambda sessions: sessions.edit(
            session_id, lambda project: update_clip_from_json(project, clip_id, fields)
        ),
    )
