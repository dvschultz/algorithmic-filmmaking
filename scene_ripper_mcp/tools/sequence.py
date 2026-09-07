"""Sequence/timeline manipulation MCP tools."""

import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_project_path

from scene_ripper_mcp.editorial import editorial_call as _editorial_call

logger = logging.getLogger(__name__)




@mcp.tool()
async def get_sequence(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """Get the current sequence/timeline state.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with sequence structure and clips
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        sequence = project.sequence

        # An empty sequence has no clips on any track. Treat that as
        # "no sequence yet" for the agent's purposes.
        if not sequence or sum(len(t.clips) for t in sequence.tracks) == 0:
            return json.dumps(
                {
                    "success": True,
                    "has_sequence": False,
                    "sequence": None,
                }
            )

        sources_by_id = project.sources_by_id

        # Build track data
        tracks_data = []
        for track in sequence.tracks:
            track_clips = []
            for seq_clip in track.clips:
                source = sources_by_id.get(seq_clip.source_id)
                fps = source.fps if source else 30.0

                track_clips.append(
                    {
                        "id": seq_clip.id,
                        "source_clip_id": seq_clip.source_clip_id,
                        "source_name": source.filename if source else "Unknown",
                        "timeline_start": seq_clip.start_time(fps),
                        "timeline_start_frame": seq_clip.start_frame,
                        "duration": seq_clip.duration_seconds(fps),
                        "duration_frames": seq_clip.duration_frames,
                        "in_point": seq_clip.in_point,
                        "out_point": seq_clip.out_point,
                    }
                )

            tracks_data.append(
                {
                    "id": track.id,
                    "name": track.name,
                    "clip_count": len(track.clips),
                    "clips": track_clips,
                }
            )

        return json.dumps(
            {
                "success": True,
                "has_sequence": True,
                "sequence": {
                    "id": sequence.id,
                    "name": sequence.name,
                    "fps": sequence.fps,
                    "duration_frames": sequence.duration_frames,
                    "duration_seconds": sequence.duration_seconds,
                    "track_count": len(sequence.tracks),
                    "total_clips": sum(len(t.clips) for t in sequence.tracks),
                    "tracks": tracks_data,
                },
            }
        )
    except Exception as e:
        logger.exception("Failed to get sequence")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def add_to_sequence(
    project_path: Annotated[str, "Path to project file"],
    clip_ids: Annotated[list[str], "List of clip IDs to add"],
    track_index: Annotated[int, "Track index (0 = first track)"] = 0,
    position: Annotated[Optional[str], "Position: 'end' (default), 'start', or frame number"] = "end",
    ctx: Context = None,
) -> str:
    """Add clips to the sequence timeline.

    Args:
        project_path: Path to the project file
        clip_ids: List of clip IDs to add (in order)
        track_index: Which track to add to (default: 0)
        position: Where to add - 'end', 'start', or specific frame number

    Returns:
        JSON with updated sequence state
    """
    def operation(project):
        from core.spine.timeline import insert_legacy_clips
        return insert_legacy_clips(project, clip_ids, track_index=track_index, position=position)
    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def remove_from_sequence(
    project_path: Annotated[str, "Path to project file"],
    sequence_clip_ids: Annotated[list[str], "List of sequence clip IDs to remove"],
    ctx: Context = None,
) -> str:
    """Remove clips from the sequence timeline.

    Args:
        project_path: Path to the project file
        sequence_clip_ids: List of sequence clip IDs to remove

    Returns:
        JSON with removal result
    """

    def operation(project):
        removed = project.remove_from_sequence(sequence_clip_ids, ripple=False)
        return {
            "success": True,
            "clips_removed": len(removed),
            "sequence_duration": project.sequence.duration_seconds,
        }

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def reorder_sequence(
    project_path: Annotated[str, "Path to project file"],
    clip_order: Annotated[list[str], "Sequence clip IDs in desired order"],
    track_index: Annotated[int, "Track index to reorder"] = 0,
    ctx: Context = None,
) -> str:
    """Reorder clips in the sequence.

    Rearranges clips on a track to match the specified order.
    Clips are placed sequentially with no gaps.

    Args:
        project_path: Path to the project file
        clip_order: List of sequence clip IDs in the desired order
        track_index: Which track to reorder

    Returns:
        JSON with reorder result
    """

    def operation(project):
        from core.spine.timeline import reorder_clips

        known = {
            c.id
            for i, track in enumerate(project.sequence.tracks)
            if i == track_index
            for c in track.clips
        }
        requested = [clip_id for clip_id in clip_order if clip_id in known]
        result = reorder_clips(
            project, project.sequence.id, requested, track_index=track_index
        )
        return {
            "success": True,
            "clips_reordered": len(result["clip_order"]),
            "new_order": result["clip_order"],
            "sequence_duration": project.sequence.duration_seconds,
        }

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def clear_sequence(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """Clear all clips from the sequence.

    Removes all clips from all tracks but preserves the track structure.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with clear result
    """

    def operation(project):
        removed = project.clear_sequence()
        return {
            "success": True,
            "clips_removed": removed,
            "tracks_preserved": len(project.sequence.tracks),
        }

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def shuffle_sequence(
    project_path: Annotated[str, "Path to project file"],
    method: Annotated[
        str, "Shuffle method: random, reverse, by_color, by_shot_type"
    ] = "random",
    track_index: Annotated[int, "Track index to shuffle"] = 0,
    ctx: Context = None,
) -> str:
    """Shuffle clips in the sequence using various algorithms.

    Args:
        project_path: Path to the project file
        method: Shuffle algorithm to use
        track_index: Which track to shuffle

    Returns:
        JSON with shuffle result
    """

    def operation(project):
        from core.spine.timeline import shuffle_clips

        result = shuffle_clips(
            project, project.sequence.id, method, track_index=track_index
        )
        if not result["clips_shuffled"]:
            return {
                "success": True,
                "message": "No clips to shuffle",
                "clips_shuffled": 0,
            }
        return {
            "success": True,
            "method": method,
            "clips_shuffled": result["clips_shuffled"],
            "new_order": result["clip_order"],
            "sequence_duration": project.sequence.duration_seconds,
        }

    return await _editorial_call(project_path, ctx, operation)
