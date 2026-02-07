"""Clip query and manipulation MCP tools."""

import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_project_path

logger = logging.getLogger(__name__)


@mcp.tool()
async def list_clips(
    project_path: Annotated[str, "Path to project file"],
    limit: Annotated[int, "Maximum clips to return"] = 100,
    offset: Annotated[int, "Number of clips to skip"] = 0,
    ctx: Context = None,
) -> str:
    """List all clips in a project with their metadata.

    Args:
        project_path: Path to the project file
        limit: Maximum number of clips to return
        offset: Number of clips to skip (for pagination)

    Returns:
        JSON with list of clips and their metadata
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        # Apply pagination
        paginated_clips = clips[offset : offset + limit]

        clip_list = []
        for clip in paginated_clips:
            source = sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0

            clip_data = {
                "id": clip.id,
                "source_id": clip.source_id,
                "source_name": source.filename if source else "Unknown",
                "start_time": clip.start_time(fps),
                "end_time": clip.end_time(fps),
                "duration": clip.duration_seconds(fps),
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
            }

            # Add optional metadata
            if clip.dominant_colors:
                clip_data["colors"] = [
                    {"r": c[0], "g": c[1], "b": c[2]} for c in clip.dominant_colors[:3]
                ]
            if clip.shot_type:
                clip_data["shot_type"] = clip.shot_type
            if clip.transcript:
                clip_data["has_transcript"] = True
                clip_data["transcript_preview"] = clip.get_transcript_text()[:100]
            if clip.tags:
                clip_data["tags"] = clip.tags
            if clip.notes:
                clip_data["notes"] = clip.notes

            clip_list.append(clip_data)

        return json.dumps(
            {
                "success": True,
                "total_clips": len(clips),
                "returned": len(clip_list),
                "offset": offset,
                "clips": clip_list,
            }
        )
    except Exception as e:
        logger.exception("Failed to list clips")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def filter_clips(
    project_path: Annotated[str, "Path to project file"],
    shot_type: Annotated[Optional[str], "Filter by shot type"] = None,
    has_speech: Annotated[Optional[bool], "Filter by speech presence"] = None,
    min_duration: Annotated[Optional[float], "Minimum duration in seconds"] = None,
    max_duration: Annotated[Optional[float], "Maximum duration in seconds"] = None,
    tags: Annotated[Optional[list[str]], "Filter by tags (any match)"] = None,
    has_colors: Annotated[Optional[bool], "Filter by color analysis"] = None,
    ctx: Context = None,
) -> str:
    """Filter clips by various criteria.

    Args:
        project_path: Path to the project file
        shot_type: Filter to specific shot type (wide shot, medium shot, close-up, extreme close-up)
        has_speech: True = with transcript, False = without transcript
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        tags: Filter to clips with any of these tags
        has_colors: True = with colors, False = without colors

    Returns:
        JSON with filtered clip list
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        filtered = []
        for clip in clips:
            source = sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0
            duration = clip.duration_seconds(fps)

            # Apply filters
            if shot_type and clip.shot_type != shot_type:
                continue
            if has_speech is not None:
                clip_has_speech = bool(clip.transcript and clip.get_transcript_text().strip())
                if has_speech != clip_has_speech:
                    continue
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            if tags:
                if not clip.tags or not any(t in clip.tags for t in tags):
                    continue
            if has_colors is not None:
                clip_has_colors = bool(clip.dominant_colors)
                if has_colors != clip_has_colors:
                    continue

            filtered.append(
                {
                    "id": clip.id,
                    "source_name": source.filename if source else "Unknown",
                    "duration": duration,
                    "shot_type": clip.shot_type,
                    "has_transcript": bool(clip.transcript),
                    "tags": clip.tags,
                }
            )

        return json.dumps(
            {
                "success": True,
                "total_clips": len(clips),
                "filtered_count": len(filtered),
                "filters_applied": {
                    "shot_type": shot_type,
                    "has_speech": has_speech,
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "tags": tags,
                    "has_colors": has_colors,
                },
                "clips": filtered,
            }
        )
    except Exception as e:
        logger.exception("Failed to filter clips")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def get_clip_metadata(
    project_path: Annotated[str, "Path to project file"],
    clip_id: Annotated[str, "ID of the clip"],
    ctx: Context = None,
) -> str:
    """Get detailed metadata for a specific clip.

    Args:
        project_path: Path to the project file
        clip_id: ID of the clip to retrieve

    Returns:
        JSON with full clip metadata
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Find clip
        clip = None
        for c in clips:
            if c.id == clip_id:
                clip = c
                break

        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        # Get source
        source = None
        for s in sources:
            if s.id == clip.source_id:
                source = s
                break

        fps = source.fps if source else 30.0

        clip_data = {
            "id": clip.id,
            "source": {
                "id": clip.source_id,
                "filename": source.filename if source else "Unknown",
                "path": str(source.file_path) if source else None,
                "fps": fps,
                "resolution": f"{source.width}x{source.height}" if source else None,
            },
            "timing": {
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "start_time": clip.start_time(fps),
                "end_time": clip.end_time(fps),
                "duration_frames": clip.duration_frames,
                "duration_seconds": clip.duration_seconds(fps),
            },
            "analysis": {
                "shot_type": clip.shot_type,
                "dominant_colors": (
                    [{"r": c[0], "g": c[1], "b": c[2]} for c in clip.dominant_colors]
                    if clip.dominant_colors
                    else None
                ),
            },
            "transcript": (
                {
                    "text": clip.get_transcript_text(),
                    "segments": [seg.to_dict() for seg in clip.transcript],
                }
                if clip.transcript
                else None
            ),
            "user_data": {
                "tags": clip.tags,
                "notes": clip.notes,
            },
        }

        return json.dumps({"success": True, "clip": clip_data})
    except Exception as e:
        logger.exception("Failed to get clip metadata")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def add_clip_tags(
    project_path: Annotated[str, "Path to project file"],
    clip_id: Annotated[str, "ID of the clip to modify"],
    tags: Annotated[list[str], "Tags to add"],
    ctx: Context = None,
) -> str:
    """Add tags to a clip.

    Args:
        project_path: Path to the project file
        clip_id: ID of the clip to tag
        tags: List of tags to add

    Returns:
        JSON with updated clip tags
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Find and update clip
        clip = None
        for c in clips:
            if c.id == clip_id:
                clip = c
                break

        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        # Add new tags (avoid duplicates)
        existing_tags = set(clip.tags or [])
        new_tags = [t.strip() for t in tags if t.strip() and t.strip() not in existing_tags]
        clip.tags = list(existing_tags) + new_tags

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clip_id": clip_id,
                "tags_added": new_tags,
                "all_tags": clip.tags,
            }
        )
    except Exception as e:
        logger.exception("Failed to add tags")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def remove_clip_tags(
    project_path: Annotated[str, "Path to project file"],
    clip_id: Annotated[str, "ID of the clip to modify"],
    tags: Annotated[list[str], "Tags to remove"],
    ctx: Context = None,
) -> str:
    """Remove tags from a clip.

    Args:
        project_path: Path to the project file
        clip_id: ID of the clip to modify
        tags: List of tags to remove

    Returns:
        JSON with updated clip tags
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Find and update clip
        clip = None
        for c in clips:
            if c.id == clip_id:
                clip = c
                break

        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        # Remove tags
        tags_to_remove = set(t.strip() for t in tags)
        original_count = len(clip.tags or [])
        clip.tags = [t for t in (clip.tags or []) if t not in tags_to_remove]
        removed_count = original_count - len(clip.tags)

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clip_id": clip_id,
                "tags_removed": removed_count,
                "all_tags": clip.tags,
            }
        )
    except Exception as e:
        logger.exception("Failed to remove tags")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def add_clip_note(
    project_path: Annotated[str, "Path to project file"],
    clip_id: Annotated[str, "ID of the clip to modify"],
    note: Annotated[str, "Note text to set (replaces existing note)"],
    ctx: Context = None,
) -> str:
    """Set or update the note on a clip.

    Args:
        project_path: Path to the project file
        clip_id: ID of the clip to annotate
        note: Note text (empty string clears the note)

    Returns:
        JSON with updated note
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Find and update clip
        clip = None
        for c in clips:
            if c.id == clip_id:
                clip = c
                break

        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        # Update note
        clip.notes = note.strip()

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clip_id": clip_id,
                "note": clip.notes,
            }
        )
    except Exception as e:
        logger.exception("Failed to add note")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def search_transcripts(
    project_path: Annotated[str, "Path to project file"],
    query: Annotated[str, "Text to search for in transcripts"],
    case_sensitive: Annotated[bool, "Case-sensitive search"] = False,
    ctx: Context = None,
) -> str:
    """Search clip transcripts for text.

    Args:
        project_path: Path to the project file
        query: Text to search for
        case_sensitive: Whether the search is case-sensitive

    Returns:
        JSON with matching clips and transcript excerpts
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        matches = []
        search_query = query if case_sensitive else query.lower()

        for clip in clips:
            if not clip.transcript:
                continue

            transcript_text = clip.get_transcript_text()
            search_text = transcript_text if case_sensitive else transcript_text.lower()

            if search_query in search_text:
                source = sources_by_id.get(clip.source_id)
                fps = source.fps if source else 30.0

                # Find matching segments
                matching_segments = []
                for seg in clip.transcript:
                    seg_text = seg.text if case_sensitive else seg.text.lower()
                    if search_query in seg_text:
                        matching_segments.append(
                            {
                                "text": seg.text,
                                "start_time": seg.start_time,
                                "end_time": seg.end_time,
                            }
                        )

                matches.append(
                    {
                        "clip_id": clip.id,
                        "source_name": source.filename if source else "Unknown",
                        "clip_start_time": clip.start_time(fps),
                        "duration": clip.duration_seconds(fps),
                        "matching_segments": matching_segments,
                        "full_transcript": transcript_text,
                    }
                )

        return json.dumps(
            {
                "success": True,
                "query": query,
                "match_count": len(matches),
                "matches": matches,
            }
        )
    except Exception as e:
        logger.exception("Failed to search transcripts")
        return json.dumps({"success": False, "error": str(e)})
