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
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clips = project.clips
        sources_by_id = project.sources_by_id

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
            if clip.object_labels:
                clip_data["object_labels"] = clip.object_labels[:10]
            if clip.detected_objects:
                clip_data["detected_object_labels"] = sorted(
                    {d.get("label", "") for d in clip.detected_objects if d.get("label")}
                )
            if clip.person_count is not None:
                clip_data["person_count"] = clip.person_count
            if clip.description:
                clip_data["description"] = clip.description[:500]
                clip_data["description_model"] = clip.description_model
            if clip.extracted_texts:
                clip_data["extracted_text"] = clip.combined_text
            if clip.custom_queries:
                clip_data["custom_queries"] = clip.custom_queries
            if clip.cinematography:
                clip_data["cinematography"] = clip.cinematography.to_dict()
            if clip.face_embeddings is not None:
                clip_data["face_count"] = len(clip.face_embeddings)
            if clip.gaze_category is not None:
                clip_data["gaze"] = {
                    "yaw": clip.gaze_yaw,
                    "pitch": clip.gaze_pitch,
                    "category": clip.gaze_category,
                }
            if clip.embedding is not None:
                clip_data["embedding"] = {
                    "model": clip.embedding_model,
                    "dimension": len(clip.embedding),
                }
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
    has_description: Annotated[Optional[bool], "Filter by description presence"] = None,
    has_objects: Annotated[Optional[bool], "Filter by object detection presence"] = None,
    has_faces: Annotated[Optional[bool], "Filter by face detection presence"] = None,
    gaze_category: Annotated[Optional[str], "Filter by gaze category"] = None,
    has_text: Annotated[Optional[bool], "Filter by OCR text presence"] = None,
    has_cinematography: Annotated[Optional[bool], "Filter by rich cinematography presence"] = None,
    custom_query: Annotated[Optional[str], "Filter to latest matching custom query"] = None,
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
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clips = project.clips
        sources_by_id = project.sources_by_id

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
            if has_description is not None and has_description != bool(clip.description):
                continue
            if has_objects is not None and has_objects != bool(clip.detected_objects):
                continue
            if has_faces is not None and has_faces != bool(clip.face_embeddings):
                continue
            if gaze_category and clip.gaze_category != gaze_category:
                continue
            if has_text is not None and has_text != bool(clip.combined_text):
                continue
            if has_cinematography is not None and has_cinematography != bool(clip.cinematography):
                continue
            if custom_query:
                matches = [
                    q for q in (clip.custom_queries or [])
                    if q.get("query") == custom_query
                ]
                if not matches or matches[-1].get("match") is not True:
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
                    "has_description": bool(clip.description),
                    "has_objects": bool(clip.detected_objects),
                    "person_count": clip.person_count,
                    "has_faces": bool(clip.face_embeddings),
                    "gaze_category": clip.gaze_category,
                    "has_text": bool(clip.combined_text),
                    "has_cinematography": bool(clip.cinematography),
                    "has_embedding": bool(clip.embedding),
                    "custom_queries": clip.custom_queries,
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
                    "has_description": has_description,
                    "has_objects": has_objects,
                    "has_faces": has_faces,
                    "gaze_category": gaze_category,
                    "has_text": has_text,
                    "has_cinematography": has_cinematography,
                    "custom_query": custom_query,
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
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clip = project.clips_by_id.get(clip_id)
        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        source = project.sources_by_id.get(clip.source_id)
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
                "object_labels": clip.object_labels,
                "detected_objects": clip.detected_objects,
                "person_count": clip.person_count,
                "description": clip.description,
                "description_model": clip.description_model,
                "description_frames": clip.description_frames,
                "extracted_texts": (
                    [text.to_dict() for text in clip.extracted_texts]
                    if clip.extracted_texts
                    else None
                ),
                "combined_text": clip.combined_text,
                "custom_queries": clip.custom_queries,
                "cinematography": (
                    clip.cinematography.to_dict()
                    if clip.cinematography
                    else None
                ),
                "face_count": (
                    len(clip.face_embeddings)
                    if clip.face_embeddings is not None
                    else None
                ),
                "faces": (
                    [
                        {
                            "bbox": face.get("bbox"),
                            "confidence": face.get("confidence"),
                            **(
                                {"frame_number": face["frame_number"]}
                                if "frame_number" in face
                                else {}
                            ),
                        }
                        for face in clip.face_embeddings
                    ]
                    if clip.face_embeddings
                    else None
                ),
                "gaze": (
                    {
                        "yaw": clip.gaze_yaw,
                        "pitch": clip.gaze_pitch,
                        "category": clip.gaze_category,
                    }
                    if clip.gaze_category is not None
                    else None
                ),
                "embedding": (
                    {
                        "model": clip.embedding_model,
                        "dimension": len(clip.embedding),
                    }
                    if clip.embedding is not None
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
        from core.project import MissingSourceError
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )

        try:
            project, mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clip = project.clips_by_id.get(clip_id)
        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        # Add new tags (avoid duplicates)
        existing_tags = set(clip.tags or [])
        new_tags = [t.strip() for t in tags if t.strip() and t.strip() not in existing_tags]
        clip.tags = list(existing_tags) + new_tags
        project.update_clips([clip])

        try:
            save_with_mtime_check(project, path, mtime)
        except ProjectModifiedExternally as exc:
            return json.dumps({
                "success": False,
                "error": {
                    "code": "project_modified_externally",
                    "path": str(exc.path),
                    "expected_mtime": exc.expected_mtime,
                    "current_mtime": exc.current_mtime,
                },
            })

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
        from core.project import MissingSourceError
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )

        try:
            project, mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clip = project.clips_by_id.get(clip_id)
        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        # Remove tags
        tags_to_remove = set(t.strip() for t in tags)
        original_count = len(clip.tags or [])
        clip.tags = [t for t in (clip.tags or []) if t not in tags_to_remove]
        removed_count = original_count - len(clip.tags)
        project.update_clips([clip])

        try:
            save_with_mtime_check(project, path, mtime)
        except ProjectModifiedExternally as exc:
            return json.dumps({
                "success": False,
                "error": {
                    "code": "project_modified_externally",
                    "path": str(exc.path),
                    "expected_mtime": exc.expected_mtime,
                    "current_mtime": exc.current_mtime,
                },
            })

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
        from core.project import MissingSourceError
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )

        try:
            project, mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clip = project.clips_by_id.get(clip_id)
        if not clip:
            return json.dumps({"success": False, "error": f"Clip not found: {clip_id}"})

        clip.notes = note.strip()
        project.update_clips([clip])

        try:
            save_with_mtime_check(project, path, mtime)
        except ProjectModifiedExternally as exc:
            return json.dumps({
                "success": False,
                "error": {
                    "code": "project_modified_externally",
                    "path": str(exc.path),
                    "expected_mtime": exc.expected_mtime,
                    "current_mtime": exc.current_mtime,
                },
            })

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
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clips = project.clips
        sources_by_id = project.sources_by_id

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
