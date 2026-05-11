"""Analysis MCP tools for color extraction, shot classification, and transcription."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_project_path

logger = logging.getLogger(__name__)


@mcp.tool()
async def analyze_colors(
    project_path: Annotated[str, "Path to project file"],
    num_colors: Annotated[int, "Number of dominant colors to extract (1-10)"] = 5,
    ctx: Context = None,
) -> str:
    """Extract dominant colors from all clip thumbnails in a project.

    Uses k-means clustering to identify the most dominant colors in each clip.
    Requires thumbnails to be generated first.

    Args:
        project_path: Path to the project file
        num_colors: Number of colors to extract per clip (1-10)

    Returns:
        JSON with analysis results
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    return await asyncio.to_thread(_analyze_colors_sync, path, num_colors)


def _analyze_colors_sync(path, num_colors):
    """Synchronous body for ``analyze_colors`` (offloaded via ``asyncio.to_thread``)."""
    try:
        from core.analysis.color import extract_dominant_colors
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

        clips = project.clips
        if not clips:
            return json.dumps({"success": False, "error": "No clips in project"})

        sources_by_id = project.sources_by_id

        analyzed_count = 0
        skipped_count = 0
        updated: list = []

        for clip in clips:
            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                skipped_count += 1
                continue

            # Extract colors by sampling frames from the video
            colors = extract_dominant_colors(
                video_path=source.file_path,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                n_colors=num_colors,
            )
            if colors:
                clip.dominant_colors = colors
                analyzed_count += 1
                updated.append(clip)
            else:
                skipped_count += 1

        if updated:
            project.update_clips(updated)

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
                "analyzed_clips": analyzed_count,
                "skipped_clips": skipped_count,
                "total_clips": len(clips),
            }
        )
    except Exception as e:
        logger.exception("Color analysis failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def analyze_shots(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """Classify shot types (wide, medium, close-up) for all clips.

    Uses CLIP zero-shot classification to determine shot type from thumbnails.
    Requires thumbnails to be generated first.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with classification results
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    return await asyncio.to_thread(_analyze_shots_sync, path)


def _analyze_shots_sync(path):
    """Synchronous body for ``analyze_shots`` (offloaded via ``asyncio.to_thread``)."""
    try:
        from core.analysis.shots import classify_shot_type
        from core.project import MissingSourceError
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )
        from core.thumbnail import get_thumbnail_path

        try:
            project, mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clips = project.clips
        if not clips:
            return json.dumps({"success": False, "error": "No clips in project"})

        sources_by_id = project.sources_by_id

        analyzed_count = 0
        skipped_count = 0
        shot_type_counts: dict = {}
        updated: list = []

        for clip in clips:
            source = sources_by_id.get(clip.source_id)
            if not source:
                skipped_count += 1
                continue

            thumb_path = get_thumbnail_path(source.file_path, clip.start_frame)
            if not thumb_path or not thumb_path.exists():
                if clip.thumbnail_path and Path(clip.thumbnail_path).exists():
                    thumb_path = Path(clip.thumbnail_path)
                else:
                    skipped_count += 1
                    continue

            shot_type, _confidence = classify_shot_type(thumb_path)
            if shot_type != "unknown":
                clip.shot_type = shot_type
                analyzed_count += 1
                shot_type_counts[shot_type] = shot_type_counts.get(shot_type, 0) + 1
                updated.append(clip)
            else:
                skipped_count += 1

        if updated:
            project.update_clips(updated)

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
                "analyzed_clips": analyzed_count,
                "skipped_clips": skipped_count,
                "total_clips": len(clips),
                "shot_type_distribution": shot_type_counts,
            }
        )
    except Exception as e:
        logger.exception("Shot classification failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def transcribe(
    project_path: Annotated[str, "Path to project file"],
    model: Annotated[str, "Whisper model: tiny.en, small.en, medium.en, large-v3"] = "small.en",
    language: Annotated[str, "Language code (en, auto, etc.)"] = "en",
    ctx: Context = None,
) -> str:
    """Transcribe speech in all clips using Whisper.

    Uses faster-whisper for efficient transcription. Extracts audio from each
    clip and runs speech recognition.

    Args:
        project_path: Path to the project file
        model: Whisper model size
        language: Language code or 'auto' for detection

    Returns:
        JSON with transcription results
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    return await asyncio.to_thread(_transcribe_sync, path, model, language)


def _transcribe_sync(path, model, language):
    """Synchronous body for ``transcribe`` (offloaded via ``asyncio.to_thread``)."""
    try:
        from core.transcription import is_faster_whisper_available

        if not is_faster_whisper_available():
            return json.dumps(
                {
                    "success": False,
                    "error": "faster-whisper not installed. Run: pip install faster-whisper",
                }
            )

        from core.project import MissingSourceError
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )
        from core.transcription import transcribe_clip

        try:
            project, mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clips = project.clips
        if not clips:
            return json.dumps({"success": False, "error": "No clips in project"})

        sources_by_id = project.sources_by_id

        transcribed_count = 0
        skipped_count = 0
        total_segments = 0
        updated: list = []

        for clip in clips:
            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                skipped_count += 1
                continue

            try:
                segments = transcribe_clip(
                    video_path=source.file_path,
                    start_time=clip.start_time(source.fps),
                    end_time=clip.end_time(source.fps),
                    model_name=model,
                    language=language if language != "auto" else None,
                )

                if segments:
                    clip.transcript = segments
                    transcribed_count += 1
                    total_segments += len(segments)
                    updated.append(clip)
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"Failed to transcribe clip {clip.id}: {e}")
                skipped_count += 1

        if updated:
            project.update_clips(updated)

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
                "transcribed_clips": transcribed_count,
                "skipped_clips": skipped_count,
                "total_clips": len(clips),
                "total_segments": total_segments,
                "model_used": model,
            }
        )
    except Exception as e:
        logger.exception("Transcription failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def get_analysis_status(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """Check what analysis has been run on a project.

    Returns counts of clips with colors, shot types, and transcripts.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with analysis status
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

        # Count analysis types
        has_colors = sum(1 for c in clips if c.dominant_colors)
        has_shots = sum(1 for c in clips if c.shot_type)
        has_transcripts = sum(1 for c in clips if c.transcript)
        has_classification = sum(1 for c in clips if c.object_labels)
        has_objects = sum(1 for c in clips if c.detected_objects)
        has_descriptions = sum(1 for c in clips if c.description)
        has_text = sum(1 for c in clips if c.extracted_texts)
        has_cinematography = sum(1 for c in clips if c.cinematography)
        has_faces = sum(1 for c in clips if c.face_embeddings is not None)
        has_gaze = sum(1 for c in clips if c.gaze_category is not None)
        has_embeddings = sum(1 for c in clips if c.embedding is not None)
        has_custom_queries = sum(1 for c in clips if c.custom_queries)
        has_tags = sum(1 for c in clips if c.tags)
        has_notes = sum(1 for c in clips if c.notes)

        # Shot type distribution
        shot_types: dict = {}
        for c in clips:
            if c.shot_type:
                shot_types[c.shot_type] = shot_types.get(c.shot_type, 0) + 1

        return json.dumps(
            {
                "success": True,
                "total_clips": len(clips),
                "analysis": {
                    "colors": {
                        "analyzed": has_colors,
                        "pending": len(clips) - has_colors,
                        "percentage": (has_colors / len(clips) * 100) if clips else 0,
                    },
                    "shots": {
                        "analyzed": has_shots,
                        "pending": len(clips) - has_shots,
                        "percentage": (has_shots / len(clips) * 100) if clips else 0,
                        "distribution": shot_types,
                    },
                    "transcripts": {
                        "analyzed": has_transcripts,
                        "pending": len(clips) - has_transcripts,
                        "percentage": (has_transcripts / len(clips) * 100) if clips else 0,
                    },
                    "classification": {
                        "analyzed": has_classification,
                        "pending": len(clips) - has_classification,
                    },
                    "objects": {
                        "analyzed": has_objects,
                        "pending": len(clips) - has_objects,
                    },
                    "descriptions": {
                        "analyzed": has_descriptions,
                        "pending": len(clips) - has_descriptions,
                    },
                    "text": {
                        "analyzed": has_text,
                        "pending": len(clips) - has_text,
                    },
                    "cinematography": {
                        "analyzed": has_cinematography,
                        "pending": len(clips) - has_cinematography,
                    },
                    "faces": {
                        "analyzed": has_faces,
                        "pending": len(clips) - has_faces,
                    },
                    "gaze": {
                        "analyzed": has_gaze,
                        "pending": len(clips) - has_gaze,
                    },
                    "embeddings": {
                        "analyzed": has_embeddings,
                        "pending": len(clips) - has_embeddings,
                    },
                    "custom_queries": {
                        "analyzed": has_custom_queries,
                        "pending": len(clips) - has_custom_queries,
                    },
                },
                "metadata": {
                    "clips_with_tags": has_tags,
                    "clips_with_notes": has_notes,
                },
            }
        )
    except Exception as e:
        logger.exception("Failed to get analysis status")
        return json.dumps({"success": False, "error": str(e)})
