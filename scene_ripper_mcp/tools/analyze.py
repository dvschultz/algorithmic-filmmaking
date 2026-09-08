"""Analysis MCP tools for color extraction, shot classification, and transcription."""

import asyncio
import json
import logging
from typing import Annotated

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_project_path

from core.spine.project_io import project_error, project_writer

logger = logging.getLogger(__name__)


@mcp.tool()
async def analyze_colors(
    project_path: Annotated[str, "Path to project file"],
    num_colors: Annotated[int, "Number of dominant colors to extract (1-10)"] = 5,
    ctx: Context = None,
) -> str:
    """Extract dominant colors from all clips in a project.

    Uses k-means clustering to identify the most dominant colors in each clip.
    Samples frames from each clip's source video; thumbnails are not required.

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
        with project_writer(path):
            from core.spine.analyze import analyze_colors as run_colors
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

            outcomes = run_colors(project, num_colors=num_colors, skip_existing=False)["result"]
            analyzed_count = len(outcomes["succeeded"])
            # Preserve the legacy skipped counter while also exposing actual errors.
            skipped_count = len(outcomes["skipped"]) + len(outcomes["failed"])

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

            result = {
                "success": True,
                "analyzed_clips": analyzed_count,
                "skipped_clips": skipped_count,
                "total_clips": len(clips),
            }
            if outcomes["failed"]:
                result["error_details"] = outcomes["failed"]
            return json.dumps(result)
    except Exception as e:
        logger.exception("Color analysis failed")
        return json.dumps({"success": False, "error": project_error(e)})


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

    store = ctx.request_context.lifespan_context["job_store"] if ctx else None
    return await asyncio.to_thread(_analyze_shots_sync, path, store)


def _analyze_shots_sync(path, store=None):
    """Synchronous body for ``analyze_shots`` (offloaded via ``asyncio.to_thread``)."""
    try:
        with project_writer(path):
            from threading import Event
            from core.jobs.shots import run_shot_job, shot_job_spec
            from core.jobs.store import JobStore
            from core.operations.shots import ShotTypeOptions
            from core.settings import load_settings
            from core.project import MissingSourceError
            from core.spine.project_io import (
                ProjectModifiedExternally,
                load_with_mtime,
            )

            try:
                project, _ = load_with_mtime(path)
            except MissingSourceError as e:
                return json.dumps({
                    "success": False,
                    "error": {"code": "source_files_missing", "message": str(e)},
                })

            clips = project.clips
            if not clips:
                return json.dumps({"success": False, "error": "No clips in project"})

            # The project loader resolves persisted thumbnail paths. Do not
            # regenerate images in this legacy analysis-only entry point.
            selected = [c.id for c in clips if c.source_id in project.sources_by_id]
            options = ShotTypeOptions()
            operation = shot_job_spec(project, selected, options,
                arguments={"force": True, "atomic": True})
            owned_store = store is None
            if owned_store:
                store = JobStore(load_settings().cache_dir / "jobs.db")
            try:
                outcomes = run_shot_job(store, path, selected, lambda *_: None,
                    Event(), operation=operation)["result"]
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
            finally:
                if owned_store:
                    store.close()

            completed = list(outcomes["succeeded"])
            # A saved result whose checkpoint failed is completed by this retry.
            # Its label is already in the loaded project and the job verifies it.
            completed.extend(
                {"clip_id": item["clip_id"], "shot_type": project.clips_by_id[item["clip_id"]].shot_type}
                for item in outcomes["skipped"] if item["reason"] == "already_committed"
            )
            analyzed_count = len(completed)
            skipped_count = len(clips) - analyzed_count
            shot_type_counts: dict = {}
            for item in completed:
                shot_type = item["shot_type"]
                shot_type_counts[shot_type] = shot_type_counts.get(shot_type, 0) + 1

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
        return json.dumps({"success": False, "error": project_error(e)})


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

    store = ctx.request_context.lifespan_context["job_store"] if ctx else None
    return await asyncio.to_thread(_transcribe_sync, path, model, language, store)


def _transcribe_sync(path, model, language, store=None):
    """Synchronous body for ``transcribe`` (offloaded via ``asyncio.to_thread``)."""
    try:
        with project_writer(path):
            from core.project import MissingSourceError
            from threading import Event
            from core.spine.project_io import load_with_mtime
            from core.jobs.store import JobStore
            from core.jobs.transcription import run_transcription_job
            from core.operations.transcription import TranscriptionOptions
            from core.settings import load_settings

            try:
                project, _ = load_with_mtime(path)
            except MissingSourceError as e:
                return json.dumps({
                    "success": False,
                    "error": {"code": "source_files_missing", "message": str(e)},
                })

            clips = project.clips
            if not clips:
                return json.dumps({"success": False, "error": "No clips in project"})

            batch = run_transcription_job(
                store if store is not None else JobStore(load_settings().cache_dir / "jobs.db"),
                path, None, TranscriptionOptions(model=model, language=language),
                lambda *_: None, Event(), force=True,
            )["result"]
            transcribed_count = len(batch["succeeded"])
            total_segments = sum(item["segment_count"] for item in batch["succeeded"])
            skipped_count = len(clips) - transcribed_count
            dependency_errors = [item for item in batch["failed"] if item["code"] == "dependency_missing"]
            if not transcribed_count and dependency_errors:
                return json.dumps({"success": False, "error": dependency_errors[0]["message"]})

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
        return json.dumps({"success": False, "error": project_error(e)})


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
        from core.analysis_availability import operation_is_complete_for_clip

        has_cinematography = sum(
            operation_is_complete_for_clip("cinematography", c, source=project.sources_by_id.get(c.source_id))
            for c in clips
        )
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
