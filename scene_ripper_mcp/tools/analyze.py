"""Analysis MCP tools for color extraction, shot classification, and transcription."""

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

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

    try:
        from core.project import load_project, save_project
        from core.analysis.color import extract_dominant_colors, classify_color_palette
        from core.thumbnail import get_thumbnail_path

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, sequence, metadata, ui_state = load_project(path)

        if not clips:
            return json.dumps({"success": False, "error": "No clips in project"})

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        analyzed_count = 0
        skipped_count = 0

        for i, clip in enumerate(clips):
            if ctx:
                progress = 0.1 + (0.8 * i / len(clips))
                await ctx.report_progress(progress, f"Analyzing clip {i + 1}/{len(clips)}...")

            # Get thumbnail path
            source = sources_by_id.get(clip.source_id)
            if not source:
                skipped_count += 1
                continue

            thumb_path = get_thumbnail_path(source.file_path, clip.start_frame)
            if not thumb_path or not thumb_path.exists():
                # Try to find any existing thumbnail
                if clip.thumbnail_path and Path(clip.thumbnail_path).exists():
                    thumb_path = Path(clip.thumbnail_path)
                else:
                    skipped_count += 1
                    continue

            # Extract colors
            colors = extract_dominant_colors(thumb_path, n_colors=num_colors)
            if colors:
                clip.dominant_colors = colors
                analyzed_count += 1
            else:
                skipped_count += 1

        if ctx:
            await ctx.report_progress(0.9, "Saving project...")

        # Save updated project
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

        if ctx:
            await ctx.report_progress(1.0, "Complete")

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

    try:
        from core.project import load_project, save_project
        from core.analysis.shots import classify_shot_type
        from core.thumbnail import get_thumbnail_path

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, sequence, metadata, ui_state = load_project(path)

        if not clips:
            return json.dumps({"success": False, "error": "No clips in project"})

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        analyzed_count = 0
        skipped_count = 0
        shot_type_counts = {}

        for i, clip in enumerate(clips):
            if ctx:
                progress = 0.1 + (0.8 * i / len(clips))
                await ctx.report_progress(progress, f"Classifying clip {i + 1}/{len(clips)}...")

            # Get thumbnail path
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

            # Classify shot type
            shot_type, confidence = classify_shot_type(thumb_path)
            if shot_type != "unknown":
                clip.shot_type = shot_type
                analyzed_count += 1
                shot_type_counts[shot_type] = shot_type_counts.get(shot_type, 0) + 1
            else:
                skipped_count += 1

        if ctx:
            await ctx.report_progress(0.9, "Saving project...")

        # Save updated project
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

        if ctx:
            await ctx.report_progress(1.0, "Complete")

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

    try:
        from core.transcription import is_faster_whisper_available

        if not is_faster_whisper_available():
            return json.dumps(
                {
                    "success": False,
                    "error": "faster-whisper not installed. Run: pip install faster-whisper",
                }
            )

        from core.project import load_project, save_project
        from core.transcription import transcribe_clip

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, sequence, metadata, ui_state = load_project(path)

        if not clips:
            return json.dumps({"success": False, "error": "No clips in project"})

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        transcribed_count = 0
        skipped_count = 0
        total_segments = 0

        for i, clip in enumerate(clips):
            if ctx:
                progress = 0.1 + (0.8 * i / len(clips))
                await ctx.report_progress(progress, f"Transcribing clip {i + 1}/{len(clips)}...")

            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                skipped_count += 1
                continue

            try:
                # Transcribe the clip
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
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"Failed to transcribe clip {clip.id}: {e}")
                skipped_count += 1

        if ctx:
            await ctx.report_progress(0.9, "Saving project...")

        # Save updated project
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

        if ctx:
            await ctx.report_progress(1.0, "Complete")

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
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state = load_project(path)

        # Count analysis types
        has_colors = sum(1 for c in clips if c.dominant_colors)
        has_shots = sum(1 for c in clips if c.shot_type)
        has_transcripts = sum(1 for c in clips if c.transcript)
        has_tags = sum(1 for c in clips if c.tags)
        has_notes = sum(1 for c in clips if c.notes)

        # Shot type distribution
        shot_types = {}
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
