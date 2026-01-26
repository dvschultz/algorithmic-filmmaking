"""Export MCP tools for clips, sequences, EDL, and datasets."""

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_path, validate_project_path

logger = logging.getLogger(__name__)


@mcp.tool()
async def export_clips(
    project_path: Annotated[str, "Path to project file"],
    output_dir: Annotated[str, "Directory for exported clip files"],
    clip_ids: Annotated[Optional[list[str]], "Specific clip IDs to export (None = all)"] = None,
    accurate: Annotated[bool, "Frame-accurate cutting (slower but precise)"] = True,
    ctx: Context = None,
) -> str:
    """Export clips as individual video files.

    Extracts each clip from its source video and saves as a separate file.

    Args:
        project_path: Path to the project file
        output_dir: Directory to save exported clips
        clip_ids: Optional list of specific clip IDs to export
        accurate: Whether to use frame-accurate (re-encode) or keyframe (stream copy) cutting

    Returns:
        JSON with export results
    """
    valid, error, proj_path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Project: {error}"})

    valid, error, out_path = validate_path(output_dir, must_be_dir=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        from core.project import load_project
        from core.ffmpeg import FFmpegProcessor

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, _, _, _ = load_project(proj_path)

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        # Filter clips if specific IDs provided
        if clip_ids:
            clip_ids_set = set(clip_ids)
            clips_to_export = [c for c in clips if c.id in clip_ids_set]
        else:
            clips_to_export = clips

        if not clips_to_export:
            return json.dumps({"success": False, "error": "No clips to export"})

        # Ensure output directory exists
        out_path.mkdir(parents=True, exist_ok=True)

        ffmpeg = FFmpegProcessor()

        exported = []
        failed = []

        for i, clip in enumerate(clips_to_export):
            if ctx:
                progress = 0.1 + (0.8 * i / len(clips_to_export))
                await ctx.report_progress(progress, f"Exporting clip {i + 1}/{len(clips_to_export)}...")

            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                failed.append({"clip_id": clip.id, "error": "Source not found"})
                continue

            # Generate output filename
            start_time = clip.start_time(source.fps)
            output_file = out_path / f"{source.file_path.stem}_clip_{i:04d}_{start_time:.2f}s.mp4"

            # Extract clip
            success = ffmpeg.extract_clip(
                input_path=source.file_path,
                output_path=output_file,
                start_seconds=start_time,
                duration_seconds=clip.duration_seconds(source.fps),
                fps=source.fps,
                accurate=accurate,
            )

            if success:
                exported.append(
                    {
                        "clip_id": clip.id,
                        "file_path": str(output_file),
                        "duration": clip.duration_seconds(source.fps),
                    }
                )
            else:
                failed.append({"clip_id": clip.id, "error": "FFmpeg failed"})

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        return json.dumps(
            {
                "success": len(exported) > 0,
                "output_dir": str(out_path),
                "exported_count": len(exported),
                "failed_count": len(failed),
                "exported": exported,
                "failed": failed if failed else None,
            }
        )
    except Exception as e:
        logger.exception("Clip export failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def export_sequence(
    project_path: Annotated[str, "Path to project file"],
    output_path: Annotated[str, "Path for output video file"],
    ctx: Context = None,
) -> str:
    """Export the sequence as a single video file.

    Renders all clips in the sequence timeline to a single video.

    Args:
        project_path: Path to the project file
        output_path: Path for the output video file

    Returns:
        JSON with export result
    """
    valid, error, proj_path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Project: {error}"})

    valid, error, out_path = validate_path(output_path, must_be_file=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        from core.project import load_project
        from core.sequence_export import export_sequence as do_export

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, sequence, _, _ = load_project(proj_path)

        if not sequence:
            return json.dumps({"success": False, "error": "No sequence in project"})

        if not sequence.get_all_clips():
            return json.dumps({"success": False, "error": "Sequence is empty"})

        # Build lookups
        sources_dict = {s.id: s for s in sources}
        clips_dict = {}
        for clip in clips:
            source = sources_dict.get(clip.source_id)
            if source:
                clips_dict[clip.id] = (clip, source)

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if ctx:
            await ctx.report_progress(0.2, "Rendering sequence...")

        # Progress wrapper
        async def progress_cb(progress: float, message: str):
            if ctx:
                # Scale to 0.2-1.0 range
                await ctx.report_progress(0.2 + progress * 0.8, message)

        success = do_export(
            sequence=sequence,
            sources=sources_dict,
            clips=clips_dict,
            output_path=out_path,
        )

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        if success:
            return json.dumps(
                {
                    "success": True,
                    "output_path": str(out_path),
                    "duration_seconds": sequence.duration_seconds,
                    "clip_count": len(sequence.get_all_clips()),
                }
            )
        else:
            return json.dumps({"success": False, "error": "FFmpeg export failed"})
    except Exception as e:
        logger.exception("Sequence export failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def export_edl(
    project_path: Annotated[str, "Path to project file"],
    output_path: Annotated[str, "Path for output EDL file"],
    title: Annotated[str, "EDL title"] = "Scene Ripper Export",
    ctx: Context = None,
) -> str:
    """Export the sequence as an EDL (Edit Decision List) file.

    Creates a CMX 3600 format EDL that can be imported into NLE software
    like Premiere Pro, DaVinci Resolve, or Final Cut Pro.

    Args:
        project_path: Path to the project file
        output_path: Path for the output EDL file
        title: Title to include in the EDL header

    Returns:
        JSON with export result
    """
    valid, error, proj_path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Project: {error}"})

    valid, error, out_path = validate_path(output_path, must_be_file=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        from core.project import load_project
        from core.edl_export import export_edl as do_export, EDLExportConfig

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, sequence, _, _ = load_project(proj_path)

        if not sequence:
            return json.dumps({"success": False, "error": "No sequence in project"})

        if not sequence.get_all_clips():
            return json.dumps({"success": False, "error": "Sequence is empty"})

        # Build source lookup
        sources_dict = {s.id: s for s in sources}

        # Ensure output has .edl extension
        if not out_path.suffix.lower() == ".edl":
            out_path = out_path.with_suffix(".edl")

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if ctx:
            await ctx.report_progress(0.5, "Generating EDL...")

        config = EDLExportConfig(output_path=out_path, title=title)
        success = do_export(sequence=sequence, sources=sources_dict, config=config)

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        if success:
            return json.dumps(
                {
                    "success": True,
                    "output_path": str(out_path),
                    "clip_count": len(sequence.get_all_clips()),
                    "title": title,
                }
            )
        else:
            return json.dumps({"success": False, "error": "EDL export failed"})
    except Exception as e:
        logger.exception("EDL export failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def export_dataset(
    project_path: Annotated[str, "Path to project file"],
    output_path: Annotated[str, "Path for output JSON file"],
    include_thumbnails: Annotated[bool, "Include thumbnail paths"] = True,
    ctx: Context = None,
) -> str:
    """Export clip metadata as a JSON dataset.

    Creates a structured JSON file containing all clip metadata including
    timing, colors, shot types, and transcripts. Useful for analysis or
    machine learning workflows.

    Args:
        project_path: Path to the project file
        output_path: Path for the output JSON file
        include_thumbnails: Whether to include thumbnail paths

    Returns:
        JSON with export result
    """
    valid, error, proj_path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Project: {error}"})

    valid, error, out_path = validate_path(output_path, must_be_file=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        from core.project import load_project
        from core.dataset_export import export_dataset as do_export, DatasetExportConfig

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, _, _, _ = load_project(proj_path)

        if not sources:
            return json.dumps({"success": False, "error": "No sources in project"})

        # Ensure output has .json extension
        if not out_path.suffix.lower() == ".json":
            out_path = out_path.with_suffix(".json")

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if ctx:
            await ctx.report_progress(0.2, "Building dataset...")

        # Export for first source (dataset_export expects single source)
        # For multi-source projects, export each source separately or use a wrapper
        source = sources[0]
        source_clips = [c for c in clips if c.source_id == source.id]

        config = DatasetExportConfig(
            output_path=out_path,
            include_thumbnails=include_thumbnails,
            pretty_print=True,
        )

        success = do_export(source=source, clips=source_clips, config=config)

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        if success:
            return json.dumps(
                {
                    "success": True,
                    "output_path": str(out_path),
                    "source": source.filename,
                    "clip_count": len(source_clips),
                }
            )
        else:
            return json.dumps({"success": False, "error": "Dataset export failed"})
    except Exception as e:
        logger.exception("Dataset export failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def export_full_dataset(
    project_path: Annotated[str, "Path to project file"],
    output_path: Annotated[str, "Path for output JSON file"],
    ctx: Context = None,
) -> str:
    """Export complete project data including all sources and clips.

    Creates a comprehensive JSON export with all project metadata,
    useful for backup, analysis, or data interchange.

    Args:
        project_path: Path to the project file
        output_path: Path for the output JSON file

    Returns:
        JSON with export result
    """
    valid, error, proj_path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Project: {error}"})

    valid, error, out_path = validate_path(output_path, must_be_file=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        from datetime import datetime
        from core.project import load_project

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        sources, clips, sequence, metadata, ui_state = load_project(proj_path)

        # Build comprehensive export
        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "project": {
                "name": metadata.name,
                "id": metadata.id,
                "created_at": metadata.created_at,
                "modified_at": metadata.modified_at,
            },
            "sources": [],
            "clips": [],
            "sequence": None,
        }

        if ctx:
            await ctx.report_progress(0.3, "Processing sources...")

        # Build source lookup
        sources_by_id = {s.id: s for s in sources}

        # Add sources
        for source in sources:
            export_data["sources"].append(
                {
                    "id": source.id,
                    "filename": source.filename,
                    "path": str(source.file_path),
                    "duration_seconds": source.duration_seconds,
                    "fps": source.fps,
                    "width": source.width,
                    "height": source.height,
                }
            )

        if ctx:
            await ctx.report_progress(0.5, "Processing clips...")

        # Add clips
        for clip in clips:
            source = sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0

            clip_data = {
                "id": clip.id,
                "source_id": clip.source_id,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "start_time": clip.start_time(fps),
                "end_time": clip.end_time(fps),
                "duration_seconds": clip.duration_seconds(fps),
            }

            if clip.dominant_colors:
                clip_data["dominant_colors"] = [
                    {"r": int(c[0]), "g": int(c[1]), "b": int(c[2])} for c in clip.dominant_colors
                ]
            if clip.shot_type:
                clip_data["shot_type"] = clip.shot_type
            if clip.transcript:
                clip_data["transcript"] = {
                    "text": clip.get_transcript_text(),
                    "segments": [s.to_dict() for s in clip.transcript],
                }
            if clip.tags:
                clip_data["tags"] = clip.tags
            if clip.notes:
                clip_data["notes"] = clip.notes

            export_data["clips"].append(clip_data)

        if ctx:
            await ctx.report_progress(0.7, "Processing sequence...")

        # Add sequence if present
        if sequence:
            seq_data = {
                "id": sequence.id,
                "name": sequence.name,
                "fps": sequence.fps,
                "duration_seconds": sequence.duration_seconds,
                "tracks": [],
            }

            for track in sequence.tracks:
                track_data = {
                    "id": track.id,
                    "name": track.name,
                    "clips": [
                        {
                            "id": sc.id,
                            "source_clip_id": sc.source_clip_id,
                            "source_id": sc.source_id,
                            "start_frame": sc.start_frame,
                            "in_point": sc.in_point,
                            "out_point": sc.out_point,
                        }
                        for sc in track.clips
                    ],
                }
                seq_data["tracks"].append(track_data)

            export_data["sequence"] = seq_data

        if ctx:
            await ctx.report_progress(0.9, "Writing file...")

        # Ensure output has .json extension
        if not out_path.suffix.lower() == ".json":
            out_path = out_path.with_suffix(".json")

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        import json as json_module

        with open(out_path, "w", encoding="utf-8") as f:
            json_module.dump(export_data, f, indent=2, ensure_ascii=False)

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        return json.dumps(
            {
                "success": True,
                "output_path": str(out_path),
                "source_count": len(sources),
                "clip_count": len(clips),
                "has_sequence": sequence is not None,
            }
        )
    except Exception as e:
        logger.exception("Full dataset export failed")
        return json.dumps({"success": False, "error": str(e)})
