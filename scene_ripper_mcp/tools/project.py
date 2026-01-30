"""Project management MCP tools."""

import json
import logging
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_path, validate_project_path, validate_video_path

logger = logging.getLogger(__name__)


@mcp.tool()
async def get_project_info(
    project_path: Annotated[str, "Absolute path to .json project file"],
    ctx: Context,
) -> str:
    """Get information about a Scene Ripper project.

    Returns project metadata including source count, clip count,
    sequence length, and analysis status.

    Args:
        project_path: Absolute path to the project file

    Returns:
        JSON with project information
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state = load_project(path)

        # Compute analysis statistics
        has_colors = sum(1 for c in clips if c.dominant_colors)
        has_shots = sum(1 for c in clips if c.shot_type)
        has_transcripts = sum(1 for c in clips if c.transcript)

        # Get sequence clip count
        sequence_clip_count = 0
        if sequence:
            sequence_clip_count = sum(len(track.clips) for track in sequence.tracks)

        return json.dumps(
            {
                "success": True,
                "project": {
                    "path": str(path),
                    "name": metadata.name,
                    "source_count": len(sources),
                    "clip_count": len(clips),
                    "sequence_clip_count": sequence_clip_count,
                    "clips_with_colors": has_colors,
                    "clips_with_shots": has_shots,
                    "clips_with_transcripts": has_transcripts,
                    "created_at": metadata.created_at,
                    "modified_at": metadata.modified_at,
                },
            }
        )
    except Exception as e:
        logger.exception("Failed to get project info")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def detect_scenes(
    video_path: Annotated[str, "Absolute path to video file"],
    output_project: Annotated[str, "Path for output project JSON"],
    sensitivity: Annotated[float, "Detection sensitivity (1.0=more scenes, 10.0=fewer)"] = 3.0,
    min_scene_length: Annotated[float, "Minimum scene length in seconds"] = 0.5,
    ctx: Context = None,
) -> str:
    """Detect scenes in a video file and create a new project.

    Uses adaptive scene detection to identify shot boundaries.
    Creates a project file with the detected clips.

    Args:
        video_path: Path to the video file
        output_project: Path for the new project file
        sensitivity: Detection sensitivity (1.0-10.0, lower = more scenes)
        min_scene_length: Minimum scene duration in seconds

    Returns:
        JSON with detection results and clip count
    """
    if ctx:
        await ctx.report_progress(0.0, "Validating paths...")

    valid, error, video = validate_video_path(video_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Video: {error}"})

    valid, error, output = validate_project_path(output_project, must_exist=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    try:
        if ctx:
            await ctx.report_progress(0.1, "Running scene detection...")

        from core.scene_detect import SceneDetector, DetectionConfig
        from core.project import save_project, ProjectMetadata

        # Configure and run detection
        config = DetectionConfig(
            threshold=sensitivity,
            min_scene_length=int(min_scene_length * 30),  # Convert to frames at 30fps estimate
            use_adaptive=True,
        )
        detector = SceneDetector(config)
        source, clips = detector.detect_scenes(video)

        if ctx:
            await ctx.report_progress(0.8, "Creating project...")

        # Create metadata
        metadata = ProjectMetadata(name=video.stem)

        # Save project
        output.parent.mkdir(parents=True, exist_ok=True)
        success = save_project(
            filepath=output,
            sources=[source],
            clips=clips,
            sequence=None,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        if ctx:
            await ctx.report_progress(1.0, "Complete")

        # Calculate total duration
        total_duration = sum(c.duration_seconds(source.fps) for c in clips)

        return json.dumps(
            {
                "success": True,
                "project_path": str(output),
                "source": str(video),
                "clip_count": len(clips),
                "total_duration": total_duration,
                "fps": source.fps,
            }
        )
    except Exception as e:
        logger.exception("Scene detection failed")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def create_project(
    name: Annotated[str, "Project name"],
    output_path: Annotated[str, "Path for the new project file"],
    ctx: Context = None,
) -> str:
    """Create a new empty project.

    Args:
        name: Name for the project
        output_path: Path where the project file will be created

    Returns:
        JSON with creation result
    """
    valid, error, path = validate_project_path(output_path, must_exist=False)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import save_project, ProjectMetadata

        metadata = ProjectMetadata(name=name)

        path.parent.mkdir(parents=True, exist_ok=True)
        success = save_project(
            filepath=path,
            sources=[],
            clips=[],
            sequence=None,
            metadata=metadata,
        )

        if success:
            return json.dumps(
                {
                    "success": True,
                    "project_path": str(path),
                    "name": name,
                }
            )
        else:
            return json.dumps({"success": False, "error": "Failed to save project"})
    except Exception as e:
        logger.exception("Failed to create project")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def list_projects(
    directory: Annotated[str, "Directory to search for project files"],
    ctx: Context = None,
) -> str:
    """List Scene Ripper projects in a directory.

    Searches for .sceneripper and .json files that appear to be Scene Ripper projects.

    Args:
        directory: Directory path to search

    Returns:
        JSON with list of project files found
    """
    valid, error, dir_path = validate_path(directory, must_exist=True, must_be_dir=True)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        projects = []

        # Search for both .sceneripper and .json files (non-recursive for safety)
        for pattern in ["*.sceneripper", "*.json"]:
            for project_file in dir_path.glob(pattern):
                try:
                    with open(project_file, "r") as f:
                        data = json.load(f)

                    # Check if it looks like a Scene Ripper project
                    if "version" in data and ("sources" in data or "clips" in data):
                        projects.append(
                            {
                                "path": str(project_file),
                                "name": data.get("project_name", project_file.stem),
                                "modified": project_file.stat().st_mtime,
                            }
                        )
                except (json.JSONDecodeError, OSError):
                    continue

        # Sort by modification time (newest first)
        projects.sort(key=lambda p: p["modified"], reverse=True)

        return json.dumps(
            {
                "success": True,
                "directory": str(dir_path),
                "count": len(projects),
                "projects": projects,
            }
        )
    except Exception as e:
        logger.exception("Failed to list projects")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def import_video(
    project_path: Annotated[str, "Path to project file"],
    video_path: Annotated[str, "Path to video file to import"],
    detect_scenes_flag: Annotated[bool, "Run scene detection after import"] = True,
    sensitivity: Annotated[float, "Scene detection sensitivity"] = 3.0,
    ctx: Context = None,
) -> str:
    """Import a video file into an existing project.

    Optionally runs scene detection to create clips from the video.

    Args:
        project_path: Path to the project file
        video_path: Path to the video file to import
        detect_scenes_flag: Whether to run scene detection
        sensitivity: Detection sensitivity (1.0-10.0)

    Returns:
        JSON with import result
    """
    valid, error, proj_path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Project: {error}"})

    valid, error, vid_path = validate_video_path(video_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Video: {error}"})

    try:
        from core.project import load_project, save_project
        from core.scene_detect import SceneDetector, DetectionConfig
        from models.clip import Source

        if ctx:
            await ctx.report_progress(0.1, "Loading project...")

        # Load existing project
        sources, clips, sequence, metadata, ui_state = load_project(proj_path)

        if ctx:
            await ctx.report_progress(0.2, "Analyzing video...")

        # Create source and optionally detect scenes
        if detect_scenes_flag:
            config = DetectionConfig(
                threshold=sensitivity,
                min_scene_length=15,
                use_adaptive=True,
            )
            detector = SceneDetector(config)
            source, new_clips = detector.detect_scenes(vid_path)
            clips.extend(new_clips)
        else:
            # Create source without scene detection
            from scenedetect.backends.opencv import VideoStreamCv2

            video = VideoStreamCv2(str(vid_path))
            source = Source(
                file_path=vid_path,
                duration_seconds=video.duration.get_seconds(),
                fps=video.frame_rate,
                width=video.frame_size[0],
                height=video.frame_size[1],
            )
            new_clips = []

        sources.append(source)

        if ctx:
            await ctx.report_progress(0.8, "Saving project...")

        # Save updated project
        success = save_project(
            filepath=proj_path,
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
                "source_id": source.id,
                "source_path": str(vid_path),
                "clips_created": len(new_clips),
                "total_sources": len(sources),
                "total_clips": len(clips),
            }
        )
    except Exception as e:
        logger.exception("Failed to import video")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def list_sources(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """List all video sources in a project.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with list of sources and their metadata
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project

        sources, clips, _, _, _ = load_project(path)

        source_list = []
        for source in sources:
            # Count clips for this source
            source_clips = [c for c in clips if c.source_id == source.id]

            source_list.append(
                {
                    "id": source.id,
                    "filename": source.filename,
                    "path": str(source.file_path),
                    "duration_seconds": source.duration_seconds,
                    "fps": source.fps,
                    "resolution": f"{source.width}x{source.height}",
                    "clip_count": len(source_clips),
                    "exists": source.file_path.exists(),
                }
            )

        return json.dumps(
            {
                "success": True,
                "count": len(source_list),
                "sources": source_list,
            }
        )
    except Exception as e:
        logger.exception("Failed to list sources")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def remove_source(
    project_path: Annotated[str, "Path to project file"],
    source_id: Annotated[str, "ID of the source to remove"],
    ctx: Context = None,
) -> str:
    """Remove a video source from a project.

    Also removes all clips associated with the source.

    Args:
        project_path: Path to the project file
        source_id: ID of the source to remove

    Returns:
        JSON with removal result
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state = load_project(path)

        # Find and remove source
        source_to_remove = None
        for i, source in enumerate(sources):
            if source.id == source_id:
                source_to_remove = sources.pop(i)
                break

        if not source_to_remove:
            return json.dumps({"success": False, "error": f"Source not found: {source_id}"})

        # Remove associated clips
        original_clip_count = len(clips)
        clips = [c for c in clips if c.source_id != source_id]
        removed_clips = original_clip_count - len(clips)

        # Remove from sequence if present
        if sequence:
            for track in sequence.tracks:
                track.clips = [c for c in track.clips if c.source_id != source_id]

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

        return json.dumps(
            {
                "success": True,
                "removed_source": source_to_remove.filename,
                "removed_clips": removed_clips,
                "remaining_sources": len(sources),
                "remaining_clips": len(clips),
            }
        )
    except Exception as e:
        logger.exception("Failed to remove source")
        return json.dumps({"success": False, "error": str(e)})
