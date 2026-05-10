"""Project management MCP tools."""

import asyncio
import json
import logging
from typing import Annotated

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_path, validate_project_path, validate_video_path

logger = logging.getLogger(__name__)


@mcp.tool()
async def get_project_info(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
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
        metadata = project.metadata
        sequence = project.sequence

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
                    "source_count": len(project.sources),
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
    output_project: Annotated[str, "Path for output .sceneripper project file"],
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
    valid, error, video = validate_video_path(video_path)
    if not valid:
        return json.dumps({"success": False, "error": f"Video: {error}"})

    valid, error, output = validate_project_path(output_project, must_exist=False)
    if not valid:
        return json.dumps({"success": False, "error": f"Output: {error}"})

    return await asyncio.to_thread(
        _detect_scenes_sync, video, output, sensitivity, min_scene_length
    )


def _detect_scenes_sync(video, output, sensitivity, min_scene_length):
    """Synchronous body for ``detect_scenes`` (offloaded via ``asyncio.to_thread``)."""
    try:
        from core.spine.detect import detect_scenes_new_project

        result = detect_scenes_new_project(
            video,
            output,
            sensitivity=sensitivity,
        )
        if not result.get("success"):
            return json.dumps(result)

        payload = result["result"]
        return json.dumps(
            {
                "success": True,
                "project_path": payload["project_path"],
                "source": str(video),
                "clip_count": payload["clip_count"],
                "source_id": payload["source_id"],
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
        from core.project import Project, ProjectMetadata

        project = Project.new(name=name)
        project.metadata = ProjectMetadata(name=name)

        path.parent.mkdir(parents=True, exist_ok=True)
        success = project.save(path)

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
    directory: Annotated[str, "Directory to search for .sceneripper project files"],
    ctx: Context = None,
) -> str:
    """List Scene Ripper projects in a directory.

    Searches for .sceneripper files that are Scene Ripper projects.

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

        # Search for .sceneripper files (non-recursive for safety)
        for project_file in dir_path.glob("*.sceneripper"):
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

    return await asyncio.to_thread(
        _import_video_sync, proj_path, vid_path, detect_scenes_flag, sensitivity
    )


def _import_video_sync(proj_path, vid_path, detect_scenes_flag, sensitivity):
    """Synchronous body for ``import_video`` (offloaded via ``asyncio.to_thread``)."""
    try:
        from core.project import MissingSourceError
        from core.scene_detect import DetectionConfig, SceneDetector
        from core.spine.project_io import (
            ProjectModifiedExternally,
            load_with_mtime,
            save_with_mtime_check,
        )
        from models.clip import Source

        try:
            project, mtime = load_with_mtime(proj_path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        # Create source and optionally detect scenes
        if detect_scenes_flag:
            config = DetectionConfig(
                threshold=sensitivity,
                min_scene_length=15,
                use_adaptive=True,
            )
            detector = SceneDetector(config)
            source, new_clips = detector.detect_scenes(vid_path)
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

        project.add_source(source)
        if new_clips:
            project.add_clips(new_clips)

        try:
            save_with_mtime_check(project, proj_path, mtime)
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
                "source_id": source.id,
                "source_path": str(vid_path),
                "clips_created": len(new_clips),
                "total_sources": len(project.sources),
                "total_clips": len(project.clips),
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
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        clips_by_source = project.clips_by_source

        source_list = []
        for source in project.sources:
            source_clips = clips_by_source.get(source.id, [])
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

        source_to_remove = project.sources_by_id.get(source_id)
        if source_to_remove is None:
            return json.dumps({"success": False, "error": f"Source not found: {source_id}"})

        original_clip_count = len(project.clips)

        # Remove the source (also drops associated clips and frames).
        project.remove_source(source_id)

        removed_clips = original_clip_count - len(project.clips)

        # Remove from sequence if present
        if project.sequence is not None:
            for track in project.sequence.tracks:
                track.clips = [c for c in track.clips if c.source_id != source_id]

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
                "removed_source": source_to_remove.filename,
                "removed_clips": removed_clips,
                "remaining_sources": len(project.sources),
                "remaining_clips": len(project.clips),
            }
        )
    except Exception as e:
        logger.exception("Failed to remove source")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def list_audio_sources(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    ctx: Context = None,
) -> str:
    """List all imported audio sources in a Scene Ripper project.

    Audio sources are imported audio files (music, podcasts, voiceovers).
    They are not cut into clips and never appear in the sequencer output —
    they exist to feed audio tools like Staccato and transcription.

    Args:
        project_path: Absolute path to the project file.

    Returns:
        JSON with success flag, audio source count, and per-source metadata
        (id, filename, duration, sample_rate, channels, transcribed flag).
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

        payload = []
        for a in project.audio_sources:
            payload.append({
                "id": a.id,
                "filename": a.filename,
                "duration": a.duration_seconds,
                "duration_str": a.duration_str,
                "sample_rate": a.sample_rate,
                "channels": a.channels,
                "transcribed": bool(a.transcript),
                "transcript_segment_count": len(a.transcript) if a.transcript else 0,
            })

        return json.dumps({
            "success": True,
            "audio_sources": payload,
            "count": len(payload),
        })
    except Exception as e:
        logger.exception("Failed to list audio sources")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def get_audio_source(
    project_path: Annotated[str, "Absolute path to .sceneripper project file"],
    audio_source_id: Annotated[str, "ID of the audio source (use list_audio_sources to find)"],
    ctx: Context = None,
) -> str:
    """Get full details for a single audio source, including its transcript
    if one has been generated.

    Args:
        project_path: Absolute path to the project file.
        audio_source_id: ID of the audio source.

    Returns:
        JSON with the full audio source record, including any transcript
        segments. Returns success=False with a guidance message if the ID
        is unknown.
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

        match = project.get_audio_source(audio_source_id)
        if match is None:
            return json.dumps({
                "success": False,
                "error": (
                    f"Audio source '{audio_source_id}' not found. "
                    "Use list_audio_sources to see available IDs."
                ),
            })

        transcript_payload = None
        if match.transcript:
            transcript_payload = [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "confidence": seg.confidence,
                }
                for seg in match.transcript
            ]

        return json.dumps({
            "success": True,
            "audio_source": {
                "id": match.id,
                "filename": match.filename,
                "file_path": str(match.file_path),
                "duration": match.duration_seconds,
                "duration_str": match.duration_str,
                "sample_rate": match.sample_rate,
                "channels": match.channels,
                "transcript": transcript_payload,
            },
        })
    except Exception as e:
        logger.exception("Failed to get audio source")
        return json.dumps({"success": False, "error": str(e)})
