"""Tool definitions for the agent chat system.

Tools are registered via decorators and can be executed by the agent
during chat interactions. Tools are split into two categories:

1. GUI State Tools - Modify the live Project instance, triggering observer
   callbacks that update the UI in real-time.

2. CLI Tools - Execute batch operations via subprocess, suitable for
   long-running or heavy operations.
"""

import inspect
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, get_type_hints

from core.youtube_api import (
    YouTubeSearchClient,
    YouTubeAPIError,
    QuotaExceededError,
    InvalidAPIKeyError,
)
from core.downloader import VideoDownloader
from core.settings import load_settings

logger = logging.getLogger(__name__)


def _validate_path(path_str: str, must_exist: bool = False, allow_relative: bool = False) -> tuple[bool, str, Optional[Path]]:
    """Validate a file path for security.

    Args:
        path_str: Path string to validate
        must_exist: Whether the path must exist
        allow_relative: Whether to allow relative paths

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    if not path_str:
        return False, "Path cannot be empty", None

    try:
        path = Path(path_str)

        # Resolve to absolute path
        resolved = path.resolve()

        # Check for absolute path requirement
        if not allow_relative and not path.is_absolute():
            return False, f"Only absolute paths are allowed: {path_str}", None

        # Check for path traversal attempts (.. in original path)
        if ".." in str(path):
            return False, f"Path traversal not allowed: {path_str}", None

        # Check existence if required
        if must_exist and not resolved.exists():
            return False, f"Path does not exist: {path_str}", None

        # Ensure path is within user's home directory or common safe locations
        home = Path.home()
        safe_roots = [
            home,
            Path("/tmp").resolve(),  # Resolve symlinks (macOS: /tmp -> /private/tmp)
            Path(tempfile.gettempdir()).resolve(),
        ]

        # On macOS, also allow /var/folders (temp), /Volumes (external drives), /private/tmp
        import sys
        if sys.platform == "darwin":
            safe_roots.extend([
                Path("/var/folders"),
                Path("/Volumes"),
                Path("/private/tmp"),
            ])

        is_safe = any(
            _is_path_under(resolved, safe_root)
            for safe_root in safe_roots
        )

        if not is_safe:
            return False, f"Path must be within home directory or temp: {path_str}", None

        return True, "", resolved

    except Exception as e:
        return False, f"Invalid path: {e}", None


def _is_path_under(path: Path, root: Path) -> bool:
    """Check if path is under root directory."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# Timeout values for CLI tools (in seconds)
TOOL_TIMEOUTS = {
    "detect_scenes": 600,      # 10 minutes for large videos
    "download_video": 1800,    # 30 minutes for long videos
    "search_youtube": 30,      # 30 seconds
    "analyze_colors": 300,     # 5 minutes
    "analyze_shots": 300,      # 5 minutes
    "transcribe": 1200,        # 20 minutes
    "export_clips": 600,       # 10 minutes
}


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent."""
    name: str
    description: str
    func: Callable
    parameters: dict
    requires_project: bool = True
    modifies_gui_state: bool = False


def _python_type_to_json_schema(python_type: type) -> dict:
    """Convert Python type hint to JSON schema type."""
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    # Handle Optional types
    origin = getattr(python_type, "__origin__", None)
    if origin is type(None):
        return {"type": "null"}

    # Handle Union types (like Optional[str] = Union[str, None])
    if origin is type(None) or str(origin) == "typing.Union":
        args = getattr(python_type, "__args__", ())
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        return {"type": "string"}  # Fallback

    # Handle list[T] -> array of T
    if origin is list:
        args = getattr(python_type, "__args__", ())
        if args:
            return {
                "type": "array",
                "items": _python_type_to_json_schema(args[0])
            }
        return {"type": "array"}

    # Handle dict[K, V] -> object
    if origin is dict:
        return {"type": "object"}

    # Basic types
    return type_map.get(python_type, {"type": "string"})


class ToolRegistry:
    """Registry for agent tools with decorator-based registration."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        description: str,
        requires_project: bool = True,
        modifies_gui_state: bool = False
    ):
        """Decorator to register a tool function.

        Args:
            description: Description shown to the LLM for tool selection
            requires_project: Whether this tool needs an active project
            modifies_gui_state: Whether this tool modifies the GUI state
                (GUI state tools should use core functions, not CLI)

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            schema = self._generate_schema(func)
            tool = ToolDefinition(
                name=func.__name__,
                description=description,
                func=func,
                parameters=schema,
                requires_project=requires_project,
                modifies_gui_state=modifies_gui_state
            )
            self._tools[tool.name] = tool
            return func
        return decorator

    def _generate_schema(self, func: Callable) -> dict:
        """Generate JSON schema for function parameters."""
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip 'project' parameter - it's injected by executor
            if param_name == "project":
                continue

            # Get type hint or default to string
            param_type = hints.get(param_name, str)
            schema = _python_type_to_json_schema(param_type)

            # Get description from docstring if available
            properties[param_name] = schema

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def to_openai_format(self) -> list[dict]:
        """Convert all tools to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self._tools.values()
        ]

    def all_tools(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())


# Global tool registry instance
tools = ToolRegistry()


# =============================================================================
# GUI State Tools - These modify the live Project instance
# =============================================================================

@tools.register(
    description="Get current project state including sources, clips, and sequence.",
    requires_project=True,
    modifies_gui_state=False
)
def get_project_state(project) -> dict:
    """Get current project information."""
    return {
        "name": project.metadata.name,
        "path": str(project.path) if project.path else None,
        "sources": [
            {
                "id": s.id,
                "name": s.file_path.name,
                "duration": s.duration_seconds,
                "fps": s.fps,
                "analyzed": s.analyzed,
            }
            for s in project.sources
        ],
        "clip_count": len(project.clips),
        "sequence_length": len(project.sequence.tracks[0].clips) if project.sequence else 0,
        "is_dirty": project.is_dirty
    }


@tools.register(
    description="Add clips to the timeline sequence by their IDs. Clips will appear in the Sequence tab.",
    requires_project=True,
    modifies_gui_state=True
)
def add_to_sequence(project, clip_ids: list[str]) -> dict:
    """Add clips to the timeline sequence."""
    # Validate clip IDs
    valid_ids = [cid for cid in clip_ids if cid in project.clips_by_id]
    invalid_ids = [cid for cid in clip_ids if cid not in project.clips_by_id]

    if invalid_ids:
        logger.warning(f"Invalid clip IDs: {invalid_ids}")

    if not valid_ids:
        return {
            "success": False,
            "error": "No valid clip IDs provided",
            "invalid_ids": invalid_ids
        }

    project.add_to_sequence(valid_ids)

    return {
        "success": True,
        "added": valid_ids,
        "invalid_ids": invalid_ids,
        "sequence_length": len(project.sequence.tracks[0].clips) if project.sequence else 0
    }


@tools.register(
    description="Filter clips by criteria. Returns matching clips with their metadata. Available filters: shot_type (close_up, medium_shot, wide_shot, etc.), has_speech (true/false), min_duration (seconds), max_duration (seconds).",
    requires_project=True,
    modifies_gui_state=False
)
def filter_clips(
    project,
    shot_type: Optional[str] = None,
    has_speech: Optional[bool] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None
) -> list[dict]:
    """Filter clips by various criteria."""
    results = []

    for clip in project.clips:
        # Get source for FPS
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0

        # Calculate duration in seconds
        duration = (clip.end_frame - clip.start_frame) / fps

        # Apply shot_type filter
        if shot_type and getattr(clip, 'shot_type', None) != shot_type:
            continue

        # Apply has_speech filter
        if has_speech is not None:
            clip_has_speech = bool(getattr(clip, 'transcript', None))
            if clip_has_speech != has_speech:
                continue

        # Apply duration filters
        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue

        results.append({
            "id": clip.id,
            "source_id": clip.source_id,
            "source_name": source.file_path.name if source else "Unknown",
            "duration_seconds": round(duration, 2),
            "shot_type": getattr(clip, 'shot_type', None),
            "has_speech": bool(getattr(clip, 'transcript', None)),
            "dominant_colors": getattr(clip, 'dominant_colors', None),
        })

    return results


@tools.register(
    description="List all clips in the project with their metadata.",
    requires_project=True,
    modifies_gui_state=False
)
def list_clips(project) -> list[dict]:
    """List all clips with metadata."""
    results = []

    for clip in project.clips:
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0
        duration = (clip.end_frame - clip.start_frame) / fps

        results.append({
            "id": clip.id,
            "source_id": clip.source_id,
            "source_name": source.file_path.name if source else "Unknown",
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "duration_seconds": round(duration, 2),
            "shot_type": getattr(clip, 'shot_type', None),
            "has_speech": bool(getattr(clip, 'transcript', None)),
        })

    return results


@tools.register(
    description="Remove clips from the timeline sequence by their sequence clip IDs.",
    requires_project=True,
    modifies_gui_state=True
)
def remove_from_sequence(project, clip_ids: list[str]) -> dict:
    """Remove clips from the timeline sequence."""
    if not clip_ids:
        return {
            "success": False,
            "error": "No clip IDs provided"
        }

    removed = project.remove_from_sequence(clip_ids)

    return {
        "success": len(removed) > 0,
        "removed": removed,
        "not_found": [cid for cid in clip_ids if cid not in removed],
        "sequence_length": len(project.sequence.tracks[0].clips) if project.sequence else 0
    }


@tools.register(
    description="Clear all clips from the timeline sequence, resetting it to empty.",
    requires_project=True,
    modifies_gui_state=True
)
def clear_sequence(project) -> dict:
    """Clear the timeline sequence."""
    count = project.clear_sequence()

    return {
        "success": True,
        "clips_removed": count,
        "message": f"Cleared {count} clips from the sequence"
    }


@tools.register(
    description="Reorder clips in the timeline sequence. Provide sequence clip IDs in the desired order.",
    requires_project=True,
    modifies_gui_state=True
)
def reorder_sequence(project, clip_ids: list[str]) -> dict:
    """Reorder clips in the sequence."""
    if not clip_ids:
        return {
            "success": False,
            "error": "No clip IDs provided"
        }

    if project.sequence is None:
        return {
            "success": False,
            "error": "No sequence exists"
        }

    success = project.reorder_sequence(clip_ids)

    if success:
        return {
            "success": True,
            "message": f"Reordered {len(clip_ids)} clips",
            "new_order": clip_ids
        }
    else:
        return {
            "success": False,
            "error": "One or more clip IDs not found in sequence"
        }


@tools.register(
    description="Get the current state of the timeline sequence including all clips, their positions, and durations.",
    requires_project=True,
    modifies_gui_state=False
)
def get_sequence_state(project) -> dict:
    """Return detailed sequence state."""
    if project.sequence is None:
        return {
            "has_sequence": False,
            "clips": [],
            "total_duration_seconds": 0,
            "clip_count": 0
        }

    sequence = project.sequence
    fps = sequence.fps
    clips_data = []

    for track in sequence.tracks:
        for seq_clip in track.clips:
            # Get source clip info
            source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
            source = project.sources_by_id.get(seq_clip.source_id)

            clips_data.append({
                "id": seq_clip.id,
                "source_clip_id": seq_clip.source_clip_id,
                "source_id": seq_clip.source_id,
                "source_name": source.file_path.name if source else "Unknown",
                "track_index": seq_clip.track_index,
                "start_frame": seq_clip.start_frame,
                "start_time_seconds": round(seq_clip.start_time(fps), 2),
                "duration_frames": seq_clip.duration_frames,
                "duration_seconds": round(seq_clip.duration_seconds(fps), 2),
                "in_point": seq_clip.in_point,
                "out_point": seq_clip.out_point,
            })

    return {
        "has_sequence": True,
        "name": sequence.name,
        "fps": fps,
        "clips": clips_data,
        "total_duration_frames": sequence.duration_frames,
        "total_duration_seconds": round(sequence.duration_seconds, 2),
        "clip_count": len(clips_data)
    }


@tools.register(
    description="Select clips in the browser by their IDs. This updates the GUI selection state.",
    requires_project=True,
    modifies_gui_state=True
)
def select_clips(project, clip_ids: list[str], gui_state=None) -> dict:
    """Update GUI selection to specified clips."""
    if gui_state is None:
        return {
            "success": False,
            "error": "GUI state not available"
        }

    # Validate clip IDs
    valid_ids = [cid for cid in clip_ids if cid in project.clips_by_id]
    invalid_ids = [cid for cid in clip_ids if cid not in project.clips_by_id]

    if invalid_ids:
        logger.warning(f"Invalid clip IDs for selection: {invalid_ids}")

    # Update GUI state
    gui_state.selected_clip_ids = valid_ids

    return {
        "success": True,
        "selected": valid_ids,
        "invalid_ids": invalid_ids,
        "selection_count": len(valid_ids)
    }


@tools.register(
    description="Switch to a specific tab in the application. Valid tabs: collect, cut, analyze, sequence, generate, render",
    requires_project=False,
    modifies_gui_state=True
)
def navigate_to_tab(tab_name: str, gui_state=None) -> dict:
    """Switch active tab."""
    valid_tabs = ["collect", "cut", "analyze", "sequence", "generate", "render"]

    if tab_name not in valid_tabs:
        return {
            "success": False,
            "error": f"Invalid tab name '{tab_name}'. Valid tabs: {', '.join(valid_tabs)}"
        }

    if gui_state is None:
        return {
            "success": False,
            "error": "GUI state not available"
        }

    gui_state.active_tab = tab_name

    return {
        "success": True,
        "active_tab": tab_name,
        "message": f"Switched to {tab_name} tab"
    }


# =============================================================================
# CLI Tools - Execute via subprocess for batch operations
# =============================================================================

@tools.register(
    description="Detect scenes in a video file. Creates clips from detected scene boundaries. Returns the output project file path and clip count.",
    requires_project=False,
    modifies_gui_state=False
)
def detect_scenes(
    video_path: str,
    sensitivity: float = 3.0,
    output_path: Optional[str] = None
) -> dict:
    """Run scene detection via CLI."""
    # Validate video path
    valid, error, video = _validate_path(video_path, must_exist=True)
    if not valid:
        return {"error": error}

    if not video.is_file():
        return {"error": f"Path is not a file: {video_path}"}

    # Validate and set output path
    if output_path is None:
        output_path = str(video.with_suffix(".json"))
    else:
        valid, error, validated_output = _validate_path(output_path)
        if not valid:
            return {"error": f"Invalid output path: {error}"}
        output_path = str(validated_output)

    cmd = [
        "scene_ripper", "detect", str(video),
        "--sensitivity", str(sensitivity),
        "--output", output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUTS.get("detect_scenes", 600)
        )
    except subprocess.TimeoutExpired:
        return {"error": "Scene detection timed out. The video may be too large."}
    except FileNotFoundError:
        return {"error": "scene_ripper CLI not found. Is it installed?"}

    if result.returncode != 0:
        return {"error": result.stderr or "Detection failed"}

    # Try to count clips from output
    try:
        with open(output_path) as f:
            data = json.load(f)
            clip_count = len(data.get("clips", []))
    except Exception:
        clip_count = "unknown"

    return {
        "success": True,
        "output_path": output_path,
        "clips_detected": clip_count,
        "message": f"Detected scenes in {video.name}"
    }


@tools.register(
    description="Search YouTube for videos matching a query. Returns video titles, IDs, durations, and URLs.",
    requires_project=False,
    modifies_gui_state=False
)
def search_youtube(query: str, max_results: int = 10) -> dict:
    """Search YouTube using the Python API."""
    # Get API key from settings
    settings = load_settings()
    api_key = settings.youtube_api_key

    if not api_key:
        return {
            "error": "YouTube API key not configured. Add it in Settings > YouTube API Key."
        }

    try:
        client = YouTubeSearchClient(api_key)
        result = client.search(query, max_results=min(max_results, 50))

        # Convert to serializable format
        videos = []
        for video in result.videos:
            videos.append({
                "video_id": video.video_id,
                "title": video.title,
                "channel": video.channel_title,
                "duration": video.duration_str,
                "url": video.youtube_url,
                "thumbnail": video.thumbnail_url,
                "view_count": video.view_count,
            })

        return {
            "success": True,
            "query": query,
            "results": videos,
            "total_results": result.total_results
        }

    except QuotaExceededError:
        return {"error": "YouTube API quota exceeded. Try again tomorrow."}
    except InvalidAPIKeyError:
        return {"error": "Invalid YouTube API key. Check your settings."}
    except YouTubeAPIError as e:
        return {"error": f"YouTube API error: {e}"}
    except Exception as e:
        logger.exception("YouTube search failed")
        return {"error": f"Search failed: {e}"}


@tools.register(
    description="Download a video from YouTube or Vimeo URL. Returns the downloaded file path. Uses the default download directory from settings unless output_dir is specified.",
    requires_project=False,
    modifies_gui_state=False
)
def download_video(url: str, output_dir: Optional[str] = None) -> dict:
    """Download video using the Python API."""
    # Determine download directory
    if output_dir:
        valid, error, validated_dir = _validate_path(output_dir)
        if not valid:
            return {"error": f"Invalid output directory: {error}"}
        download_path = validated_dir
    else:
        # Use settings download directory
        settings = load_settings()
        download_path = settings.download_dir

    try:
        downloader = VideoDownloader(download_dir=download_path)

        # Validate URL first
        valid, error = downloader.is_valid_url(url)
        if not valid:
            return {"error": error}

        # Download the video
        result = downloader.download(url)

        if not result.success:
            return {"error": result.error or "Download failed"}

        return {
            "success": True,
            "file_path": str(result.file_path) if result.file_path else None,
            "title": result.title,
            "duration": result.duration,
            "message": f"Downloaded: {result.title}"
        }

    except RuntimeError as e:
        # yt-dlp not found or other runtime errors
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Video download failed")
        return {"error": f"Download failed: {e}"}


@tools.register(
    description="Run color palette extraction on project clips. Adds dominant_colors metadata to clips.",
    requires_project=False,
    modifies_gui_state=False
)
def analyze_colors(project_path: str) -> dict:
    """Run color analysis via CLI."""
    # Validate project path
    valid, error, validated_path = _validate_path(project_path, must_exist=True)
    if not valid:
        return {"error": error}

    cmd = ["scene_ripper", "analyze", "colors", str(validated_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUTS.get("analyze_colors", 300)
        )
    except subprocess.TimeoutExpired:
        return {"error": "Color analysis timed out."}
    except FileNotFoundError:
        return {"error": "scene_ripper CLI not found."}

    if result.returncode != 0:
        return {"error": result.stderr or "Analysis failed"}

    return {
        "success": True,
        "message": "Color analysis complete",
        "output": result.stdout.strip()
    }


@tools.register(
    description="Run shot type classification on project clips. Identifies close-ups, wide shots, medium shots, etc.",
    requires_project=False,
    modifies_gui_state=False
)
def analyze_shots(project_path: str) -> dict:
    """Run shot classification via CLI."""
    # Validate project path
    valid, error, validated_path = _validate_path(project_path, must_exist=True)
    if not valid:
        return {"error": error}

    cmd = ["scene_ripper", "analyze", "shots", str(validated_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUTS.get("analyze_shots", 300)
        )
    except subprocess.TimeoutExpired:
        return {"error": "Shot analysis timed out."}
    except FileNotFoundError:
        return {"error": "scene_ripper CLI not found."}

    if result.returncode != 0:
        return {"error": result.stderr or "Analysis failed"}

    return {
        "success": True,
        "message": "Shot classification complete",
        "output": result.stdout.strip()
    }


@tools.register(
    description="Transcribe speech in project clips using Whisper. Adds transcript metadata to clips.",
    requires_project=False,
    modifies_gui_state=False
)
def transcribe(project_path: str, model: str = "small.en") -> dict:
    """Run transcription via CLI."""
    # Validate project path
    valid, error, validated_path = _validate_path(project_path, must_exist=True)
    if not valid:
        return {"error": error}

    cmd = ["scene_ripper", "transcribe", str(validated_path), "--model", model]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUTS.get("transcribe", 1200)
        )
    except subprocess.TimeoutExpired:
        return {"error": "Transcription timed out."}
    except FileNotFoundError:
        return {"error": "scene_ripper CLI not found."}

    if result.returncode != 0:
        return {"error": result.stderr or "Transcription failed"}

    return {
        "success": True,
        "message": "Transcription complete",
        "output": result.stdout.strip()
    }


@tools.register(
    description="Export clips from a project to individual video files.",
    requires_project=False,
    modifies_gui_state=False
)
def export_clips(
    project_path: str,
    output_dir: str,
    clip_ids: Optional[list[str]] = None
) -> dict:
    """Export clips via CLI."""
    # Validate project path
    valid, error, validated_project = _validate_path(project_path, must_exist=True)
    if not valid:
        return {"error": f"Invalid project path: {error}"}

    # Validate output directory
    valid, error, validated_output = _validate_path(output_dir)
    if not valid:
        return {"error": f"Invalid output directory: {error}"}

    cmd = ["scene_ripper", "export", "clips", str(validated_project), "--output-dir", str(validated_output)]

    if clip_ids:
        cmd.extend(["--clips", ",".join(clip_ids)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUTS.get("export_clips", 600)
        )
    except subprocess.TimeoutExpired:
        return {"error": "Export timed out."}
    except FileNotFoundError:
        return {"error": "scene_ripper CLI not found."}

    if result.returncode != 0:
        return {"error": result.stderr or "Export failed"}

    return {
        "success": True,
        "output_dir": output_dir,
        "message": result.stdout.strip()
    }
