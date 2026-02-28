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
from datetime import datetime
from typing import Any, Callable, Optional, get_type_hints

from core.youtube_api import (
    YouTubeSearchClient,
    YouTubeAPIError,
    QuotaExceededError,
    InvalidAPIKeyError,
)
from core.downloader import VideoDownloader
from core.settings import load_settings
from core.analysis.shots import SHOT_TYPES
from core.constants import (
    VALID_ASPECT_RATIOS,
    VALID_COLOR_PALETTES,
    VALID_SHOT_TYPES,
    VALID_SORT_ORDERS,
)
from core.gui_state import NameProjectThenPlanAction
from core.plan_controller import PlanController

logger = logging.getLogger(__name__)

def _get_plan_controller(main_window) -> PlanController:
    """Get or create a PlanController for the given main_window."""
    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state is None:
        raise ValueError("No GUI state available")
    return PlanController(gui_state)


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

    # Check for path traversal attempts in raw string BEFORE any parsing
    # This prevents bypass via Path() normalization
    if ".." in path_str:
        return False, f"Path traversal not allowed: {path_str}", None

    try:
        path = Path(path_str)

        # Resolve to absolute path
        resolved = path.resolve()

        # Check for absolute path requirement
        if not allow_relative and not path.is_absolute():
            return False, f"Only absolute paths are allowed: {path_str}", None

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

        # Platform-specific safe roots
        import sys
        if sys.platform == "darwin":
            safe_roots.extend([
                Path("/var/folders"),
                Path("/Volumes"),
                Path("/private/tmp"),
            ])
        elif sys.platform == "win32":
            # Allow all existing drive roots (users store videos on D:\, E:\, etc.)
            import string
            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    safe_roots.append(drive)

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


# Timeout values for tools (in seconds)
# Used by both CLI subprocess calls and GUI tool async workers
TOOL_TIMEOUTS = {
    "detect_scenes": 600,      # 10 minutes for large videos
    "detect_scenes_live": 600, # 10 minutes for large videos
    "download_video": 1800,    # 30 minutes for long videos
    "download_videos": 7200,   # 2 hours for bulk downloads (10 videos × 10 min timeout + buffer)
    "search_youtube": 30,      # 30 seconds
    "analyze_colors_live": 300,     # 5 minutes
    "analyze_shots_live": 300,      # 5 minutes
    "classify_content_live": 300,   # 5 minutes for classification
    "detect_objects_live": 300,     # 5 minutes for object detection
    "count_people_live": 300,       # 5 minutes for person detection
    "describe_content_live": 600,   # 10 minutes for descriptions
    "transcribe_live": 1200,        # 20 minutes
    "transcribe_clips": 1200,       # 20 minutes
    "export_sequence": 600,    # 10 minutes
}

# Default timeout for tools not in TOOL_TIMEOUTS
DEFAULT_TOOL_TIMEOUT = 60  # 1 minute


def get_tool_timeout(tool_name: str) -> float:
    """Get the timeout for a tool in seconds.

    Args:
        tool_name: Name of the tool

    Returns:
        Timeout in seconds
    """
    return TOOL_TIMEOUTS.get(tool_name, DEFAULT_TOOL_TIMEOUT)


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent."""
    name: str
    description: str
    func: Callable
    parameters: dict
    requires_project: bool = True
    modifies_gui_state: bool = False
    modifies_project_state: bool = False


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
        modifies_gui_state: bool = False,
        modifies_project_state: bool = False
    ):
        """Decorator to register a tool function.

        Args:
            description: Description shown to the LLM for tool selection
            requires_project: Whether this tool needs an active project
            modifies_gui_state: Whether this tool modifies the GUI state
                (GUI state tools should use core functions, not CLI)
            modifies_project_state: Whether this tool modifies project data
                (triggers auto-save after successful execution)

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
                modifies_gui_state=modifies_gui_state,
                modifies_project_state=modifies_project_state
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

        # Parameters that are injected by the system, not provided by the LLM
        injected_params = {"project", "gui_state", "main_window"}

        for param_name, param in sig.parameters.items():
            # Skip injected parameters - they're provided by the executor
            if param_name in injected_params:
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
    description=(
        "REQUIRED: Call this to create any multi-step workflow plan. "
        "Describing steps in text is NOT sufficient - you MUST call this tool to create a plan object. "
        "After calling this, wait for user confirmation, then call start_plan_execution. "
        "If the project is unnamed, this will prompt you to ask for a name first."
    ),
    requires_project=False,
    modifies_gui_state=True
)
def present_plan(main_window, steps: list[str], summary: str) -> dict:
    """Display an editable plan widget in the chat panel.

    Args:
        steps: List of human-readable step descriptions in execution order
        summary: Brief description of what the plan accomplishes

    Returns:
        Plan ID and instructions for the LLM to wait for confirmation
    """
    controller = _get_plan_controller(main_window)
    project = getattr(main_window, 'project', None)
    return controller.present(steps, summary, project=project)


@tools.register(
    description="Start executing a confirmed plan. Call this after the user confirms the plan. Returns the first step to execute.",
    requires_project=False,
    modifies_gui_state=True
)
def start_plan_execution(main_window) -> dict:
    """Begin executing the current plan.

    Call this after present_plan and user confirmation.

    Returns:
        Current step info and instructions
    """
    controller = _get_plan_controller(main_window)
    return controller.start()


@tools.register(
    description="Mark the current plan step as complete and get the next step. Call this after successfully completing each step in the plan.",
    requires_project=False,
    modifies_gui_state=True
)
def complete_plan_step(main_window, result_summary: Optional[str] = None) -> dict:
    """Mark current step complete and advance to next.

    Args:
        result_summary: Brief description of what was accomplished (optional)

    Returns:
        Next step info or completion status
    """
    controller = _get_plan_controller(main_window)
    return controller.advance(result_summary)


@tools.register(
    description="Get current plan execution status including which step is active. Use this to check progress or remind yourself what step you're on.",
    requires_project=False,
    modifies_gui_state=False
)
def get_plan_status(main_window) -> dict:
    """Get current plan status and step information.

    Returns:
        Plan status, current step, and remaining steps
    """
    controller = _get_plan_controller(main_window)
    return controller.get_status()


@tools.register(
    description="Mark the current plan step as failed. Use this if a step cannot be completed. You can optionally retry or skip to handle the failure.",
    requires_project=False,
    modifies_gui_state=True
)
def fail_plan_step(main_window, error: str, action: str = "stop") -> dict:
    """Mark current step as failed and handle the failure.

    Args:
        error: Description of what went wrong
        action: How to handle - 'stop' (halt plan), 'retry' (try step again), 'skip' (move to next step)

    Returns:
        Updated plan status
    """
    controller = _get_plan_controller(main_window)
    return controller.fail(error, action)


@tools.register(
    description="Get current project state including sources, clips, and sequence.",
    requires_project=True,
    modifies_gui_state=False
)
def get_project_state(project) -> dict:
    """Get current project information."""
    return {
        "success": True,
        "name": project.metadata.name,
        "path": str(project.path) if project.path else "Unsaved",
        "sources": [
            {
                "id": s.id,
                "name": s.file_path.name if s.file_path else "Unknown",
                "duration": s.duration_seconds,
                "fps": s.fps,
                "analyzed": s.analyzed,
                "clips": len(project.clips_by_source.get(s.id, [])),
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
    modifies_gui_state=True,
    modifies_project_state=True
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
    description="Generate natural language descriptions for video clips using Vision-Language Models. "
                "Useful for understanding clip content beyond simple tags. "
                "Supports tiers: 'cpu' (Moondream), 'gpu' (LLaVA), 'cloud' (GPT-4o/Claude).",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def describe_content_live(
    main_window,
    project,
    clip_ids: list[str],
    tier: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    """Generate descriptions for clips.

    Args:
        clip_ids: List of clip IDs to analyze
        tier: Model tier ('cpu', 'gpu', 'cloud')
        prompt: Custom prompt for the model

    Returns:
        Dict with status and wait token
    """
    # Validate clip IDs
    valid_ids = [cid for cid in clip_ids if cid in project.clips_by_id]
    if not valid_ids:
        return {
            "success": False,
            "error": "No valid clip IDs provided"
        }

    # Check if worker already running
    if main_window.description_worker and main_window.description_worker.isRunning():
        return {"success": False, "error": "Description generation already in progress"}

    # Set default prompt if not provided
    final_prompt = prompt or "Describe this video frame in 3 sentences or less. Focus on the main subjects, action, and setting."

    return {
        "_wait_for_worker": "description",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "tier": tier,
        "prompt": final_prompt,
        "message": f"Started generating descriptions for {len(valid_ids)} clips..."
    }


# Aspect ratio tolerance ranges (5% tolerance) for filtering
ASPECT_RATIO_RANGES = {
    "16:9": (1.69, 1.87),   # 1.778 ± 5%
    "4:3": (1.27, 1.40),     # 1.333 ± 5%
    "9:16": (0.53, 0.59),    # 0.5625 ± 5%
}


@tools.register(
    description="Filter clips by criteria. Returns matching clips with their metadata. Available filters: shot_type, has_speech, min_duration, max_duration, aspect_ratio, search_query, has_object (e.g., 'dog', 'car'), min_people, max_people, search_description.",
    requires_project=True,
    modifies_gui_state=False
)
def filter_clips(
    project,
    shot_type: Optional[str] = None,
    has_speech: Optional[bool] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    search_query: Optional[str] = None,
    has_object: Optional[str] = None,
    min_people: Optional[int] = None,
    max_people: Optional[int] = None,
    search_description: Optional[str] = None,
) -> list[dict]:
    """Filter clips by various criteria including content analysis.

    Args:
        project: Project instance
        shot_type: Filter by shot type (e.g., 'close-up', 'wide shot')
        has_speech: Filter by whether clip has transcribed speech
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        aspect_ratio: Filter by aspect ratio ('16:9', '4:3', '9:16')
        search_query: Search text in transcripts
        has_object: Filter by object label (e.g., 'dog', 'car', 'person')
        min_people: Minimum number of people detected
        max_people: Maximum number of people detected
        search_description: Search text in visual descriptions

    Returns:
        List of matching clips with metadata
    """
    results = []

    for clip in project.clips:
        # Get source for FPS and dimensions
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

        # Apply aspect ratio filter
        if aspect_ratio and aspect_ratio in ASPECT_RATIO_RANGES:
            if not source or source.width == 0 or source.height == 0:
                continue  # Skip clips without dimensions
            source_aspect = source.width / source.height
            min_ratio, max_ratio = ASPECT_RATIO_RANGES[aspect_ratio]
            if not (min_ratio <= source_aspect <= max_ratio):
                continue

        # Apply transcript search filter
        if search_query:
            transcript_text = clip.get_transcript_text()
            if not transcript_text:
                continue
            if search_query.lower() not in transcript_text.lower():
                continue

        # Apply description search filter
        if search_description:
            description = getattr(clip, 'description', None)
            if not description:
                continue
            if search_description.lower() not in description.lower():
                continue

        # Apply has_object filter (checks both object_labels and detected_objects)
        if has_object is not None:
            object_labels = getattr(clip, 'object_labels', None) or []
            detected_objects = getattr(clip, 'detected_objects', None) or []
            detected_labels = [d.get("label", "") for d in detected_objects]
            all_labels = set(label.lower() for label in object_labels + detected_labels)
            if has_object.lower() not in all_labels:
                continue

        # Apply min_people filter
        if min_people is not None:
            person_count = getattr(clip, 'person_count', None) or 0
            if person_count < min_people:
                continue

        # Apply max_people filter
        if max_people is not None:
            person_count = getattr(clip, 'person_count', None) or 0
            if person_count > max_people:
                continue

        # Calculate aspect ratio for output
        clip_aspect_ratio = None
        if source and source.height > 0:
            clip_aspect_ratio = round(source.width / source.height, 3)

        results.append({
            "id": clip.id,
            "source_id": clip.source_id,
            "source_name": source.file_path.name if source else "Unknown",
            "duration_seconds": round(duration, 2),
            "shot_type": getattr(clip, 'shot_type', None),
            "has_speech": bool(getattr(clip, 'transcript', None)),
            "dominant_colors": getattr(clip, 'dominant_colors', None),
            "object_labels": getattr(clip, 'object_labels', None),
            "person_count": getattr(clip, 'person_count', None),
            "description": getattr(clip, 'description', None),
            "width": source.width if source else None,
            "height": source.height if source else None,
            "aspect_ratio": clip_aspect_ratio,
        })

    return results


@tools.register(
    description="List clips in the project with their metadata. "
                "Supports filtering by source, shot type, and analysis status. "
                "Returns clip details including duration, shot type, colors, transcript, etc.",
    requires_project=True,
    modifies_gui_state=True  # Needs main_window to check detection status
)
def list_clips(
    main_window,
    project,
    source_id: Optional[str] = None,
    shot_type: Optional[str] = None,
    has_description: Optional[bool] = None,
    has_transcript: Optional[bool] = None,
    has_colors: Optional[bool] = None,
    sort_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """List clips with metadata, with optional filtering and sorting.

    Args:
        source_id: Filter by source ID
        shot_type: Filter by shot type (e.g. "close-up", "wide shot")
        has_description: Filter to clips with (True) or without (False) descriptions
        has_transcript: Filter to clips with (True) or without (False) transcripts
        has_colors: Filter to clips with (True) or without (False) color analysis
        sort_by: Sort by "duration", "start_frame", or "shot_type"
        limit: Maximum number of clips to return

    Returns:
        Dict with clips list, count, and detection status if relevant
    """
    results = []

    clips = project.clips

    # Apply filters
    if source_id is not None:
        clips = [c for c in clips if c.source_id == source_id]
    if shot_type is not None:
        clips = [c for c in clips if c.shot_type == shot_type]
    if has_description is True:
        clips = [c for c in clips if c.description is not None]
    elif has_description is False:
        clips = [c for c in clips if c.description is None]
    if has_transcript is True:
        clips = [c for c in clips if c.transcript is not None]
    elif has_transcript is False:
        clips = [c for c in clips if c.transcript is None]
    if has_colors is True:
        clips = [c for c in clips if c.dominant_colors is not None]
    elif has_colors is False:
        clips = [c for c in clips if c.dominant_colors is None]

    # Apply sorting
    if sort_by == "duration":
        clips = sorted(clips, key=lambda c: c.end_frame - c.start_frame, reverse=True)
    elif sort_by == "start_frame":
        clips = sorted(clips, key=lambda c: (c.source_id, c.start_frame))
    elif sort_by == "shot_type":
        clips = sorted(clips, key=lambda c: c.shot_type or "")

    # Apply limit
    total_matching = len(clips)
    if limit is not None and limit > 0:
        clips = clips[:limit]

    for clip in clips:
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
            "dominant_colors": getattr(clip, 'dominant_colors', None),
            "object_labels": getattr(clip, 'object_labels', None),
            "person_count": getattr(clip, 'person_count', None),
            "description": getattr(clip, 'description', None),
            "transcript": clip.get_transcript_text() if clip.transcript else None,
            "notes": getattr(clip, 'notes', None),
            "tags": getattr(clip, 'tags', []),
        })

    # Check if detection is in progress when no clips found
    if not results and main_window is not None:
        is_running = (
            main_window.detection_worker is not None and
            main_window.detection_worker.isRunning()
        )
        queue_remaining = len(getattr(main_window, '_analyze_queue', []))

        if is_running or queue_remaining > 0:
            return {
                "success": True,
                "clips": [],
                "count": 0,
                "message": f"No clips yet. Scene detection is still running "
                          f"({queue_remaining} sources queued). "
                          "Call check_detection_status to monitor progress, then retry.",
                "detection_in_progress": True
            }

        # Check if there are unanalyzed sources
        unanalyzed_count = sum(1 for s in project.sources if not s.analyzed)
        if unanalyzed_count > 0:
            return {
                "success": True,
                "clips": [],
                "count": 0,
                "message": f"No clips found. {unanalyzed_count} sources have not been analyzed. "
                          "Call detect_all_unanalyzed to detect scenes.",
                "detection_in_progress": False,
                "unanalyzed_sources": unanalyzed_count
            }

    result = {
        "success": True,
        "clips": results,
        "count": len(results),
    }
    if limit is not None and total_matching > len(results):
        result["total_matching"] = total_matching
    return result


@tools.register(
    description="Remove clips from the timeline sequence by their sequence clip IDs.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
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
    modifies_gui_state=True,
    modifies_project_state=True
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
    modifies_gui_state=True,
    modifies_project_state=True
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
    description="Update a sequence clip's trim points or position on the timeline. "
                "Use this to trim clips (in_point/out_point), reposition them (start_frame), "
                "change their track (track_index), or set hold duration for frame entries (hold_frames).",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def update_sequence_clip(
    project,
    clip_id: str,
    in_point: Optional[int] = None,
    out_point: Optional[int] = None,
    start_frame: Optional[int] = None,
    track_index: Optional[int] = None,
    hold_frames: Optional[int] = None,
) -> dict:
    """Update a sequence clip's trim points or position.

    Args:
        clip_id: ID of the sequence clip to update
        in_point: New trim start (frames into source clip)
        out_point: New trim end (frames into source clip)
        start_frame: New position on timeline (in frames)
        track_index: Move to a different track
        hold_frames: For frame entries, number of timeline frames to hold

    Returns:
        Dict with success status and updated clip info
    """
    if project.sequence is None:
        return {"success": False, "error": "No sequence exists"}

    # Find the clip across all tracks
    target_clip = None
    for track in project.sequence.tracks:
        for seq_clip in track.clips:
            if seq_clip.id == clip_id:
                target_clip = seq_clip
                break
        if target_clip:
            break

    if target_clip is None:
        return {
            "success": False,
            "error": f"Sequence clip '{clip_id}' not found. Use list_sequence_clips to see available clips."
        }

    # Validate and apply updates
    updated_fields = {}

    if in_point is not None:
        if in_point < 0:
            return {"success": False, "error": f"in_point must be >= 0, got {in_point}"}
        target_clip.in_point = in_point
        updated_fields["in_point"] = in_point

    if out_point is not None:
        if out_point <= (in_point if in_point is not None else target_clip.in_point):
            return {"success": False, "error": "out_point must be greater than in_point"}
        target_clip.out_point = out_point
        updated_fields["out_point"] = out_point

    if start_frame is not None:
        if start_frame < 0:
            return {"success": False, "error": f"start_frame must be >= 0, got {start_frame}"}
        target_clip.start_frame = start_frame
        updated_fields["start_frame"] = start_frame

    if track_index is not None:
        if track_index < 0 or track_index >= len(project.sequence.tracks):
            return {
                "success": False,
                "error": f"track_index {track_index} out of range (0-{len(project.sequence.tracks) - 1})"
            }
        target_clip.track_index = track_index
        updated_fields["track_index"] = track_index

    if hold_frames is not None:
        if hold_frames < 1:
            return {"success": False, "error": f"hold_frames must be >= 1, got {hold_frames}"}
        target_clip.hold_frames = hold_frames
        updated_fields["hold_frames"] = hold_frames

    if not updated_fields:
        return {"success": False, "error": "No fields provided to update"}

    # Notify observers
    project._dirty = True
    project._notify_observers("sequence_changed", [clip_id])

    return {
        "success": True,
        "message": f"Updated sequence clip {clip_id[:8]}...",
        "updated_fields": updated_fields,
        "duration_frames": target_clip.duration_frames,
    }


@tools.register(
    description=(
        "Sort clips in the sequence by a specific criterion. "
        "Supported criteria: 'color' (sort by dominant color hue), "
        "'duration' (shortest to longest), 'shot_type' (alphabetical), "
        "'random' (shuffle randomly). Use reverse=True for descending order."
    ),
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def sort_sequence(project, sort_by: str, reverse: bool = False) -> dict:
    """Sort clips in the sequence by various criteria."""
    import colorsys
    import random

    if project.sequence is None:
        return {"success": False, "error": "No sequence exists"}

    # Get all sequence clips with their source clip data
    all_clips = []
    for track in project.sequence.tracks:
        for seq_clip in track.clips:
            source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
            all_clips.append((seq_clip, source_clip))

    if not all_clips:
        return {"success": False, "error": "Sequence is empty"}

    # Define sort key functions
    def color_key(item):
        seq_clip, source_clip = item
        if source_clip and source_clip.dominant_colors:
            # Use first dominant color's hue for sorting
            r, g, b = source_clip.dominant_colors[0]
            h, _, _ = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            return h
        return 0.5  # Mid-gray for clips without color

    def duration_key(item):
        seq_clip, _ = item
        return seq_clip.duration_frames

    def shot_type_key(item):
        _, source_clip = item
        if source_clip and source_clip.shot_type:
            return source_clip.shot_type.lower()
        return "zzz"  # Sort unknowns to end

    # Sort based on criterion
    sort_by_lower = sort_by.lower()
    if sort_by_lower == "color":
        all_clips.sort(key=color_key, reverse=reverse)
    elif sort_by_lower == "duration":
        all_clips.sort(key=duration_key, reverse=reverse)
    elif sort_by_lower == "shot_type":
        all_clips.sort(key=shot_type_key, reverse=reverse)
    elif sort_by_lower == "random":
        random.shuffle(all_clips)
    else:
        return {
            "success": False,
            "error": f"Unknown sort criterion: {sort_by}. Use: color, duration, shot_type, random"
        }

    # Reorder the sequence using the sorted IDs
    sorted_ids = [seq_clip.id for seq_clip, _ in all_clips]
    success = project.reorder_sequence(sorted_ids)

    if success:
        return {
            "success": True,
            "message": f"Sorted {len(sorted_ids)} clips by {sort_by}" + (" (descending)" if reverse else ""),
            "new_order": sorted_ids
        }
    else:
        return {
            "success": False,
            "error": "Failed to reorder sequence after sorting"
        }


@tools.register(
    description=(
        "Filter the sequence tab to show only clips of a specific shot type. "
        "Valid shot types: 'wide shot', 'full shot', 'medium shot', 'close-up', 'extreme close-up'. "
        "Use shot_type=None to show all clips (remove filter)."
    ),
    requires_project=True,
    modifies_gui_state=True,
)
def set_sequence_shot_filter(
    project,
    gui_state,
    main_window,
    shot_type: Optional[str] = None,
) -> dict:
    """Set the shot type filter for the sequence tab."""
    from core.analysis.shots import SHOT_TYPES

    if shot_type and shot_type not in SHOT_TYPES:
        return {
            "success": False,
            "error": f"Invalid shot type '{shot_type}'. Valid types: {SHOT_TYPES}",
        }

    sequence_tab = main_window.sequence_tab
    filtered_count = sequence_tab.apply_shot_type_filter(shot_type)

    gui_state.sequence_shot_filter = shot_type

    return {
        "success": True,
        "result": {
            "shot_type": shot_type or "all",
            "clip_count": filtered_count,
            "message": (
                f"Showing {filtered_count} clips"
                + (f" of type '{shot_type}'" if shot_type else " (all types)")
            ),
        },
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

            clip_data = {
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
            }

            # Include source clip metadata for sorting/filtering
            if source_clip:
                if source_clip.dominant_colors:
                    clip_data["dominant_colors"] = [
                        f"#{r:02x}{g:02x}{b:02x}"
                        for r, g, b in source_clip.dominant_colors
                    ]
                if source_clip.shot_type:
                    clip_data["shot_type"] = source_clip.shot_type
                if source_clip.description:
                    clip_data["description"] = source_clip.description

            clips_data.append(clip_data)

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
    description="Switch to a specific tab in the application. Valid tabs: collect, cut, analyze, frames, sequence, render",
    requires_project=False,
    modifies_gui_state=True
)
def navigate_to_tab(tab_name: str, gui_state=None) -> dict:
    """Switch active tab."""
    valid_tabs = ["collect", "cut", "analyze", "frames", "sequence", "render"]

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


@tools.register(
    description="Apply filters to the clip browser in the active tab (Cut or Analyze). "
                "Filters clips by duration range, aspect ratio, shot type, color palette, and/or transcript search. "
                "Use clear_all=True to reset all filters.",
    requires_project=True,
    modifies_gui_state=True
)
def apply_filters(
    main_window,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    shot_type: Optional[str] = None,
    color_palette: Optional[str] = None,
    search_query: Optional[str] = None,
    clear_all: bool = False,
) -> dict:
    """Apply filters to the clip browser in the active tab.

    Args:
        min_duration: Minimum duration in seconds (None = no minimum)
        max_duration: Maximum duration in seconds (None = no maximum)
        aspect_ratio: Filter by aspect ratio ('16:9', '4:3', '9:16', or None)
        shot_type: Filter by shot type ('Wide Shot', 'Medium Shot', 'Close-up', 'Extreme CU', or None)
        color_palette: Filter by color palette ('Warm', 'Cool', 'Neutral', 'Vibrant', or None)
        search_query: Filter by transcript text (case-insensitive substring search)
        clear_all: If True, clears all filters instead of applying new ones

    Returns:
        Dict with success status, active filters, and clip counts
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Validate aspect_ratio if provided
    if aspect_ratio and aspect_ratio not in VALID_ASPECT_RATIOS:
        return {
            "success": False,
            "error": f"Invalid aspect_ratio '{aspect_ratio}'. Valid options: {', '.join(VALID_ASPECT_RATIOS)}"
        }

    # Validate shot_type if provided
    if shot_type and shot_type != "All" and shot_type not in VALID_SHOT_TYPES:
        return {
            "success": False,
            "error": f"Invalid shot_type '{shot_type}'. Valid options: {', '.join(VALID_SHOT_TYPES)}"
        }

    # Validate color_palette if provided
    if color_palette and color_palette != "All" and color_palette not in VALID_COLOR_PALETTES:
        return {
            "success": False,
            "error": f"Invalid color_palette '{color_palette}'. Valid options: {', '.join(VALID_COLOR_PALETTES)}"
        }

    # Get the active tab info
    active_tab = main_window._gui_state.active_tab if main_window._gui_state else "unknown"

    # Get the active tab's clip browser via public API
    clip_browser = main_window.get_active_clip_browser()
    if clip_browser is None:
        return {
            "success": False,
            "error": f"No clip browser available in '{active_tab}' tab. Switch to Cut or Analyze tab first."
        }

    if clear_all:
        clip_browser.clear_all_filters()
        return {
            "success": True,
            "message": "All filters cleared",
            "active_filters": clip_browser.get_active_filters(),
            "visible_clips": clip_browser.get_visible_clip_count(),
            "total_clips": len(clip_browser.thumbnails),
        }

    # Build filters dict for public API
    filters = {}
    if min_duration is not None:
        filters['min_duration'] = min_duration
    if max_duration is not None:
        filters['max_duration'] = max_duration
    if aspect_ratio:
        filters['aspect_ratio'] = aspect_ratio
    if shot_type:
        filters['shot_type'] = shot_type
    if color_palette:
        filters['color_palette'] = color_palette
    if search_query is not None:
        filters['search_query'] = search_query

    # Apply filters via public API
    clip_browser.apply_filters(filters)

    return {
        "success": True,
        "message": "Filters applied",
        "active_filters": clip_browser.get_active_filters(),
        "visible_clips": clip_browser.get_visible_clip_count(),
        "total_clips": len(clip_browser.thumbnails),
        "active_tab": active_tab,
    }


@tools.register(
    description="Set the sort order for clips in the active tab's clip browser (Cut or Analyze). "
                "Options: 'Timeline' (original order), 'Color' (grouped by dominant hue), 'Duration' (shortest to longest).",
    requires_project=False,
    modifies_gui_state=True
)
def set_clip_sort_order(
    main_window,
    sort_by: str,
) -> dict:
    """Set the sort order for clips in the clip browser.

    Args:
        sort_by: Sort method - 'Timeline', 'Color', or 'Duration'

    Returns:
        Dict with success status and current sort order
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Validate sort option
    if sort_by not in VALID_SORT_ORDERS:
        return {
            "success": False,
            "error": f"Invalid sort_by '{sort_by}'. Valid options: {', '.join(VALID_SORT_ORDERS)}"
        }

    # Get the active tab's clip browser via public API
    clip_browser = main_window.get_active_clip_browser()
    if clip_browser is None:
        active_tab = main_window._gui_state.active_tab if main_window._gui_state else "unknown"
        return {
            "success": False,
            "error": f"No clip browser available in '{active_tab}' tab. Switch to Cut or Analyze tab first."
        }

    # Set sort order via public API
    clip_browser.set_sort_order(sort_by)

    active_tab = main_window._gui_state.active_tab if main_window._gui_state else "unknown"
    return {
        "success": True,
        "message": f"Clips sorted by {sort_by}",
        "sort_order": sort_by,
        "active_tab": active_tab,
    }


@tools.register(
    description="Clear all clips from the Analyze tab. Use this to reset the analysis view before adding new clips.",
    requires_project=False,
    modifies_gui_state=True
)
def clear_analyze_clips(main_window) -> dict:
    """Clear all clips from the Analyze tab.

    Returns:
        Dict with success status
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    if not hasattr(main_window, 'analyze_tab'):
        return {"success": False, "error": "Analyze tab not available"}

    clip_count = main_window.get_analyze_clip_count()
    main_window.analyze_tab.clear_clips()
    main_window.analyze_tab.clips_cleared.emit()

    return {
        "success": True,
        "message": f"Cleared {clip_count} clips from Analyze tab",
        "cleared_count": clip_count,
    }


@tools.register(
    description="Send clips from Cut tab to Analyze tab for analysis. "
                "If no clip_ids provided, sends currently selected clips from Cut tab. "
                "Automatically switches to Analyze tab after adding clips.",
    requires_project=True,
    modifies_gui_state=True
)
def send_to_analyze(
    main_window,
    project,
    clip_ids: Optional[list[str]] = None,
    auto_analyze: bool = False,
) -> dict:
    """Send clips to the Analyze tab.

    Args:
        clip_ids: List of clip IDs to send (optional, defaults to selected clips in Cut tab)
        auto_analyze: If True, automatically start "Analyze All" on the sent clips

    Returns:
        Dict with success status and sent clip count
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    if not hasattr(main_window, 'analyze_tab') or not hasattr(main_window, 'cut_tab'):
        return {"success": False, "error": "Required tabs not available"}

    # Get clip IDs from selection if not provided
    if clip_ids is None:
        selected_clips = main_window.get_selected_clips()
        if not selected_clips:
            return {
                "success": False,
                "error": "No clips selected. Select clips first or provide clip_ids."
            }
        clip_ids = [clip.id for clip in selected_clips]

    # Validate clip IDs exist
    valid_ids = []
    for cid in clip_ids:
        if cid in project.clips_by_id:
            valid_ids.append(cid)

    if not valid_ids:
        return {
            "success": False,
            "error": "No valid clip IDs found. Check that clips exist in the project."
        }

    # Add clips to Analyze tab
    main_window.analyze_tab.add_clips(valid_ids)

    # Optionally start analysis via the pipeline
    if auto_analyze:
        from core.analysis_operations import DEFAULT_SELECTED
        return {
            "_wait_for_worker": "analyze_all",
            "clip_ids": valid_ids,
            "clip_count": len(valid_ids),
            "operations": list(DEFAULT_SELECTED),
            "sent_count": len(valid_ids),
        }

    return {
        "success": True,
        "message": f"Sent {len(valid_ids)} clips to Analyze tab. Use navigate_to_tab('analyze') to switch.",
        "sent_count": len(valid_ids),
        "clip_ids": valid_ids,
        "auto_analyze": auto_analyze,
    }


@tools.register(
    description="Clear all filters in the active tab's clip browser (Cut or Analyze). "
                "Resets duration, aspect ratio, shot type, color palette, and search filters.",
    requires_project=False,
    modifies_gui_state=True
)
def clear_filters(main_window, tab: Optional[str] = None) -> dict:
    """Clear all filters in the clip browser.

    Args:
        tab: Target tab - 'cut', 'analyze', or None for active tab

    Returns:
        Dict with success status
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Determine which tab to use
    gui_state = getattr(main_window, '_gui_state', None)
    target_tab = tab or (gui_state.active_tab if gui_state else "cut")

    clip_browser = main_window.get_clip_browser(target_tab)
    if clip_browser is None:
        return {
            "success": False,
            "error": f"No clip browser available in '{target_tab}' tab."
        }

    clip_browser.clear_all_filters()

    return {
        "success": True,
        "message": f"Cleared all filters in {target_tab} tab",
        "tab": target_tab,
        "visible_clips": clip_browser.get_visible_clip_count(),
        "total_clips": len(clip_browser.thumbnails),
    }


@tools.register(
    description="Select all visible clips in the active tab's clip browser (Cut or Analyze). "
                "Equivalent to Cmd+A keyboard shortcut.",
    requires_project=False,
    modifies_gui_state=True
)
def select_all_clips(main_window, tab: Optional[str] = None) -> dict:
    """Select all visible clips in the clip browser.

    Args:
        tab: Target tab - 'cut', 'analyze', or None for active tab

    Returns:
        Dict with success status and selected count
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    gui_state = getattr(main_window, '_gui_state', None)
    target_tab = tab or (gui_state.active_tab if gui_state else "cut")

    clip_browser = main_window.get_clip_browser(target_tab)
    if clip_browser is None:
        return {
            "success": False,
            "error": f"No clip browser available in '{target_tab}' tab."
        }

    clip_browser.select_all()
    selected_count = len(clip_browser.get_selected_clips())

    # Update GUI state
    if gui_state:
        gui_state.selected_clip_ids = [c.id for c in clip_browser.get_selected_clips()]

    return {
        "success": True,
        "message": f"Selected {selected_count} clips in {target_tab} tab",
        "selected_count": selected_count,
        "tab": target_tab,
    }


@tools.register(
    description="Deselect all clips in the active tab's clip browser (Cut or Analyze). "
                "Equivalent to Cmd+Shift+A keyboard shortcut.",
    requires_project=False,
    modifies_gui_state=True
)
def deselect_all_clips(main_window, tab: Optional[str] = None) -> dict:
    """Deselect all clips in the clip browser.

    Args:
        tab: Target tab - 'cut', 'analyze', or None for active tab

    Returns:
        Dict with success status
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    gui_state = getattr(main_window, '_gui_state', None)
    target_tab = tab or (gui_state.active_tab if gui_state else "cut")

    clip_browser = main_window.get_clip_browser(target_tab)
    if clip_browser is None:
        return {
            "success": False,
            "error": f"No clip browser available in '{target_tab}' tab."
        }

    clip_browser.clear_selection()

    # Update GUI state
    if gui_state:
        gui_state.selected_clip_ids = []

    return {
        "success": True,
        "message": f"Deselected all clips in {target_tab} tab",
        "tab": target_tab,
    }


# =============================================================================
# Playback Control Tools
# =============================================================================

@tools.register(
    description="Start or resume video playback in the sequence preview player.",
    requires_project=False,
    modifies_gui_state=True
)
def play_preview(main_window) -> dict:
    """Start video playback in the preview player.

    Returns:
        Dict with success status and playback state
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    player = main_window.get_video_player()
    if player is None:
        return {"success": False, "error": "Video player not available"}

    player.play()

    return {
        "success": True,
        "message": "Playback started",
        "state": "playing",
    }


@tools.register(
    description="Pause video playback in the sequence preview player.",
    requires_project=False,
    modifies_gui_state=True
)
def pause_preview(main_window) -> dict:
    """Pause video playback in the preview player.

    Returns:
        Dict with success status and playback state
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    player = main_window.get_video_player()
    if player is None:
        return {"success": False, "error": "Video player not available"}

    player.pause()

    return {
        "success": True,
        "message": "Playback paused",
        "state": "paused",
    }


@tools.register(
    description="Seek to a specific position in the video preview. Time is in seconds.",
    requires_project=False,
    modifies_gui_state=True
)
def seek_to_time(main_window, seconds: float) -> dict:
    """Seek to a position in the video preview.

    Args:
        seconds: Position to seek to in seconds

    Returns:
        Dict with success status and current position
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    player = main_window.get_video_player()
    if player is None:
        return {"success": False, "error": "Video player not available"}

    if seconds < 0:
        return {"success": False, "error": "Position cannot be negative"}

    # Check upper bound - get duration from player
    duration_ms = player.duration_ms
    if duration_ms > 0:
        duration_s = duration_ms / 1000.0
        if seconds > duration_s:
            return {
                "success": False,
                "error": f"Position {seconds:.2f}s exceeds video duration ({duration_s:.2f}s)"
            }

    player.seek_to(seconds)

    return {
        "success": True,
        "message": f"Seeked to {seconds:.2f}s",
        "position_seconds": seconds,
    }


@tools.register(
    description="Remove a video source from the library. This also removes all clips generated from this source. "
                "Use list_sources first to find the source ID.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def remove_source(
    main_window,
    project,
    source_id: str,
) -> dict:
    """Remove a video source from the library.

    Args:
        source_id: ID of the source to remove

    Returns:
        Dict with success status and removed source info
    """
    if project is None:
        return {"success": False, "error": "No project loaded"}

    # Find the source
    source = project.sources_by_id.get(source_id)
    if source is None:
        # Try matching by filename
        for s in project.sources:
            if s.file_path and (s.file_path.name == source_id or s.file_path.stem == source_id):
                source = s
                source_id = s.id
                break

    if source is None:
        return {
            "success": False,
            "error": f"Source '{source_id}' not found. Use list_sources to see available sources."
        }

    # Count clips that will be removed
    clips_to_remove = len([c for c in project.clips if c.source_id == source_id])
    source_name = source.filename if source.file_path else f"Source {source_id[:8]}"

    # Remove from project
    project.remove_source(source_id)

    # Update UI if available
    if main_window:
        main_window.remove_source_from_library(source_id)

    return {
        "success": True,
        "message": f"Removed source '{source_name}' and {clips_to_remove} associated clips",
        "removed_source": source_name,
        "removed_clips_count": clips_to_remove,
    }


@tools.register(
    description="Update metadata fields on a source video. "
                "Supports updating: color_profile ('color', 'grayscale', 'sepia'), "
                "fps (float), analyzed (bool).",
    requires_project=True,
    modifies_gui_state=False
)
def update_source(
    project,
    source_id: str,
    color_profile: Optional[str] = None,
    fps: Optional[float] = None,
    analyzed: Optional[bool] = None,
) -> dict:
    """Update metadata fields on a source video.

    Args:
        source_id: ID of the source to update
        color_profile: New color profile ('color', 'grayscale', 'sepia')
        fps: Corrected frames per second
        analyzed: Whether source has been analyzed

    Returns:
        Dict with success status and updated fields
    """
    if project is None:
        return {"success": False, "error": "No project loaded"}

    source = project.sources_by_id.get(source_id)
    if source is None:
        return {
            "success": False,
            "error": f"Source '{source_id}' not found. Use list_sources to see available sources."
        }

    # Validate color_profile
    valid_profiles = ["color", "grayscale", "sepia"]
    if color_profile is not None and color_profile not in valid_profiles:
        return {
            "success": False,
            "error": f"Invalid color_profile '{color_profile}'. Valid options: {', '.join(valid_profiles)}"
        }

    # Validate fps
    if fps is not None and (fps <= 0 or fps > 240):
        return {
            "success": False,
            "error": f"Invalid fps {fps}. Must be between 0 and 240."
        }

    # Build kwargs for update
    kwargs = {}
    if color_profile is not None:
        kwargs["color_profile"] = color_profile
    if fps is not None:
        kwargs["fps"] = fps
    if analyzed is not None:
        kwargs["analyzed"] = analyzed

    if not kwargs:
        return {"success": False, "error": "No fields provided to update"}

    updated = project.update_source(source_id, **kwargs)
    if updated is None:
        return {"success": False, "error": "Failed to update source"}

    return {
        "success": True,
        "message": f"Updated source {source_id[:8]}...",
        "updated_fields": kwargs,
    }


# =============================================================================
# Phase 3: Export & Project Management Tools
# =============================================================================

@tools.register(
    description="Export the current sequence as a video file (MP4). This renders all clips in the timeline into a single video. "
                "Parameters: quality='low'/'medium'/'high' (affects CRF and encoding speed), "
                "resolution='original'/'1080p'/'720p'/'480p' (target resolution), "
                "fps=float (target frame rate, defaults to sequence fps). "
                "Runs in background - may take significant time for long sequences.",
    requires_project=True,
    modifies_gui_state=True  # Needs main_window to access sequence_tab
)
def export_sequence(
    main_window,
    project,
    output_path: Optional[str] = None,
    quality: str = "medium",
    resolution: Optional[str] = None,
    fps: Optional[float] = None,
) -> dict:
    """Export the current sequence as a video file.

    Args:
        output_path: Path for output video (optional, defaults to export_dir)
        quality: Export quality - 'low', 'medium', or 'high'
        resolution: Target resolution - 'original', '1080p', '720p', or '480p' (optional)
        fps: Target frame rate (optional, defaults to sequence fps)

    Returns:
        Dict with success status and output path
    """
    from core.sequence_export import SequenceExporter, ExportConfig

    # Validate quality parameter
    valid_qualities = ["low", "medium", "high"]
    if quality not in valid_qualities:
        return {
            "success": False,
            "error": f"Invalid quality '{quality}'. Valid options: {', '.join(valid_qualities)}"
        }

    # Validate resolution if provided
    valid_resolutions = ["original", "1080p", "720p", "480p"]
    if resolution and resolution not in valid_resolutions:
        return {
            "success": False,
            "error": f"Invalid resolution '{resolution}'. Valid options: {', '.join(valid_resolutions)}"
        }

    # Get sequence from sequence tab
    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {"success": False, "error": "Sequence tab not available"}

    sequence = main_window.sequence_tab.get_sequence()
    all_clips = sequence.get_all_clips()

    if not all_clips:
        return {
            "success": False,
            "error": "No clips in timeline to export. Add clips to the sequence first."
        }

    # Determine output path
    if output_path:
        valid, error, validated_path = _validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        video_path = validated_path
    else:
        settings = load_settings()
        project_name = project.metadata.name or "sequence_export"
        video_path = settings.export_dir / f"{project_name}.mp4"

    # Ensure .mp4 extension
    if video_path.suffix.lower() != ".mp4":
        video_path = video_path.with_suffix(".mp4")

    # Ensure parent directory exists
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Quality presets
    quality_presets = {
        "high": {"crf": 18, "preset": "slow", "bitrate": "8M"},
        "medium": {"crf": 23, "preset": "medium", "bitrate": "4M"},
        "low": {"crf": 28, "preset": "fast", "bitrate": "2M"},
    }
    preset = quality_presets[quality]

    # Resolution presets
    resolution_presets = {
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "480p": (854, 480),
        "original": (None, None),
    }
    width, height = resolution_presets.get(resolution, (None, None)) if resolution else (None, None)

    # Build sources and clips dictionaries
    sources = project.sources_by_id.copy()
    clips = {}
    for clip in project.clips:
        source = sources.get(clip.source_id)
        if source:
            clips[clip.id] = (clip, source)

    config = ExportConfig(
        output_path=video_path,
        fps=fps if fps else sequence.fps,
        width=width,
        height=height,
        crf=preset["crf"],
        preset=preset["preset"],
        video_bitrate=preset["bitrate"],
    )

    # Check if export is already running
    if main_window.export_worker and main_window.export_worker.isRunning():
        return {"success": False, "error": "Export already in progress"}

    # Start async export via worker
    started = main_window.start_agent_export(sequence, sources, clips, config)
    if not started:
        return {"success": False, "error": "Failed to start export worker"}

    # Return marker that tells GUI handler to wait for worker completion
    return {
        "_wait_for_worker": "export",
        "output_path": str(video_path),
        "clip_count": len(all_clips),
        "quality": quality,
    }


@tools.register(
    description="Export the current sequence as an EDL (Edit Decision List) file for use in external video editors like DaVinci Resolve, Premiere Pro, or Final Cut.",
    requires_project=True,
    modifies_gui_state=False
)
def export_edl(project, output_path: Optional[str] = None) -> dict:
    """Export sequence to EDL format."""
    from core.edl_export import export_edl as do_export, EDLExportConfig

    if project.sequence is None or not project.sequence.get_all_clips():
        return {
            "success": False,
            "error": "No sequence to export. Add clips to the sequence first."
        }

    # Determine output path
    if output_path:
        valid, error, validated_path = _validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        edl_path = validated_path
    else:
        # Use settings export_dir with project name
        settings = load_settings()
        project_name = project.metadata.name or "untitled"
        edl_path = settings.export_dir / f"{project_name}.edl"

    # Ensure parent directory exists
    edl_path.parent.mkdir(parents=True, exist_ok=True)

    config = EDLExportConfig(
        output_path=edl_path,
        title=project.metadata.name or "Scene Ripper Export"
    )

    success = do_export(
        sequence=project.sequence,
        sources=project.sources_by_id,
        config=config
    )

    if success:
        return {
            "success": True,
            "output_path": str(edl_path),
            "clip_count": len(project.sequence.get_all_clips()),
            "message": f"Exported {len(project.sequence.get_all_clips())} clips to EDL"
        }
    else:
        return {
            "success": False,
            "error": "Failed to write EDL file"
        }


@tools.register(
    description="Export clip metadata as a JSON dataset file. Useful for training AI models or external analysis. "
                "Includes clip timings, colors, shot types, and optionally thumbnail paths.",
    requires_project=True,
    modifies_gui_state=False
)
def export_dataset(
    project,
    output_path: Optional[str] = None,
    include_thumbnails: bool = True,
    source_id: Optional[str] = None,
) -> dict:
    """Export clip metadata as JSON dataset.

    Args:
        output_path: Path for the JSON output file (optional, defaults to export_dir)
        include_thumbnails: Whether to include thumbnail paths in the export
        source_id: Export clips from a specific source only (optional, defaults to all)

    Returns:
        Dict with success status and export info
    """
    from core.dataset_export import export_dataset as do_export, DatasetExportConfig

    # Get clips to export
    if source_id:
        source = project.sources_by_id.get(source_id)
        if source is None:
            return {"success": False, "error": f"Source '{source_id}' not found"}
        clips = [c for c in project.clips if c.source_id == source_id]
        source_name = source.path.stem
    else:
        # Export all clips, using first source as primary
        clips = project.clips
        source = project.sources[0] if project.sources else None
        source_name = "all_clips"

    if not clips:
        return {"success": False, "error": "No clips to export"}

    if source is None:
        return {"success": False, "error": "No source video found"}

    # Determine output path
    if output_path:
        valid, error, validated_path = _validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        json_path = validated_path
    else:
        settings = load_settings()
        json_path = settings.export_dir / f"{source_name}_dataset.json"

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    config = DatasetExportConfig(
        output_path=json_path,
        include_thumbnails=include_thumbnails,
        pretty_print=True,
    )

    success = do_export(source=source, clips=clips, config=config)

    if success:
        return {
            "success": True,
            "output_path": str(json_path),
            "clip_count": len(clips),
            "message": f"Exported {len(clips)} clips to JSON dataset"
        }
    else:
        return {
            "success": False,
            "error": "Failed to write dataset file"
        }


@tools.register(
    description="Set or update the project name. Use this when the project is unnamed ('Untitled Project') to give it a meaningful name. The project will be automatically saved after naming.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def set_project_name(main_window, project, name: str) -> dict:
    """Set the project name and auto-save."""
    import re

    if not name or not name.strip():
        return {
            "success": False,
            "error": "Project name cannot be empty"
        }

    # Sanitize the name to remove invalid filesystem characters
    clean_name = re.sub(r'[<>:"/\\|?*]', '', name.strip())
    if not clean_name:
        return {
            "success": False,
            "error": "Project name cannot be empty or contain only invalid characters"
        }

    old_name = project.metadata.name
    project.metadata.name = clean_name
    project.mark_dirty()

    # Auto-save the project if it hasn't been saved yet
    # This enables future auto-saves (which require project.path to be set)
    save_path = None
    save_error = None
    if not project.path:
        settings = load_settings()
        export_dir = settings.export_dir

        # Validate export directory is writable before attempting save
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = export_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            save_error = (
                f"Cannot save project: No write access to export directory '{export_dir}'. "
                "Please update your export directory in Settings > Directories, or ensure the "
                "external drive is mounted and writable."
            )
        except OSError as e:
            save_error = (
                f"Cannot save project: Export directory '{export_dir}' is not accessible ({e}). "
                "Please check the path in Settings > Directories."
            )

        if not save_error:
            save_path = export_dir / f"{clean_name}.sceneripper"
            try:
                project.save(path=save_path)
                logger.info(f"Auto-saved new project to {save_path}")
            except Exception as e:
                logger.error(f"Failed to auto-save project: {e}")
                save_error = f"Failed to save project: {e}"
                save_path = None

    result = {
        "success": True,
        "old_name": old_name,
        "new_name": project.metadata.name,
        "message": f"Project renamed to '{project.metadata.name}'"
    }

    if save_path:
        result["saved_to"] = str(save_path)
        result["message"] += f" and saved to {save_path.name}"
    elif save_error:
        # Naming succeeded but saving failed - include warning
        result["warning"] = save_error
        result["message"] += f". Warning: {save_error}"

    # Check for pending action that requires follow-up after naming
    if hasattr(main_window, '_gui_state') and main_window._gui_state.pending_action:
        pending = main_window._gui_state.pending_action
        if isinstance(pending, NameProjectThenPlanAction):
            result["next_action"] = "present_plan"
            result["next_action_args"] = {
                "steps": pending.pending_steps,
                "summary": pending.pending_summary
            }
            result["message"] += ". IMPORTANT: Now call present_plan with the pending steps to continue."
            # Clear the pending action
            main_window._gui_state.clear_pending_action()

    return result


@tools.register(
    description="Save the current project to disk. Uses the existing path if project was previously saved, or saves to the export directory for new projects.",
    requires_project=True,
    modifies_gui_state=False
)
def save_project(project, path: Optional[str] = None) -> dict:
    """Save project state to JSON file."""
    # Determine save path
    if path:
        valid, error, validated_path = _validate_path(path)
        if not valid:
            return {"success": False, "error": f"Invalid path: {error}"}
        save_path = validated_path
    elif project.path:
        save_path = project.path
    else:
        # New project - use export dir
        settings = load_settings()
        project_name = project.metadata.name or "untitled"
        save_path = settings.export_dir / f"{project_name}.sceneripper"

    # Ensure .sceneripper extension
    if save_path.suffix.lower() != ".sceneripper":
        save_path = save_path.with_suffix(".sceneripper")

    # Validate and create parent directory
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = save_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        return {
            "success": False,
            "error": f"Cannot save project: No write access to '{save_path.parent}'. "
                     "Please check directory permissions or choose a different location."
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"Cannot save project: Directory '{save_path.parent}' is not accessible ({e}). "
                     "Please check the path or choose a different location."
        }

    try:
        success = project.save(path=save_path)
        if success:
            return {
                "success": True,
                "path": str(save_path),
                "message": f"Project saved to {save_path.name}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to save project"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tools.register(
    description="Load a project from a .sceneripper file. This replaces the current project.",
    requires_project=False,
    modifies_gui_state=True
)
def load_project(path: str, main_window=None) -> dict:
    """Load project from file."""
    from core.project import Project, ProjectLoadError, MissingSourceError

    valid, error, validated_path = _validate_path(path, must_exist=True)
    if not valid:
        return {"success": False, "error": f"Invalid path: {error}"}

    if not validated_path.is_file():
        return {"success": False, "error": f"Path is not a file: {path}"}

    if main_window is None:
        return {"success": False, "error": "Cannot load project: main window not available"}

    try:
        # Load the project
        new_project = Project.load(validated_path)

        if not new_project.sources:
            return {"success": False, "error": "No valid sources found in project"}

        # Clear existing UI state
        main_window._clear_project_state()

        # Set the new project and update adapter
        main_window.project = new_project
        main_window._project_adapter.set_project(main_window.project)

        # Refresh UI components with new project data
        main_window._refresh_ui_from_project()

        return {
            "success": True,
            "path": str(validated_path),
            "name": new_project.metadata.name,
            "sources": len(new_project.sources),
            "clips": len(new_project.clips),
            "message": f"Loaded project: {new_project.metadata.name}"
        }
    except ProjectLoadError as e:
        return {"success": False, "error": f"Failed to load project: {e}"}
    except MissingSourceError as e:
        return {"success": False, "error": f"Missing source video: {e.source_path}"}
    except Exception as e:
        logger.exception("Failed to load project")
        return {"success": False, "error": str(e)}


@tools.register(
    description="Create a fresh empty project. This clears the current project state.",
    requires_project=False,
    modifies_gui_state=True
)
def new_project(name: str = "Untitled Project", main_window=None) -> dict:
    """Create a new empty project."""
    from core.project import Project

    if main_window is None:
        return {"success": False, "error": "Cannot create project: main window not available"}

    # Clear all existing project state
    main_window._clear_project_state()

    # Create and set new project
    new_proj = Project.new(name=name)
    main_window.project = new_proj
    main_window._project_adapter.set_project(main_window.project)

    # Update window title
    main_window._update_window_title()

    return {
        "success": True,
        "name": name,
        "message": f"Created new project: {name}"
    }


@tools.register(
    description="Add tags to one or more clips for organization and filtering.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def add_tags(project, clip_ids: list[str], tags: list[str]) -> dict:
    """Add tags to specified clips."""
    if not clip_ids:
        return {"success": False, "error": "No clip IDs provided"}
    if not tags:
        return {"success": False, "error": "No tags provided"}

    updated = []
    not_found = []

    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None:
            not_found.append(clip_id)
            continue

        # Add new tags (avoid duplicates)
        for tag in tags:
            if tag not in clip.tags:
                clip.tags.append(tag)
        updated.append(clip_id)

    if updated:
        project.update_clips([project.clips_by_id[cid] for cid in updated])

    return {
        "success": len(updated) > 0,
        "updated": updated,
        "not_found": not_found,
        "tags_added": tags,
        "message": f"Added {len(tags)} tag(s) to {len(updated)} clip(s)"
    }


@tools.register(
    description="Remove tags from one or more clips.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def remove_tags(project, clip_ids: list[str], tags: list[str]) -> dict:
    """Remove tags from specified clips."""
    if not clip_ids:
        return {"success": False, "error": "No clip IDs provided"}
    if not tags:
        return {"success": False, "error": "No tags provided"}

    updated = []
    not_found = []

    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None:
            not_found.append(clip_id)
            continue

        # Remove tags
        removed_any = False
        for tag in tags:
            if tag in clip.tags:
                clip.tags.remove(tag)
                removed_any = True

        if removed_any:
            updated.append(clip_id)

    if updated:
        project.update_clips([project.clips_by_id[cid] for cid in updated])

    return {
        "success": len(updated) > 0,
        "updated": updated,
        "not_found": not_found,
        "tags_removed": tags,
        "message": f"Removed tag(s) from {len(updated)} clip(s)"
    }


@tools.register(
    description="Add or update a note on a clip.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def add_note(project, clip_id: str, note: str) -> dict:
    """Set note text for a clip."""
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        return {"success": False, "error": f"Clip not found: {clip_id}"}

    clip.notes = note
    project.update_clips([clip])

    return {
        "success": True,
        "clip_id": clip_id,
        "note": note,
        "message": "Note updated" if note else "Note cleared"
    }


# Valid shot types for update_clip validation (imported from core.analysis.shots)
VALID_SHOT_TYPES = set(SHOT_TYPES)


@tools.register(
    description="Update clip metadata. Only specified fields are updated. Shot type must be one of: 'wide shot', 'medium shot', 'close-up', 'extreme close-up', or empty string to clear.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def update_clip(
    project,
    clip_id: str,
    name: Optional[str] = None,
    shot_type: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[list[str]] = None
) -> dict:
    """Update clip metadata fields.

    Args:
        project: The current project
        clip_id: ID of the clip to update
        name: New clip name (None to skip, empty string to clear)
        shot_type: New shot type (None to skip, empty string to clear)
        notes: New notes (None to skip, empty string to clear)
        tags: New tags list (None to skip, replaces all existing tags)

    Returns:
        Dict with success status and updated fields
    """
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        return {"success": False, "error": f"Clip not found: {clip_id}"}

    updated_fields = []

    # Update name if provided
    if name is not None:
        clip.name = name
        updated_fields.append("name")

    # Validate and update shot_type if provided
    if shot_type is not None:
        if shot_type == "":
            # Empty string clears the shot type
            clip.shot_type = None
            updated_fields.append("shot_type")
        elif shot_type in VALID_SHOT_TYPES:
            clip.shot_type = shot_type
            updated_fields.append("shot_type")
        else:
            return {
                "success": False,
                "error": f"Invalid shot type: '{shot_type}'. Must be one of: {', '.join(sorted(VALID_SHOT_TYPES))} or empty string to clear."
            }

    # Update notes if provided
    if notes is not None:
        clip.notes = notes
        updated_fields.append("notes")

    # Update tags if provided (replaces all existing tags)
    if tags is not None:
        clip.tags = list(tags)
        updated_fields.append("tags")

    if updated_fields:
        project.update_clips([clip])

    return {
        "success": True,
        "clip_id": clip_id,
        "updated_fields": updated_fields,
        "message": f"Updated {', '.join(updated_fields)}" if updated_fields else "No fields updated"
    }


@tools.register(
    description="Update the text of a transcript segment by index. Use list_clips to see clip info including transcript segments.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def update_clip_transcript(
    project,
    clip_id: str,
    segment_index: int,
    text: str
) -> dict:
    """Update the text of a specific transcript segment.

    Args:
        project: The current project
        clip_id: ID of the clip containing the transcript
        segment_index: Zero-based index of the segment to update
        text: New text for the segment

    Returns:
        Dict with success status, old text, and new text
    """
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        return {"success": False, "error": f"Clip not found: {clip_id}"}

    if not clip.transcript:
        return {"success": False, "error": f"Clip has no transcript: {clip_id}"}

    if segment_index < 0 or segment_index >= len(clip.transcript):
        return {
            "success": False,
            "error": f"Segment index {segment_index} out of range. Clip has {len(clip.transcript)} segments (0-{len(clip.transcript)-1})."
        }

    if not text or not text.strip():
        return {"success": False, "error": "Text cannot be empty"}

    segment = clip.transcript[segment_index]
    old_text = segment.text
    segment.text = text.strip()

    project.update_clips([clip])

    return {
        "success": True,
        "clip_id": clip_id,
        "segment_index": segment_index,
        "old_text": old_text,
        "new_text": segment.text,
        "message": f"Updated transcript segment {segment_index}"
    }


@tools.register(
    description="Open the clip details sidebar to show a specific clip's information.",
    requires_project=True,
    modifies_gui_state=True
)
def show_clip_details(project, clip_id: str) -> dict:
    """Open the clip details sidebar for a specific clip.

    Args:
        project: The current project
        clip_id: ID of the clip to show details for

    Returns:
        Dict with marker for GUI to open sidebar
    """
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        return {"success": False, "error": f"Clip not found: {clip_id}"}

    source = project.sources_by_id.get(clip.source_id)
    if source is None:
        return {"success": False, "error": f"Source not found for clip: {clip_id}"}

    return {
        "success": True,
        "_show_clip_details": True,
        "clip_id": clip_id,
        "source_id": clip.source_id,
        "message": f"Opening clip details for {clip.display_name(source.filename, source.fps)}"
    }


@tools.register(
    description="Generate a human-readable summary of the current project including sources, clips, analysis status, and sequence information.",
    requires_project=True,
    modifies_gui_state=False
)
def get_project_summary(project) -> dict:
    """Generate project summary."""
    # Build summary
    lines = []
    lines.append(f"# {project.metadata.name}")
    lines.append("")

    # Project info
    lines.append("## Project Info")
    lines.append(f"- **Path**: {project.path or 'Unsaved'}")
    lines.append(f"- **Created**: {project.metadata.created_at[:10]}")
    lines.append(f"- **Modified**: {project.metadata.modified_at[:10]}")
    lines.append(f"- **Unsaved changes**: {'Yes' if project.is_dirty else 'No'}")
    lines.append("")

    # Sources
    lines.append(f"## Sources ({len(project.sources)} videos)")
    if project.sources:
        total_duration = sum(s.duration_seconds for s in project.sources)
        lines.append(f"- **Total duration**: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        lines.append("")
        for source in project.sources:
            clip_count = len(project.clips_by_source.get(source.id, []))
            analyzed = "✓" if source.analyzed else "✗"
            lines.append(f"- {source.filename} ({source.duration_seconds:.1f}s, {clip_count} clips) [{analyzed}]")
    else:
        lines.append("- No sources imported yet")
    lines.append("")

    # Clips
    lines.append(f"## Clips ({len(project.clips)} total)")
    if project.clips:
        # Count analysis status
        with_colors = sum(1 for c in project.clips if c.dominant_colors)
        with_shots = sum(1 for c in project.clips if c.shot_type)
        with_transcript = sum(1 for c in project.clips if c.transcript)
        with_tags = sum(1 for c in project.clips if c.tags)
        with_notes = sum(1 for c in project.clips if c.notes)

        lines.append(f"- **Color analyzed**: {with_colors}/{len(project.clips)}")
        lines.append(f"- **Shot classified**: {with_shots}/{len(project.clips)}")
        lines.append(f"- **Transcribed**: {with_transcript}/{len(project.clips)}")
        lines.append(f"- **Tagged**: {with_tags}/{len(project.clips)}")
        lines.append(f"- **With notes**: {with_notes}/{len(project.clips)}")

        # List unique tags
        all_tags = set()
        for clip in project.clips:
            all_tags.update(clip.tags)
        if all_tags:
            lines.append(f"- **Tags used**: {', '.join(sorted(all_tags))}")
    else:
        lines.append("- No clips detected yet")
    lines.append("")

    # Sequence
    lines.append("## Sequence")
    if project.sequence and project.sequence.get_all_clips():
        seq_clips = project.sequence.get_all_clips()
        lines.append(f"- **Clips in sequence**: {len(seq_clips)}")
        lines.append(f"- **Total duration**: {project.sequence.duration_seconds:.1f}s")
        lines.append(f"- **FPS**: {project.sequence.fps}")
    else:
        lines.append("- No sequence built yet")

    summary_text = "\n".join(lines)

    return {
        "success": True,
        "summary": summary_text,
        "stats": {
            "sources": len(project.sources),
            "clips": len(project.clips),
            "sequence_clips": len(project.sequence.get_all_clips()) if project.sequence else 0,
            "is_dirty": project.is_dirty
        }
    }


# =============================================================================
# CLI Tools - Execute via subprocess for batch operations
# =============================================================================

@tools.register(
    description="Detect scenes in a video file and add clips to the project. Creates clips from detected scene boundaries. Returns the clip count and clip IDs.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def detect_scenes(
    project,
    video_path: str,
    sensitivity: float = 3.0,
    luma_only: bool | None = None,
) -> dict:
    """Run scene detection using Python API and add clips to project."""
    from core.scene_detect import SceneDetector, DetectionConfig

    # Validate video path
    valid, error, video = _validate_path(video_path, must_exist=True)
    if not valid:
        return {"success": False, "error": error}

    if not video.is_file():
        return {"success": False, "error": f"Path is not a file: {video_path}"}

    try:
        # Check if source already exists in project (by resolved file path)
        # Use resolve() to handle symlinks and relative paths consistently
        resolved_video = video.resolve()
        existing_source = None
        for s in project.sources:
            try:
                if s.file_path.resolve() == resolved_video:
                    existing_source = s
                    break
            except (OSError, ValueError):
                # Handle edge cases where resolve() might fail
                if s.file_path == video:
                    existing_source = s
                    break

        # Create detector with configured sensitivity
        config = DetectionConfig(threshold=sensitivity, luma_only=luma_only)
        detector = SceneDetector(config)

        # Run detection
        source, clips = detector.detect_scenes(video)

        # If source already exists, use that source ID and update clips
        if existing_source:
            source = existing_source
            # Mark source as analyzed
            source.analyzed = True
            # Update clip source IDs to match existing source
            for clip in clips:
                clip.source_id = source.id
        else:
            # Mark source as analyzed
            source.analyzed = True
            # Add new source using proper Project method (invalidates caches, notifies observers)
            project.add_source(source)

        # Secondary fallback: ensure at least one clip exists (belt and suspenders)
        # This handles edge cases where scene_detect might be bypassed or return empty
        is_fallback = False
        if not clips:
            from models.clip import Clip
            total_frames = source.total_frames
            clips = [Clip(
                source_id=source.id,
                start_frame=0,
                end_frame=total_frames,
            )]
            is_fallback = True

        # Check if the single clip is a full-video fallback
        if len(clips) == 1 and clips[0].start_frame == 0 and clips[0].end_frame == source.total_frames:
            is_fallback = True

        # Add clips using proper Project method (invalidates caches, notifies observers)
        project.add_clips(clips)

        # Build informative message
        if is_fallback:
            message = f"No scene cuts detected in {video.name} - created single clip spanning full video"
        else:
            message = f"Detected {len(clips)} scenes in {video.name} and added to project"

        return {
            "success": True,
            "clips_detected": len(clips),
            "clip_ids": [clip.id for clip in clips],
            "source_id": source.id,
            "is_fallback_clip": is_fallback,
            "message": message
        }

    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Scene detection failed: {e}"}


@tools.register(
    description="""Search YouTube for videos matching a query. Returns video titles, IDs, durations, and URLs.
Optional filters (require fetching metadata from each video, which is slower):
- aspect_ratio: "any", "16:9", "4:3", "9:16", or "1:1"
- resolution: "any", "4k", "1080p", "720p", or "480p" (minimum resolution)
- max_size: "any", "100mb", "500mb", or "1gb" (maximum file size)""",
    requires_project=False,
    modifies_gui_state=False
)
def search_youtube(
    query: str,
    max_results: int = 10,
    aspect_ratio: str = "any",
    resolution: str = "any",
    max_size: str = "any"
) -> dict:
    """Search YouTube using the Python API with optional filters."""
    # Get API key from settings
    settings = load_settings()
    api_key = settings.youtube_api_key

    if not api_key:
        return {
            "success": False,
            "error": "YouTube API key not configured. Add it in Settings > YouTube API Key.",
        }

    try:
        client = YouTubeSearchClient(api_key)
        result = client.search(query, max_results=min(max_results, 50))

        # Check if filtering is needed
        filters_active = any([
            aspect_ratio != "any",
            resolution != "any",
            max_size != "any",
        ])

        videos_to_process = result.videos

        # Apply filters if set (requires fetching metadata)
        if filters_active and videos_to_process:
            try:
                downloader = VideoDownloader()
                filtered = []
                for video in videos_to_process:
                    try:
                        info = downloader.get_video_info(
                            video.youtube_url,
                            include_format_details=True
                        )
                        video.width = info.get("width")
                        video.height = info.get("height")
                        video.aspect_ratio = info.get("aspect_ratio")
                        video.filesize_approx = info.get("filesize_approx")
                        video.has_detailed_info = True

                        if (video.matches_aspect_ratio(aspect_ratio) and
                            video.matches_resolution(resolution) and
                            video.matches_max_size(max_size)):
                            filtered.append(video)
                    except Exception as e:
                        logger.warning(f"Failed to fetch metadata for {video.video_id}: {e}")
                videos_to_process = filtered
            except Exception as e:
                logger.warning(f"Could not apply filters: {e}")

        # Convert to serializable format
        videos = []
        for video in videos_to_process:
            video_data = {
                "video_id": video.video_id,
                "title": video.title,
                "channel": video.channel_title,
                "duration": video.duration_str,
                "url": video.youtube_url,
                "thumbnail": video.thumbnail_url,
                "view_count": video.view_count,
            }
            # Include metadata if available
            if video.has_detailed_info:
                video_data.update({
                    "width": video.width,
                    "height": video.height,
                    "resolution": video.resolution_str,
                    "aspect_ratio": video.aspect_ratio,
                    "filesize_approx": video.filesize_approx,
                })
            videos.append(video_data)

        return {
            "success": True,
            "query": query,
            "filters_applied": filters_active,
            "results": videos,
            "total_results": result.total_results,
            "filtered_count": len(videos)
        }

    except QuotaExceededError as e:
        return {"success": False, "error": f"YouTube API quota exceeded. {e}"}
    except InvalidAPIKeyError as e:
        return {"success": False, "error": f"Invalid YouTube API key. {e}"}
    except YouTubeAPIError as e:
        return {"success": False, "error": f"YouTube API error: {e}"}
    except Exception as e:
        logger.exception("YouTube search failed")
        return {"success": False, "error": f"Search failed: {e}"}


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
    description="Download multiple videos from YouTube URLs. Downloads in background thread to keep UI responsive. "
                "Returns status for each video. Use search_youtube first to find video URLs.",
    requires_project=False,
    modifies_gui_state=True
)
def download_videos(
    main_window,
    urls: list[str],
    output_dir: Optional[str] = None,
) -> dict:
    """Download multiple videos from YouTube URLs.

    Args:
        urls: List of YouTube video URLs to download
        output_dir: Optional output directory (defaults to settings.download_dir)

    Returns:
        Dict with success status and results for each video (after worker completes)
    """
    if not urls:
        return {"success": False, "error": "No URLs provided"}

    if len(urls) > 10:
        return {
            "success": False,
            "error": "Maximum 10 videos per batch. Split into multiple calls for larger batches."
        }

    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Determine download directory
    if output_dir:
        valid, error, validated_dir = _validate_path(output_dir)
        if not valid:
            return {"success": False, "error": f"Invalid output directory: {error}"}
        download_path = validated_dir
    else:
        settings = load_settings()
        download_path = settings.download_dir

    # Check if download already running
    if main_window.url_bulk_download_worker and main_window.url_bulk_download_worker.isRunning():
        return {"success": False, "error": "Bulk download already in progress"}

    # Start async download via worker
    started = main_window.start_agent_bulk_download(urls, download_path)
    if not started:
        return {"success": False, "error": "Failed to start download worker"}

    # Return marker that tells GUI handler to wait for worker completion
    return {
        "_wait_for_worker": "download",
        "url_count": len(urls),
        "download_dir": str(download_path),
    }


# =============================================================================
# GUI-Aware Tools - Trigger workers and wait for completion
#
# Note: CLI-based analyze_colors, analyze_shots, transcribe, and export_clips
# tools were removed because they required the scene_ripper CLI to be installed.
# Use the *_live versions below instead (analyze_colors_live, analyze_shots_live,
# transcribe_live) which work directly with the in-memory project.
# =============================================================================

@tools.register(
    description="Import a local video file into the project library. The video will appear in the Collect tab.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def import_video(main_window, path: str) -> dict:
    """Import a video file to the library.

    Args:
        path: Absolute path to the video file

    Returns:
        Dict with source_id and metadata if successful
    """
    # Validate path
    valid, error, video_path = _validate_path(path, must_exist=True)
    if not valid:
        return {"success": False, "error": error}

    if not video_path.is_file():
        return {"success": False, "error": f"Path is not a file: {path}"}

    if video_path.suffix.lower() not in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
        return {"success": False, "error": f"Unsupported video format: {video_path.suffix}"}

    # Check if already in library
    for source in main_window.project.sources:
        if source.file_path == video_path:
            return {
                "success": True,
                "source_id": source.id,
                "message": "Video already in library",
                "filename": source.filename,
                "already_imported": True
            }

    # Add to library (reuse existing method)
    main_window._add_video_to_library(video_path)

    # Find the newly added source
    new_source = None
    for source in main_window.project.sources:
        if source.file_path == video_path:
            new_source = source
            break

    if new_source:
        return {
            "success": True,
            "source_id": new_source.id,
            "filename": new_source.filename,
            "message": f"Imported {new_source.filename}"
        }
    else:
        return {"success": False, "error": "Failed to add video to library"}


@tools.register(
    description="Select a video source as the current active source for detection and editing.",
    requires_project=True,
    modifies_gui_state=True
)
def select_source(main_window, source_id: str) -> dict:
    """Select a source as the current active source.

    Args:
        source_id: ID of the source to select

    Returns:
        Dict with success status and source info
    """
    # Find source
    source = None
    for s in main_window.project.sources:
        if s.id == source_id:
            source = s
            break

    if not source:
        return {"success": False, "error": f"Source not found: {source_id}"}

    # Select it
    main_window._select_source(source)

    return {
        "success": True,
        "source_id": source.id,
        "filename": source.filename,
        "duration_seconds": source.duration_seconds,
        "analyzed": source.analyzed,
        "clip_count": len(main_window.project.clips_by_source.get(source.id, []))
    }


@tools.register(
    description="Detect scenes in a video source with live GUI update. Supports visual detection (adaptive/content) "
                "and text-based detection (karaoke) for videos with changing text overlays. "
                "For karaoke mode, set mode='karaoke' and optionally adjust roi_top, text_threshold, etc.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def detect_scenes_live(
    main_window,
    source_id: str,
    mode: str = "adaptive",
    sensitivity: float = 3.0,
    luma_only: bool | None = None,
    roi_top: float = 0.0,
    text_threshold: float = 60.0,
    confirm_frames: int = 3,
    cut_offset: int = 5,
) -> dict:
    """Detect scenes in a source video with live GUI update.

    Args:
        source_id: ID of the source to analyze
        mode: Detection mode - 'adaptive' (visual), 'content' (visual), or 'karaoke' (text-based)
        sensitivity: Detection sensitivity for visual modes (1.0=sensitive, 10.0=less sensitive)
        luma_only: Force luma-only detection for B&W video. None=auto-detect.
        roi_top: For karaoke mode - top of text region (0.0=full frame, 0.75=bottom 25%)
        text_threshold: For karaoke mode - text similarity threshold (lower=more cuts)
        confirm_frames: For karaoke mode - frames to confirm text change (reduces false positives)
        cut_offset: For karaoke mode - shift cuts backward to catch fade-in starts

    Returns:
        Dict with detected clip count and IDs (after worker completes)
    """
    # Validate mode
    valid_modes = ["adaptive", "content", "karaoke"]
    if mode not in valid_modes:
        return {"success": False, "error": f"Invalid mode '{mode}'. Valid modes: {valid_modes}"}

    # Find source
    source = None
    for s in main_window.project.sources:
        if s.id == source_id:
            source = s
            break

    if not source:
        return {"success": False, "error": f"Source not found: {source_id}"}

    # Check if detection already running
    if main_window.detection_worker and main_window.detection_worker.isRunning():
        return {"success": False, "error": "Scene detection already in progress"}

    # Build config dict based on mode
    if mode == "karaoke":
        config = {
            "roi_top_percent": roi_top,
            "text_similarity_threshold": text_threshold,
            "confirm_frames": confirm_frames,
            "cut_offset": cut_offset,
        }
    else:
        config = {
            "threshold": sensitivity,
            "use_adaptive": (mode == "adaptive"),
            "luma_only": luma_only,
        }

    # Return marker — GUI layer handles source selection, detection start, and tab switch
    return {
        "_wait_for_worker": "detection",
        "source_id": source_id,
        "mode": mode,
        "config": config,
    }


@tools.register(
    description="Detect scenes in all unanalyzed video sources. Queues all sources that haven't been analyzed yet "
                "and processes them sequentially. Equivalent to 'Cut New Videos' button in Collect tab.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def detect_all_unanalyzed(
    main_window,
    project,
    sensitivity: float = 3.0,
    luma_only: bool | None = None,
) -> dict:
    """Detect scenes in all unanalyzed video sources.

    Args:
        sensitivity: Detection sensitivity (1.0=sensitive, 10.0=less sensitive)
        luma_only: Force luma-only detection for B&W video. None=auto-detect.

    Returns:
        Dict with queued source count
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Find unanalyzed sources
    unanalyzed = [s for s in project.sources if not s.analyzed]

    if not unanalyzed:
        return {
            "success": True,
            "message": "All sources are already analyzed",
            "queued_count": 0,
            "source_ids": [],
        }

    # Check if detection already running
    if main_window.detection_worker and main_window.detection_worker.isRunning():
        return {"success": False, "error": "Scene detection already in progress. Wait for it to complete."}

    # Update default sensitivity for the batch
    main_window.settings.default_sensitivity = sensitivity

    # Queue all unanalyzed sources
    source_ids = [s.id for s in unanalyzed]
    main_window._on_analyze_requested(source_ids)

    return {
        "success": True,
        "message": f"Queued {len(unanalyzed)} sources for scene detection. "
                   "IMPORTANT: Detection runs in background. "
                   "Call check_detection_status to monitor progress before calling list_clips.",
        "queued_count": len(unanalyzed),
        "source_ids": source_ids,
        "sensitivity": sensitivity,
        "next_action": "Call check_detection_status periodically until all_complete is True, then call list_clips"
    }


@tools.register(
    description="Check if scene detection is currently running and get progress. "
                "Call ONCE after detect_all_unanalyzed to confirm it started, then inform the user and WAIT for them to ask for updates. "
                "Do NOT call this repeatedly in a loop - you cannot actually wait between calls. "
                "Scene detection takes 1-5 minutes PER VIDEO. Only conclude failure if is_running=False AND sources_analyzed < sources_total.",
    requires_project=True,
    modifies_gui_state=True  # Needs main_window to check worker status
)
def check_detection_status(main_window, project) -> dict:
    """Check scene detection status and progress.

    Returns:
        Dict with detection status, queue info, and progress
    """
    import time

    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Check if detection worker is running
    is_running = (
        main_window.detection_worker is not None and
        main_window.detection_worker.isRunning()
    )

    # Check queue status
    queue_remaining = len(getattr(main_window, '_analyze_queue', []))
    queue_total = getattr(main_window, '_analyze_queue_total', 0)

    # Get elapsed time and current video progress
    start_time = getattr(main_window, '_detection_start_time', None)
    current_progress = getattr(main_window, '_detection_current_progress', 0.0)
    elapsed_seconds = int(time.time() - start_time) if start_time else 0

    # Count analyzed vs total sources
    analyzed_count = sum(1 for s in project.sources if s.analyzed)
    total_count = len(project.sources)

    # Count clips
    clip_count = len(project.clips)

    # Determine if all detection is complete
    all_complete = (
        analyzed_count == total_count and
        not is_running and
        queue_remaining == 0
    )

    if all_complete:
        message = f"Detection complete. All {analyzed_count} sources analyzed, {clip_count} clips available."
        patience_note = None
    elif is_running or queue_remaining > 0:
        current = queue_total - queue_remaining if queue_total > 0 else 0
        progress_pct = int(current_progress * 100)
        message = (
            f"Detection in progress: {analyzed_count}/{total_count} sources analyzed"
            f"{f', processing {current} of {queue_total}' if queue_total > 0 else ''}"
            f" (current video: {progress_pct}%)"
            f". {clip_count} clips so far."
            f" Elapsed: {elapsed_seconds // 60}m {elapsed_seconds % 60}s."
        )
        # Add patience guidance - detection is slow, especially for long videos
        remaining_videos = total_count - analyzed_count
        patience_note = (
            f"PATIENCE REQUIRED: Detection takes 1-5 minutes per video. "
            f"With {remaining_videos} videos remaining, expect up to {remaining_videos * 5} more minutes. "
            f"Keep checking status - do NOT assume failure while is_running=True."
        )
    else:
        unanalyzed = total_count - analyzed_count
        if unanalyzed > 0:
            message = f"Detection idle. {unanalyzed} sources not yet analyzed. Call detect_all_unanalyzed to start."
            patience_note = None
        else:
            message = f"All {analyzed_count} sources analyzed. {clip_count} clips available."
            patience_note = None

    result = {
        "success": True,
        "is_running": is_running,
        "queue_remaining": queue_remaining,
        "queue_total": queue_total,
        "sources_analyzed": analyzed_count,
        "sources_total": total_count,
        "clips_available": clip_count,
        "all_complete": all_complete,
        "current_video_progress": current_progress,
        "elapsed_seconds": elapsed_seconds,
        "message": message
    }
    if patience_note:
        result["patience_note"] = patience_note
    return result


@tools.register(
    description="Run one or more analysis operations on clips. Preferred over individual tools. "
                "Operations: 'colors' (dominant colors), 'shots' (shot type classification), "
                "'transcription' (speech-to-text), 'classification' (ImageNet labels), "
                "'description' (VLM description), 'objects' (YOLO object detection), "
                "'people' (person count). "
                "Example: start_clip_analysis(clip_ids=[...], operations=['colors', 'shots', 'transcription'])",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def start_clip_analysis(
    main_window,
    clip_ids: list[str],
    operations: list[str],
) -> dict:
    """Run one or more analysis operations on clips.

    Args:
        clip_ids: List of clip IDs to analyze
        operations: List of operation names to run

    Returns:
        Dict with _wait_for_worker marker for the first operation,
        or combined marker for analyze_all pipeline
    """
    valid_operations = {
        "colors", "shots", "transcription", "classification",
        "description", "objects", "people",
    }

    # Validate operations
    ops = [op for op in operations if op in valid_operations]
    if not ops:
        return {
            "success": False,
            "error": f"No valid operations. Valid: {', '.join(sorted(valid_operations))}"
        }

    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]
    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Map operation names to worker types
    op_to_worker = {
        "colors": "color_analysis",
        "shots": "shot_analysis",
        "transcription": "transcription",
        "classification": "classification",
        "description": "description",
        "objects": "object_detection",
        "people": "person_detection",
    }

    # Single operation: return specific worker marker
    if len(ops) == 1:
        op = ops[0]
        worker_type = op_to_worker[op]

        # Check specific worker availability
        worker_checks = {
            "colors": ("color_worker", "Color analysis"),
            "shots": ("shot_type_worker", "Shot type analysis"),
            "transcription": ("transcription_worker", "Transcription"),
            "classification": ("classification_worker", "Classification"),
            "description": ("description_worker", "Description generation"),
            "objects": ("detection_worker_yolo", "Object detection"),
            "people": ("detection_worker_yolo", "Person detection"),
        }
        worker_attr, worker_name = worker_checks[op]
        worker = getattr(main_window, worker_attr, None)
        if worker and worker.isRunning():
            return {"success": False, "error": f"{worker_name} already in progress"}

        result = {
            "_wait_for_worker": worker_type,
            "clip_ids": valid_ids,
            "clip_count": len(valid_ids),
        }
        # Add extra params for specific operations
        if op == "classification":
            result["top_k"] = 5
        elif op == "objects":
            result["confidence"] = 0.5
        return result

    # Multiple operations: use analyze_all pipeline
    # Map our operation names to analyze_all operation keys
    pipeline_op_map = {
        "colors": "colors",
        "shots": "shots",
        "transcription": "transcribe",
        "classification": "classify",
        "description": "describe",
        "objects": "detect_objects",
        "people": "detect_objects",  # people detection uses same YOLO pipeline
    }
    pipeline_ops = list(dict.fromkeys(pipeline_op_map[op] for op in ops))

    # Return marker — GUI layer handles tab switch, clip loading, and pipeline start
    return {
        "_wait_for_worker": "analyze_all",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "operations": pipeline_ops,
    }


@tools.register(
    description="Extract dominant colors from clips with live GUI update. Updates clip metadata. "
                "Prefer start_clip_analysis(operations=['colors']) instead.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def analyze_colors_live(main_window, clip_ids: list[str]) -> dict:
    """Extract dominant colors from clips with live GUI update.

    Args:
        clip_ids: List of clip IDs to analyze

    Returns:
        Dict with analysis results (after worker completes)
    """
    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.color_worker and main_window.color_worker.isRunning():
        return {"success": False, "error": "Color analysis already in progress"}

    # Return instruction for MainWindow to start the worker
    return {"_wait_for_worker": "color_analysis", "clip_ids": valid_ids, "clip_count": len(valid_ids)}


@tools.register(
    description="Classify shot types (close-up, medium, wide, etc.) for clips with live GUI update. "
                "Prefer start_clip_analysis(operations=['shots']) instead.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def analyze_shots_live(main_window, clip_ids: list[str]) -> dict:
    """Classify shot types for clips with live GUI update.

    Args:
        clip_ids: List of clip IDs to analyze

    Returns:
        Dict with analysis results (after worker completes)
    """
    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.shot_type_worker and main_window.shot_type_worker.isRunning():
        return {"success": False, "error": "Shot type analysis already in progress"}

    # Return instruction for MainWindow to start the worker
    return {"_wait_for_worker": "shot_analysis", "clip_ids": valid_ids, "clip_count": len(valid_ids)}


@tools.register(
    description="Transcribe speech in clips using Whisper with live GUI update. "
                "Prefer start_clip_analysis(operations=['transcription']) instead.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def transcribe_live(main_window, clip_ids: list[str]) -> dict:
    """Transcribe speech in clips with live GUI update.

    Args:
        clip_ids: List of clip IDs to transcribe

    Returns:
        Dict with transcription results (after worker completes)
    """
    # Check if faster-whisper is available
    from core.transcription import is_faster_whisper_available
    if not is_faster_whisper_available():
        return {
            "success": False,
            "error": "Transcription unavailable - faster-whisper not installed"
        }

    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.transcription_worker and main_window.transcription_worker.isRunning():
        return {"success": False, "error": "Transcription already in progress"}

    # Return instruction for MainWindow to start the worker
    return {"_wait_for_worker": "transcription", "clip_ids": valid_ids, "clip_count": len(valid_ids)}


@tools.register(
    description="Classify frame content using ImageNet labels. Identifies objects like 'dog', 'car', 'tree' in clips using MobileNet. Updates clip.object_labels. "
                "Prefer start_clip_analysis(operations=['classification']) instead.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def classify_content_live(main_window, clip_ids: list[str], top_k: int = 5) -> dict:
    """Classify content in clips with live GUI update.

    Uses MobileNetV3-Small to classify frames with ImageNet labels (1000 categories).
    Results are stored in clip.object_labels.

    Args:
        clip_ids: List of clip IDs to classify
        top_k: Number of top labels to return per clip (default: 5)

    Returns:
        Dict with classification results (after worker completes)
    """
    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.classification_worker and main_window.classification_worker.isRunning():
        return {"success": False, "error": "Classification already in progress"}

    # Return instruction for MainWindow to start the worker
    return {
        "_wait_for_worker": "classification",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "top_k": top_k,
    }


@tools.register(
    description="Detect and count objects in clips using YOLO. Returns object labels, counts, and bounding boxes. Updates clip.detected_objects and clip.person_count. "
                "Prefer start_clip_analysis(operations=['objects']) instead.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def detect_objects_live(main_window, clip_ids: list[str], confidence: float = 0.5) -> dict:
    """Detect objects in clips with live GUI update.

    Uses YOLOv8 to detect objects from COCO dataset (80 object classes).
    Results are stored in clip.detected_objects and clip.person_count.

    Args:
        clip_ids: List of clip IDs to analyze
        confidence: Detection confidence threshold (0.0-1.0, default: 0.5)

    Returns:
        Dict with detection results (after worker completes)
    """
    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.detection_worker_yolo and main_window.detection_worker_yolo.isRunning():
        return {"success": False, "error": "Object detection already in progress"}

    # Return instruction for MainWindow to start the worker
    return {
        "_wait_for_worker": "object_detection",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "confidence": confidence,
    }


@tools.register(
    description="Count people in clips using YOLO. Faster than full object detection when you only need person counts. Updates clip.person_count. "
                "Prefer start_clip_analysis(operations=['people']) instead.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def count_people_live(main_window, clip_ids: list[str]) -> dict:
    """Count people in clips with live GUI update.

    Uses YOLOv8 to count people in each clip. Faster than detect_objects_live
    when you only need person counts.

    Args:
        clip_ids: List of clip IDs to analyze

    Returns:
        Dict with person count results (after worker completes)
    """
    # Validate clips exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Check if worker already running
    if main_window.detection_worker_yolo and main_window.detection_worker_yolo.isRunning():
        return {"success": False, "error": "Object detection already in progress"}

    # Return instruction for MainWindow to start the worker
    return {
        "_wait_for_worker": "person_detection",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
    }


@tools.register(
    description="Run analysis operations on clips with live GUI update. "
                "Supports smart concurrency: local ops run in parallel, then sequential, then cloud. "
                "Default operations: colors, shots, transcribe. "
                "Available operations: colors, shots, classify, detect_objects, extract_text, transcribe, describe, cinematography.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def analyze_all_live(
    main_window,
    clip_ids: list[str],
    operations: Optional[list[str]] = None,
) -> dict:
    """Run analysis operations on clips with live GUI update.

    Uses the phase-based pipeline: local ops concurrent, sequential ops
    one-at-a-time, cloud ops concurrent.

    Args:
        clip_ids: List of clip IDs to analyze
        operations: List of operation keys to run (default: colors, shots, transcribe).
            Valid keys: colors, shots, classify, detect_objects, extract_text,
            transcribe, describe, cinematography.

    Returns:
        Dict with analysis summary (after all workers complete)
    """
    from core.analysis_operations import OPERATIONS_BY_KEY, DEFAULT_SELECTED

    # Default operations
    if operations is None:
        operations = list(DEFAULT_SELECTED)

    # Validate operation keys
    valid_ops = [op for op in operations if op in OPERATIONS_BY_KEY]
    if not valid_ops:
        return {
            "success": False,
            "error": f"No valid operations. Valid keys: {', '.join(OPERATIONS_BY_KEY.keys())}"
        }

    # Validate clip IDs exist
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]
    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    # Return marker — GUI layer handles tab switch, clip loading, and pipeline start
    return {
        "_wait_for_worker": "analyze_all",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "operations": valid_ops,
    }


# =============================================================================
# Sequence/Remix Tools - Generate sorted clip sequences
# =============================================================================

@tools.register(
    description="List available sorting algorithms and their current availability status. "
                "Some algorithms (like color) require clip analysis first.",
    requires_project=True,
    modifies_gui_state=False
)
def list_sorting_algorithms(project) -> dict:
    """List available sorting algorithms and whether they can be used.

    Returns:
        Dict with algorithms list showing name, key, available status, and reason if unavailable
    """
    # Check if clips have color analysis
    clips = project.clips
    has_colors = any(clip.dominant_colors for clip in clips) if clips else False

    has_brightness = any(clip.average_brightness is not None for clip in clips) if clips else False
    has_volume = any(clip.rms_volume is not None for clip in clips) if clips else False
    has_shot_type = any(clip.shot_type for clip in clips) if clips else False
    has_embeddings = any(clip.embedding is not None for clip in clips) if clips else False
    has_boundary_emb = any(
        clip.first_frame_embedding is not None and clip.last_frame_embedding is not None
        for clip in clips
    ) if clips else False
    has_text = any(clip.extracted_texts for clip in clips) if clips else False
    has_descriptions = any(clip.description for clip in clips) if clips else False

    algorithms = [
        {
            "key": "shuffle",
            "name": "Dice Roll",
            "description": "Randomly shuffle clips into a new order",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "seed", "type": "integer", "description": "Random seed for reproducibility (0 = random)", "default": 0}
            ]
        },
        {
            "key": "sequential",
            "name": "Time Capsule",
            "description": "Keep clips in their original order",
            "available": True,
            "reason": None,
            "parameters": []
        },
        {
            "key": "duration",
            "name": "Tempo Shift",
            "description": "Order clips from shortest to longest (or reverse)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["short_first", "long_first"], "default": "short_first"}
            ]
        },
        {
            "key": "color",
            "name": "Chromatic Flow",
            "description": "Arrange clips along a color gradient",
            "available": has_colors,
            "reason": None if has_colors else "Run color analysis on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["rainbow", "warm_to_cool", "cool_to_warm"], "default": "rainbow"}
            ]
        },
        {
            "key": "color_cycle",
            "name": "Color Cycle",
            "description": "Curate clips with strong color identity and cycle through the spectrum",
            "available": has_colors,
            "reason": None if has_colors else "Run color analysis on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["rainbow", "warm_to_cool", "cool_to_warm"], "default": "rainbow"}
            ]
        },
        {
            "key": "brightness",
            "name": "Into the Dark",
            "description": "Arrange clips from light to shadow, or shadow to light (auto-computes if needed)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["light_to_dark", "dark_to_light"], "default": "light_to_dark"}
            ]
        },
        {
            "key": "volume",
            "name": "Crescendo",
            "description": "Build from silence to thunder, or thunder to silence (auto-computes if needed)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["quiet_to_loud", "loud_to_quiet"], "default": "quiet_to_loud"}
            ]
        },
        {
            "key": "shot_type",
            "name": "Focal Ladder",
            "description": "Arrange clips by camera shot scale",
            "available": has_shot_type,
            "reason": None if has_shot_type else "Run shot type classification on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["wide_to_close", "close_to_wide"], "default": "wide_to_close"}
            ]
        },
        {
            "key": "proximity",
            "name": "Up Close and Personal",
            "description": "Glide from distant vistas to intimate close-ups",
            "available": has_shot_type,
            "reason": None if has_shot_type else "Run shot type classification on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["far_to_near", "near_to_far"], "default": "far_to_near"}
            ]
        },
        {
            "key": "similarity_chain",
            "name": "Human Centipede",
            "description": "Chain clips together by visual similarity (auto-computes embeddings if needed)",
            "available": True,
            "reason": None,
            "parameters": []
        },
        {
            "key": "match_cut",
            "name": "Match Cut",
            "description": "Find hidden connections between clips using boundary frame similarity (auto-computes if needed)",
            "available": True,
            "reason": None,
            "parameters": []
        },
        {
            "key": "exquisite_corpus",
            "name": "Exquisite Corpus",
            "description": "Generate a poem from on-screen text",
            "available": has_text,
            "reason": None if has_text else "Run OCR/text extraction on clips first",
            "parameters": []
        },
        {
            "key": "storyteller",
            "name": "Storyteller",
            "description": "Create a narrative from clip descriptions",
            "available": has_descriptions,
            "reason": None if has_descriptions else "Run clip description analysis first",
            "parameters": []
        },
    ]

    return {
        "algorithms": algorithms,
        "clip_count": len(clips),
        "has_color_analysis": has_colors,
        "reference_guided_available": True,
        "reference_guided_note": (
            "For matching clips to a reference video's structure, use the "
            "generate_reference_guided tool instead of generate_remix."
        ),
    }


@tools.register(
    description="Generate a sequence using a sorting algorithm and apply it to the timeline. "
                "Available algorithms: color, color_cycle, duration, brightness, volume, "
                "shuffle, sequential, shot_type, proximity, similarity_chain, match_cut, "
                "exquisite_corpus, storyteller. Use list_sorting_algorithms to check availability.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_remix(
    project,
    main_window,
    algorithm: str,
    clip_count: int = 10,
    direction: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Generate a sequence using the specified algorithm and apply to timeline.

    Args:
        algorithm: One of the 13 sorting algorithms (e.g. "color", "brightness",
                   "similarity_chain", "match_cut", etc.)
        clip_count: Number of clips to include (1-100)
        direction: Algorithm-specific direction (e.g. "rainbow", "short_first",
                   "light_to_dark", "quiet_to_loud", "wide_to_close")
        seed: For shuffle: random seed for reproducibility (0 = random)

    Returns:
        Dict with success status, applied clips, and algorithm used
    """
    valid_algorithms = [
        "color", "color_cycle", "duration", "brightness", "volume",
        "shuffle", "sequential", "shot_type", "proximity",
        "similarity_chain", "match_cut", "exquisite_corpus", "storyteller",
    ]
    if algorithm not in valid_algorithms:
        return {
            "success": False,
            "error": f"Invalid algorithm '{algorithm}'. Valid options: {', '.join(valid_algorithms)}"
        }

    # Validate clip count
    if clip_count < 1 or clip_count > 100:
        return {
            "success": False,
            "error": "clip_count must be between 1 and 100"
        }

    # Check color algorithm requirements
    if algorithm == "color":
        has_colors = any(clip.dominant_colors for clip in project.clips)
        if not has_colors:
            return {
                "success": False,
                "error": "Color sorting requires color analysis. Run analyze_colors_live first."
            }

    # Check if we have clips
    if not project.clips:
        return {
            "success": False,
            "error": "No clips available. Detect scenes first."
        }

    # Get sequence tab from main window
    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {
            "success": False,
            "error": "Main window not available"
        }

    # Use sequence tab's generate_and_apply method
    result = main_window.sequence_tab.generate_and_apply(
        algorithm=algorithm,
        clip_count=clip_count,
        direction=direction,
        seed=seed
    )

    return result


@tools.register(
    description="Check which matching dimensions have analysis data for the current clips. "
                "Call this before generate_reference_guided to discover which dimension weights "
                "are valid (e.g. color requires color analysis, embedding requires embeddings).",
    requires_project=True,
    modifies_gui_state=False
)
def get_available_dimensions(project, main_window) -> dict:
    """Check which reference-guided matching dimensions have data.

    Returns:
        Dict with available dimension keys and their descriptions
    """
    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {"success": False, "error": "Main window not available"}

    clips = main_window.sequence_tab._available_clips
    if not clips:
        return {"success": False, "error": "No clips available for sequencing"}

    from core.remix.reference_match import get_active_dimensions_for_clips
    clip_objects = [clip for clip, _ in clips]
    available = get_active_dimensions_for_clips(clip_objects)

    return {
        "success": True,
        "available_dimensions": sorted(available),
        "clip_count": len(clip_objects),
    }


@tools.register(
    description="Generate a reference-guided sequence that matches your clips to a reference "
                "video's structure across weighted dimensions (color, brightness, shot_scale, "
                "audio, embedding, movement, duration). Use get_available_dimensions first to "
                "check which dimensions have data. Use list_sources to find source IDs.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_reference_guided(
    project,
    main_window,
    reference_source_id: str,
    weights: Optional[dict] = None,
    allow_repeats: bool = False,
) -> dict:
    """Generate a sequence by matching user clips to a reference video's structure.

    Args:
        reference_source_id: Source ID of the reference video (use list_sources to find IDs)
        weights: Dimension weights as {"dimension": 0.0-1.0}. Available dimensions:
            color, brightness, shot_scale, audio, embedding, movement, duration.
            Defaults to {"embedding": 1.0, "brightness": 0.4, "duration": 0.6}
        allow_repeats: Allow same clip to match multiple reference positions

    Returns:
        Dict with success status, matched clips, and unmatched count
    """
    if weights is None:
        weights = {"embedding": 1.0, "brightness": 0.4, "duration": 0.6}

    # Validate source exists
    if reference_source_id not in project.sources_by_id:
        return {
            "success": False,
            "error": f"Source '{reference_source_id}' not found. Use list_sources to find valid IDs."
        }

    # Validate weights
    valid_dims = {"color", "brightness", "shot_scale", "audio", "embedding", "movement", "duration"}
    invalid = set(weights.keys()) - valid_dims
    if invalid:
        return {
            "success": False,
            "error": f"Invalid dimensions: {invalid}. Valid: {sorted(valid_dims)}"
        }

    for dim, val in weights.items():
        if not isinstance(val, (int, float)):
            return {
                "success": False,
                "error": f"Weight for '{dim}' must be a number, got {type(val).__name__}"
            }
        if not (0.0 <= val <= 1.0):
            return {
                "success": False,
                "error": f"Weight for '{dim}' must be between 0.0 and 1.0, got {val}"
            }

    if not any(v > 0 for v in weights.values()):
        return {
            "success": False,
            "error": "At least one dimension must have weight > 0"
        }

    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {"success": False, "error": "Main window not available"}

    result = main_window.sequence_tab.generate_reference_guided(
        reference_source_id=reference_source_id,
        weights=weights,
        allow_repeats=allow_repeats,
    )

    return result


@tools.register(
    description="Get the current state of the sequence tab including selected algorithm, "
                "parameters, preview clips, and timeline clips.",
    requires_project=True,
    modifies_gui_state=False
)
def get_remix_state(project, main_window) -> dict:
    """Get current state of the remix/sequence UI.

    Returns:
        Dict with current algorithm, parameters, preview clip count, and timeline state
    """
    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {
            "success": False,
            "error": "Main window not available"
        }

    state = main_window.sequence_tab.get_sorting_state()
    state["success"] = True

    return state


@tools.register(
    description="Get current application settings (non-sensitive). Returns directories, quality presets, "
                "theme preference, and other user-configurable options. Does not expose API keys.",
    requires_project=False,
    modifies_gui_state=False
)
def get_settings() -> dict:
    """Get current application settings.

    Returns non-sensitive settings including directories, export quality,
    theme preference, and detection parameters.

    Returns:
        Dict with success status and settings values
    """
    settings = load_settings()

    return {
        "success": True,
        "settings": {
            # Directories
            "download_dir": str(settings.download_dir),
            "export_dir": str(settings.export_dir),
            "thumbnail_cache_dir": str(settings.thumbnail_cache_dir),
            # Detection
            "default_sensitivity": settings.default_sensitivity,
            "min_scene_length_seconds": settings.min_scene_length_seconds,
            # Export
            "export_quality": settings.export_quality,
            "export_resolution": settings.export_resolution,
            "export_fps": settings.export_fps,
            # Transcription
            "transcription_model": settings.transcription_model,
            "transcription_language": settings.transcription_language,
            # Appearance
            "theme_preference": settings.theme_preference,
            # YouTube (no API key)
            "youtube_results_count": settings.youtube_results_count,
            "youtube_parallel_downloads": settings.youtube_parallel_downloads,
            # LLM (no API keys)
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
        }
    }


# Safe settings that can be modified by agent (no API keys, no paths)
SAFE_SETTINGS = {
    "default_sensitivity": (float, 1.0, 10.0),
    "min_scene_length_seconds": (float, 0.1, 10.0),
    "export_quality": (str, ["low", "medium", "high"]),
    "export_resolution": (str, ["original", "1080p", "720p", "480p"]),
    "export_fps": (str, ["original", "24", "30", "60"]),
    "transcription_model": (str, ["tiny.en", "small.en", "medium.en", "large-v3"]),
    "transcription_language": (str, None),  # Any string allowed
    "theme_preference": (str, ["system", "light", "dark"]),
    "youtube_results_count": (int, 10, 50),
    "youtube_parallel_downloads": (int, 1, 3),
    "llm_provider": (str, ["local", "openai", "anthropic", "gemini", "openrouter"]),
    "llm_model": (str, None),  # Any string allowed
    "llm_temperature": (float, 0.0, 2.0),
}


@tools.register(
    description="Update application settings (safe settings only, no API keys or paths). "
                "Use get_settings first to see current values. "
                "Settings: default_sensitivity (1.0-10.0), export_quality ('low'/'medium'/'high'), "
                "export_resolution ('original'/'1080p'/'720p'/'480p'), theme_preference ('system'/'light'/'dark'), "
                "llm_provider ('local'/'openai'/'anthropic'/'gemini'/'openrouter'), llm_model, llm_temperature (0.0-2.0).",
    requires_project=False,
    modifies_gui_state=False
)
def update_settings(setting_name: str, value) -> dict:
    """Update an application setting.

    Args:
        setting_name: Name of the setting to update (must be in SAFE_SETTINGS)
        value: New value for the setting

    Returns:
        Dict with success status and updated value
    """
    if setting_name not in SAFE_SETTINGS:
        return {
            "success": False,
            "error": f"Setting '{setting_name}' cannot be modified. "
                     f"Safe settings: {', '.join(sorted(SAFE_SETTINGS.keys()))}"
        }

    spec = SAFE_SETTINGS[setting_name]
    expected_type = spec[0]

    # Type validation
    if expected_type == float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return {"success": False, "error": f"Setting '{setting_name}' requires a number"}
        min_val, max_val = spec[1], spec[2]
        if not (min_val <= value <= max_val):
            return {
                "success": False,
                "error": f"Setting '{setting_name}' must be between {min_val} and {max_val}"
            }
    elif expected_type == int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return {"success": False, "error": f"Setting '{setting_name}' requires an integer"}
        min_val, max_val = spec[1], spec[2]
        if not (min_val <= value <= max_val):
            return {
                "success": False,
                "error": f"Setting '{setting_name}' must be between {min_val} and {max_val}"
            }
    elif expected_type == str:
        value = str(value)
        allowed_values = spec[1]
        if allowed_values is not None and value not in allowed_values:
            return {
                "success": False,
                "error": f"Setting '{setting_name}' must be one of: {', '.join(allowed_values)}"
            }

    # Load current settings
    settings = load_settings()

    # Update the setting
    old_value = getattr(settings, setting_name)
    setattr(settings, setting_name, value)

    # Save settings
    from core.settings import save_settings
    save_settings(settings)

    return {
        "success": True,
        "message": f"Updated {setting_name}: {old_value} -> {value}",
        "setting": setting_name,
        "old_value": old_value,
        "new_value": value,
    }


# =============================================================================
# Content-Aware Tools - Search, similarity, and grouping
# =============================================================================


@tools.register(
    description="Search clip transcripts for specific words or phrases. Returns clips containing the search term with timestamp and context. Useful for finding clips where specific things are said.",
    requires_project=True,
    modifies_gui_state=False
)
def search_transcripts(
    project,
    query: str,
    case_sensitive: bool = False,
    context_chars: int = 50
) -> dict:
    """Search transcripts for matching content.

    Args:
        query: Text to search for in transcripts
        case_sensitive: Whether to match case exactly (default: False)
        context_chars: Number of characters of context around match (default: 50)

    Returns:
        Dictionary with success status, query, match count, and list of matches
    """
    if not query:
        return {"success": False, "error": "Query cannot be empty"}

    search_query = query if case_sensitive else query.lower()
    results = []

    for clip in project.clips:
        if not clip.transcript:
            continue

        full_text = clip.get_transcript_text()
        if not full_text:
            continue

        search_text = full_text if case_sensitive else full_text.lower()

        if search_query in search_text:
            # Find match position for context
            pos = search_text.find(search_query)
            start = max(0, pos - context_chars)
            end = min(len(full_text), pos + len(query) + context_chars)
            context = full_text[start:end]

            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(full_text):
                context = context + "..."

            source = project.sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0

            results.append({
                "clip_id": clip.id,
                "source_name": source.file_path.name if source else "Unknown",
                "match_context": context,
                "duration_seconds": round(clip.duration_seconds(fps), 2),
                "start_time": round(clip.start_time(fps), 2),
                "shot_type": clip.shot_type,
            })

    return {
        "success": True,
        "query": query,
        "match_count": len(results),
        "matches": results
    }


@tools.register(
    description="Find clips visually similar to a reference clip based on color palette, shot type, or duration. Returns a ranked list of similar clips.",
    requires_project=True,
    modifies_gui_state=False
)
def find_similar_clips(
    project,
    clip_id: str,
    criteria: Optional[list[str]] = None,
    limit: int = 10
) -> dict:
    """Find clips similar to a reference clip by visual/temporal criteria.

    Args:
        clip_id: ID of the reference clip to find similar clips to
        criteria: List of criteria to compare: 'color', 'shot_type', 'duration'
                  (default: ['color', 'shot_type'])
        limit: Maximum number of similar clips to return (default: 10)

    Returns:
        Dictionary with success status, reference clip ID, criteria used,
        and list of similar clips with similarity scores
    """
    from core.analysis.color import get_primary_hue

    if criteria is None:
        criteria = ["color", "shot_type"]

    # Validate criteria
    valid_criteria = ["color", "shot_type", "duration"]
    for c in criteria:
        if c not in valid_criteria:
            return {
                "success": False,
                "error": f"Invalid criterion '{c}'. Valid criteria: {', '.join(valid_criteria)}"
            }

    reference = project.clips_by_id.get(clip_id)
    if not reference:
        return {"success": False, "error": f"Clip '{clip_id}' not found"}

    ref_source = project.sources_by_id.get(reference.source_id)
    ref_fps = ref_source.fps if ref_source else 30.0
    ref_duration = reference.duration_seconds(ref_fps)

    scores = []
    for clip in project.clips:
        if clip.id == clip_id:
            continue

        score = 0.0
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0

        # Score by shot type match
        if "shot_type" in criteria:
            if clip.shot_type and reference.shot_type:
                if clip.shot_type == reference.shot_type:
                    score += 1.0
                # Partial match for similar shot types
                elif clip.shot_type and reference.shot_type:
                    # Close types get partial score
                    close_types = {
                        "close_up": ["extreme_close_up", "medium_close_up"],
                        "medium_shot": ["medium_close_up", "medium_long_shot"],
                        "wide_shot": ["medium_long_shot", "extreme_wide_shot"],
                    }
                    if reference.shot_type in close_types:
                        if clip.shot_type in close_types[reference.shot_type]:
                            score += 0.5

        # Score by color similarity
        if "color" in criteria and reference.dominant_colors and clip.dominant_colors:
            ref_hue = get_primary_hue(reference.dominant_colors)
            clip_hue = get_primary_hue(clip.dominant_colors)
            hue_diff = abs(ref_hue - clip_hue)
            if hue_diff > 180:
                hue_diff = 360 - hue_diff
            # Similar within 60 degrees gets full score, degrades linearly
            score += max(0, 1.0 - hue_diff / 60)

        # Score by duration similarity
        if "duration" in criteria:
            clip_duration = clip.duration_seconds(fps)
            if ref_duration > 0 and clip_duration > 0:
                duration_ratio = min(ref_duration, clip_duration) / max(ref_duration, clip_duration)
                score += duration_ratio

        if score > 0:
            scores.append((clip, score, source))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for clip, score, source in scores[:limit]:
        fps = source.fps if source else 30.0
        results.append({
            "clip_id": clip.id,
            "source_name": source.file_path.name if source else "Unknown",
            "similarity_score": round(score, 2),
            "shot_type": clip.shot_type,
            "duration_seconds": round(clip.duration_seconds(fps), 2),
            "has_speech": bool(clip.transcript),
        })

    return {
        "success": True,
        "reference_clip_id": clip_id,
        "criteria": criteria,
        "similar_clips": results
    }


@tools.register(
    description="Group clips by a specific criterion: 'color' (dominant palette), 'shot_type', 'duration' (short/medium/long), or 'source'. Returns groups with clip IDs and counts.",
    requires_project=True,
    modifies_gui_state=False
)
def group_clips_by(project, criterion: str) -> dict:
    """Group clips by specified criterion.

    Args:
        criterion: How to group clips - 'color', 'shot_type', 'duration', or 'source'

    Returns:
        Dictionary with success status, criterion used, group count,
        and groups with clip IDs and counts
    """
    from core.analysis.color import classify_color_palette

    valid_criteria = ["color", "shot_type", "duration", "source"]
    if criterion not in valid_criteria:
        return {
            "success": False,
            "error": f"Invalid criterion '{criterion}'. Valid criteria: {', '.join(valid_criteria)}"
        }

    groups: dict[str, list[str]] = {}

    for clip in project.clips:
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0

        if criterion == "shot_type":
            key = clip.shot_type if clip.shot_type else "unknown"
        elif criterion == "color":
            if clip.dominant_colors:
                key = classify_color_palette(clip.dominant_colors)
            else:
                key = "unanalyzed"
        elif criterion == "duration":
            duration = clip.duration_seconds(fps)
            if duration < 2:
                key = "short (<2s)"
            elif duration < 10:
                key = "medium (2-10s)"
            else:
                key = "long (>10s)"
        elif criterion == "source":
            key = source.file_path.name if source else "unknown"
        else:
            key = "unknown"

        if key not in groups:
            groups[key] = []
        groups[key].append(clip.id)

    # Format output with counts
    formatted_groups = {
        k: {"clip_ids": v, "count": len(v)}
        for k, v in sorted(groups.items())
    }

    return {
        "success": True,
        "criterion": criterion,
        "group_count": len(groups),
        "groups": formatted_groups
    }


# =============================================================================
# Film Language & Glossary Tools
# =============================================================================

@tools.register(
    description=(
        "Look up the definition of a film/cinematography term. "
        "Accepts term keys (like 'CU', 'low_angle') or display names (like 'Close-Up'). "
        "Returns the term's name, category, and definition."
    ),
    requires_project=False,
    modifies_gui_state=False
)
def get_film_term_definition(term: str) -> dict:
    """Look up a film term definition from the glossary.

    Args:
        term: The term to look up (case-insensitive, matches key or display name)

    Returns:
        Dict with success status and term data (name, category, definition)
    """
    from core.film_glossary import get_term_definition

    if not term or not term.strip():
        return {
            "success": False,
            "error": "No term provided. Please specify a film term to look up."
        }

    result = get_term_definition(term)

    if result:
        return {
            "success": True,
            "key": result["key"],
            "name": result["name"],
            "category": result["category"],
            "definition": result["definition"]
        }
    else:
        return {
            "success": False,
            "error": f"Term '{term}' not found in glossary.",
            "suggestion": "Try searching with search_glossary for partial matches."
        }


@tools.register(
    description=(
        "Search the film glossary for terms matching a query. "
        "Searches term names and definitions. Optionally filter by category: "
        "'Shot Sizes', 'Camera Angles', 'Camera Movement', 'Composition', "
        "'Lighting', 'Focus', 'Sound', 'Editing'."
    ),
    requires_project=False,
    modifies_gui_state=False
)
def search_glossary(query: str, category: Optional[str] = None) -> dict:
    """Search the film glossary for matching terms.

    Args:
        query: Search string to match against term names and definitions
        category: Optional category filter (e.g., 'Shot Sizes', 'Lighting')

    Returns:
        Dict with success status and list of matching terms
    """
    from core.film_glossary import search_glossary as do_search, GLOSSARY_CATEGORIES

    if not query or not query.strip():
        return {
            "success": False,
            "error": "No search query provided."
        }

    # Validate category if provided
    if category and category != "All" and category not in GLOSSARY_CATEGORIES:
        return {
            "success": False,
            "error": f"Invalid category '{category}'.",
            "valid_categories": GLOSSARY_CATEGORIES
        }

    results = do_search(query, category)

    if results:
        return {
            "success": True,
            "query": query,
            "category_filter": category,
            "result_count": len(results),
            "terms": [
                {
                    "key": r["key"],
                    "name": r["name"],
                    "category": r["category"],
                    "definition": r["definition"]
                }
                for r in results
            ]
        }
    else:
        return {
            "success": True,
            "query": query,
            "category_filter": category,
            "result_count": 0,
            "terms": [],
            "message": f"No terms found matching '{query}'."
        }


# =============================================================================
# Audio-Guided Sequencing Tools
# =============================================================================

@tools.register(
    description=(
        "Analyze an audio or video file for beats, tempo, and onsets. "
        "Returns beat timestamps for rhythm-based video editing. "
        "Supports MP3, WAV, FLAC audio files and video files with audio tracks."
    ),
    requires_project=False,
    modifies_gui_state=False
)
def detect_audio_beats(
    audio_path: str,
    include_onsets: bool = True
) -> dict:
    """Analyze audio for beats, tempo, and onset detection.

    Args:
        audio_path: Path to audio file (MP3, WAV, FLAC) or video file
        include_onsets: Whether to detect onsets/transients (good for cut points)

    Returns:
        Dict with tempo_bpm, beat_times, onset_times, downbeat_times, duration
    """
    from pathlib import Path
    from core.analysis.audio import (
        analyze_music_file,
        analyze_audio_from_video,
        has_audio_track,
    )

    path = Path(audio_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {audio_path}"
        }

    # Determine if audio or video file
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    try:
        if path.suffix.lower() in audio_extensions:
            # Direct audio file
            analysis = analyze_music_file(path, include_onsets=include_onsets)
        elif path.suffix.lower() in video_extensions:
            # Video file - extract and analyze audio
            if not has_audio_track(path):
                return {
                    "success": False,
                    "error": f"Video file has no audio track: {audio_path}"
                }
            analysis = analyze_audio_from_video(path, include_onsets=include_onsets)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {path.suffix}. "
                         f"Supported: {audio_extensions | video_extensions}"
            }

        return {
            "success": True,
            "file": str(path),
            "tempo_bpm": round(analysis.tempo_bpm, 1),
            "beat_count": len(analysis.beat_times),
            "beat_times": [round(t, 3) for t in analysis.beat_times[:20]],  # First 20
            "beat_times_truncated": len(analysis.beat_times) > 20,
            "downbeat_count": len(analysis.downbeat_times),
            "downbeat_times": [round(t, 3) for t in analysis.downbeat_times[:10]],
            "onset_count": len(analysis.onset_times) if include_onsets else 0,
            "onset_times": [round(t, 3) for t in analysis.onset_times[:20]] if include_onsets else [],
            "duration_seconds": round(analysis.duration_seconds, 2),
            "message": (
                f"Detected {analysis.tempo_bpm:.1f} BPM with "
                f"{len(analysis.beat_times)} beats over {analysis.duration_seconds:.1f}s"
            )
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Audio analysis failed: {str(e)}"
        }


@tools.register(
    description=(
        "Suggest beat-aligned cut points for clips in the current sequence. "
        "Analyzes an audio track and suggests adjustments to align clip transitions "
        "with musical beats. Strategies: 'nearest' (snap to closest beat), "
        "'downbeat' (prefer strong beats), 'onset' (align to transients/hits)."
    ),
    requires_project=True,
    modifies_gui_state=False
)
def align_sequence_to_audio(
    audio_path: str,
    strategy: str = "nearest",
    max_adjustment: float = 0.5
) -> dict:
    """Suggest beat-aligned adjustments for sequence clip cut points.

    Args:
        audio_path: Path to music/audio file to align with
        strategy: Alignment strategy - 'nearest', 'downbeat', or 'onset'
        max_adjustment: Maximum time shift in seconds (default 0.5)

    Returns:
        Dict with alignment suggestions for each clip
    """
    from pathlib import Path
    from core.analysis.audio import analyze_music_file, analyze_audio_from_video, has_audio_track
    from core.remix.audio_sync import suggest_beat_aligned_cuts

    # Validate strategy
    valid_strategies = ("nearest", "downbeat", "onset")
    if strategy not in valid_strategies:
        return {
            "success": False,
            "error": f"Invalid strategy '{strategy}'. Use: {valid_strategies}"
        }

    # Check sequence has clips
    if not project.sequence or not project.sequence.clips:
        return {
            "success": False,
            "error": "No clips in sequence. Add clips to the sequence first."
        }

    # Load and analyze audio
    path = Path(audio_path)
    if not path.exists():
        return {
            "success": False,
            "error": f"Audio file not found: {audio_path}"
        }

    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    try:
        if path.suffix.lower() in audio_extensions:
            audio_analysis = analyze_music_file(path)
        elif path.suffix.lower() in video_extensions:
            if not has_audio_track(path):
                return {
                    "success": False,
                    "error": f"Video file has no audio track: {audio_path}"
                }
            audio_analysis = analyze_audio_from_video(path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {path.suffix}"
            }

        # Build clip end times from sequence
        # Calculate cumulative end times based on clip durations
        clip_end_times = []
        current_time = 0.0

        for seq_clip in project.sequence.clips:
            # Find the source clip to get FPS
            source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
            source = project.sources_by_id.get(seq_clip.source_id)

            if source_clip and source:
                duration = source_clip.duration_seconds(source.fps)
                current_time += duration
                clip_end_times.append((seq_clip.id, current_time))

        if not clip_end_times:
            return {
                "success": False,
                "error": "Could not calculate clip durations. Check source clips exist."
            }

        # Get alignment suggestions
        suggestions = suggest_beat_aligned_cuts(
            clip_end_times=clip_end_times,
            audio_analysis=audio_analysis,
            strategy=strategy,
            max_adjustment=max_adjustment,
        )

        return {
            "success": True,
            "audio_file": str(path),
            "tempo_bpm": round(audio_analysis.tempo_bpm, 1),
            "strategy": strategy,
            "max_adjustment": max_adjustment,
            "sequence_clip_count": len(project.sequence.clips),
            "suggestions_count": len(suggestions),
            "suggestions": [s.to_dict() for s in suggestions],
            "message": (
                f"Found {len(suggestions)} clips that could be adjusted to align with "
                f"{audio_analysis.tempo_bpm:.1f} BPM beats (strategy: {strategy})"
            )
        }

    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Alignment analysis failed: {str(e)}"
        }


# =============================================================================
# Sequence Analysis Tools
# =============================================================================

@tools.register(
    description=(
        "Analyze the current sequence for pacing, continuity, and visual consistency. "
        "Returns shot duration statistics, potential continuity issues, and advisory suggestions. "
        "Optionally compare pacing to genre norms (action, drama, documentary, music_video)."
    ),
    requires_project=True,
    modifies_gui_state=False
)
def get_sequence_analysis(
    genre_comparison: Optional[str] = None
) -> dict:
    """Analyze the sequence for pacing and continuity metrics.

    Args:
        genre_comparison: Optional genre to compare pacing against
            (action, drama, documentary, music_video, commercial, art_film)

    Returns:
        Dict with pacing stats, continuity warnings, and suggestions
    """
    from core.analysis.sequence import analyze_sequence, get_pacing_curve
    from models.sequence_analysis import GENRE_PACING_NORMS

    # Check sequence exists
    if not project.sequence:
        return {
            "success": False,
            "error": "No sequence exists. Create a sequence first."
        }

    all_clips = project.sequence.get_all_clips()
    if not all_clips:
        return {
            "success": False,
            "error": "Sequence is empty. Add clips to the sequence first."
        }

    try:
        # Run analysis
        analysis = analyze_sequence(project.sequence, project)

        result = {
            "success": True,
            "sequence_name": project.sequence.name,
            "clip_count": analysis.pacing.clip_count,
            "pacing": analysis.pacing.to_dict(),
            "visual_consistency": analysis.visual_consistency.to_dict(),
            "continuity_warning_count": len(analysis.continuity_warnings),
            "continuity_warnings": [w.to_dict() for w in analysis.continuity_warnings],
            "suggestions": analysis.suggestions,
        }

        # Add genre comparison if requested
        if genre_comparison:
            genre_lower = genre_comparison.lower()
            if genre_lower in GENRE_PACING_NORMS:
                comparison = analysis.compare_to_genre(genre_lower)
                if comparison:
                    result["genre_comparison"] = comparison.to_dict()
            else:
                result["genre_comparison_error"] = (
                    f"Unknown genre '{genre_comparison}'. "
                    f"Valid genres: {list(GENRE_PACING_NORMS.keys())}"
                )

        # Add pacing curve for visualization
        result["pacing_curve"] = get_pacing_curve(project.sequence, project)

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Sequence analysis failed: {str(e)}"
        }


@tools.register(
    description=(
        "Check the current sequence for potential continuity issues. "
        "Detects similar consecutive shots (jump cuts), large shot size jumps, "
        "and other editing patterns that may need attention. "
        "All warnings are advisory - many 'issues' are valid artistic choices."
    ),
    requires_project=True,
    modifies_gui_state=False
)
def check_continuity_issues() -> dict:
    """Check sequence for potential continuity problems.

    Returns:
        Dict with list of continuity warnings and their severities
    """
    from core.analysis.sequence import check_continuity, _resolve_source_clips

    # Check sequence exists
    if not project.sequence:
        return {
            "success": False,
            "error": "No sequence exists. Create a sequence first."
        }

    all_clips = project.sequence.get_all_clips()
    if len(all_clips) < 2:
        return {
            "success": True,
            "message": "Need at least 2 clips to check continuity",
            "warning_count": 0,
            "warnings": []
        }

    try:
        # Resolve clips and check continuity
        resolved = _resolve_source_clips(all_clips, project)
        warnings = check_continuity(resolved)

        # Group by severity
        by_severity = {"low": 0, "medium": 0, "high": 0}
        for w in warnings:
            by_severity[w.severity] = by_severity.get(w.severity, 0) + 1

        return {
            "success": True,
            "sequence_name": project.sequence.name,
            "clip_count": len(all_clips),
            "warning_count": len(warnings),
            "warnings_by_severity": by_severity,
            "warnings": [w.to_dict() for w in warnings],
            "message": (
                f"Found {len(warnings)} potential continuity issues "
                f"({by_severity['high']} high, {by_severity['medium']} medium, {by_severity['low']} low)"
                if warnings else "No continuity issues detected"
            )
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Continuity check failed: {str(e)}"
        }


# =============================================================================
# Scene Report Tools
# =============================================================================

@tools.register(
    description=(
        "Generate a film analysis report for the current sequence or selected clips. "
        "Returns a markdown-formatted report with cinematography analysis, pacing metrics, "
        "continuity notes, and advisory suggestions. "
        "Sections: overview, cinematography, pacing, visual_consistency, continuity, recommendations, clip_details."
    ),
    requires_project=True,
    modifies_gui_state=False
)
def generate_analysis_report(
    sections: Optional[list[str]] = None,
    include_clip_details: bool = False,
    output_format: str = "markdown",
    clip_ids: Optional[list[str]] = None,
) -> dict:
    """Generate a film analysis report.

    Args:
        sections: Which sections to include (default: overview, cinematography, pacing, recommendations)
        include_clip_details: Whether to include per-clip breakdown
        output_format: 'markdown' (default) or 'html'
        clip_ids: Optional specific clips to report on (default: entire sequence)

    Returns:
        Dict with report content and metadata
    """
    from core.scene_report import (
        generate_sequence_report,
        generate_clips_report,
        report_to_html,
        REPORT_SECTIONS,
        DEFAULT_SECTIONS,
    )

    # Validate sections
    valid_sections = list(REPORT_SECTIONS.keys())
    if sections:
        invalid = [s for s in sections if s not in valid_sections]
        if invalid:
            return {
                "success": False,
                "error": f"Invalid sections: {invalid}. Valid sections: {valid_sections}"
            }
    else:
        sections = DEFAULT_SECTIONS

    # Validate output format
    if output_format not in ("markdown", "html"):
        return {
            "success": False,
            "error": f"Invalid output_format: {output_format}. Use 'markdown' or 'html'"
        }

    try:
        if clip_ids:
            # Report on specific clips
            clips = [project.clips_by_id.get(cid) for cid in clip_ids]
            clips = [c for c in clips if c is not None]

            if not clips:
                return {
                    "success": False,
                    "error": "No valid clips found for the provided IDs"
                }

            report = generate_clips_report(clips, project, title="Selected Clips Analysis")
        else:
            # Report on entire sequence
            if not project.sequence:
                return {
                    "success": False,
                    "error": "No sequence exists. Create a sequence first."
                }

            if not project.sequence.get_all_clips():
                return {
                    "success": False,
                    "error": "Sequence is empty. Add clips to the sequence first."
                }

            report = generate_sequence_report(
                project.sequence,
                project,
                sections=sections,
                include_clip_details=include_clip_details,
            )

        # Convert to HTML if requested
        if output_format == "html":
            report = report_to_html(report)

        # Calculate word count (for markdown)
        word_count = len(report.split()) if output_format == "markdown" else 0

        return {
            "success": True,
            "format": output_format,
            "sections_included": sections,
            "word_count": word_count,
            "report": report,
            "message": f"Generated {output_format} report with {len(sections)} sections"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Report generation failed: {str(e)}"
        }


# =============================================================================
# Frame Tools - Manage extracted/imported frames
# =============================================================================

@tools.register(
    description="Extract frames from a video source. Modes: 'interval' (every N frames), "
                "'all' (every frame from a clip), or 'smart' (key frames only). "
                "Optionally restrict to a specific clip.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def extract_frames(
    project, main_window,
    source_id: str,
    mode: str = "interval",
    interval: int = 10,
    clip_id: Optional[str] = None,
) -> dict:
    """Extract frames from a video source.

    Args:
        source_id: ID of the source video
        mode: Extraction mode - "all", "interval", or "smart"
        interval: Frame interval for "interval" mode (default 10)
        clip_id: Optional clip ID to extract frames from specific clip
    """
    valid_modes = ("all", "interval", "smart")
    if mode not in valid_modes:
        return {
            "success": False,
            "error": f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}"
        }

    source = project.sources_by_id.get(source_id)
    if not source:
        return {
            "success": False,
            "error": f"Source not found: {source_id}"
        }

    if clip_id:
        clip = project.clips_by_id.get(clip_id)
        if not clip:
            return {
                "success": False,
                "error": f"Clip not found: {clip_id}"
            }
        if clip.source_id != source_id:
            return {
                "success": False,
                "error": f"Clip {clip_id} does not belong to source {source_id}"
            }

    if interval < 1:
        return {
            "success": False,
            "error": "Interval must be at least 1"
        }

    # Return a marker for the GUI to start the async worker
    return {
        "success": True,
        "_wait_for_worker": "extract_frames",
        "_source_id": source_id,
        "_mode": mode,
        "_interval": interval,
        "_clip_id": clip_id,
        "message": f"Starting frame extraction from source '{source.file_path.name}' "
                   f"(mode={mode}, interval={interval})"
    }


@tools.register(
    description="Import image files as frames into the project.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def import_frames(project, file_paths: list[str]) -> dict:
    """Import image files as frames.

    Args:
        file_paths: List of absolute paths to image files
    """
    from models.frame import Frame
    import shutil

    if not file_paths:
        return {"success": False, "error": "No file paths provided"}

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    frames_created = []
    errors = []

    # Determine project frames directory
    project_dir = project.project_dir
    if project_dir is None:
        return {"success": False, "error": "Project must be saved before importing frames"}

    frames_dir = project_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for path_str in file_paths:
        valid, err_msg, resolved = _validate_path(path_str, must_exist=True)
        if not valid:
            errors.append(f"{path_str}: {err_msg}")
            continue

        if resolved.suffix.lower() not in valid_extensions:
            errors.append(f"{path_str}: Unsupported image format '{resolved.suffix}'")
            continue

        # Copy to project directory
        dest = frames_dir / resolved.name
        if dest.exists():
            stem = resolved.stem
            suffix = resolved.suffix
            counter = 1
            while dest.exists():
                dest = frames_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            shutil.copy2(str(resolved), str(dest))
        except OSError as e:
            errors.append(f"{path_str}: Copy failed: {e}")
            continue

        frame = Frame(file_path=dest)
        frames_created.append(frame)

    if frames_created:
        project.add_frames(frames_created)

    result = {
        "success": len(frames_created) > 0,
        "imported_count": len(frames_created),
        "frame_ids": [f.id for f in frames_created],
    }
    if errors:
        result["errors"] = errors
    return result


@tools.register(
    description="List frames in the project with optional filters. "
                "Filter by source_id, clip_id, shot_type, or whether frames have descriptions.",
    requires_project=True,
)
def list_frames(
    project,
    source_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    shot_type: Optional[str] = None,
    has_description: Optional[bool] = None,
) -> dict:
    """List frames in the project with optional filters.

    Args:
        source_id: Filter by source video ID
        clip_id: Filter by clip ID
        shot_type: Filter by shot type classification
        has_description: Filter by whether frame has a description
    """
    frames = list(project.frames)

    if source_id:
        frames = [f for f in frames if f.source_id == source_id]
    if clip_id:
        frames = [f for f in frames if f.clip_id == clip_id]
    if shot_type:
        frames = [f for f in frames if f.shot_type == shot_type]
    if has_description is not None:
        if has_description:
            frames = [f for f in frames if f.description]
        else:
            frames = [f for f in frames if not f.description]

    results = []
    for frame in frames:
        results.append({
            "id": frame.id,
            "display_name": frame.display_name(),
            "source_id": frame.source_id,
            "clip_id": frame.clip_id,
            "frame_number": frame.frame_number,
            "analyzed": frame.analyzed,
            "shot_type": frame.shot_type,
            "has_description": bool(frame.description),
            "tags": frame.tags,
        })

    return {
        "success": True,
        "frames": results,
        "count": len(results),
        "total_in_project": len(project.frames),
    }


@tools.register(
    description="Select frames in the Frames tab browser by their IDs.",
    requires_project=True,
    modifies_gui_state=True
)
def select_frames(project, frame_ids: list[str], gui_state=None) -> dict:
    """Select frames in the Frames tab browser.

    Args:
        frame_ids: List of frame IDs to select
    """
    if gui_state is None:
        return {"success": False, "error": "GUI state not available"}

    valid_ids = [fid for fid in frame_ids if fid in project.frames_by_id]
    invalid_ids = [fid for fid in frame_ids if fid not in project.frames_by_id]

    if invalid_ids:
        logger.warning(f"Invalid frame IDs for selection: {invalid_ids}")

    gui_state.selected_frame_ids = valid_ids

    return {
        "success": True,
        "selected": valid_ids,
        "invalid_ids": invalid_ids,
        "selection_count": len(valid_ids),
    }


@tools.register(
    description="Analyze frames with specified operations (describe, classify shot type, "
                "detect colors, detect objects). If no operations specified, runs all.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def analyze_frames(
    project, main_window,
    frame_ids: list[str],
    operations: Optional[list[str]] = None,
) -> dict:
    """Analyze frames with specified operations.

    Args:
        frame_ids: List of frame IDs to analyze
        operations: List of operations to run. Valid: "describe", "classify",
                    "colors", "objects". If None, runs all operations.
    """
    if not frame_ids:
        return {"success": False, "error": "No frame IDs provided"}

    valid_ops = {"describe", "classify", "colors", "objects"}
    if operations:
        invalid_ops = [op for op in operations if op not in valid_ops]
        if invalid_ops:
            return {
                "success": False,
                "error": f"Invalid operations: {invalid_ops}. Valid: {sorted(valid_ops)}"
            }
    else:
        operations = list(valid_ops)

    # Validate frame IDs
    valid_ids = [fid for fid in frame_ids if fid in project.frames_by_id]
    if not valid_ids:
        return {"success": False, "error": "No valid frame IDs provided"}

    return {
        "success": True,
        "_wait_for_worker": "analyze_frames",
        "_frame_ids": valid_ids,
        "_operations": operations,
        "message": f"Starting analysis of {len(valid_ids)} frames "
                   f"(operations: {', '.join(operations)})"
    }


@tools.register(
    description="Add frames to the timeline sequence with a configurable hold duration.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def add_frames_to_sequence(
    project,
    frame_ids: list[str],
    hold_frames: int = 1,
) -> dict:
    """Add frames to the timeline sequence.

    Args:
        frame_ids: List of frame IDs to add
        hold_frames: Number of timeline frames each frame occupies (default 1)
    """
    if not frame_ids:
        return {"success": False, "error": "No frame IDs provided"}

    if hold_frames < 1:
        return {"success": False, "error": "hold_frames must be at least 1"}

    # Validate frame IDs
    valid_ids = [fid for fid in frame_ids if fid in project.frames_by_id]
    invalid_ids = [fid for fid in frame_ids if fid not in project.frames_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid frame IDs provided"}

    project.add_frames_to_sequence(valid_ids, hold_frames=hold_frames)

    seq_length = len(project.sequence.get_all_clips()) if project.sequence else 0

    result = {
        "success": True,
        "added_count": len(valid_ids),
        "hold_frames": hold_frames,
        "sequence_length": seq_length,
        "message": f"Added {len(valid_ids)} frames to sequence "
                   f"(hold={hold_frames} frames each)"
    }
    if invalid_ids:
        result["invalid_ids"] = invalid_ids
    return result


@tools.register(
    description="Navigate to the Frames tab.",
    requires_project=False,
    modifies_gui_state=True
)
def navigate_to_frames_tab(gui_state=None) -> dict:
    """Navigate to the Frames tab."""
    if gui_state is None:
        return {"success": False, "error": "GUI state not available"}

    gui_state.active_tab = "frames"

    return {
        "success": True,
        "active_tab": "frames",
        "message": "Switched to Frames tab"
    }


@tools.register(
    description="Delete frames from the project by their IDs.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def delete_frames(project, frame_ids: list[str]) -> dict:
    """Delete frames from the project.

    Args:
        frame_ids: List of frame IDs to delete
    """
    if not frame_ids:
        return {"success": False, "error": "No frame IDs provided"}

    removed = project.remove_frames(frame_ids)

    return {
        "success": len(removed) > 0,
        "removed_count": len(removed),
        "removed_ids": [f.id for f in removed],
        "not_found": [fid for fid in frame_ids if fid not in {f.id for f in removed}],
        "remaining_frames": len(project.frames),
    }


@tools.register(
    description="Delete clips from the project by their IDs. Clips in the active sequence are rejected unless force=True.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def delete_clips(project, clip_ids: list[str], force: bool = False) -> dict:
    """Delete clips from the project.

    Args:
        clip_ids: List of clip IDs to delete
        force: If True, also remove clips from the sequence before deleting
    """
    if not clip_ids:
        return {"success": False, "error": "No clip IDs provided"}

    # Check if any clips are in the active sequence
    sequence_clip_ids = set()
    if project.sequence and project.sequence.tracks:
        for track in project.sequence.tracks:
            for sc in track.clips:
                if sc.source_clip_id:
                    sequence_clip_ids.add(sc.source_clip_id)

    in_sequence = [cid for cid in clip_ids if cid in sequence_clip_ids]
    if in_sequence and not force:
        return {
            "success": False,
            "error": f"{len(in_sequence)} clip(s) are in the active sequence. "
                     f"Use force=True to remove them from the sequence and delete, "
                     f"or use remove_from_sequence first.",
            "clips_in_sequence": in_sequence,
        }

    # If force, remove from sequence first
    if in_sequence and force:
        for track in project.sequence.tracks:
            track.clips = [
                sc for sc in track.clips
                if sc.source_clip_id not in set(clip_ids)
            ]

    removed = project.remove_clips(clip_ids)

    return {
        "success": len(removed) > 0,
        "removed_count": len(removed),
        "removed_ids": [c.id for c in removed],
        "not_found": [cid for cid in clip_ids if cid not in {c.id for c in removed}],
        "removed_from_sequence": len(in_sequence) if force else 0,
        "remaining_clips": len(project.clips),
    }