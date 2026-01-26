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
    description="Display a plan to the user for review and editing. Call this after breaking down a complex request into steps. The user can edit, reorder, or delete steps before confirming. Wait for user confirmation before executing the plan.",
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
    from models.plan import Plan

    if not steps:
        return {
            "success": False,
            "error": "No steps provided for the plan"
        }

    if len(steps) > 20:
        return {
            "success": False,
            "error": f"Plan has {len(steps)} steps, maximum is 20. Please break into smaller plans."
        }

    # Create the plan
    plan = Plan.from_steps(steps, summary)

    # Store plan in GUI state for tracking
    if hasattr(main_window, '_gui_state'):
        main_window._gui_state.current_plan = plan

    # Return marker that tells GUI handler to display the plan widget
    return {
        "_display_plan": True,
        "plan_id": plan.id,
        "summary": summary,
        "steps": steps,
        "step_count": len(steps),
        "message": "Plan presented to user. Waiting for confirmation or edits."
    }


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
        "path": str(project.path) if project.path else None,
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


# Aspect ratio tolerance ranges (5% tolerance) for filtering
ASPECT_RATIO_RANGES = {
    "16:9": (1.69, 1.87),   # 1.778 ± 5%
    "4:3": (1.27, 1.40),     # 1.333 ± 5%
    "9:16": (0.53, 0.59),    # 0.5625 ± 5%
}


@tools.register(
    description="Filter clips by criteria. Returns matching clips with their metadata. Available filters: shot_type (close_up, medium_shot, wide_shot, etc.), has_speech (true/false), min_duration (seconds), max_duration (seconds), aspect_ratio ('16:9', '4:3', '9:16').",
    requires_project=True,
    modifies_gui_state=False
)
def filter_clips(
    project,
    shot_type: Optional[str] = None,
    has_speech: Optional[bool] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    aspect_ratio: Optional[str] = None
) -> list[dict]:
    """Filter clips by various criteria."""
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
            "width": source.width if source else None,
            "height": source.height if source else None,
            "aspect_ratio": clip_aspect_ratio,
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
            "dominant_colors": getattr(clip, 'dominant_colors', None),
            "transcript": clip.get_transcript_text() if clip.transcript else None,
            "notes": getattr(clip, 'notes', None),
            "tags": getattr(clip, 'tags', []),
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
    valid_aspects = ["16:9", "4:3", "9:16"]
    if aspect_ratio and aspect_ratio not in valid_aspects:
        return {
            "success": False,
            "error": f"Invalid aspect_ratio '{aspect_ratio}'. Valid options: {', '.join(valid_aspects)}"
        }

    # Validate shot_type if provided
    valid_shots = ["All", "Wide Shot", "Medium Shot", "Close-up", "Extreme CU"]
    if shot_type and shot_type not in valid_shots:
        return {
            "success": False,
            "error": f"Invalid shot_type '{shot_type}'. Valid options: {', '.join(valid_shots[1:])}"
        }

    # Validate color_palette if provided
    valid_palettes = ["All", "Warm", "Cool", "Neutral", "Vibrant"]
    if color_palette and color_palette not in valid_palettes:
        return {
            "success": False,
            "error": f"Invalid color_palette '{color_palette}'. Valid options: {', '.join(valid_palettes[1:])}"
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
    valid_sorts = ["Timeline", "Color", "Duration"]
    if sort_by not in valid_sorts:
        return {
            "success": False,
            "error": f"Invalid sort_by '{sort_by}'. Valid options: {', '.join(valid_sorts)}"
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

    # Switch to Analyze tab
    main_window._switch_to_tab("analyze")

    # Update GUI state
    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.active_tab = "analyze"

    # Optionally start analysis
    if auto_analyze:
        main_window._on_analyze_all_from_tab()

    return {
        "success": True,
        "message": f"Sent {len(valid_ids)} clips to Analyze tab",
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

    player.player.play()

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

    player.player.pause()

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
    duration_ms = player.player.duration()
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
    modifies_gui_state=True
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
    modifies_gui_state=False
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

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
    description="Load a project from a .sceneripper or .json file. This replaces the current project.",
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
    modifies_gui_state=True
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
    modifies_gui_state=True
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
    modifies_gui_state=True
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
            "success": False,
            "error": "YouTube API key not configured. Add it in Settings > YouTube API Key.",
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


# =============================================================================
# GUI-Aware Tools - Trigger workers and wait for completion
# =============================================================================

@tools.register(
    description="Import a local video file into the project library. The video will appear in the Collect tab.",
    requires_project=True,
    modifies_gui_state=True
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
    description="Detect scenes in a video source with live GUI update. Updates the project with detected clips. "
                "This may take a while for long videos.",
    requires_project=True,
    modifies_gui_state=True
)
def detect_scenes_live(
    main_window,
    source_id: str,
    sensitivity: float = 3.0
) -> dict:
    """Detect scenes in a source video with live GUI update.

    Args:
        source_id: ID of the source to analyze
        sensitivity: Detection sensitivity (1.0=sensitive, 10.0=less sensitive)

    Returns:
        Dict with detected clip count and IDs (after worker completes)
    """
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

    # Set current source and start detection
    main_window._select_source(source)

    # Mark that we're waiting for detection result via agent
    main_window._pending_agent_detection = True

    # Start detection (this returns immediately, worker runs in background)
    main_window._start_detection(sensitivity)

    # Switch to Cut tab to show progress
    main_window._switch_to_tab("cut")

    # Return marker that tells GUI handler to wait for worker completion
    return {"_wait_for_worker": "detection", "source_id": source_id}


@tools.register(
    description="Detect scenes in all unanalyzed video sources. Queues all sources that haven't been analyzed yet "
                "and processes them sequentially. Equivalent to 'Cut New Videos' button in Collect tab.",
    requires_project=True,
    modifies_gui_state=True
)
def detect_all_unanalyzed(
    main_window,
    project,
    sensitivity: float = 3.0
) -> dict:
    """Detect scenes in all unanalyzed video sources.

    Args:
        sensitivity: Detection sensitivity (1.0=sensitive, 10.0=less sensitive)

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
        "message": f"Queued {len(unanalyzed)} sources for scene detection",
        "queued_count": len(unanalyzed),
        "source_ids": source_ids,
        "sensitivity": sensitivity,
    }


@tools.register(
    description="Extract dominant colors from clips with live GUI update. Updates clip metadata.",
    requires_project=True,
    modifies_gui_state=True
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
    description="Classify shot types (close-up, medium, wide, etc.) for clips with live GUI update.",
    requires_project=True,
    modifies_gui_state=True
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
    description="Transcribe speech in clips using Whisper with live GUI update.",
    requires_project=True,
    modifies_gui_state=True
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
    description="Run all analysis (colors, shots, transcription) on clips sequentially with live GUI update.",
    requires_project=True,
    modifies_gui_state=True
)
def analyze_all_live(main_window, clip_ids: list[str]) -> dict:
    """Run all analysis on clips with live GUI update.

    Runs colors, shots, and transcription sequentially.

    Args:
        clip_ids: List of clip IDs to analyze

    Returns:
        Dict with analysis summary (after all workers complete)
    """
    # Resolve clips
    clips = []
    for clip_id in clip_ids:
        clip = main_window.project.clips_by_id.get(clip_id)
        if clip:
            clips.append(clip)

    if not clips:
        return {"success": False, "error": "No valid clips found"}

    # Check if any analysis already running
    if main_window.color_worker and main_window.color_worker.isRunning():
        return {"success": False, "error": "Color analysis already in progress"}
    if main_window.shot_type_worker and main_window.shot_type_worker.isRunning():
        return {"success": False, "error": "Shot analysis already in progress"}
    if main_window.transcription_worker and main_window.transcription_worker.isRunning():
        return {"success": False, "error": "Transcription already in progress"}

    # Use existing analyze_all mechanism
    main_window._analyze_all_pending = ["colors", "shots", "transcribe"]
    main_window._analyze_all_clips = clips

    # Mark that agent is waiting for analyze_all completion
    main_window._pending_agent_analyze_all = True

    # Add clips to Analyze tab (mirrors "Analyze Selected" behavior)
    main_window.analyze_tab.add_clips(clip_ids)

    # Update UI state
    main_window.analyze_tab.set_analyzing(True, "all")

    # Switch to Analyze tab to show progress
    main_window._switch_to_tab("analyze")

    # Start the sequential analysis
    main_window._start_next_analyze_all_step()

    return {"_wait_for_worker": "analyze_all", "clip_count": len(clips)}


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

    algorithms = [
        {
            "key": "color",
            "name": "Color",
            "description": "Sort clips by dominant color along the color wheel",
            "available": has_colors,
            "reason": None if has_colors else "Run color analysis on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["rainbow", "warm_to_cool", "cool_to_warm"], "default": "rainbow"}
            ]
        },
        {
            "key": "duration",
            "name": "Duration",
            "description": "Sort clips by length (shortest or longest first)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["short_first", "long_first"], "default": "short_first"}
            ]
        },
        {
            "key": "shuffle",
            "name": "Shuffle",
            "description": "Randomize clip order with no repeating sources back-to-back",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "seed", "type": "integer", "description": "Random seed for reproducibility (0 = random)", "default": 0}
            ]
        },
        {
            "key": "sequential",
            "name": "Sequential",
            "description": "Keep clips in their original detection order",
            "available": True,
            "reason": None,
            "parameters": []
        },
    ]

    return {
        "algorithms": algorithms,
        "clip_count": len(clips),
        "has_color_analysis": has_colors,
    }


@tools.register(
    description="Generate a sequence using a sorting algorithm and apply it to the timeline. "
                "Available algorithms: color, duration, shuffle, sequential. "
                "Returns the generated sequence with clip details.",
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
        algorithm: One of "color", "duration", "shuffle", "sequential"
        clip_count: Number of clips to include (1-100)
        direction: For color: "rainbow", "warm_to_cool", "cool_to_warm"
                   For duration: "short_first", "long_first"
        seed: For shuffle: random seed for reproducibility (0 = random)

    Returns:
        Dict with success status, applied clips, and algorithm used
    """
    valid_algorithms = ["color", "duration", "shuffle", "sequential"]
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
