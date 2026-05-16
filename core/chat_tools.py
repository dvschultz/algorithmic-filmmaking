"""Tool definitions for the agent chat system.

Tools are registered via decorators and can be executed by the agent
during chat interactions. Tools are split into two categories:

1. GUI State Tools - Modify the live Project instance, triggering observer
   callbacks that update the UI in real-time.

2. CLI Tools - Execute batch operations via subprocess, suitable for
   long-running or heavy operations.
"""

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, get_type_hints

from core.youtube_api import (
    YouTubeSearchClient,
    YouTubeAPIError,
    QuotaExceededError,
    InvalidAPIKeyError,
)
from core.downloader import VideoDownloader
from core.settings import load_settings
from core.spine.security import validate_path
from core.constants import (
    VALID_ASPECT_RATIOS,
    VALID_COLOR_PALETTES,
    VALID_SHOT_TYPES,
    VALID_SORT_ORDERS,
)
from core.plan_controller import PlanController

logger = logging.getLogger(__name__)

def _get_plan_controller(main_window) -> PlanController:
    """Get the cached PlanController from main_window."""
    controller = getattr(main_window, 'plan_controller', None)
    if controller is None:
        raise AttributeError(
            "main_window.plan_controller not found — "
            "PlanController must be initialized during MainWindow setup"
        )
    return controller


# Agent-formatting helpers live in core.spine._agent_formatting so spine impls
# don't have to reach back into this module. Re-exported here as private names
# to keep legacy imports working.
from core.spine._agent_formatting import (
    add_sequence_summary_for_agent as _add_sequence_summary_for_agent,
    append_gaze_fields as _append_gaze_fields,
    clip_summary_for_agent as _clip_summary_for_agent,
    summarize_clip_sequence_for_agent as _summarize_clip_sequence_for_agent,
    summarize_report_for_agent as _summarize_report_for_agent,
    truncate_for_agent as _truncate_for_agent,
)


# Timeout values for tools (in seconds)
# Used by both CLI subprocess calls and GUI tool async workers
TOOL_TIMEOUTS = {
    "detect_scenes": 600,      # 10 minutes for large videos
    "detect_scenes_live": 600, # 10 minutes for large videos
    "download_video": 1800,    # 30 minutes for long videos
    "download_videos": 7200,   # 2 hours for bulk downloads (10 videos × 10 min timeout + buffer)
    "search_youtube": 30,      # 30 seconds
    "describe_content_live": 600,   # 10 minutes for descriptions
    "transcribe_clips": 1200,       # 20 minutes
    "export_sequence": 600,    # 10 minutes
    "export_bundle": 1800,     # 30 minutes (copies video files)
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
    conflicts_with_workers: bool = False
    emits_gui_sync: bool = False  # Tool results trigger GUI sync signals (e.g., populating search results)


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
        # Bare ``list`` annotation — emit a permissive but valid ``items``
        # schema. Vertex AI / Gemini reject array schemas without ``items``
        # (litellm.BadRequestError "missing field"). ``string`` is a safe
        # fallback for most agent use cases; tools that actually pass dicts
        # or numbers should annotate as ``list[dict]`` / ``list[int]``.
        return {"type": "array", "items": {"type": "string"}}

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
        modifies_project_state: bool = False,
        conflicts_with_workers: bool = False,
        emits_gui_sync: bool = False
    ):
        """Decorator to register a tool function.

        Args:
            description: Description shown to the LLM for tool selection
            requires_project: Whether this tool needs an active project
            modifies_gui_state: Whether this tool modifies the GUI state
                (GUI state tools should use core functions, not CLI)
            modifies_project_state: Whether this tool modifies project data
                (triggers auto-save after successful execution)
            conflicts_with_workers: Whether this tool conflicts with running
                GUI workers (e.g., DetectionWorker, DownloadWorker)
            emits_gui_sync: Whether tool results trigger GUI sync signals
                (e.g., populating search results or import panels)

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
                modifies_project_state=modifies_project_state,
                conflicts_with_workers=conflicts_with_workers,
                emits_gui_sync=emits_gui_sync
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
    modifies_gui_state=True
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
    from core.spine.queries import get_project_state as _impl
    return _impl(project)


@tools.register(
    description="List all video sources in the project with their metadata",
    requires_project=True,
    modifies_gui_state=False,
    modifies_project_state=False
)
def list_sources(project) -> dict:
    """List all video sources in the project."""
    from core.spine.sources import list_sources as _impl
    return _impl(project)


@tools.register(
    description=(
        "List all imported audio sources (music, podcasts, voiceovers) in the project. "
        "Audio sources are not cut into clips; they feed audio tools like Staccato and "
        "transcription."
    ),
    requires_project=True,
    modifies_gui_state=False,
    modifies_project_state=False,
)
def list_audio_sources(project) -> dict:
    """List all imported audio sources in the project."""
    from core.spine.audio_sources import list_audio_sources as _impl
    return _impl(project)


@tools.register(
    description=(
        "Get full details for a single audio source by ID, including its transcript "
        "if one has been generated."
    ),
    requires_project=True,
    modifies_gui_state=False,
    modifies_project_state=False,
)
def get_audio_source(project, audio_source_id: str) -> dict:
    """Return detailed information about an audio source.

    Args:
        audio_source_id: ID of the audio source (use list_audio_sources to find IDs).
    """
    from core.spine.audio_sources import get_audio_source as _impl
    return _impl(project, audio_source_id)


@tools.register(
    description=(
        "Import an audio file (mp3/wav/flac/m4a/aac/ogg) into the project. "
        "The file is added to the project's audio library and becomes available "
        "to Staccato and transcription. Returns the new audio source ID."
    ),
    requires_project=True,
    modifies_gui_state=False,
    modifies_project_state=True,
)
def import_audio_source(project, file_path: str) -> dict:
    """Synchronously import an audio file and add it to the project.

    Args:
        file_path: Absolute or project-relative path to the audio file.
    """
    from core.spine.audio_sources import import_audio_source as _impl
    return _impl(project, file_path)


@tools.register(
    description="Add clips to the timeline sequence by their IDs. Clips will appear in the Sequence tab.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def add_to_sequence(project, clip_ids: list[str]) -> dict:
    """Add clips to the timeline sequence."""
    # Validate clip IDs and exclude disabled clips
    valid_ids = [cid for cid in clip_ids if cid in project.clips_by_id and not project.clips_by_id[cid].disabled]
    disabled_ids = [cid for cid in clip_ids if cid in project.clips_by_id and project.clips_by_id[cid].disabled]
    invalid_ids = [cid for cid in clip_ids if cid not in project.clips_by_id]

    if invalid_ids:
        logger.warning(f"Invalid clip IDs: {invalid_ids}")

    if not valid_ids:
        return {
            "success": False,
            "error": "No valid clip IDs provided",
            "invalid_ids": invalid_ids,
            "disabled_ids": disabled_ids,
        }

    project.add_to_sequence(valid_ids)

    result = {
        "success": True,
        "added": valid_ids,
        "invalid_ids": invalid_ids,
        "sequence_length": len(project.sequence.tracks[0].clips) if project.sequence else 0,
    }
    if disabled_ids:
        result["disabled_ids"] = disabled_ids
    return result


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


def _append_gaze_fields(clip, clip_data: dict) -> None:
    """Append gaze fields to clip_data dict if present on the clip."""
    if clip.gaze_yaw is not None:
        clip_data["gaze_yaw"] = round(clip.gaze_yaw, 2)
    if clip.gaze_pitch is not None:
        clip_data["gaze_pitch"] = round(clip.gaze_pitch, 2)
    if clip.gaze_category is not None:
        clip_data["gaze_category"] = clip.gaze_category


@tools.register(
    description=(
        "Filter clips by criteria. Returns matching clips with their metadata. "
        "Available filters: shot_type, has_speech, min_duration, max_duration, "
        "aspect_ratio, search_query, has_object (case-insensitive substring, e.g., 'car' matches 'racecar'), "
        "min_people, max_people, search_description, has_faces, "
        "gaze_category (e.g., 'at_camera', 'looking_left'), "
        "min_brightness/max_brightness (0.0-1.0), "
        "search_ocr_text (substring in OCR text), "
        "min_volume/max_volume (dB, typically -60 to 0), "
        "search_tags (substring in tags), search_notes (substring in notes), "
        "cinematography_shot_size, cinematography_camera_angle, "
        "cinematography_camera_movement, cinematography_lighting_style, "
        "cinematography_subject_count, cinematography_emotional_intensity, "
        "cinematography_suggested_pacing (exact match on cinematography fields), "
        "similar_to_clip_id (rank results by DINOv2 embedding similarity to given clip). "
        "All filters combine with AND logic."
    ),
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
    has_faces: Optional[bool] = None,
    gaze_category: Optional[str] = None,
    min_brightness: Optional[float] = None,
    max_brightness: Optional[float] = None,
    search_ocr_text: Optional[str] = None,
    min_volume: Optional[float] = None,
    max_volume: Optional[float] = None,
    search_tags: Optional[str] = None,
    search_notes: Optional[str] = None,
    cinematography_shot_size: Optional[str] = None,
    cinematography_camera_angle: Optional[str] = None,
    cinematography_camera_movement: Optional[str] = None,
    cinematography_lighting_style: Optional[str] = None,
    cinematography_subject_count: Optional[str] = None,
    cinematography_emotional_intensity: Optional[str] = None,
    cinematography_suggested_pacing: Optional[str] = None,
    similar_to_clip_id: Optional[str] = None,
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
        has_object: Filter by object label substring (e.g., 'car' matches 'car' and 'racecar')
        min_people: Minimum number of people detected
        max_people: Maximum number of people detected
        search_description: Search text in visual descriptions
        has_faces: Filter by whether clip has detected faces
        gaze_category: Filter by gaze direction (e.g., 'at_camera', 'looking_left')
        min_brightness: Minimum average brightness (0.0-1.0)
        max_brightness: Maximum average brightness (0.0-1.0)
        search_ocr_text: Search text in OCR extracted text (case-insensitive substring)
        min_volume: Minimum RMS volume in dB
        max_volume: Maximum RMS volume in dB
        search_tags: Search text in clip tags (case-insensitive substring)
        search_notes: Search text in clip notes (case-insensitive substring)
        cinematography_shot_size: Filter by cinematography shot size (e.g., 'CU', 'MS')
        cinematography_camera_angle: Filter by camera angle (e.g., 'eye_level', 'low_angle')
        cinematography_camera_movement: Filter by camera movement (e.g., 'static', 'pan')
        cinematography_lighting_style: Filter by lighting style (e.g., 'dramatic', 'natural')
        cinematography_subject_count: Filter by subject count (e.g., 'single', 'group')
        cinematography_emotional_intensity: Filter by emotional intensity (e.g., 'low', 'high')
        cinematography_suggested_pacing: Filter by suggested pacing (e.g., 'fast', 'slow')
        similar_to_clip_id: Rank results by embedding similarity to this clip

    Returns:
        List of matching clips with metadata
    """
    from core.spine.clips import filter_clips as _impl
    return _impl(project, shot_type, has_speech, min_duration, max_duration, aspect_ratio, search_query, has_object, min_people, max_people, search_description, has_faces, gaze_category, min_brightness, max_brightness, search_ocr_text, min_volume, max_volume, search_tags, search_notes, cinematography_shot_size, cinematography_camera_angle, cinematography_camera_movement, cinematography_lighting_style, cinematography_subject_count, cinematography_emotional_intensity, cinematography_suggested_pacing, similar_to_clip_id)


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

    clips = [c for c in project.clips if not c.disabled]

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

        clip_data = {
            "id": clip.id,
            "source_id": clip.source_id,
            "source_name": source.file_path.name if source else "Unknown",
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "duration_seconds": round(duration, 2),
            "shot_type": getattr(clip, 'shot_type', None),
            "has_speech": bool(getattr(clip, 'transcript', None)),
            "has_word_alignment": bool(
                clip.transcript and any(getattr(seg, "words", None) is not None for seg in clip.transcript)
            ),
            "dominant_colors": getattr(clip, 'dominant_colors', None),
            "object_labels": getattr(clip, 'object_labels', None),
            "person_count": getattr(clip, 'person_count', None),
            "description": getattr(clip, 'description', None),
            "transcript": clip.get_transcript_text() if clip.transcript else None,
            "notes": getattr(clip, 'notes', None),
            "tags": getattr(clip, 'tags', []),
            "has_face_embeddings": clip.face_embeddings is not None and len(clip.face_embeddings) > 0,
            "face_count": len(clip.face_embeddings) if clip.face_embeddings else 0,
            "custom_queries": clip.custom_queries if clip.custom_queries else None,
        }
        _append_gaze_fields(clip, clip_data)
        results.append(clip_data)

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
                          f"({queue_remaining} sources queued).",
                "detection_in_progress": True
            }

        # Check if there are unanalyzed sources
        unanalyzed_count = sum(1 for s in project.sources if not s.analyzed)
        if unanalyzed_count > 0:
            return {
                "success": True,
                "clips": [],
                "count": 0,
                "message": f"No clips found. {unanalyzed_count} sources have not been analyzed.",
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
    description="Get the full cinematography analysis for a single clip. Returns all fields from "
                "the CinematographyAnalysis including shot size, camera angle, movement, composition, "
                "lighting, and emotional properties. Requires the clip to have been analyzed with "
                "the cinematography analyzer.",
    requires_project=True,
    modifies_gui_state=False,
    modifies_project_state=False
)
def get_clip_cinematography(project, clip_id: str) -> dict:
    """Get the full CinematographyAnalysis for a single clip.

    Args:
        clip_id: ID of the clip to read cinematography data from

    Returns:
        Dict with success status and all cinematography fields
    """
    from core.spine.clips import get_clip_cinematography as _impl
    return _impl(project, clip_id)


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
    description="Update a sequence clip's trim points, position, or transform flags on the timeline. "
                "Use this to trim clips (in_point/out_point), reposition them (start_frame), "
                "change their track (track_index), set hold duration for frame entries (hold_frames), "
                "or toggle transforms (hflip, vflip, reverse).",
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
    hflip: Optional[bool] = None,
    vflip: Optional[bool] = None,
    reverse: Optional[bool] = None,
) -> dict:
    """Update a sequence clip's trim points, position, or transform flags.

    Args:
        clip_id: ID of the sequence clip to update
        in_point: New trim start (frames into source clip)
        out_point: New trim end (frames into source clip)
        start_frame: New position on timeline (in frames)
        track_index: Move to a different track
        hold_frames: For frame entries, number of timeline frames to hold
        hflip: Horizontal flip transform
        vflip: Vertical flip transform
        reverse: Reverse playback transform

    Note:
        Setting hflip/vflip/reverse only updates the boolean flags on the
        SequenceClip. The pre-rendered clip (prerendered_path) is NOT
        regenerated here — that requires a background worker. After changing
        transform flags, the prerendered_path is cleared so the render
        pipeline knows it must be regenerated before export.

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

    if hflip is not None:
        if not isinstance(hflip, bool):
            return {"success": False, "error": f"hflip must be a boolean, got {type(hflip).__name__}"}
        target_clip.hflip = hflip
        # Invalidate pre-rendered clip since transforms changed
        target_clip.prerendered_path = None
        updated_fields["hflip"] = hflip

    if vflip is not None:
        if not isinstance(vflip, bool):
            return {"success": False, "error": f"vflip must be a boolean, got {type(vflip).__name__}"}
        target_clip.vflip = vflip
        target_clip.prerendered_path = None
        updated_fields["vflip"] = vflip

    if reverse is not None:
        if not isinstance(reverse, bool):
            return {"success": False, "error": f"reverse must be a boolean, got {type(reverse).__name__}"}
        target_clip.reverse = reverse
        target_clip.prerendered_path = None
        updated_fields["reverse"] = reverse

    if not updated_fields:
        return {"success": False, "error": "No fields provided to update"}

    # Notify observers
    project.mark_dirty()
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
    description=(
        "Filter the sequence tab to show only clips with a specific gaze direction. "
        "Valid gaze categories: 'at_camera', 'looking_left', 'looking_right', "
        "'looking_up', 'looking_down'. "
        "Use gaze_category=None to show all clips (remove filter)."
    ),
    requires_project=True,
    modifies_gui_state=True,
)
def set_sequence_gaze_filter(
    project,
    gui_state,
    main_window,
    gaze_category: Optional[str] = None,
) -> dict:
    """Set the gaze direction filter for the sequence tab."""
    from core.analysis.gaze import GAZE_CATEGORIES

    if gaze_category and gaze_category not in GAZE_CATEGORIES:
        return {
            "success": False,
            "error": f"Invalid gaze category '{gaze_category}'. Valid categories: {list(GAZE_CATEGORIES)}",
        }

    sequence_tab = main_window.sequence_tab
    filtered_count = sequence_tab.apply_gaze_filter(gaze_category)

    gui_state.sequence_gaze_filter = gaze_category

    return {
        "success": True,
        "result": {
            "gaze_category": gaze_category or "all",
            "clip_count": filtered_count,
            "message": (
                f"Showing {filtered_count} clips"
                + (f" with gaze '{gaze_category}'" if gaze_category else " (all gaze directions)")
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
    from core.spine.queries import get_sequence_state as _impl
    return _impl(project)


@tools.register(
    description="Update sequence-level metadata such as name, fps, music path, or allow_repeats. "
                "Does not affect the clips in the sequence.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def update_sequence(
    project,
    name: Optional[str] = None,
    fps: Optional[float] = None,
    music_path: Optional[str] = None,
    allow_repeats: Optional[bool] = None,
) -> dict:
    """Update sequence-level metadata.

    Args:
        name: New sequence name
        fps: New sequence frame rate
        music_path: Path to music file for the sequence
        allow_repeats: Whether to allow repeated clips in the sequence

    Returns:
        Dict with success status and updated fields
    """
    if project.sequence is None:
        return {"success": False, "error": "No sequence exists. Use create_sequence first."}

    updated_fields = {}

    if name is not None:
        project.sequence.name = name
        updated_fields["name"] = name

    if fps is not None:
        if fps <= 0:
            return {"success": False, "error": f"fps must be > 0, got {fps}"}
        project.sequence.fps = fps
        updated_fields["fps"] = fps

    if music_path is not None:
        is_valid, err_msg, validated_path = validate_path(music_path, must_exist=True)
        if not is_valid:
            return {"success": False, "error": err_msg}
        project.sequence.music_path = str(validated_path)
        updated_fields["music_path"] = str(validated_path)

    if allow_repeats is not None:
        project.sequence.allow_repeats = allow_repeats
        updated_fields["allow_repeats"] = allow_repeats

    if not updated_fields:
        return {"success": False, "error": "No fields provided to update"}

    project.mark_dirty()
    project._notify_observers("sequence_changed", [])

    return {
        "success": True,
        "message": f"Updated sequence: {', '.join(updated_fields.keys())}",
        "updated_fields": updated_fields,
    }


@tools.register(
    description="Create a new empty sequence, replacing any existing one. "
                "Optionally set a name and frame rate.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def create_sequence(
    project,
    name: Optional[str] = None,
    fps: Optional[float] = None,
) -> dict:
    """Create a new empty sequence.

    Args:
        name: Sequence name (defaults to project name)
        fps: Frame rate (defaults to first source FPS or 30.0)

    Returns:
        Dict with success status and sequence info
    """
    from models.sequence import Sequence

    seq_name = name or project.metadata.name or "Untitled Sequence"
    seq_fps = fps or (project.sources[0].fps if project.sources else 30.0)

    if seq_fps <= 0:
        return {"success": False, "error": f"fps must be > 0, got {seq_fps}"}

    new_seq = Sequence(name=seq_name, fps=seq_fps)
    project.add_sequence(new_seq)
    project.set_active_sequence(len(project.sequences) - 1)

    return {
        "success": True,
        "message": f"Created sequence '{seq_name}' at {seq_fps} fps (now active)",
        "name": seq_name,
        "fps": seq_fps,
        "sequence_index": project.active_sequence_index,
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
                "Filters clips by duration range, aspect ratio, shot type, color palette, "
                "transcript search, gaze direction, object labels, and/or description text. "
                "Use clear_all=True to reset all filters.",
    requires_project=True,
    modifies_gui_state=True
)
def _validate_enum_arg(value, valid_set, field_name: str):
    """Accept str or list[str] for an agent-tool enum filter. Return (ok_value, error)."""
    if value is None:
        return None, None
    # Normalise single-string and list/tuple inputs to a list
    if isinstance(value, str):
        items = [value] if value not in ("", "All") else []
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v not in (None, "", "All")]
    else:
        return None, f"Invalid {field_name}: expected str or list[str], got {type(value).__name__}"
    for item in items:
        if item not in valid_set:
            return None, (
                f"Invalid {field_name} value '{item}'. "
                f"Valid options: {', '.join(sorted(valid_set))}"
            )
    return items if len(items) != 1 else items[0], None


def apply_filters(
    main_window,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    aspect_ratio=None,
    shot_type=None,
    color_palette=None,
    search_query: Optional[str] = None,
    gaze=None,
    object_search: Optional[str] = None,
    description_search: Optional[str] = None,
    clear_all: bool = False,
) -> dict:
    """Apply filters to the clip browser in the active tab.

    Args:
        min_duration: Minimum duration in seconds (None = no minimum)
        max_duration: Maximum duration in seconds (None = no maximum)
        aspect_ratio: Filter by aspect ratio — str, list[str], or None.
          Valid values: '16:9', '4:3', '9:16'. Multi-select chips use list.
        shot_type: Filter by shot type — str, list[str], or None.
          Valid values: 'Wide Shot', 'Medium Shot', 'Close-up', 'Extreme CU'.
        color_palette: Filter by color palette — str, list[str], or None.
          Valid values: 'Warm', 'Cool', 'Neutral', 'Vibrant'.
        search_query: Filter by transcript text (case-insensitive substring search)
        gaze: Filter by gaze direction — str, list[str], or None.
          Valid internal keys: 'at_camera', 'looking_left', 'looking_right',
          'looking_up', 'looking_down'. Display labels also accepted.
        object_search: Filter by detected object labels (case-insensitive substring)
        description_search: Filter by clip description text (case-insensitive substring)
        clear_all: If True, clears all filters instead of applying new ones

    Returns:
        Dict with success status, active filters, and clip counts. Note that
        `active_filters` may contain a `list[str]` for multi-select enum fields
        — feed that back directly into a subsequent call.
    """
    if main_window is None:
        return {"success": False, "error": "Main window not available"}

    # Validate enum args (accept str, list[str], or None)
    aspect_ratio, err = _validate_enum_arg(aspect_ratio, VALID_ASPECT_RATIOS, "aspect_ratio")
    if err:
        return {"success": False, "error": err}
    shot_type, err = _validate_enum_arg(shot_type, VALID_SHOT_TYPES, "shot_type")
    if err:
        return {"success": False, "error": err}
    color_palette, err = _validate_enum_arg(color_palette, VALID_COLOR_PALETTES, "color_palette")
    if err:
        return {"success": False, "error": err}

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
    if gaze is not None:
        filters['gaze'] = gaze
    if object_search is not None:
        filters['object_search'] = object_search
    if description_search is not None:
        filters['description_search'] = description_search

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

    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.update_playback_state(is_playing=True)

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

    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.update_playback_state(is_playing=False)

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
    modifies_gui_state=True
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
    from core.sequence_export import ExportConfig

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
        valid, error, validated_path = validate_path(output_path)
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
        show_chromatic_color_bar=(
            bool(getattr(sequence, "show_chromatic_color_bar", False))
            and sequence.algorithm == "color"
        ),
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
    from core.spine.exports import export_edl as _impl
    return _impl(project, output_path)


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
    from core.spine.exports import export_dataset as _impl
    return _impl(project, output_path, include_thumbnails, source_id)


@tools.register(
    description="Set or update the project name. Use this when the project is unnamed ('Untitled Project') to give it a meaningful name. The project will be automatically saved after naming.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def set_project_name(main_window, project, name: str) -> dict:
    """Set the project name.

    This only renames the project. Call save_project separately to persist.
    """
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

    # Refresh the window title to reflect the new project name
    if hasattr(main_window, '_update_window_title'):
        main_window._update_window_title()

    return {
        "success": True,
        "old_name": old_name,
        "new_name": project.metadata.name,
        "message": f"Project renamed to '{project.metadata.name}'",
        "needs_save": not bool(project.path),
    }


@tools.register(
    description="Save the current project to disk. Uses the existing path if project was previously saved, or saves to the export directory for new projects.",
    requires_project=True,
    modifies_gui_state=False
)
def save_project(project, path: Optional[str] = None) -> dict:
    """Save project state to JSON file."""
    from core.spine.project_save import save_project as _impl
    return _impl(project, path)


@tools.register(
    description="Load a project from a .sceneripper file. This replaces the current project.",
    requires_project=False,
    modifies_gui_state=True
)
def load_project(path: str, main_window=None) -> dict:
    """Load project from file."""
    from core.project import Project, ProjectLoadError, MissingSourceError

    valid, error, validated_path = validate_path(path, must_be_file=True)
    if not valid:
        return {"success": False, "error": f"Invalid path: {error}"}

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
    from core.spine.queries import get_project_summary as _impl
    return _impl(project)


# =============================================================================
# CLI Tools - Execute via subprocess for batch operations
# =============================================================================

@tools.register(
    description="Detect scenes in a video file and add clips to the project. Creates clips from detected scene boundaries. Returns the clip count and clip IDs.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True,
    conflicts_with_workers=True
)
def detect_scenes(
    project,
    video_path: str,
    sensitivity: float = 3.0,
    luma_only: bool | None = None,
) -> dict:
    """Run scene detection using Python API and add clips to project."""
    valid, error, video = validate_path(video_path, must_be_file=True)
    if not valid:
        return {"success": False, "error": error}

    try:
        from core.spine.detect import detect_scenes_for_video

        result = detect_scenes_for_video(
            project, video, sensitivity=sensitivity, luma_only=luma_only
        )
        if not result.get("success"):
            err = result.get("error", {})
            return {"success": False, "error": err.get("message") or err.get("code") or "detection failed"}

        payload = result["result"]
        source = project.sources_by_id.get(payload["source_id"])
        clips = [c for c in project.clips if c.source_id == payload["source_id"]]

        if payload["clip_count"] == 0:
            return {
                "success": True,
                "clips_detected": 0,
                "source_id": payload["source_id"],
                "source_name": payload["source_name"],
                "message": f"No scene cuts detected in {payload['source_name']}. "
                           "Try a lower sensitivity value, or the video may be a single continuous shot.",
            }

        return {
            "success": True,
            "clips_detected": payload["clip_count"],
            "clip_ids": payload["clip_ids"],
            "source_id": payload["source_id"],
            "source_name": payload["source_name"],
            "detected_clips": [
                _clip_summary_for_agent(project, clip, source)
                for clip in clips[:20]
            ],
            "response_guidance": (
                "Summarize scene detection using only the detected clip IDs, "
                "source name, and timing ranges. Do not invent clip descriptions."
            ),
            "is_fallback_clip": False,
            "message": f"Detected {payload['clip_count']} scenes in {payload['source_name']} and added to project",
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
    modifies_gui_state=False,
    emits_gui_sync=True
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
    description="Search the Internet Archive for videos matching a query. "
                "Returns video titles, identifiers, durations, and URLs. "
                "No API key required. Results include movies, feature films, "
                "short films, and animation.",
    requires_project=False,
    modifies_gui_state=False,
    modifies_project_state=False
)
def search_internet_archive(
    query: str,
    max_results: int = 10,
) -> dict:
    """Search Internet Archive for videos.

    Args:
        query: Search term
        max_results: Maximum number of results (default 10, max 50)

    Returns:
        Dict with success status and list of matching videos
    """
    from core.internet_archive_api import InternetArchiveClient, InternetArchiveError

    if not query.strip():
        return {"success": False, "error": "Query cannot be empty"}

    try:
        client = InternetArchiveClient()
        results = client.search(query, max_results=min(max_results, 50))

        videos = []
        for video in results:
            videos.append({
                "identifier": video.identifier,
                "title": video.title,
                "description": video.description,
                "creator": video.creator,
                "date": video.date,
                "duration": video.duration_str,
                "duration_seconds": video.duration_seconds,
                "url": video.item_url,
                "download_url": video.download_url,
                "thumbnail_url": video.thumbnail_url,
            })

        return {
            "success": True,
            "query": query,
            "results": videos,
            "count": len(videos),
        }

    except InternetArchiveError as e:
        return {"success": False, "error": f"Internet Archive search failed: {e}"}
    except Exception as e:
        logger.exception("Internet Archive search failed")
        return {"success": False, "error": f"Search failed: {e}"}


@tools.register(
    description="Download a video from YouTube or Vimeo URL. Returns the downloaded file path. Uses the default download directory from settings unless output_dir is specified.",
    requires_project=False,
    modifies_gui_state=False,
    conflicts_with_workers=True,
    emits_gui_sync=True
)
def download_video(url: str, output_dir: Optional[str] = None) -> dict:
    """Download video using the Python API."""
    if output_dir:
        valid, error, validated_dir = validate_path(output_dir)
        if not valid:
            return {"error": f"Invalid output directory: {error}"}
        download_path = validated_dir
    else:
        settings = load_settings()
        download_path = settings.download_dir

    try:
        from core.spine.downloads import download_videos as _impl

        result = _impl([url], download_path)
        if not result.get("success"):
            err = result.get("error", {})
            return {"error": err.get("message") or err.get("code") or "Download failed"}

        payload = result["result"]
        if payload["succeeded"]:
            entry = payload["succeeded"][0]
            return {
                "success": True,
                "file_path": entry["file_path"],
                "title": entry["title"],
                "duration": entry["duration"],
                "message": f"Downloaded: {entry['title']}",
            }

        if payload["failed"]:
            entry = payload["failed"][0]
            return {"error": entry.get("error_message") or "Download failed"}

        return {"error": "Download did not produce a result"}

    except RuntimeError as e:
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
        valid, error, validated_dir = validate_path(output_dir)
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
# Use start_clip_analysis or analyze_all_live instead, which work directly
# with the in-memory project.
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
    valid, error, video_path = validate_path(path, must_be_file=True)
    if not valid:
        return {"success": False, "error": error}

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


@tools.register(
    description="Import all video files from a folder. Scans the directory for supported "
                "video formats (.mp4, .mkv, .avi, .mov, .webm) and imports each one.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def import_folder(project, main_window, folder_path: str) -> dict:
    """Import all video files from a folder.

    Args:
        folder_path: Path to the folder to scan for video files

    Returns:
        Dict with success status, imported count, and file list
    """
    from pathlib import Path

    folder = Path(folder_path)
    if not folder.is_dir():
        return {"success": False, "error": f"Not a directory: {folder_path}"}

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv"}
    video_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    )

    if not video_files:
        return {"success": False, "error": f"No video files found in {folder_path}"}

    imported = []
    skipped = []
    for vf in video_files:
        # Check if already imported
        existing = next(
            (s for s in project.sources if Path(s.file_path).resolve() == vf.resolve()),
            None
        )
        if existing:
            skipped.append(vf.name)
            continue

        from core.scene_detect import load_source
        source = load_source(str(vf))
        if source:
            project.add_source(source)
            imported.append(vf.name)

    return {
        "success": True,
        "imported_count": len(imported),
        "imported_files": imported,
        "skipped_count": len(skipped),
        "skipped_files": skipped,
        "total_sources": len(project.sources),
    }


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
        "message": f"Queued {len(unanalyzed)} sources for scene detection.",
        "queued_count": len(unanalyzed),
        "source_ids": source_ids,
        "sensitivity": sensitivity,
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
    else:
        unanalyzed = total_count - analyzed_count
        if unanalyzed > 0:
            message = f"Detection idle. {unanalyzed} sources not yet analyzed."
        else:
            message = f"All {analyzed_count} sources analyzed. {clip_count} clips available."

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
        "description", "objects", "people", "face_embeddings",
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
        "face_embeddings": "face_detection",
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
            "face_embeddings": ("face_detection_worker", "Face detection"),
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
        "face_embeddings": "face_embeddings",
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
    description="Run analysis operations on clips with live GUI update. "
                "Supports smart concurrency: local ops run in parallel, then sequential, then cloud. "
                "Default operations: colors, shots, transcribe. "
                "Available operations: colors, shots, classify, detect_objects, face_embeddings, extract_text, transcribe, describe, cinematography.",
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


@tools.register(
    description="Run the Custom Visual Query analysis operation across clips. "
                "Use this tool when the user asks to visually search for something "
                "inside clips, such as 'eye', 'blue flower', or 'person wearing a hat'. "
                "This is NOT a sorting/sequencing algorithm and NOT a metadata-only "
                "description search. It runs a VLM yes/no query on clip thumbnails and "
                "returns actual match/non-match results with confidence. Results "
                "accumulate on clips.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def custom_visual_query(
    main_window,
    query: str,
    clip_ids: Optional[list[str]] = None,
) -> dict:
    """Run a custom visual query on clips.

    Args:
        query: Natural language visual query (e.g., 'blue flower', 'outdoor scene')
        clip_ids: Optional list of clip IDs to query. If None, queries all clips
            in the Analyze tab.

    Returns:
        Dict with _wait_for_worker marker for async execution.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Query text is required"}

    # Resolve clips
    if clip_ids:
        valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]
        if not valid_ids:
            return {"success": False, "error": "No valid clip IDs found"}
        clips = [main_window.project.clips_by_id[cid] for cid in valid_ids]
    else:
        clips = main_window.analyze_tab.get_clips()
        if not clips:
            return {
                "success": False,
                "error": "No clips in Analyze tab. Send clips to Analyze first."
            }

    # Store the query text so the worker launch can access it
    main_window._custom_query_text = query.strip()

    return {
        "_wait_for_worker": "analyze_all",
        "clip_ids": [c.id for c in clips],
        "clip_count": len(clips),
        "operations": ["custom_query"],
    }


@tools.register(
    description="Detect gaze direction (where subjects are looking) for selected clips "
                "using MediaPipe Face Mesh iris tracking. Produces yaw/pitch angles and "
                "categorical labels (at_camera, looking_left, looking_right, looking_up, "
                "looking_down).",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def analyze_gaze(
    main_window,
    clip_ids: Optional[list[str]] = None,
) -> dict:
    """Run gaze direction analysis on clips.

    Args:
        clip_ids: List of clip IDs to analyze. If None, analyzes all clips
            in the Analyze tab.

    Returns:
        Dict with _wait_for_worker marker for async execution.
    """
    # Resolve clips
    if clip_ids:
        valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]
        if not valid_ids:
            return {"success": False, "error": "No valid clip IDs found"}
    else:
        clips = main_window.analyze_tab.get_clips()
        if not clips:
            return {
                "success": False,
                "error": "No clips in Analyze tab. Send clips to Analyze first."
            }
        valid_ids = [c.id for c in clips]

    # Return marker — GUI layer handles tab switch, clip loading, and pipeline start
    return {
        "_wait_for_worker": "analyze_all",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "operations": ["gaze"],
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
    from core.spine.settings_io import list_sorting_algorithms as _impl
    return _impl(project)


@tools.register(
    description="Generate a sequence using a sorting algorithm and apply it to the timeline. "
                "Available algorithms: color, duration, brightness, volume, "
                "shuffle, sequential, shot_type, proximity, similarity_chain, match_cut, "
                "exquisite_corpus, storyteller, gaze_sort, gaze_consistency. "
                "Use list_sorting_algorithms to check availability.",
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
    no_color_handling: Optional[str] = None,
    random_hflip: bool = False,
    random_vflip: bool = False,
    random_reverse: bool = False,
) -> dict:
    """Generate a sequence using the specified algorithm and apply to timeline.

    Args:
        algorithm: One of the sorting algorithms (e.g. "color", "brightness",
                   "similarity_chain", "match_cut", etc.)
        clip_count: Number of clips to include (1-100)
        direction: Algorithm-specific direction (e.g. "rainbow", "complementary",
                   "short_first", "light_to_dark", "quiet_to_loud", "wide_to_close")
        seed: For shuffle: random seed for reproducibility (0 = random)
        no_color_handling: For color algorithm — how to handle clips without color data.
                   "append_end" (default), "exclude", or "sort_inline"
        random_hflip: For shuffle: randomly flip ~50% of clips horizontally at export
        random_vflip: For shuffle: randomly flip ~50% of clips vertically at export
        random_reverse: For shuffle: randomly reverse ~50% of clips at export

    Returns:
        Dict with success status, applied clips, and algorithm used
    """
    valid_algorithms = [
        "color", "duration", "brightness", "volume",
        "shuffle", "sequential", "shot_type", "proximity",
        "similarity_chain", "match_cut", "exquisite_corpus", "storyteller",
        "gaze_sort", "gaze_consistency",
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
                "error": "Color sorting requires color analysis. Run start_clip_analysis(operations=['colors']) first."
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

    # Build transform options if any are enabled
    transform_options = None
    if random_hflip or random_vflip or random_reverse:
        transform_options = {
            "hflip": random_hflip,
            "vflip": random_vflip,
            "reverse": random_reverse,
        }

    # Use sequence tab's generate_and_apply method
    result = main_window.sequence_tab.generate_and_apply(
        algorithm=algorithm,
        clip_count=clip_count,
        direction=direction,
        seed=seed,
        no_color_handling=no_color_handling,
        transform_options=transform_options,
    )
    _add_sequence_summary_for_agent(project, result)

    return result


@tools.register(
    description="Generate an Eyes Without a Face sequence using gaze-based algorithms. "
                "Three modes: 'eyeline_match' (shot/reverse-shot pairing by negated yaw), "
                "'gaze_filter' (keep clips matching a gaze category), "
                "'gaze_rotation' (arrange clips in monotonic angle progression). "
                "Requires gaze analysis. Use analyze_clips with 'gaze' first.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_eyes_without_a_face(
    project,
    main_window,
    mode: str = "eyeline_match",
    tolerance: float = 20.0,
    category: Optional[str] = None,
    axis: str = "yaw",
    range_start: float = -30.0,
    range_end: float = 30.0,
    ascending: bool = True,
) -> dict:
    """Generate an Eyes Without a Face gaze-based sequence.

    Args:
        mode: 'eyeline_match', 'gaze_filter', or 'gaze_rotation'
        tolerance: For eyeline_match — max abs(yaw_a + yaw_b) to pair (default 20.0)
        category: For gaze_filter — gaze category to keep ('at_camera', 'looking_left',
                  'looking_right', 'looking_up', 'looking_down')
        axis: For gaze_rotation — 'yaw' or 'pitch' (default 'yaw')
        range_start: For gaze_rotation — minimum angle in degrees (default -30.0)
        range_end: For gaze_rotation — maximum angle in degrees (default 30.0)
        ascending: For gaze_rotation — True for ascending angle order

    Returns:
        Dict with success status and applied clip info
    """
    valid_modes = ("eyeline_match", "gaze_filter", "gaze_rotation")
    if mode not in valid_modes:
        return {
            "success": False,
            "error": f"Invalid mode '{mode}'. Valid options: {', '.join(valid_modes)}"
        }

    if mode == "gaze_filter" and not category:
        return {
            "success": False,
            "error": "category is required for gaze_filter mode "
                     "(e.g. 'at_camera', 'looking_left', 'looking_right')"
        }

    valid_categories = ("at_camera", "looking_left", "looking_right", "looking_up", "looking_down")
    if category and category not in valid_categories:
        return {
            "success": False,
            "error": f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}"
        }

    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {"success": False, "error": "Main window not available"}

    # Get selected clips
    gui_state = main_window._gui_state
    selected_ids = []
    if gui_state:
        selected_ids = gui_state.analyze_selected_ids or gui_state.cut_selected_ids or []

    if not selected_ids:
        return {"success": False, "error": "No clips selected in Analyze or Cut tab"}

    seq_tab = main_window.sequence_tab
    clips = seq_tab._resolve_selected_clips(selected_ids)
    if not clips:
        return {"success": False, "error": "Selected clips not available for sequencing"}

    has_gaze = any(clip.gaze_category is not None for clip, _ in clips)
    if not has_gaze:
        return {"success": False, "error": "No clips have gaze data. Run gaze analysis first."}

    from core.remix.gaze import eyeline_match, gaze_filter, gaze_rotation

    try:
        if mode == "eyeline_match":
            sorted_clips = eyeline_match(clips, tolerance=tolerance)
        elif mode == "gaze_filter":
            sorted_clips = gaze_filter(clips, category=category)
        else:
            sorted_clips = gaze_rotation(
                clips, axis=axis, range_start=range_start,
                range_end=range_end, ascending=ascending,
            )

        seq_tab.timeline.clear_timeline()
        current_frame = 0
        for clip, source in sorted_clips:
            seq_tab.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
            current_frame += clip.duration_frames
        seq_tab.timeline._on_zoom_fit()
        seq_tab._set_state(seq_tab.STATE_TIMELINE)

        return _add_sequence_summary_for_agent(project, {
            "success": True,
            "algorithm": f"eyes_without_a_face ({mode})",
            "clip_count": len(sorted_clips),
            "mode": mode,
        }, sorted_clips)
    except Exception as e:
        return {"success": False, "error": str(e)}


@tools.register(
    description="Check which matching dimensions have analysis data for the current clips. "
                "Call this before generate_reference_guided to discover which dimension weights "
                "are valid (e.g. color requires color analysis, embedding requires embeddings).",
    requires_project=True,
    modifies_gui_state=True
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
    description="Generate an Exquisite Corpus poem-sequence from clips with extracted text. "
                "Requires text extraction analysis. The LLM arranges on-screen text fragments "
                "into a poem following the specified mood, length, and form.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_exquisite_corpus(
    project,
    main_window,
    mood: str = "dreamy and contemplative",
    length: str = "medium",
    form: str = "free_verse",
) -> dict:
    """Generate an Exquisite Corpus poem-sequence.

    Args:
        mood: Description of the desired mood/vibe for the poem
        length: Target length — 'short' (up to 11 lines), 'medium' (12-25), 'long' (26+)
        form: Poetic form — 'free_verse', 'couplets', 'haiku_chain', 'concrete',
              'found_poem', 'cut_up', 'erasure'

    Returns:
        Dict with success status and poem details
    """
    gui_state = main_window._gui_state if main_window else None
    selected_ids = []
    if gui_state:
        selected_ids = gui_state.analyze_selected_ids or gui_state.cut_selected_ids or []

    if not selected_ids:
        return {"success": False, "error": "No clips selected. Select clips with extracted text first."}

    clips_with_text = []
    for cid in selected_ids:
        clip = project.clips_by_id.get(cid)
        if clip and clip.combined_text:
            clips_with_text.append((clip, clip.combined_text))

    if not clips_with_text:
        return {"success": False, "error": "No selected clips have extracted text. Run text extraction first."}

    from core.remix.exquisite_corpus import generate_poem
    try:
        poem_lines = generate_poem(clips_with_text, mood, length=length, form=form)
        if not poem_lines:
            return {"success": False, "error": "LLM could not generate a poem from the available text."}

        # Apply poem order to timeline
        seq_tab = main_window.sequence_tab
        seq_tab.timeline.clear_timeline()
        current_frame = 0
        applied = 0
        applied_entries = []
        for line in poem_lines:
            clip = line.clip
            source = project.sources_by_id.get(clip.source_id)
            if source:
                seq_tab.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
                current_frame += clip.duration_frames
                applied += 1
                applied_entries.append((clip, source))
        seq_tab.timeline._on_zoom_fit()
        seq_tab._set_state(seq_tab.STATE_TIMELINE)

        return _add_sequence_summary_for_agent(project, {
            "success": True,
            "algorithm": "exquisite_corpus",
            "clip_count": applied,
            "poem_lines": len(poem_lines),
            "mood": mood,
            "form": form,
        }, applied_entries)
    except Exception as e:
        return {"success": False, "error": str(e)}


@tools.register(
    description="Generate a Storyteller narrative-driven sequence from clips with descriptions. "
                "Requires description analysis. The LLM selects and orders clips to create "
                "a coherent story following the specified narrative structure.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_storyteller(
    project,
    main_window,
    theme: Optional[str] = None,
    structure: str = "auto",
    target_duration_minutes: Optional[int] = None,
) -> dict:
    """Generate a Storyteller narrative sequence.

    Args:
        theme: Optional theme or topic for the narrative (e.g. 'journey', 'conflict')
        structure: Narrative structure — 'three_act', 'chronological', 'thematic', 'auto'
        target_duration_minutes: Target total duration in minutes (None = use all clips)

    Returns:
        Dict with success status and narrative details
    """
    gui_state = main_window._gui_state if main_window else None
    selected_ids = []
    if gui_state:
        selected_ids = gui_state.analyze_selected_ids or gui_state.cut_selected_ids or []

    if not selected_ids:
        return {"success": False, "error": "No clips selected. Select clips with descriptions first."}

    clips_with_desc = []
    for cid in selected_ids:
        clip = project.clips_by_id.get(cid)
        if clip and clip.description:
            source = project.sources_by_id.get(clip.source_id)
            if source:
                clip._duration_seconds = clip.duration_seconds(source.fps)
                clips_with_desc.append((clip, clip.description))

    if not clips_with_desc:
        return {"success": False, "error": "No selected clips have descriptions. Run Describe analysis first."}

    from core.remix.storyteller import generate_narrative, sequence_by_narrative
    try:
        narrative_lines = generate_narrative(
            clips_with_desc,
            target_duration_minutes=target_duration_minutes,
            narrative_structure=structure,
            theme=theme,
        )
        if not narrative_lines:
            return {"success": False, "error": "LLM could not generate a narrative from the available clips."}

        sequence = sequence_by_narrative(
            narrative_lines,
            project.clips_by_id,
            project.sources_by_id,
        )
        if not sequence:
            return {"success": False, "error": "Could not resolve generated narrative to project clips."}

        # Apply narrative order to timeline
        seq_tab = main_window.sequence_tab
        seq_tab.timeline.clear_timeline()
        current_frame = 0
        applied = 0
        for clip, source in sequence:
            seq_tab.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
            current_frame += clip.duration_frames
            applied += 1
        seq_tab.timeline._on_zoom_fit()
        seq_tab._set_state(seq_tab.STATE_TIMELINE)

        return _add_sequence_summary_for_agent(project, {
            "success": True,
            "algorithm": "storyteller",
            "clip_count": applied,
            "structure": structure,
            "theme": theme,
        }, sequence)
    except Exception as e:
        return {"success": False, "error": str(e)}


@tools.register(
    description="Generate a Cassette Tape sequence: find clips that say specific phrases. "
                "For each phrase the agent supplies, the top-N best-matching transcript "
                "segments are selected and assembled into a sequence of sub-clips trimmed "
                "to just the matched lines. Requires transcribe analysis. Phrases is a list "
                "of {phrase: str, count: int} where count is 1-5 matches per phrase.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_cassette_tape(
    project,
    main_window,
    phrases: list[dict],
) -> dict:
    """Generate a Cassette Tape phrase-driven sequence.

    Args:
        phrases: List of {"phrase": str, "count": int} dicts. Count must be 1-5;
            values outside that range are clamped. Empty/whitespace-only phrases
            are skipped.

    Returns:
        Dict with success status, applied sub-clip count, and per-phrase counts.
    """
    if not phrases:
        return {"success": False, "error": "No phrases provided."}

    from core.remix.cassette_tape import SLIDER_DEFAULT, clamp_count

    # Validate + normalize input. The agent may pass weird shapes; defend at the boundary.
    phrases_with_counts: list[tuple[str, int]] = []
    for entry in phrases:
        if not isinstance(entry, dict):
            continue
        phrase = str(entry.get("phrase", "") or "").strip()
        if not phrase:
            continue
        try:
            count = int(entry.get("count", SLIDER_DEFAULT))
        except (TypeError, ValueError):
            count = SLIDER_DEFAULT
        phrases_with_counts.append((phrase, clamp_count(count)))

    if not phrases_with_counts:
        return {"success": False, "error": "All supplied phrases were empty after trimming."}

    gui_state = main_window._gui_state if main_window else None
    selected_ids = []
    if gui_state:
        selected_ids = gui_state.analyze_selected_ids or gui_state.cut_selected_ids or []

    # If clips are selected, use the selection; otherwise use the whole project.
    if selected_ids:
        candidate_clips = [project.clips_by_id[cid] for cid in selected_ids
                           if cid in project.clips_by_id]
    else:
        candidate_clips = list(project.clips_by_id.values())

    transcribed = [c for c in candidate_clips if c.transcript and not c.disabled]
    if not transcribed:
        return {
            "success": False,
            "error": "No transcribed clips available. Run Transcribe analysis on clips first.",
        }

    from core.remix.cassette_tape import (
        build_sequence_data,
        flatten_matches_in_phrase_order,
        match_phrases,
    )

    try:
        results = match_phrases(phrases_with_counts, transcribed)
        if not results:
            return {"success": False, "error": "No matches found for the supplied phrases."}

        flat = flatten_matches_in_phrase_order(results)  # all matches enabled
        sequence_data = build_sequence_data(flat, project.clips_by_id, project.sources_by_id)
        if not sequence_data:
            return {"success": False, "error": "Could not resolve matches to project clips."}

        seq_tab = main_window.sequence_tab
        seq_tab._apply_cassette_tape_sequence(sequence_data)

        return _add_sequence_summary_for_agent(project, {
            "success": True,
            "algorithm": "cassette_tape",
            "clip_count": len(sequence_data),
            "phrases": [
                {"phrase": p, "match_count": len(results.get(p, []))}
                for p, _ in phrases_with_counts
            ],
        }, sequence_data)
    except Exception as e:
        logger.exception("generate_cassette_tape failed")
        return {"success": False, "error": str(e)}


@tools.register(
    description="Generate a Signature Style sequence from a reference image. "
                "Samples color and pacing from the image, then matches clips to create "
                "a sequence that follows the image's visual structure. "
                "In parametric mode, samples the image directly. In VLM mode, uses a "
                "vision model to interpret the image's style.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_signature_style(
    project,
    main_window,
    reference_image_path: str,
    mode: str = "parametric",
    sample_count: int = 64,
) -> dict:
    """Generate a Signature Style sequence from a reference image.

    Args:
        reference_image_path: Path to the reference image (can be an extracted frame)
        mode: 'parametric' (sample colors directly) or 'vlm' (use vision model)
        sample_count: Number of samples along the image width (8-128, default 64)

    Returns:
        Dict with success status and sequence details
    """
    from pathlib import Path

    image_path = Path(reference_image_path)
    if not image_path.is_file():
        return {"success": False, "error": f"Reference image not found: {reference_image_path}"}

    if mode not in ("parametric", "vlm"):
        return {"success": False, "error": f"Invalid mode '{mode}'. Use 'parametric' or 'vlm'."}

    sample_count = max(8, min(128, sample_count))

    gui_state = main_window._gui_state if main_window else None
    selected_ids = []
    if gui_state:
        selected_ids = gui_state.analyze_selected_ids or gui_state.cut_selected_ids or []

    if not selected_ids:
        return {"success": False, "error": "No clips selected for sequencing."}

    clip_pairs = []
    for cid in selected_ids:
        clip = project.clips_by_id.get(cid)
        if clip:
            source = project.sources_by_id.get(clip.source_id)
            if source:
                clip_pairs.append((clip, source))

    if not clip_pairs:
        return {"success": False, "error": "No valid clips found for sequencing."}

    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        from core.remix.signature_style import (
            sample_drawing_parametric, match_clips_to_segments,
            build_sequence_from_matches,
        )

        segments = sample_drawing_parametric(image, sample_count=sample_count)
        matches = match_clips_to_segments(segments, clip_pairs)
        sequence = build_sequence_from_matches(matches)

        # Apply to timeline
        seq_tab = main_window.sequence_tab
        seq_tab.timeline.clear_timeline()
        current_frame = 0
        for clip, source, in_pt, out_pt in sequence:
            seq_tab.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
            current_frame += (out_pt - in_pt)
        seq_tab.timeline._on_zoom_fit()
        seq_tab._set_state(seq_tab.STATE_TIMELINE)

        return _add_sequence_summary_for_agent(project, {
            "success": True,
            "algorithm": f"signature_style ({mode})",
            "clip_count": len(sequence),
            "sample_count": sample_count,
            "reference_image": str(image_path),
        }, [(clip, source, (out_pt - in_pt) / source.fps) for clip, source, in_pt, out_pt in sequence])
    except Exception as e:
        return {"success": False, "error": str(e)}


@tools.register(
    description="Generate a reference-guided sequence that matches your clips to a reference "
                "video's structure across weighted dimensions (color, brightness, shot_scale, "
                "audio, embedding, description, transcript, movement, duration). Use "
                "get_available_dimensions first to check which dimensions have data. "
                "Use list_sources to find source IDs.",
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
            color, brightness, shot_scale, audio, embedding, description,
            transcript, movement, duration.
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
    valid_dims = {
        "color",
        "brightness",
        "shot_scale",
        "audio",
        "embedding",
        "description",
        "transcript",
        "movement",
        "duration",
    }
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

    return _add_sequence_summary_for_agent(project, result)


@tools.register(
    description="Generate a Rose Hobart sequence filtering clips by a specific person's face. "
                "Requires a reference image path. Use list_sorting_algorithms to check if "
                "face_embeddings are available.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_rose_hobart(
    project,
    main_window,
    reference_image_path: str | None = None,
    reference_image_paths: list[str] | None = None,
    sensitivity: str = "balanced",
    ordering: str = "original",
    sampling_interval: float = 1.0,
) -> dict:
    """Generate a Rose Hobart sequence filtering clips by a specific person's face.

    Args:
        reference_image_path: (Deprecated) Single reference image path, use reference_image_paths instead
        reference_image_paths: Paths to 1-3 reference images of the person
        sensitivity: Match sensitivity - "strict", "balanced", or "loose"
        ordering: Result ordering - "original", "duration", "color", "brightness",
                 "confidence", or "random"
        sampling_interval: Seconds between frame samples (0.25-5.0, default 1.0)

    Returns:
        Dict with success status and matched clip count
    """
    from core.analysis.faces import (
        average_embeddings,
        compare_faces,
        extract_faces_from_image,
        order_matched_clips,
        SENSITIVITY_PRESETS,
    )

    # Support both singular and plural reference image params
    paths = reference_image_paths or ([reference_image_path] if reference_image_path else [])
    if not paths:
        return {"success": False, "error": "At least one reference image path is required"}

    if sensitivity not in SENSITIVITY_PRESETS:
        return {"success": False, "error": f"Invalid sensitivity: {sensitivity}. Use: strict, balanced, loose"}

    valid_orderings = {"original", "duration", "color", "brightness", "confidence", "random"}
    if ordering not in valid_orderings:
        return {"success": False, "error": f"Invalid ordering: {ordering}. Use: {sorted(valid_orderings)}"}

    # Extract reference faces from all provided images
    ref_embeddings = []
    for path_str in paths:
        is_valid, err_msg, validated_path = validate_path(path_str, must_exist=True)
        if not is_valid:
            return {"success": False, "error": err_msg}
        ref_faces = extract_faces_from_image(validated_path)
        if ref_faces:
            best = max(ref_faces, key=lambda f: f["confidence"])
            ref_embeddings.append(best["embedding"])

    if not ref_embeddings:
        return {"success": False, "error": "No face detected in any reference image"}

    if len(ref_embeddings) > 1:
        ref_embeddings = [average_embeddings(ref_embeddings)]

    threshold = SENSITIVITY_PRESETS[sensitivity]

    # Match against clips
    clips = project.clips
    sources_by_id = project.sources_by_id
    matched = []

    for clip in clips:
        if clip.disabled:
            continue
        source = sources_by_id.get(clip.source_id)
        if not source:
            continue

        # Only use pre-computed face embeddings
        clip_faces = clip.face_embeddings
        if clip_faces is None:
            continue  # Skip clips without pre-computed face embeddings

        is_match, confidence = compare_faces(ref_embeddings, clip_faces, threshold)
        if is_match:
            matched.append((clip, source, confidence))

    if not matched:
        return {
            "success": True,
            "matched_count": 0,
            "message": "No clips matched the reference person. Try 'loose' sensitivity.",
        }

    # Order using shared function
    sequence_clips = order_matched_clips(matched, ordering)

    if main_window and hasattr(main_window, 'sequence_tab'):
        main_window.sequence_tab._apply_dialog_sequence(
            sequence_clips, "rose_hobart", "Rose Hobart"
        )

    return _add_sequence_summary_for_agent(project, {
        "success": True,
        "matched_count": len(matched),
        "total_clips": len(clips),
        "sensitivity": sensitivity,
        "ordering": ordering,
    }, [(clip, source) for clip, source, _confidence in sequence_clips])


@tools.register(
    description="Get the current state of the sequence tab including selected algorithm, "
                "parameters, preview clips, and timeline clips.",
    requires_project=True,
    modifies_gui_state=True
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
    from core.spine.settings_io import get_settings as _impl
    return _impl()


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
    from core.spine.settings_io import update_settings as _impl
    return _impl(setting_name, value)


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
    from core.spine.queries import search_transcripts as _impl
    return _impl(project, query, case_sensitive, context_chars)


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
    from core.spine.clips import find_similar_clips as _impl
    return _impl(project, clip_id, criteria, limit)


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
    from core.spine.clips import group_clips_by as _impl
    return _impl(project, criterion)


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
    from core.spine.glossary import get_film_term_definition as _impl
    return _impl(term)


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
    from core.spine.glossary import search_glossary as _impl
    return _impl(query, category)


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
    from core.spine.sequence_analysis import detect_audio_beats as _impl
    return _impl(audio_path, include_onsets)


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
    project,
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
    from core.spine.sequence_analysis import align_sequence_to_audio as _impl
    return _impl(project, audio_path, strategy, max_adjustment)


@tools.register(
    description="Generate a Staccato beat-driven sequence that assigns clips to beat intervals "
                "from a music track. Onset strength at each cut determines visual contrast "
                "between consecutive clips (measured by DINOv2 embedding distance). "
                "Requires an audio file and clips with embeddings. "
                "Strategies: 'beats' (every beat), 'downbeats' (strong beats only), "
                "'onsets' (transient hits).",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def generate_staccato(
    project,
    main_window,
    audio_path: str,
    strategy: str = "beats",
    clip_ids: Optional[list[str]] = None,
) -> dict:
    """Generate a Staccato beat-driven sequence.

    Args:
        audio_path: Path to music file (MP3, WAV, FLAC)
        strategy: Beat strategy - 'beats', 'downbeats', or 'onsets'
        clip_ids: Optional list of clip IDs to use (defaults to all enabled clips)

    Returns:
        Dict with success status, clip count, and slot count
    """
    from core.analysis.audio import analyze_music_file
    from core.remix.staccato import generate_staccato_sequence

    # Validate strategy
    valid_strategies = ("beats", "downbeats", "onsets")
    if strategy not in valid_strategies:
        return {
            "success": False,
            "error": f"Invalid strategy '{strategy}'. Use: {', '.join(valid_strategies)}"
        }

    # Validate audio path
    is_valid, err_msg, validated_path = validate_path(audio_path, must_exist=True)
    if not is_valid:
        return {"success": False, "error": err_msg}

    # Analyze audio
    try:
        audio_analysis = analyze_music_file(validated_path)
    except FileNotFoundError:
        return {"success": False, "error": f"Audio file not found: {audio_path}"}
    except Exception as e:
        return {"success": False, "error": f"Audio analysis failed: {e}"}

    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {"success": False, "error": "Main window not available"}

    # Build clip list
    if clip_ids:
        clips = []
        for cid in clip_ids:
            clip = project.clips_by_id.get(cid)
            if clip is None:
                return {
                    "success": False,
                    "error": f"Clip '{cid}' not found. Use list_clips to see available clips."
                }
            if clip.disabled:
                continue
            source = project.sources_by_id.get(clip.source_id)
            if source:
                clips.append((clip, source))
    else:
        clips = []
        for clip in project.clips:
            if clip.disabled:
                continue
            source = project.sources_by_id.get(clip.source_id)
            if source:
                clips.append((clip, source))

    if not clips:
        return {"success": False, "error": "No clips available for sequencing"}

    missing_embeddings = [
        getattr(clip, "id", "<unknown>")
        for clip, _source in clips
        if getattr(clip, "embedding", None) is None
    ]
    if missing_embeddings:
        return {
            "success": False,
            "error": (
                f"Staccato requires DINOv2 embeddings, but {len(missing_embeddings)} "
                "selected clips are missing them. Run embedding analysis first."
            ),
        }

    # Generate the staccato sequence
    try:
        result = generate_staccato_sequence(
            clips=clips,
            audio_analysis=audio_analysis,
            strategy=strategy,
        )
    except Exception as e:
        return {"success": False, "error": f"Staccato generation failed: {e}"}

    if not result:
        return {
            "success": False,
            "error": "No sequence generated. Check that the audio has detectable beats."
        }

    # Apply to the sequence tab
    sequence_clips = list(result)
    main_window.sequence_tab._apply_staccato_sequence(
        sequence_clips, str(validated_path)
    )

    response = {
        "success": True,
        "clip_count": len(sequence_clips),
        "slot_count": result.debug.total_slots if result.debug else len(sequence_clips),
        "strategy": strategy,
        "audio_file": str(validated_path),
        "tempo_bpm": round(audio_analysis.tempo_bpm, 1),
        "message": (
            f"Generated Staccato sequence: {len(sequence_clips)} clips "
            f"across {result.debug.total_slots if result.debug else len(sequence_clips)} "
            f"beat slots at {audio_analysis.tempo_bpm:.1f} BPM"
        ),
    }
    if result.debug:
        response["staccato_debug_summary"] = {
            "total_slots": result.debug.total_slots,
            "total_clips_available": result.debug.total_clips_available,
            "looped_slot_count": sum(1 for slot in result.debug.slots if slot.needs_loop),
            "slots": [
                {
                    "slot_index": slot.slot_index,
                    "start_time": round(slot.start_time, 3),
                    "end_time": round(slot.end_time, 3),
                    "clip_id": slot.clip_id,
                    "source_name": slot.source_filename,
                    "onset_strength": round(slot.onset_strength, 4),
                    "cosine_distance": (
                        round(slot.cosine_distance, 4)
                        if slot.cosine_distance is not None
                        else None
                    ),
                    "needs_loop": slot.needs_loop,
                }
                for slot in result.debug.slots[:20]
            ],
        }
    return _add_sequence_summary_for_agent(project, response, sequence_clips)


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
    project,
    genre_comparison: Optional[str] = None
) -> dict:
    """Analyze the sequence for pacing and continuity metrics.

    Args:
        genre_comparison: Optional genre to compare pacing against
            (action, drama, documentary, music_video, commercial, art_film)

    Returns:
        Dict with pacing stats, continuity warnings, and suggestions
    """
    from core.spine.sequence_analysis import get_sequence_analysis as _impl
    return _impl(project, genre_comparison)


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
def check_continuity_issues(project) -> dict:
    """Check sequence for potential continuity problems.

    Returns:
        Dict with list of continuity warnings and their severities
    """
    from core.spine.sequence_analysis import check_continuity_issues as _impl
    return _impl(project)


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
    project,
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
    from core.spine.sequence_analysis import generate_analysis_report as _impl
    return _impl(project, sections, include_clip_details, output_format, clip_ids)


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
        valid, err_msg, resolved = validate_path(path_str, must_exist=True)
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
    from core.spine.queries import list_frames as _impl
    return _impl(project, source_id, clip_id, shot_type, has_description)


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


@tools.register(
    description="Clear all custom visual query results from specified clips. "
                "Use when re-running queries with a new model or clearing stale results.",
    requires_project=True,
    modifies_project_state=True
)
def clear_custom_queries(project, clip_ids: Optional[list[str]] = None) -> dict:
    """Clear custom query results from clips.

    Args:
        clip_ids: List of clip IDs to clear. If None, clears all clips.

    Returns:
        Dict with success status and count of cleared clips
    """
    from core.spine.settings_io import clear_custom_queries as _impl
    return _impl(project, clip_ids)


# =============================================================================
# Video Playback Tools - Control the video player
# =============================================================================

@tools.register(
    description="Stop video playback and return to the clip start position.",
    requires_project=False,
    modifies_gui_state=True
)
def stop_playback(main_window) -> dict:
    """Stop the video player.

    Returns:
        Dict with success status
    """
    player = main_window.get_video_player()
    if not player:
        return {"success": False, "error": "Video player not available"}

    player.stop()
    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.clear_playback_state()
    return {"success": True, "message": "Playback stopped"}


@tools.register(
    description="Step one frame forward in the video. Pauses playback if playing.",
    requires_project=False,
    modifies_gui_state=True
)
def frame_step_forward(main_window) -> dict:
    """Advance the video by one frame.

    Returns:
        Dict with success status
    """
    player = main_window.get_video_player()
    if not player:
        return {"success": False, "error": "Video player not available"}

    player.frame_step_forward()
    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.update_playback_state(is_playing=False)
    return {"success": True, "message": "Stepped one frame forward"}


@tools.register(
    description="Step one frame backward in the video. Pauses playback if playing.",
    requires_project=False,
    modifies_gui_state=True
)
def frame_step_backward(main_window) -> dict:
    """Step the video back by one frame.

    Returns:
        Dict with success status
    """
    player = main_window.get_video_player()
    if not player:
        return {"success": False, "error": "Video player not available"}

    player.frame_step_backward()
    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.update_playback_state(is_playing=False)
    return {"success": True, "message": "Stepped one frame backward"}


@tools.register(
    description="Set video playback speed. Valid speeds: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0",
    requires_project=False,
    modifies_gui_state=True
)
def set_playback_speed(main_window, speed: float) -> dict:
    """Set the video playback speed.

    Args:
        speed: Playback speed multiplier (0.25 to 4.0)

    Returns:
        Dict with success status and new speed
    """
    from core.constants import PLAYBACK_SPEEDS

    if speed not in PLAYBACK_SPEEDS:
        return {
            "success": False,
            "error": f"Invalid speed {speed}. Valid speeds: {PLAYBACK_SPEEDS}",
        }

    player = main_window.get_video_player()
    if not player:
        return {"success": False, "error": "Video player not available"}

    player.set_speed(speed)

    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.update_playback_state(speed=speed)

    return {"success": True, "speed": speed, "message": f"Playback speed set to {speed}x"}


@tools.register(
    description="Set A/B loop markers on the video player for looping a section. "
                "Both a_seconds and b_seconds are in seconds. "
                "Call with a_seconds=0 and b_seconds=0 to clear the loop.",
    requires_project=False,
    modifies_gui_state=True
)
def set_ab_loop(main_window, a_seconds: float, b_seconds: float) -> dict:
    """Set or clear A/B loop markers.

    Args:
        a_seconds: Loop start in seconds (0 to clear)
        b_seconds: Loop end in seconds (0 to clear)

    Returns:
        Dict with success status
    """
    player = main_window.get_video_player()
    if not player:
        return {"success": False, "error": "Video player not available"}

    if a_seconds == 0 and b_seconds == 0:
        player.clear_ab_loop()
        gui_state = getattr(main_window, '_gui_state', None)
        if gui_state:
            gui_state.update_playback_state(
                ab_loop_start_ms=None, ab_loop_end_ms=None,
            )
        return {"success": True, "message": "A/B loop cleared"}

    if b_seconds <= a_seconds:
        return {"success": False, "error": "b_seconds must be greater than a_seconds"}

    if a_seconds < 0:
        return {"success": False, "error": "a_seconds cannot be negative"}

    duration_ms = getattr(player, 'duration_ms', 0)
    if duration_ms > 0:
        duration_s = duration_ms / 1000.0
        if b_seconds > duration_s:
            return {"success": False, "error": f"b_seconds ({b_seconds:.2f}s) exceeds video duration ({duration_s:.2f}s)"}

    player.set_ab_loop(a_seconds, b_seconds)
    gui_state = getattr(main_window, '_gui_state', None)
    if gui_state:
        gui_state.update_playback_state(
            ab_loop_start_ms=int(a_seconds * 1000),
            ab_loop_end_ms=int(b_seconds * 1000),
        )
    return {
        "success": True,
        "a_seconds": a_seconds,
        "b_seconds": b_seconds,
        "message": f"A/B loop set: {a_seconds:.1f}s - {b_seconds:.1f}s",
    }


# ============================================================================
# Clip Disable Tool
# ============================================================================

@tools.register(
    description="Toggle the disabled state of clips. Disabled clips are excluded from sequences and exports.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def toggle_clip_disabled(
    project,
    clip_ids: list[str],
    disabled: Optional[bool] = None,
) -> dict:
    """Toggle or set the disabled state of clips.

    Args:
        project: The current project
        clip_ids: IDs of clips to toggle
        disabled: If True/False, set explicitly. If None, toggle current state.

    Returns:
        Dict with updated clip states
    """
    if not clip_ids:
        return {"success": False, "error": "No clip IDs provided"}

    updated = []
    not_found = []

    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None:
            not_found.append(clip_id)
            continue

        if disabled is None:
            clip.disabled = not clip.disabled
        else:
            clip.disabled = disabled

        updated.append({"id": clip.id, "disabled": clip.disabled})

    if updated:
        clips_to_update = [
            project.clips_by_id[u["id"]] for u in updated
        ]
        project.update_clips(clips_to_update)

    result = {
        "success": True,
        "updated": updated,
        "updated_count": len(updated),
    }
    if not_found:
        result["not_found"] = not_found
    return result


# ============================================================================
# Frame Update Tool
# ============================================================================

@tools.register(
    description="Update metadata fields on a frame (tags, notes, shot_type).",
    requires_project=True,
    modifies_gui_state=False,
    modifies_project_state=True
)
def update_frame(
    project,
    frame_id: str,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
    shot_type: Optional[str] = None,
) -> dict:
    """Update frame metadata fields.

    Args:
        project: The current project
        frame_id: ID of the frame to update
        tags: New tags list (None to skip, replaces all existing tags)
        notes: New notes (None to skip, empty string to clear)
        shot_type: New shot type (None to skip, empty string to clear)

    Returns:
        Dict with success status and updated fields
    """
    from core.spine.frames import update_frame as _impl
    return _impl(project, frame_id, tags, notes, shot_type)


# ============================================================================
# Cancel Plan Tool
# ============================================================================

@tools.register(
    description="Cancel the current plan and clear plan state.",
    requires_project=False,
    modifies_gui_state=True,
)
def cancel_plan(main_window) -> dict:
    """Cancel the current plan and clear plan state.

    Returns:
        Dict with cancellation status
    """
    controller = _get_plan_controller(main_window)
    plan = controller.current_plan

    if plan is None:
        return {"success": False, "error": "No active plan to cancel."}

    plan_id = plan.id
    plan_summary = plan.summary
    plan_status = plan.status

    # Clear the plan from gui_state
    controller._gui_state.clear_plan_state()

    return {
        "success": True,
        "cancelled_plan_id": plan_id,
        "cancelled_summary": plan_summary,
        "previous_status": plan_status,
        "message": f"Plan cancelled: {plan_summary}",
    }


# ============================================================================
# Export SRT Tool
# ============================================================================

@tools.register(
    description="Export sequence clips as an SRT subtitle file with clip metadata.",
    requires_project=True,
    modifies_gui_state=True,  # Needs main_window to access sequence_tab
)
def export_srt(
    main_window,
    project,
    output_path: Optional[str] = None,
) -> dict:
    """Export sequence metadata as SRT subtitle file.

    Args:
        main_window: Main window reference
        project: The current project
        output_path: Path for SRT output (optional, defaults to export_dir)

    Returns:
        Dict with success status and export details
    """
    from core.srt_export import export_srt as _export_srt, SRTExportConfig

    if main_window is None or not hasattr(main_window, 'sequence_tab'):
        return {"success": False, "error": "Sequence tab not available"}

    sequence = main_window.sequence_tab.get_sequence()
    all_clips = sequence.get_all_clips()

    if not all_clips:
        return {"success": False, "error": "No clips in timeline to export."}

    # Determine output path
    if output_path:
        valid, error, validated_path = validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        srt_path = validated_path
    else:
        settings = load_settings()
        project_name = project.metadata.name or "sequence_export"
        srt_path = settings.export_dir / f"{project_name}.srt"

    # Ensure .srt extension
    if srt_path.suffix.lower() != ".srt":
        srt_path = srt_path.with_suffix(".srt")

    # Ensure parent directory exists
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookups
    clips_lookup = {clip.id: clip for clip in project.clips}
    sources_lookup = project.sources_by_id.copy()
    frames_lookup = project.frames_by_id.copy() if project._frames else None

    config = SRTExportConfig(output_path=srt_path)
    success, exported, skipped = _export_srt(
        sequence, clips_lookup, sources_lookup, config, frames=frames_lookup
    )

    if not success:
        return {"success": False, "error": "SRT export failed."}

    return {
        "success": True,
        "output_path": str(srt_path),
        "exported_count": exported,
        "skipped_count": skipped,
        "message": f"Exported {exported} subtitle entries to {srt_path.name}",
    }


# ============================================================================
# Export Clips Tool
# ============================================================================

@tools.register(
    description="Export individual clips as separate video files. Can export specific clips or all enabled clips.",
    requires_project=True,
    modifies_gui_state=True,  # Needs main_window for source context
)
def export_clips(
    main_window,
    project,
    clip_ids: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Export individual clips as separate video files.

    Args:
        main_window: Main window reference
        project: The current project
        clip_ids: IDs of clips to export (None exports all enabled clips)
        output_dir: Output directory path (optional, defaults to export_dir)

    Returns:
        Dict with success status and export details
    """
    from core.ffmpeg import FFmpegProcessor

    # Determine which clips to export
    if clip_ids:
        clips_to_export = []
        not_found = []
        for cid in clip_ids:
            clip = project.clips_by_id.get(cid)
            if clip is None:
                not_found.append(cid)
            else:
                clips_to_export.append(clip)
        if not_found:
            return {"success": False, "error": f"Clips not found: {', '.join(not_found)}"}
    else:
        clips_to_export = [c for c in project.clips if not c.disabled]

    if not clips_to_export:
        return {"success": False, "error": "No clips to export."}

    # Determine output directory
    if output_dir:
        valid, error, validated_path = validate_path(output_dir)
        if not valid:
            return {"success": False, "error": f"Invalid output directory: {error}"}
        out_path = validated_path
    else:
        settings = load_settings()
        project_name = project.metadata.name or "clips_export"
        out_path = settings.export_dir / project_name

    out_path.mkdir(parents=True, exist_ok=True)

    processor = FFmpegProcessor()
    exported = 0
    failed = 0
    exported_files = []

    for i, clip in enumerate(clips_to_export):
        source = project.sources_by_id.get(clip.source_id)
        if source is None:
            failed += 1
            continue

        fps = source.fps
        start = clip.start_time(fps)
        duration = clip.duration_seconds(fps)
        source_name = source.file_path.stem
        output_file = out_path / f"{source_name}_scene_{i + 1:03d}.mp4"

        success = processor.extract_clip(
            input_path=source.file_path,
            output_path=output_file,
            start_seconds=start,
            duration_seconds=duration,
            fps=fps,
        )
        if success:
            exported += 1
            exported_files.append(str(output_file))
        else:
            failed += 1

    return {
        "success": True,
        "output_dir": str(out_path),
        "exported_count": exported,
        "failed_count": failed,
        "total_clips": len(clips_to_export),
        "exported_files": exported_files[:10],  # First 10 for brevity
        "message": f"Exported {exported}/{len(clips_to_export)} clips to {out_path}",
    }


@tools.register(
    description="Export the current project as a self-contained bundle folder. "
                "The bundle includes the project file, thumbnails, and optionally "
                "source video files and trimmed clip media. Use lightweight=True to "
                "skip copying video files and exporting trimmed clips (project file "
                "+ thumbnails only). "
                "Runs in background - may take significant time for large projects.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=False
)
def export_bundle(
    main_window,
    project,
    output_path: Optional[str] = None,
    lightweight: bool = False,
    include_clips: Optional[bool] = None,
) -> dict:
    """Export the project as a self-contained bundle folder.

    Args:
        output_path: Destination directory path (optional, defaults to export_dir)
        lightweight: If True, skip copying source video files and trimmed clips
            by default.
        include_clips: Whether to export trimmed clip media. Defaults to
            not lightweight.

    Returns:
        Dict with _wait_for_worker marker for async execution
    """
    # Determine output path
    if output_path:
        valid, error, validated_path = validate_path(output_path)
        if not valid:
            return {"success": False, "error": f"Invalid output path: {error}"}
        dest_dir = validated_path
    else:
        settings = load_settings()
        project_name = project.metadata.name or "project_bundle"
        dest_dir = settings.export_dir / f"{project_name}_bundle"

    # Ensure parent directory exists
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing destination if present (agent skips confirmation dialog)
    if dest_dir.exists():
        import shutil
        shutil.rmtree(dest_dir)

    if main_window is None:
        return {"success": False, "error": "Main window not available for bundle export"}

    # Check if bundle export is already running
    if getattr(main_window, 'export_bundle_worker', None) and main_window.export_bundle_worker.isRunning():
        return {"success": False, "error": "Bundle export already in progress"}

    include_videos = not lightweight
    should_include_clips = not lightweight if include_clips is None else include_clips

    # Start async export via worker
    started = main_window.start_agent_export_bundle(
        dest_dir,
        include_videos,
        should_include_clips,
    )
    if not started:
        return {"success": False, "error": "Failed to start bundle export worker"}

    # Return marker that tells GUI handler to wait for worker completion
    return {
        "_wait_for_worker": "export_bundle",
        "output_path": str(dest_dir),
        "include_videos": include_videos,
        "include_clips": should_include_clips,
    }


@tools.register(
    description="Update cinematography analysis fields on a single clip. "
                "Only specified fields are updated; unspecified fields remain unchanged. "
                "If the clip has no existing cinematography analysis, a new one is created "
                "with default values before applying the updates. "
                "Valid shot_size values: ELS, VLS, LS, MLS, MS, MCU, CU, BCU, ECU, Insert. "
                "Valid camera_angle values: low_angle, eye_level, high_angle, dutch_angle, birds_eye, worms_eye. "
                "Valid camera_movement values: static, pan, tilt, track, handheld, crane, arc, n/a. "
                "Valid lighting_style values: high_key, low_key, natural, dramatic.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def update_clip_cinematography(
    project,
    clip_id: str,
    shot_size: Optional[str] = None,
    camera_angle: Optional[str] = None,
    angle_effect: Optional[str] = None,
    camera_movement: Optional[str] = None,
    movement_direction: Optional[str] = None,
    dutch_tilt: Optional[str] = None,
    camera_position: Optional[str] = None,
    subject_position: Optional[str] = None,
    headroom: Optional[str] = None,
    lead_room: Optional[str] = None,
    balance: Optional[str] = None,
    subject_count: Optional[str] = None,
    subject_type: Optional[str] = None,
    focus_type: Optional[str] = None,
    background_type: Optional[str] = None,
    estimated_lens_type: Optional[str] = None,
    lighting_style: Optional[str] = None,
    lighting_direction: Optional[str] = None,
    light_quality: Optional[str] = None,
    color_temperature: Optional[str] = None,
    emotional_intensity: Optional[str] = None,
    suggested_pacing: Optional[str] = None,
) -> dict:
    """Update cinematography analysis fields on a clip.

    Args:
        clip_id: ID of the clip to update
        shot_size: Shot size (ELS, VLS, LS, MLS, MS, MCU, CU, BCU, ECU, Insert)
        camera_angle: Camera angle (low_angle, eye_level, high_angle, dutch_angle, birds_eye, worms_eye)
        angle_effect: Angle narrative effect (power, neutral, vulnerability, disorientation, omniscience, extreme_power)
        camera_movement: Camera movement type (static, pan, tilt, track, handheld, crane, arc, n/a)
        movement_direction: Movement direction (left, right, up, down, forward, backward, clockwise, counterclockwise)
        dutch_tilt: Horizon tilt (none, slight, moderate, extreme, unknown)
        camera_position: Camera position relative to subject (frontal, three_quarter, profile, back, unknown)
        subject_position: Subject position in frame (left_third, center, right_third, distributed)
        headroom: Headroom spacing (tight, normal, excessive, n/a)
        lead_room: Lead room spacing (tight, normal, excessive, n/a)
        balance: Visual balance (balanced, left_heavy, right_heavy, symmetrical)
        subject_count: Subject count (empty, single, two_shot, group)
        subject_type: Subject type (person, object, landscape, text, mixed)
        focus_type: Focus type (deep, shallow, rack_focus)
        background_type: Background type (blurred, sharp, cluttered, plain)
        estimated_lens_type: Lens type (wide, normal, telephoto, unknown)
        lighting_style: Lighting style (high_key, low_key, natural, dramatic)
        lighting_direction: Lighting direction (front, three_quarter, side, back, below)
        light_quality: Light quality (hard, soft, mixed, unknown)
        color_temperature: Color temperature (warm, neutral, cool, unknown)
        emotional_intensity: Emotional intensity (low, medium, high)
        suggested_pacing: Suggested pacing (fast, medium, slow)

    Returns:
        Dict with success status and updated fields
    """
    from models.cinematography import (
        CinematographyAnalysis,
        SHOT_SIZES, CAMERA_ANGLES, ANGLE_EFFECTS, CAMERA_MOVEMENTS,
        MOVEMENT_DIRECTIONS, DUTCH_TILT_VALUES, CAMERA_POSITION_VALUES,
        SUBJECT_POSITIONS, SPACING_VALUES, BALANCE_VALUES,
        SUBJECT_COUNTS, SUBJECT_TYPES, FOCUS_TYPES, BACKGROUND_TYPES,
        LENS_TYPE_VALUES, LIGHTING_STYLES, LIGHTING_DIRECTIONS,
        LIGHT_QUALITY_VALUES, COLOR_TEMPERATURE_VALUES,
        INTENSITY_LEVELS, PACING_VALUES,
    )

    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        return {"success": False, "error": f"Clip not found: {clip_id}"}

    # Validation map: field_name -> (value, valid_values_list)
    validations = {
        "shot_size": (shot_size, SHOT_SIZES),
        "camera_angle": (camera_angle, CAMERA_ANGLES),
        "angle_effect": (angle_effect, ANGLE_EFFECTS),
        "camera_movement": (camera_movement, CAMERA_MOVEMENTS),
        "movement_direction": (movement_direction, MOVEMENT_DIRECTIONS),
        "dutch_tilt": (dutch_tilt, DUTCH_TILT_VALUES),
        "camera_position": (camera_position, CAMERA_POSITION_VALUES),
        "subject_position": (subject_position, SUBJECT_POSITIONS),
        "headroom": (headroom, SPACING_VALUES),
        "lead_room": (lead_room, SPACING_VALUES),
        "balance": (balance, BALANCE_VALUES),
        "subject_count": (subject_count, SUBJECT_COUNTS),
        "subject_type": (subject_type, SUBJECT_TYPES),
        "focus_type": (focus_type, FOCUS_TYPES),
        "background_type": (background_type, BACKGROUND_TYPES),
        "estimated_lens_type": (estimated_lens_type, LENS_TYPE_VALUES),
        "lighting_style": (lighting_style, LIGHTING_STYLES),
        "lighting_direction": (lighting_direction, LIGHTING_DIRECTIONS),
        "light_quality": (light_quality, LIGHT_QUALITY_VALUES),
        "color_temperature": (color_temperature, COLOR_TEMPERATURE_VALUES),
        "emotional_intensity": (emotional_intensity, INTENSITY_LEVELS),
        "suggested_pacing": (suggested_pacing, PACING_VALUES),
    }

    # Validate all provided fields
    for field_name, (value, valid_list) in validations.items():
        if value is not None and value not in valid_list:
            return {
                "success": False,
                "error": f"Invalid {field_name} '{value}'. Valid values: {', '.join(valid_list)}"
            }

    # Create cinematography object if it doesn't exist
    if clip.cinematography is None:
        clip.cinematography = CinematographyAnalysis()

    # Apply updates
    updated_fields = []
    for field_name, (value, _) in validations.items():
        if value is not None:
            setattr(clip.cinematography, field_name, value)
            updated_fields.append(field_name)

    if updated_fields:
        project.update_clips([clip])

    return {
        "success": True,
        "clip_id": clip_id,
        "updated_fields": updated_fields,
        "message": f"Updated cinematography: {', '.join(updated_fields)}" if updated_fields else "No fields updated"
    }


@tools.register(
    description="Clear (remove) cinematography analysis from one or more clips. "
                "Sets each clip's cinematography field to None.",
    requires_project=True,
    modifies_gui_state=True,
    modifies_project_state=True
)
def clear_clip_cinematography(
    project,
    clip_ids: list[str],
) -> dict:
    """Clear cinematography analysis from clips.

    Args:
        clip_ids: List of clip IDs to clear cinematography from

    Returns:
        Dict with success status and count of cleared clips
    """
    if not clip_ids:
        return {"success": False, "error": "No clip IDs provided"}

    cleared = []
    not_found = []
    already_clear = []

    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is None:
            not_found.append(clip_id)
            continue
        if clip.cinematography is None:
            already_clear.append(clip_id)
            continue
        clip.cinematography = None
        cleared.append(clip_id)

    # Update all modified clips at once
    if cleared:
        modified_clips = [project.clips_by_id[cid] for cid in cleared]
        project.update_clips(modified_clips)

    result = {
        "success": True,
        "cleared_count": len(cleared),
        "cleared_ids": cleared,
        "message": f"Cleared cinematography from {len(cleared)} clip(s)"
    }

    if not_found:
        result["not_found"] = not_found
    if already_clear:
        result["already_clear"] = already_clear

    return result


@tools.register(
    description="Undo the last undoable action. Returns the name of the action undone, "
                "or reports if there is nothing to undo.",
    requires_project=False,
    modifies_gui_state=True
)
def undo(main_window) -> dict:
    """Undo the last action.

    Returns:
        Dict with success status and the undone action name
    """
    if not hasattr(main_window, 'undo_stack'):
        return {"success": False, "error": "Undo not available"}

    if not main_window.undo_stack.canUndo():
        return {"success": False, "error": "Nothing to undo"}

    action_text = main_window.undo_stack.undoText()
    main_window.undo_stack.undo()
    return {"success": True, "undone": action_text or "last action"}


@tools.register(
    description="Redo the last undone action. Returns the name of the action redone, "
                "or reports if there is nothing to redo.",
    requires_project=False,
    modifies_gui_state=True
)
def redo(main_window) -> dict:
    """Redo the last undone action.

    Returns:
        Dict with success status and the redone action name
    """
    if not hasattr(main_window, 'undo_stack'):
        return {"success": False, "error": "Redo not available"}

    if not main_window.undo_stack.canRedo():
        return {"success": False, "error": "Nothing to redo"}

    action_text = main_window.undo_stack.redoText()
    main_window.undo_stack.redo()
    return {"success": True, "redone": action_text or "last action"}
