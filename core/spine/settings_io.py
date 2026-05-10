"""Settings + sorting-algorithm catalog spine impls.

Read/write for the limited set of safe app settings, plus a catalog of
sequencer algorithms keyed off what analysis a project's clips have.
"""

from __future__ import annotations

from typing import Optional

from core.settings import load_settings


# Safe settings that can be modified by the agent (no API keys, no paths)
SAFE_SETTINGS: dict[str, tuple] = {
    "default_sensitivity": (float, 1.0, 10.0),
    "min_scene_length_seconds": (float, 0.1, 10.0),
    "export_quality": (str, ["low", "medium", "high"]),
    "export_resolution": (str, ["original", "1080p", "720p", "480p"]),
    "export_fps": (str, ["original", "24", "30", "60"]),
    "transcription_model": (str, ["tiny.en", "small.en", "medium.en", "large-v3"]),
    "transcription_language": (str, None),
    "theme_preference": (str, ["system", "light", "dark"]),
    "youtube_results_count": (int, 10, 50),
    "youtube_parallel_downloads": (int, 1, 3),
    "llm_provider": (str, ["local", "openai", "anthropic", "gemini", "openrouter"]),
    "llm_model": (str, None),
    "llm_temperature": (float, 0.0, 2.0),
}


def get_settings() -> dict:
    """Return the non-sensitive subset of current settings."""
    settings = load_settings()
    return {
        "success": True,
        "settings": {
            "download_dir": str(settings.download_dir),
            "export_dir": str(settings.export_dir),
            "thumbnail_cache_dir": str(settings.thumbnail_cache_dir),
            "default_sensitivity": settings.default_sensitivity,
            "min_scene_length_seconds": settings.min_scene_length_seconds,
            "export_quality": settings.export_quality,
            "export_resolution": settings.export_resolution,
            "export_fps": settings.export_fps,
            "transcription_model": settings.transcription_model,
            "transcription_language": settings.transcription_language,
            "theme_preference": settings.theme_preference,
            "youtube_results_count": settings.youtube_results_count,
            "youtube_parallel_downloads": settings.youtube_parallel_downloads,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
        },
    }


def update_settings(setting_name: str, value) -> dict:
    """Update a single safe setting."""
    if setting_name not in SAFE_SETTINGS:
        return {
            "success": False,
            "error": (
                f"Setting '{setting_name}' cannot be modified. "
                f"Safe settings: {', '.join(sorted(SAFE_SETTINGS.keys()))}"
            ),
        }

    spec = SAFE_SETTINGS[setting_name]
    expected_type = spec[0]

    if expected_type is float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": f"Setting '{setting_name}' requires a number",
            }
        min_val, max_val = spec[1], spec[2]
        if not (min_val <= value <= max_val):
            return {
                "success": False,
                "error": (
                    f"Setting '{setting_name}' must be between "
                    f"{min_val} and {max_val}"
                ),
            }
    elif expected_type is int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": f"Setting '{setting_name}' requires an integer",
            }
        min_val, max_val = spec[1], spec[2]
        if not (min_val <= value <= max_val):
            return {
                "success": False,
                "error": (
                    f"Setting '{setting_name}' must be between "
                    f"{min_val} and {max_val}"
                ),
            }
    elif expected_type is str:
        value = str(value)
        allowed_values = spec[1]
        if allowed_values is not None and value not in allowed_values:
            return {
                "success": False,
                "error": (
                    f"Setting '{setting_name}' must be one of: "
                    f"{', '.join(allowed_values)}"
                ),
            }

    settings = load_settings()
    old_value = getattr(settings, setting_name)
    setattr(settings, setting_name, value)

    from core.settings import save_settings

    save_settings(settings)

    return {
        "success": True,
        "message": f"Updated {setting_name}: {old_value} -> {value}",
        "setting": setting_name,
        "old_value": old_value,
        "new_value": value,
    }


def clear_custom_queries(project, clip_ids: Optional[list[str]] = None) -> dict:
    """Clear custom visual query results from clips."""
    clips = (
        project.clips
        if clip_ids is None
        else [
            project.clips_by_id[cid]
            for cid in clip_ids
            if cid in project.clips_by_id
        ]
    )

    cleared = 0
    for clip in clips:
        if clip.custom_queries:
            clip.custom_queries = []
            cleared += 1

    if cleared > 0:
        project.mark_dirty()

    return {
        "success": True,
        "cleared_count": cleared,
        "total_checked": len(clips),
    }


def list_sorting_algorithms(project) -> dict:
    """List all sequencer algorithms with availability flags."""
    clips = project.clips
    has_colors = any(clip.dominant_colors for clip in clips) if clips else False
    has_transcripts = any(clip.transcript for clip in clips) if clips else False

    has_shot_type = any(clip.shot_type for clip in clips) if clips else False
    has_text = any(clip.extracted_texts for clip in clips) if clips else False
    has_descriptions = any(clip.description for clip in clips) if clips else False
    has_face_embeddings = (
        any(clip.face_embeddings for clip in clips) if clips else False
    )
    has_gaze = (
        any(clip.gaze_category is not None for clip in clips) if clips else False
    )

    algorithms = [
        {
            "key": "shuffle",
            "name": "Hatchet Job",
            "description": "Randomly shuffle clips into a new order",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "seed", "type": "integer", "description": "Random seed for reproducibility (0 = random)", "default": 0},
                {"name": "random_hflip", "type": "boolean", "description": "Randomly flip ~50% of clips horizontally at export", "default": False},
                {"name": "random_vflip", "type": "boolean", "description": "Randomly flip ~50% of clips vertically at export", "default": False},
                {"name": "random_reverse", "type": "boolean", "description": "Randomly reverse ~50% of clips at export", "default": False},
            ],
        },
        {
            "key": "sequential",
            "name": "Time Capsule",
            "description": "Keep clips in their original order",
            "available": True,
            "reason": None,
            "parameters": [],
        },
        {
            "key": "duration",
            "name": "Tempo Shift",
            "description": "Order clips from shortest to longest (or reverse)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["short_first", "long_first"], "default": "short_first"},
            ],
        },
        {
            "key": "color",
            "name": "Chromatics",
            "description": "Arrange clips along a color gradient or cycle through the spectrum",
            "available": has_colors,
            "reason": None if has_colors else "Run color analysis on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["rainbow", "warm_to_cool", "cool_to_warm", "complementary"], "default": "rainbow"},
                {"name": "no_color_handling", "type": "string", "options": ["append_end", "exclude", "sort_inline"], "default": "append_end"},
            ],
        },
        {
            "key": "brightness",
            "name": "Into the Dark",
            "description": "Arrange clips from light to shadow, or shadow to light (auto-computes if needed)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["light_to_dark", "dark_to_light"], "default": "light_to_dark"},
            ],
        },
        {
            "key": "volume",
            "name": "Crescendo",
            "description": "Build from silence to thunder, or thunder to silence (auto-computes if needed)",
            "available": True,
            "reason": None,
            "parameters": [
                {"name": "direction", "type": "string", "options": ["quiet_to_loud", "loud_to_quiet"], "default": "quiet_to_loud"},
            ],
        },
        {
            "key": "shot_type",
            "name": "Focal Ladder",
            "description": "Arrange clips by camera shot scale",
            "available": has_shot_type,
            "reason": None if has_shot_type else "Run shot type classification on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["wide_to_close", "close_to_wide"], "default": "wide_to_close"},
            ],
        },
        {
            "key": "proximity",
            "name": "Up Close and Personal",
            "description": "Glide from distant vistas to intimate close-ups",
            "available": has_shot_type,
            "reason": None if has_shot_type else "Run shot type classification on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["far_to_near", "near_to_far"], "default": "far_to_near"},
            ],
        },
        {
            "key": "similarity_chain",
            "name": "Human Centipede",
            "description": "Chain clips together by visual similarity (auto-computes embeddings if needed)",
            "available": True,
            "reason": None,
            "parameters": [],
        },
        {
            "key": "match_cut",
            "name": "Match Cut",
            "description": "Find hidden connections between clips using boundary frame similarity (auto-computes if needed)",
            "available": True,
            "reason": None,
            "parameters": [],
        },
        {
            "key": "exquisite_corpus",
            "name": "Exquisite Corpus",
            "description": "Generate a poem from on-screen text",
            "available": has_text,
            "reason": None if has_text else "Run OCR/text extraction on clips first",
            "parameters": [],
        },
        {
            "key": "storyteller",
            "name": "Storyteller",
            "description": "Create a narrative from clip descriptions",
            "available": has_descriptions,
            "reason": None if has_descriptions else "Run clip description analysis first",
            "parameters": [],
        },
        {
            "key": "rose_hobart",
            "name": "Rose Hobart",
            "description": "Isolate clips featuring a specific person (requires reference image)",
            "available": has_face_embeddings,
            "reason": None if has_face_embeddings else "Run face detection analysis first",
            "parameters": [
                {"name": "reference_image_paths", "type": "array", "description": "Paths to 1-3 reference images of the person"},
                {"name": "sensitivity", "type": "string", "options": ["strict", "balanced", "loose"], "default": "balanced"},
                {"name": "ordering", "type": "string", "options": ["original", "duration", "color", "brightness", "confidence", "random"], "default": "original"},
                {"name": "sampling_interval", "type": "number", "description": "Seconds between frame samples (0.25-5.0, default 1.0)"},
            ],
        },
        {
            "key": "gaze_sort",
            "name": "Gaze Sort",
            "description": "Arrange clips by gaze direction",
            "available": has_gaze,
            "reason": None if has_gaze else "Run gaze analysis on clips first",
            "parameters": [
                {"name": "direction", "type": "string", "options": ["left_to_right", "right_to_left", "up_to_down", "down_to_up"], "default": "left_to_right"},
            ],
        },
        {
            "key": "gaze_consistency",
            "name": "Gaze Consistency",
            "description": "Group clips by matching gaze direction",
            "available": has_gaze,
            "reason": None if has_gaze else "Run gaze analysis on clips first",
            "parameters": [],
        },
        {
            "key": "cassette_tape",
            "name": "Cassette Tape",
            "description": "Find clips that say specific phrases (transcript-driven mixtape)",
            "available": has_transcripts,
            "reason": None if has_transcripts else "Run transcribe analysis on clips first",
            "parameters": [
                {
                    "name": "phrases",
                    "type": "array",
                    "description": "List of {phrase: str, count: int} dicts. count is 1-5 matches per phrase. Use generate_cassette_tape, not generate_remix.",
                },
            ],
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


__all__ = [
    "SAFE_SETTINGS",
    "clear_custom_queries",
    "get_settings",
    "list_sorting_algorithms",
    "update_settings",
]
