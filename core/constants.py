"""Shared constants for agent tools, analysis, and UI.

Single source of truth for enum values used across chat_tools,
analysis modules, and UI filter controls.
"""

# Canonical shot types: the lowercase values analysis writes to Clip.shot_type.
# Must equal core.analysis_model_identity.SHOT_TYPES (it feeds a fingerprint).
SHOT_TYPES = ["wide shot", "full shot", "medium shot", "close-up", "extreme close-up"]

# Kept importable for older callers; same as SHOT_TYPES.
VALID_SHOT_TYPES = SHOT_TYPES

# Short display names used by the GUI filters and clip cards
SHOT_TYPE_DISPLAY = {
    "wide shot": "Wide",
    "full shot": "Full",
    "medium shot": "Medium",
    "close-up": "Close-up",
    "extreme close-up": "Extreme CU",
}

_SHOT_TYPE_ALIASES = {
    **{value: value for value in SHOT_TYPES},
    **{display.lower(): value for value, display in SHOT_TYPE_DISPLAY.items()},
    # Older title-case labels ("Wide Shot", "Full Shot", ...) lowercase to the
    # canonical values above; "Extreme CU" lowercases to a display name.
}


def normalize_shot_type(value: str) -> str | None:
    """Map a canonical value, display name, or legacy label to the canonical value.

    Matching is case-insensitive and ignores surrounding whitespace. Returns
    None when the value is not a known shot type.
    """
    if not isinstance(value, str):
        return None
    return _SHOT_TYPE_ALIASES.get(value.strip().lower())


# Valid aspect ratios for filtering
VALID_ASPECT_RATIOS = ["16:9", "4:3", "9:16"]

# Valid color palettes for filtering
VALID_COLOR_PALETTES = ["Warm", "Cool", "Neutral", "Vibrant"]

# Valid sort orders for clip browser
VALID_SORT_ORDERS = ["Timeline", "Color", "Duration"]

# Playback speed presets
PLAYBACK_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
DEFAULT_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
