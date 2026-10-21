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
    # Cinematography's get_simple_shot_type() writes "wide"/"full"/"medium"
    # (display names, above), "close-up" and "ECU".
    "ecu": "extreme close-up",
    # Cinematography shot sizes that name one canonical size. VLS, MLS, MCU
    # and BCU sit between two canonical sizes, and Insert is not a size.
    "els": "wide shot",
    "ls": "full shot",
    "ms": "medium shot",
    "cu": "close-up",
}


def normalize_shot_type(value: str) -> str | None:
    """Map a canonical value, display name, or legacy label to the canonical value.

    Matching is case-insensitive and ignores surrounding whitespace. Returns
    None when the value is not a known shot type.
    """
    if not isinstance(value, str):
        return None
    return _SHOT_TYPE_ALIASES.get(value.strip().lower())


def shot_type_matches(stored: str | None, wanted: str | None) -> bool:
    """Whether a stored shot type is the wanted one, in any spelling.

    Known values compare canonically. When either side is unknown, the raw
    values compare case-insensitively.
    """
    if not isinstance(stored, str) or not isinstance(wanted, str):
        return False
    stored_canonical = normalize_shot_type(stored)
    wanted_canonical = normalize_shot_type(wanted)
    if stored_canonical is not None and wanted_canonical is not None:
        return stored_canonical == wanted_canonical
    return bool(stored.strip()) and stored.strip().lower() == wanted.strip().lower()


def canonical_shot_type(value: str | None) -> str | None:
    """Canonical value for a known shot type; other values unchanged."""
    return normalize_shot_type(value) or value


# Valid aspect ratios for filtering
VALID_ASPECT_RATIOS = ["16:9", "4:3", "9:16"]

# Valid color palettes for filtering
VALID_COLOR_PALETTES = ["Warm", "Cool", "Neutral", "Vibrant"]

# Valid sort orders for clip browser
VALID_SORT_ORDERS = ["Timeline", "Color", "Duration"]

# Playback speed presets
PLAYBACK_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
DEFAULT_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
