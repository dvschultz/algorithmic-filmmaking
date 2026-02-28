"""Shared constants for agent tools, analysis, and UI.

Single source of truth for enum values used across chat_tools,
analysis modules, and UI filter controls.
"""

from core.analysis.shots import SHOT_TYPES

# Shot types for UI display (title-cased)
# Maps from display name -> analysis name (lowercase in SHOT_TYPES)
VALID_SHOT_TYPES = ["Wide Shot", "Medium Shot", "Close-up", "Extreme CU", "Full Shot"]

# Mapping from UI display names to analysis-level names
SHOT_TYPE_DISPLAY_TO_ANALYSIS = {
    "Wide Shot": "wide shot",
    "Full Shot": "full shot",
    "Medium Shot": "medium shot",
    "Close-up": "close-up",
    "Extreme CU": "extreme close-up",
}

# Valid aspect ratios for filtering
VALID_ASPECT_RATIOS = ["16:9", "4:3", "9:16"]

# Valid color palettes for filtering
VALID_COLOR_PALETTES = ["Warm", "Cool", "Neutral", "Vibrant"]

# Valid sort orders for clip browser
VALID_SORT_ORDERS = ["Timeline", "Color", "Duration"]
