"""Shared constants for agent tools, analysis, and UI.

Single source of truth for enum values used across chat_tools,
analysis modules, and UI filter controls.
"""

# Shot types for UI display (title-cased)
VALID_SHOT_TYPES = ["Wide Shot", "Medium Shot", "Close-up", "Extreme CU", "Full Shot"]

# Valid aspect ratios for filtering
VALID_ASPECT_RATIOS = ["16:9", "4:3", "9:16"]

# Valid color palettes for filtering
VALID_COLOR_PALETTES = ["Warm", "Cool", "Neutral", "Vibrant"]

# Valid sort orders for clip browser
VALID_SORT_ORDERS = ["Timeline", "Color", "Duration"]

# Playback speed presets
PLAYBACK_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
DEFAULT_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
