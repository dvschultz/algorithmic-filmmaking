"""Color, visual, and audio analysis utilities."""

from .color import (
    extract_dominant_colors,
    rgb_to_hsv,
    get_primary_hue,
    classify_color_palette,
    get_palette_display_name,
    COLOR_PALETTES,
    COLOR_PALETTE_DISPLAY,
)
from .shots import (
    classify_shot_type,
    get_display_name,
    SHOT_TYPES,
)
from .audio import (
    AudioAnalysis,
    has_audio_track,
    extract_audio,
    extracted_audio,
    analyze_audio,
    analyze_audio_from_video,
    analyze_music_file,
)
from .sequence import (
    analyze_sequence,
    analyze_pacing,
    check_continuity,
    analyze_visual_consistency,
    generate_suggestions,
    get_pacing_curve,
)

__all__ = [
    "extract_dominant_colors",
    "rgb_to_hsv",
    "get_primary_hue",
    "classify_color_palette",
    "get_palette_display_name",
    "COLOR_PALETTES",
    "COLOR_PALETTE_DISPLAY",
    "classify_shot_type",
    "get_display_name",
    "SHOT_TYPES",
    # Audio analysis
    "AudioAnalysis",
    "has_audio_track",
    "extract_audio",
    "extracted_audio",
    "analyze_audio",
    "analyze_audio_from_video",
    "analyze_music_file",
    # Sequence analysis
    "analyze_sequence",
    "analyze_pacing",
    "check_continuity",
    "analyze_visual_consistency",
    "generate_suggestions",
    "get_pacing_curve",
]
