"""Signature Style — drawing-based sequencer algorithm.

Interprets a visual drawing left-to-right as an editing guide.
Two modes share the same DrawingSegment intermediate format:

- Parametric: pixel-level reading of Y-position (pacing) and color
- VLM: vision-language model interpretation of visual meaning

The clip matcher consumes DrawingSegment[] and produces (Clip, Source)[].
"""

import logging
import math
from typing import TYPE_CHECKING, Optional

from core.analysis.color import _SATURATION_THRESHOLD, rgb_to_hsv
from core.remix.drawing_segment import DrawingSegment

if TYPE_CHECKING:
    from PySide6.QtGui import QImage

    from models.clip import Clip, Source

logger = logging.getLogger(__name__)

# Parametric mode constants
_DEFAULT_SAMPLE_COUNT = 64
_PACING_MERGE_THRESHOLD = 0.08  # Merge segments within this pacing distance
_COLOR_MERGE_THRESHOLD = 30.0  # Merge segments within this RGB euclidean distance

# Clip matcher weights (defaults)
WEIGHT_DURATION = 0.3
WEIGHT_COLOR = 0.35
WEIGHT_SHOT_TYPE = 0.15
WEIGHT_BRIGHTNESS = 0.1
WEIGHT_ENERGY = 0.1

# Shot type string to numeric distance lookup (for matching)
_SHOT_TYPE_NUMERIC = {
    "extreme close-up": 0.0,
    "close-up": 0.2,
    "medium shot": 0.5,
    "full shot": 0.7,
    "wide shot": 1.0,
    # 10-class cinematography abbreviations
    "ECU": 0.0,
    "BCU": 0.1,
    "CU": 0.2,
    "MCU": 0.3,
    "MS": 0.5,
    "MLS": 0.6,
    "LS": 0.7,
    "VLS": 0.85,
    "ELS": 1.0,
    "Insert": 0.4,
}


# ──────────────────────────────────────────────────────────────
# Parametric sampling
# ──────────────────────────────────────────────────────────────


def sample_drawing_parametric(
    image: "QImage",
    total_duration_seconds: float,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
) -> list[DrawingSegment]:
    """Sample a drawing image left-to-right to produce DrawingSegments.

    For each sample column:
    - Y-axis: highest non-background pixel maps to pacing (top=fast, bottom=slow)
    - Color: average color in the region; low saturation → B&W

    Args:
        image: QImage of the drawing canvas (white background)
        total_duration_seconds: Total target output duration
        sample_count: Number of X-positions to sample (granularity)

    Returns:
        List of DrawingSegments (merged from raw samples)
    """
    width = image.width()
    height = image.height()

    if width == 0 or height == 0:
        return []

    # Clamp sample_count to image width
    sample_count = min(sample_count, width)
    if sample_count < 1:
        return []

    step = width / sample_count
    raw_samples: list[dict] = []

    for i in range(sample_count):
        x_center = int(i * step + step / 2)
        x_start = int(i * step)
        x_end = int((i + 1) * step)

        # Sample a column: find highest non-white pixel, average color
        pacing, color, is_bw = _sample_column(image, x_center, height)

        raw_samples.append({
            "x_start": x_start,
            "x_end": min(x_end, width),
            "pacing": pacing,
            "color": color,
            "is_bw": is_bw,
        })

    # Merge adjacent similar samples
    merged = _merge_samples(raw_samples)

    # Convert to DrawingSegments with duration allocation
    segments = _allocate_durations(merged, total_duration_seconds)

    return segments


def _sample_column(
    image: "QImage",
    x: int,
    height: int,
    sample_radius: int = 3,
) -> tuple[float, Optional[tuple[int, int, int]], bool]:
    """Sample a single column of the drawing.

    Args:
        image: QImage to sample
        x: X pixel position
        height: Image height
        sample_radius: Radius for color averaging

    Returns:
        (pacing, color_rgb_or_None, is_bw)
    """
    # Find highest non-white pixel (scan top to bottom)
    highest_y = None
    for y in range(height):
        pixel = image.pixelColor(x, y)
        # Consider non-background if not close to white
        if pixel.red() < 240 or pixel.green() < 240 or pixel.blue() < 240:
            highest_y = y
            break

    if highest_y is None:
        # Blank column — default to mid pacing, no color
        return 0.5, None, True

    # Pacing: top = 1.0 (fast), bottom = 0.0 (slow)
    pacing = 1.0 - (highest_y / max(height - 1, 1))

    # Color: average in a small region around the highest pixel
    r_sum, g_sum, b_sum, count = 0, 0, 0, 0
    for dy in range(-sample_radius, sample_radius + 1):
        for dx in range(-sample_radius, sample_radius + 1):
            sx = x + dx
            sy = highest_y + dy
            if 0 <= sx < image.width() and 0 <= sy < height:
                pixel = image.pixelColor(sx, sy)
                if pixel.red() < 240 or pixel.green() < 240 or pixel.blue() < 240:
                    r_sum += pixel.red()
                    g_sum += pixel.green()
                    b_sum += pixel.blue()
                    count += 1

    if count == 0:
        return pacing, None, True

    avg_r = r_sum // count
    avg_g = g_sum // count
    avg_b = b_sum // count

    # Check if B&W using HSV saturation
    h, s, v = rgb_to_hsv((avg_r, avg_g, avg_b))
    # rgb_to_hsv returns s in 0-1 range; _SATURATION_THRESHOLD uses 0-255 scale
    is_bw = (s * 255) < _SATURATION_THRESHOLD

    color = None if is_bw else (avg_r, avg_g, avg_b)

    return pacing, color, is_bw


def _merge_samples(raw_samples: list[dict]) -> list[dict]:
    """Merge adjacent samples with similar pacing and color.

    Returns merged samples with consolidated x_start/x_end ranges.
    """
    if not raw_samples:
        return []

    merged: list[dict] = [raw_samples[0].copy()]

    for sample in raw_samples[1:]:
        prev = merged[-1]

        # Check pacing similarity
        pacing_close = abs(sample["pacing"] - prev["pacing"]) <= _PACING_MERGE_THRESHOLD

        # Check color similarity
        color_close = _colors_similar(prev["color"], sample["color"])

        # Check B&W consistency
        bw_same = prev["is_bw"] == sample["is_bw"]

        if pacing_close and color_close and bw_same:
            # Merge: extend x_end, average pacing and color
            prev["x_end"] = sample["x_end"]
            count = prev.get("_count", 1) + 1
            prev["pacing"] = (prev["pacing"] * (count - 1) + sample["pacing"]) / count
            if prev["color"] and sample["color"]:
                pr, pg, pb = prev["color"]
                sr, sg, sb = sample["color"]
                prev["color"] = (
                    (pr * (count - 1) + sr) // count,
                    (pg * (count - 1) + sg) // count,
                    (pb * (count - 1) + sb) // count,
                )
            prev["_count"] = count
        else:
            merged.append(sample.copy())

    return merged


def _colors_similar(
    c1: Optional[tuple[int, int, int]],
    c2: Optional[tuple[int, int, int]],
) -> bool:
    """Check if two colors are similar enough to merge."""
    if c1 is None and c2 is None:
        return True
    if c1 is None or c2 is None:
        return False
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    return dist <= _COLOR_MERGE_THRESHOLD


def _allocate_durations(
    merged_samples: list[dict],
    total_duration_seconds: float,
) -> list[DrawingSegment]:
    """Convert merged samples to DrawingSegments with allocated durations.

    Duration is proportional to segment width relative to total canvas width.
    """
    if not merged_samples:
        return []

    total_width = sum(s["x_end"] - s["x_start"] for s in merged_samples)
    if total_width == 0:
        return []

    segments: list[DrawingSegment] = []
    for sample in merged_samples:
        seg_width = sample["x_end"] - sample["x_start"]
        proportion = seg_width / total_width
        duration = total_duration_seconds * proportion

        segments.append(DrawingSegment(
            x_start=sample["x_start"],
            x_end=sample["x_end"],
            target_duration_seconds=duration,
            target_pacing=sample["pacing"],
            target_color=sample["color"],
            is_bw=sample["is_bw"],
        ))

    return segments


# ──────────────────────────────────────────────────────────────
# Clip matching
# ──────────────────────────────────────────────────────────────


def match_clips_to_segments(
    segments: list[DrawingSegment],
    clips: list[tuple["Clip", "Source"]],
    allow_reuse: bool = True,
) -> list[tuple["Clip", "Source", DrawingSegment]]:
    """Match clips to drawing segments via weighted multi-criteria scoring.

    Greedy left-to-right assignment. Clips can be reused if allow_reuse is True.

    Args:
        segments: Drawing segments from parametric or VLM interpretation
        clips: Available (Clip, Source) pairs
        allow_reuse: Whether clips can be matched to multiple segments

    Returns:
        List of (Clip, Source, DrawingSegment) tuples in segment order
    """
    if not segments or not clips:
        return []

    used_indices: set[int] = set()
    result: list[tuple["Clip", "Source", DrawingSegment]] = []

    for segment in segments:
        best_score = float("inf")
        best_idx: Optional[int] = None

        for idx, (clip, source) in enumerate(clips):
            if not allow_reuse and idx in used_indices:
                continue

            score = _compute_match_score(segment, clip, source)
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            used_indices.add(best_idx)
            clip, source = clips[best_idx]
            result.append((clip, source, segment))
            logger.debug(
                f"Segment [{segment.x_start}-{segment.x_end}]: "
                f"matched clip {clip.id[:8]} (score={best_score:.4f})"
            )

    return result


def _compute_match_score(
    segment: DrawingSegment,
    clip: "Clip",
    source: "Source",
) -> float:
    """Compute weighted distance between a segment and a clip.

    Lower score = better match.
    """
    total = 0.0
    total_weight = 0.0

    # Duration fit
    clip_duration = clip.duration_seconds(source.fps)
    if clip_duration > 0 and segment.target_duration_seconds > 0:
        # Ratio-based: how far is clip duration from target?
        ratio = clip_duration / segment.target_duration_seconds
        # Log-scale distance: 1.0 = perfect, 0.5 and 2.0 are equally bad
        duration_dist = abs(math.log2(max(ratio, 0.01)))
        # Normalize to 0-1 range (log2(16) ≈ 4 is very far)
        duration_dist = min(duration_dist / 4.0, 1.0)
        total += WEIGHT_DURATION * duration_dist
        total_weight += WEIGHT_DURATION

    # Color distance
    if not segment.is_bw and segment.target_color and clip.dominant_colors:
        clip_color = clip.dominant_colors[0]  # Most dominant
        color_dist = _hue_distance(segment.target_color, clip_color)
        total += WEIGHT_COLOR * color_dist
        total_weight += WEIGHT_COLOR
    elif segment.is_bw:
        # Prefer B&W clips for B&W segments
        if source.color_profile in ("grayscale", "sepia"):
            total += WEIGHT_COLOR * 0.0  # Perfect match
        else:
            total += WEIGHT_COLOR * 0.5  # Partial penalty
        total_weight += WEIGHT_COLOR

    # Shot type (VLM segments only)
    if segment.shot_type and clip.shot_type:
        seg_val = _SHOT_TYPE_NUMERIC.get(segment.shot_type, 0.5)
        clip_val = _SHOT_TYPE_NUMERIC.get(clip.shot_type, 0.5)
        total += WEIGHT_SHOT_TYPE * abs(seg_val - clip_val)
        total_weight += WEIGHT_SHOT_TYPE

    # Brightness (VLM segments only)
    if segment.brightness is not None and clip.average_brightness is not None:
        total += WEIGHT_BRIGHTNESS * abs(segment.brightness - clip.average_brightness)
        total_weight += WEIGHT_BRIGHTNESS

    # Energy (VLM segments only) — map to pacing
    if segment.energy is not None:
        # No direct clip energy; use pacing as proxy
        total += WEIGHT_ENERGY * abs(segment.energy - segment.target_pacing)
        total_weight += WEIGHT_ENERGY

    return total / total_weight if total_weight > 0 else float("inf")


def _hue_distance(
    rgb1: tuple[int, int, int],
    rgb2: tuple[int, int, int],
) -> float:
    """Compute perceptual color distance using HSV hue.

    Returns 0.0 (identical) to 1.0 (opposite on color wheel).
    """
    h1, s1, v1 = rgb_to_hsv(rgb1)
    h2, s2, v2 = rgb_to_hsv(rgb2)

    # Hue is circular (0-360)
    hue_diff = abs(h1 - h2)
    if hue_diff > 180:
        hue_diff = 360 - hue_diff
    hue_dist = hue_diff / 180.0  # Normalize to 0-1

    # Also factor in saturation and value differences (minor weight)
    sat_dist = abs(s1 - s2)
    val_dist = abs(v1 - v2)

    # Weighted: hue matters most for chromatic colors
    return 0.7 * hue_dist + 0.15 * sat_dist + 0.15 * val_dist


# ──────────────────────────────────────────────────────────────
# Sequence building
# ──────────────────────────────────────────────────────────────


def build_sequence_from_matches(
    matches: list[tuple["Clip", "Source", DrawingSegment]],
    fps: float = 30.0,
) -> list[tuple["Clip", "Source", int, int]]:
    """Convert matched clips into timeline-ready (Clip, Source, in_point, out_point).

    Clips are trimmed to fit segment target duration when needed.

    Args:
        matches: Output from match_clips_to_segments
        fps: Timeline FPS

    Returns:
        List of (Clip, Source, in_point_frames, out_point_frames)
    """
    result: list[tuple["Clip", "Source", int, int]] = []

    for clip, source, segment in matches:
        clip_duration_frames = clip.duration_frames
        target_frames = max(1, int(segment.target_duration_seconds * fps))

        if clip_duration_frames <= 0:
            # Degenerate clip — skip
            continue

        if target_frames >= clip_duration_frames:
            # Clip is shorter than or equal to target — use full clip
            in_point = 0
            out_point = clip_duration_frames
        else:
            # Clip is longer — trim to target duration, centering the trim
            excess = clip_duration_frames - target_frames
            in_point = excess // 2
            out_point = in_point + target_frames

        result.append((clip, source, in_point, out_point))

    return result


def check_missing_analysis(
    clips: list["Clip"],
    mode: str = "parametric",
) -> dict[str, int]:
    """Check which analysis is missing for signature style matching.

    Args:
        clips: Clip objects to check
        mode: "parametric" or "vlm"

    Returns:
        Dict mapping analysis type to count of clips needing it.
        Empty dict means all clips have required metadata.
    """
    missing: dict[str, int] = {}

    # Both modes need dominant_colors
    need_colors = sum(1 for c in clips if not c.dominant_colors)
    if need_colors > 0:
        missing["colors"] = need_colors

    # VLM mode benefits from shot_type and description
    if mode == "vlm":
        need_shots = sum(1 for c in clips if not c.shot_type)
        if need_shots > 0:
            missing["shots"] = need_shots

        need_describe = sum(1 for c in clips if not c.description)
        if need_describe > 0:
            missing["describe"] = need_describe

    return missing
