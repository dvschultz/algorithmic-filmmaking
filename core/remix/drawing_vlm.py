"""VLM interpretation mode for Signature Style drawing sequencer.

Takes a drawing image, slices it at visual change boundaries, sends
slices to a vision-language model for interpretation, and produces
DrawingSegment[] objects for clip matching.
"""

import base64
import json
import logging
import re
from typing import Callable, Optional

from core.analysis.shots import SHOT_TYPES
from core.remix.drawing_segment import DrawingSegment

try:
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QImage

    _HAS_QT = True
except ImportError:
    _HAS_QT = False

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

_WHOLE_IMAGE_PROMPT = (
    "Analyze this drawing as a visual editing guide for a video sequence. "
    "Describe the overall mood, energy arc, and any narrative progression you see. "
    "Focus on:\n"
    "- Overall color palette and mood\n"
    "- Energy/intensity progression (left to right)\n"
    "- Any recurring visual themes or patterns\n"
    "Respond in 2-3 sentences."
)

_PER_SLICE_PROMPT_TEMPLATE = (
    'This is one section of a drawing being used as a visual editing guide for video. '
    'The overall drawing was described as: "{overall_context}"\n\n'
    "For this section, analyze the visual qualities and return JSON:\n"
    "{{\n"
    '    "shot_type": "close-up" | "medium shot" | "wide shot" | "extreme close-up" | "full shot",\n'
    '    "color_mood": "warm" | "cool" | "neutral" | "vibrant" | "muted" | "dark",\n'
    '    "energy": 0.0-1.0,\n'
    '    "pacing": "fast" | "medium" | "slow",\n'
    '    "brightness": "bright" | "medium" | "dark"\n'
    "}}\n\n"
    "First briefly describe what you see, then provide the JSON."
)

# ──────────────────────────────────────────────────────────────
# VLM string-to-numeric mappings
# ──────────────────────────────────────────────────────────────

_PACING_MAP = {
    "fast": 0.8,
    "medium": 0.5,
    "slow": 0.2,
}

_ENERGY_DEFAULT = 0.5

_BRIGHTNESS_MAP = {
    "bright": 0.8,
    "light": 0.8,
    "medium": 0.5,
    "normal": 0.5,
    "dark": 0.2,
}


# ──────────────────────────────────────────────────────────────
# Adaptive slicing
# ──────────────────────────────────────────────────────────────


def slice_drawing_adaptive(
    image: "QImage",
    min_slices: int = 3,
    max_slices: int = 20,
) -> list[tuple[int, int]]:
    """Detect visual change boundaries and return slice (x_start, x_end) tuples.

    Scans vertical strips across the image computing simple color histograms.
    Where the histogram difference between adjacent strips exceeds a threshold,
    a boundary is placed.  Falls back to equal-width slicing when no significant
    changes are detected.

    Args:
        image: QImage of the drawing canvas.
        min_slices: Minimum number of slices to produce.
        max_slices: Maximum number of slices to produce.

    Returns:
        List of (x_start, x_end) pixel-position tuples.
    """
    width = image.width()
    height = image.height()

    if width == 0 or height == 0:
        return []

    strip_step = 5  # sample every 5 pixels along X
    num_strips = max(1, width // strip_step)

    # Compute a coarse 8-bin-per-channel histogram for each vertical strip
    histograms: list[list[int]] = []
    for i in range(num_strips):
        x = min(i * strip_step, width - 1)
        hist = _column_histogram(image, x, height)
        histograms.append(hist)

    # Compute differences between adjacent histograms
    diffs: list[float] = []
    for i in range(1, len(histograms)):
        diff = sum(abs(a - b) for a, b in zip(histograms[i - 1], histograms[i]))
        diffs.append(float(diff))

    if not diffs:
        return _equal_slices(width, min_slices)

    # Adaptive threshold: use mean + 1 standard deviation
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    std_diff = variance ** 0.5
    threshold = mean_diff + std_diff

    # Collect boundary positions (in pixel coordinates)
    boundary_xs: list[int] = []
    for i, diff in enumerate(diffs):
        if diff > threshold:
            pixel_x = (i + 1) * strip_step
            boundary_xs.append(min(pixel_x, width))

    # Enforce min/max slice count
    if len(boundary_xs) < min_slices - 1:
        # Not enough visual changes detected -- fall back to equal slicing
        return _equal_slices(width, min_slices)

    if len(boundary_xs) > max_slices - 1:
        # Too many boundaries -- keep the strongest ones
        indexed = sorted(
            enumerate(boundary_xs),
            key=lambda ib: diffs[ib[0]] if ib[0] < len(diffs) else 0,
            reverse=True,
        )
        keep = sorted(b[1] for b in indexed[: max_slices - 1])
        boundary_xs = keep

    # Remove duplicates and sort
    boundary_xs = sorted(set(boundary_xs))

    # Build slice tuples
    slices: list[tuple[int, int]] = []
    prev_x = 0
    for bx in boundary_xs:
        if bx > prev_x:
            slices.append((prev_x, bx))
            prev_x = bx
    if prev_x < width:
        slices.append((prev_x, width))

    return slices


def _column_histogram(
    image: "QImage",
    x: int,
    height: int,
    bins: int = 8,
) -> list[int]:
    """Compute a coarse color histogram for a single vertical strip.

    Args:
        image: Source QImage.
        x: X pixel coordinate to sample.
        height: Image height.
        bins: Number of bins per channel (R, G, B).

    Returns:
        Flat list of length ``bins * 3`` with bin counts.
    """
    hist = [0] * (bins * 3)
    bin_size = 256 // bins
    # Sample every other row for speed
    for y in range(0, height, 2):
        pixel = image.pixelColor(x, y)
        r_bin = min(pixel.red() // bin_size, bins - 1)
        g_bin = min(pixel.green() // bin_size, bins - 1)
        b_bin = min(pixel.blue() // bin_size, bins - 1)
        hist[r_bin] += 1
        hist[bins + g_bin] += 1
        hist[2 * bins + b_bin] += 1
    return hist


def _equal_slices(width: int, count: int) -> list[tuple[int, int]]:
    """Produce *count* equal-width slices spanning the full width."""
    count = max(1, count)
    step = width / count
    slices: list[tuple[int, int]] = []
    for i in range(count):
        x_start = int(i * step)
        x_end = int((i + 1) * step)
        slices.append((x_start, min(x_end, width)))
    return slices


# ──────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────


def _image_to_base64(image: "QImage") -> str:
    """Convert a QImage to a base64-encoded PNG string.

    Args:
        image: QImage to encode.

    Returns:
        Base64 string of the PNG data.
    """
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    buf.close()
    return base64.b64encode(buf.data().data()).decode("ascii")


def _extract_sub_image(image: "QImage", x_start: int, x_end: int) -> "QImage":
    """Extract a vertical slice from an image.

    Args:
        image: Source QImage.
        x_start: Left pixel boundary.
        x_end: Right pixel boundary.

    Returns:
        Cropped QImage for the slice.
    """
    return image.copy(x_start, 0, x_end - x_start, image.height())


# ──────────────────────────────────────────────────────────────
# VLM response parsing
# ──────────────────────────────────────────────────────────────


def _parse_vlm_response(response_text: str) -> dict:
    """Parse a VLM response that should contain JSON.

    Strategy:
    1. Try ``json.loads()`` on the full response.
    2. Try to extract a fenced ``json`` code block.
    3. Try to find a bare JSON object ``{...}``.
    4. Return empty dict on failure.

    Args:
        response_text: Raw text from the VLM.

    Returns:
        Parsed dict or empty dict.
    """
    if not response_text:
        return {}

    # 1. Direct parse
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Fenced code block: ```json ... ```
    fenced = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Bare JSON object
    bare = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if bare:
        try:
            data = json.loads(bare.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    logger.warning("Failed to parse VLM response as JSON: %s", response_text[:200])
    return {}


# ──────────────────────────────────────────────────────────────
# VLM data mapping
# ──────────────────────────────────────────────────────────────


def _closest_shot_type(value: str) -> str:
    """Map an arbitrary shot type string to the nearest known SHOT_TYPE.

    Args:
        value: Shot type string from the VLM.

    Returns:
        The closest match from ``SHOT_TYPES``.
    """
    if not value:
        return "medium shot"

    lower = value.strip().lower()

    # Exact match
    for st in SHOT_TYPES:
        if lower == st:
            return st

    # Substring / partial match
    for st in SHOT_TYPES:
        if lower in st or st in lower:
            return st

    # Keyword heuristics
    if "wide" in lower or "long" in lower or "establishing" in lower:
        return "wide shot"
    if "full" in lower:
        return "full shot"
    if "medium" in lower or "mid" in lower or "cowboy" in lower:
        return "medium shot"
    if "extreme" in lower or "ecu" in lower:
        return "extreme close-up"
    if "close" in lower or "tight" in lower:
        return "close-up"

    return "medium shot"


def _map_vlm_to_segment(
    vlm_data: dict,
    x_start: int,
    x_end: int,
    total_duration_seconds: float,
    canvas_width: int,
) -> DrawingSegment:
    """Convert parsed VLM dict and position info to a DrawingSegment.

    Args:
        vlm_data: Parsed JSON dict from the VLM response.
        x_start: Left pixel boundary of this slice.
        x_end: Right pixel boundary of this slice.
        total_duration_seconds: Total target output duration.
        canvas_width: Full drawing width in pixels.

    Returns:
        A DrawingSegment populated with VLM-derived values.
    """
    # Pacing
    raw_pacing = vlm_data.get("pacing", "medium")
    if isinstance(raw_pacing, (int, float)):
        pacing = max(0.0, min(1.0, float(raw_pacing)))
    else:
        pacing = _PACING_MAP.get(str(raw_pacing).strip().lower(), 0.5)

    # Energy
    raw_energy = vlm_data.get("energy", _ENERGY_DEFAULT)
    if isinstance(raw_energy, (int, float)):
        energy = max(0.0, min(1.0, float(raw_energy)))
    else:
        # Treat string values same as pacing mapping
        energy = _PACING_MAP.get(str(raw_energy).strip().lower(), 0.5)

    # Brightness
    raw_brightness = vlm_data.get("brightness", "medium")
    if isinstance(raw_brightness, (int, float)):
        brightness = max(0.0, min(1.0, float(raw_brightness)))
    else:
        brightness = _BRIGHTNESS_MAP.get(str(raw_brightness).strip().lower(), 0.5)

    # Shot type
    raw_shot_type = vlm_data.get("shot_type", "medium shot")
    shot_type = _closest_shot_type(str(raw_shot_type))

    # Color mood (pass through)
    color_mood = vlm_data.get("color_mood")
    if color_mood is not None:
        color_mood = str(color_mood).strip().lower()

    # Duration proportional to slice width
    slice_width = max(1, x_end - x_start)
    effective_canvas = max(1, canvas_width)
    target_duration = total_duration_seconds * (slice_width / effective_canvas)

    return DrawingSegment(
        x_start=x_start,
        x_end=x_end,
        target_duration_seconds=target_duration,
        target_pacing=pacing,
        target_color=None,  # VLM mode uses color_mood instead of RGB
        is_bw=False,
        shot_type=shot_type,
        energy=energy,
        brightness=brightness,
        color_mood=color_mood,
    )


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────


def interpret_drawing_vlm(
    image: "QImage",
    total_duration_seconds: float,
    llm_client,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[DrawingSegment]:
    """Interpret a drawing via a vision-language model.

    1. Slice the drawing at visual change boundaries.
    2. Send the whole image for overall mood/context.
    3. Send each slice for per-segment VLM analysis.
    4. Parse responses into DrawingSegment objects.

    This function is synchronous (designed to run in a QThread worker).
    It uses ``asyncio.run()`` internally to call the async LLM client.

    Args:
        image: QImage of the drawing canvas.
        total_duration_seconds: Total target output duration in seconds.
        llm_client: An ``LLMClient`` instance (from ``core.llm_client``).
        progress_callback: Optional ``(current, total)`` callback for UI updates.

    Returns:
        List of DrawingSegments, one per visual slice.
    """
    import asyncio

    canvas_width = image.width()
    if canvas_width == 0 or image.height() == 0:
        return []

    # Step 1: Adaptive slicing
    slices = slice_drawing_adaptive(image)
    if not slices:
        return []

    total_steps = len(slices) + 1  # +1 for the whole-image call

    # Step 2: Whole-image context
    full_b64 = _image_to_base64(image)
    overall_context = _call_vlm_sync(
        llm_client,
        full_b64,
        _WHOLE_IMAGE_PROMPT,
    )

    if progress_callback:
        progress_callback(1, total_steps)

    # Step 3: Per-slice interpretation
    segments: list[DrawingSegment] = []
    for idx, (x_start, x_end) in enumerate(slices):
        sub_image = _extract_sub_image(image, x_start, x_end)
        slice_b64 = _image_to_base64(sub_image)

        prompt = _PER_SLICE_PROMPT_TEMPLATE.format(overall_context=overall_context)
        response_text = _call_vlm_sync(
            llm_client,
            slice_b64,
            prompt,
        )

        vlm_data = _parse_vlm_response(response_text)
        segment = _map_vlm_to_segment(
            vlm_data, x_start, x_end, total_duration_seconds, canvas_width
        )
        segments.append(segment)

        if progress_callback:
            progress_callback(idx + 2, total_steps)  # +2: whole-image was step 1

    logger.info(
        "VLM interpretation complete: %d slices -> %d segments",
        len(slices),
        len(segments),
    )
    return segments


# ──────────────────────────────────────────────────────────────
# Async bridge helpers
# ──────────────────────────────────────────────────────────────


def _has_running_loop() -> bool:
    """Check whether an asyncio event loop is currently running."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        return loop is not None
    except RuntimeError:
        return False


def _call_vlm_sync(
    llm_client,
    image_b64: str,
    text_prompt: str,
) -> str:
    """Call the VLM synchronously, bridging async if necessary.

    Uses ``asyncio.run()`` when no event loop is active.  When called
    from within an already-running loop (rare), falls back to
    a background thread to avoid nested run() errors.

    Args:
        llm_client: LLMClient instance.
        image_b64: Base64-encoded PNG image.
        text_prompt: The text prompt to send.

    Returns:
        The text content of the VLM response, or empty string on error.
    """
    import asyncio

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": text_prompt,
                },
            ],
        }
    ]

    try:
        if _has_running_loop():
            # Running inside an existing event loop (e.g. pytest-asyncio).
            # Spin up a background thread to avoid nested run() errors.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, llm_client.chat(messages))
                response = future.result(timeout=120)
        else:
            response = asyncio.run(llm_client.chat(messages))

        # LiteLLM returns a ModelResponse; extract text content
        content = response.choices[0].message.content
        if content is None:
            logger.warning("VLM returned None content (possible content filter)")
            return ""
        return str(content)

    except Exception:
        logger.exception("VLM call failed")
        return ""
