"""Color extraction and video color profile detection using k-means clustering."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Constants for video color profile detection
_PRECHECK_SAMPLE_WIDTH = 160
_PRECHECK_NUM_SAMPLES = 10
_SATURATION_THRESHOLD = 15.0
_GRAYSCALE_RATIO_THRESHOLD = 0.95
_SEPIA_SAT_MIN = 3.0
_SEPIA_SAT_MAX = 40.0
_SEPIA_HUE_MIN = 15.0
_SEPIA_HUE_MAX = 45.0
_SEPIA_HUE_STD_MAX = 20.0


@dataclass
class ColorProfileResult:
    """Result of video color profile detection."""

    is_grayscale: bool
    classification: str  # "grayscale", "sepia", "mixed", "color"
    mean_saturation: float
    frame_saturations: list[float] = field(default_factory=list)


def detect_video_color_profile(
    video_path: Path,
    num_samples: int = _PRECHECK_NUM_SAMPLES,
    saturation_threshold: float = _SATURATION_THRESHOLD,
    grayscale_ratio_threshold: float = _GRAYSCALE_RATIO_THRESHOLD,
    downsample_width: int = _PRECHECK_SAMPLE_WIDTH,
) -> ColorProfileResult:
    """Detect whether a video is grayscale, sepia, mixed, or color.

    Samples N evenly-spaced frames, downsamples for speed,
    and checks HSV saturation to classify video content.

    Args:
        video_path: Path to video file
        num_samples: Number of frames to sample
        saturation_threshold: Max mean saturation to consider a frame grayscale
        grayscale_ratio_threshold: Fraction of grayscale frames required
        downsample_width: Resize width for faster processing

    Returns:
        ColorProfileResult with classification and saturation data
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.warning(f"Cannot open video for color profile: {video_path}")
            return ColorProfileResult(
                is_grayscale=False,
                classification="color",
                mean_saturation=0.0,
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning(f"Cannot determine frame count for color profile: {video_path}")
            return ColorProfileResult(
                is_grayscale=False,
                classification="color",
                mean_saturation=0.0,
            )

        # Calculate evenly-spaced sample positions
        if num_samples >= total_frames:
            sample_positions = list(range(total_frames))
        else:
            step = total_frames / (num_samples + 1)
            sample_positions = [int(step * (i + 1)) for i in range(num_samples)]

        frame_saturations: list[float] = []
        frame_hue_stds: list[float] = []
        frame_mean_hues: list[float] = []

        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Downsample for speed
            h, w = frame.shape[:2]
            if w > downsample_width:
                scale = downsample_width / w
                frame = cv2.resize(
                    frame,
                    (downsample_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            # Convert to HSV and measure saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1].astype(np.float32)
            h_channel = hsv[:, :, 0].astype(np.float32)

            mean_sat = float(s_channel.mean())
            frame_saturations.append(mean_sat)

            # Track hue distribution for sepia detection
            sat_mask = s_channel > 5
            if sat_mask.any():
                hue_std = float(h_channel[sat_mask].std())
                mean_hue = float(h_channel[sat_mask].mean())
            else:
                hue_std = 0.0
                mean_hue = 0.0
            frame_hue_stds.append(hue_std)
            frame_mean_hues.append(mean_hue)

    finally:
        cap.release()

    # Not enough readable frames to classify
    if len(frame_saturations) < 3:
        logger.warning(
            f"Too few readable frames ({len(frame_saturations)}) for color profile: {video_path}"
        )
        return ColorProfileResult(
            is_grayscale=False,
            classification="color",
            mean_saturation=0.0,
            frame_saturations=frame_saturations,
        )

    # Less than 50% of samples readable — unreliable classification
    if len(frame_saturations) < len(sample_positions) * 0.5:
        logger.warning(
            f"Only {len(frame_saturations)}/{len(sample_positions)} frames readable "
            f"for color profile: {video_path}"
        )
        return ColorProfileResult(
            is_grayscale=False,
            classification="color",
            mean_saturation=float(np.mean(frame_saturations)),
            frame_saturations=frame_saturations,
        )

    overall_mean_sat = float(np.mean(frame_saturations))
    overall_hue_std = float(np.mean(frame_hue_stds))
    overall_mean_hue = float(np.mean(frame_mean_hues))
    grayscale_frame_count = sum(
        1 for s in frame_saturations if s < saturation_threshold
    )
    grayscale_ratio = grayscale_frame_count / len(frame_saturations)

    # Classification logic
    # Check sepia first: low but non-zero saturation with narrow warm hue.
    # Sepia frames have saturation above the per-frame grayscale threshold
    # but should still use luma-only detection since hue/sat carry no useful
    # scene-change information.
    if (
        _SEPIA_SAT_MIN < overall_mean_sat < _SEPIA_SAT_MAX
        and _SEPIA_HUE_MIN < overall_mean_hue < _SEPIA_HUE_MAX
        and overall_hue_std < _SEPIA_HUE_STD_MAX
    ):
        classification = "sepia"
        is_grayscale = True
    elif grayscale_ratio >= grayscale_ratio_threshold:
        classification = "grayscale"
        is_grayscale = True
    elif grayscale_ratio > 0.3:
        classification = "mixed"
        is_grayscale = False
    else:
        classification = "color"
        is_grayscale = False

    return ColorProfileResult(
        is_grayscale=is_grayscale,
        classification=classification,
        mean_saturation=overall_mean_sat,
        frame_saturations=frame_saturations,
    )


# Color palette constants
COLOR_PALETTES = ["warm", "cool", "neutral", "vibrant"]

COLOR_PALETTE_DISPLAY = {
    "warm": "Warm",
    "cool": "Cool",
    "neutral": "Neutral",
    "vibrant": "Vibrant",
}


def extract_dominant_colors(
    image_path: Path,
    n_colors: int = 5,
    sample_size: int = 50,
) -> list[tuple[int, int, int]]:
    """
    Extract dominant colors from an image using k-means clustering.

    Args:
        image_path: Path to the image file
        n_colors: Number of dominant colors to extract
        sample_size: Size to resize image for faster processing

    Returns:
        List of RGB tuples sorted by frequency (most dominant first)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for speed
    img = cv2.resize(img, (sample_size, sample_size), interpolation=cv2.INTER_AREA)

    # Flatten to pixel array
    pixels = img.reshape(-1, 3)

    # Run k-means clustering (n_init=1 is sufficient with fixed random_state)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=1, max_iter=100)
    kmeans.fit(pixels)

    # Get cluster centers (colors) and labels
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Count pixels per cluster and sort by frequency
    counts = np.bincount(labels)
    sorted_indices = np.argsort(-counts)  # Descending order

    # Return colors as RGB tuples sorted by frequency
    return [tuple(colors[i]) for i in sorted_indices]


def rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Convert RGB to HSV color space.

    Args:
        rgb: RGB tuple (0-255 per channel)

    Returns:
        HSV tuple (H: 0-360, S: 0-1, V: 0-1)
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    # Hue calculation
    if diff == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    # Saturation calculation
    s = 0 if max_c == 0 else diff / max_c

    # Value
    v = max_c

    return (h, s, v)


def get_primary_hue(colors: list[tuple[int, int, int]]) -> float:
    """
    Get the hue of the most dominant color for sorting purposes.

    Args:
        colors: List of RGB tuples sorted by frequency

    Returns:
        Hue value (0-360) of the most dominant color
    """
    if not colors:
        return 0.0

    hsv = rgb_to_hsv(colors[0])
    return hsv[0]


def classify_color_palette(colors: list[tuple[int, int, int]]) -> str:
    """
    Classify the color palette based on HSV analysis of dominant colors.

    Classification logic:
    - Warm: Hues in 0-60 or 300-360 range (reds, oranges, yellows)
    - Cool: Hues in 180-300 range (blues, cyans, purples)
    - Neutral: Low saturation (<0.2) across dominant colors
    - Vibrant: High saturation (>0.6) across dominant colors

    Args:
        colors: List of RGB tuples sorted by frequency

    Returns:
        Color palette classification: "warm", "cool", "neutral", or "vibrant"
    """
    if not colors:
        return "neutral"

    # Analyze top 3 dominant colors (or fewer if not available)
    analyze_count = min(3, len(colors))
    hsv_colors = [rgb_to_hsv(c) for c in colors[:analyze_count]]

    # Calculate weighted averages (more weight to dominant colors)
    weights = [1.0, 0.5, 0.25][:analyze_count]
    total_weight = sum(weights)

    avg_saturation = sum(hsv[1] * w for hsv, w in zip(hsv_colors, weights)) / total_weight

    # Check for neutral first (low saturation)
    if avg_saturation < 0.2:
        return "neutral"

    # Check for vibrant (high saturation)
    if avg_saturation > 0.6:
        return "vibrant"

    # Classify by hue (warm vs cool)
    # Use the primary (most dominant) color's hue
    primary_hue = hsv_colors[0][0]

    # Warm: 0-60 (red-yellow) or 300-360 (magenta-red)
    # Cool: 180-300 (cyan-magenta)
    if (0 <= primary_hue <= 60) or (300 <= primary_hue <= 360):
        return "warm"
    elif 180 <= primary_hue <= 300:
        return "cool"
    else:
        # Transitional hues (60-180): greens and yellow-greens
        # Classify based on whether closer to warm or cool
        if primary_hue < 120:
            return "warm"
        else:
            return "cool"


def get_palette_display_name(palette: str) -> str:
    """
    Get the display name for a color palette.

    Args:
        palette: Palette identifier (e.g., "warm", "cool")

    Returns:
        Human-readable display name
    """
    return COLOR_PALETTE_DISPLAY.get(palette, palette.title())
