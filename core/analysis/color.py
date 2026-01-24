"""Color extraction using k-means clustering."""

from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


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
