"""Color extraction using k-means clustering."""

from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


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
