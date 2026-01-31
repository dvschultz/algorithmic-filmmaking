"""Fast text detection using OpenCV EAST model.

Pre-screens video frames for text presence before running expensive OCR/VLM.
Uses the EAST (Efficient and Accurate Scene Text) neural network which runs
in ~50-100ms on CPU, significantly faster than VLM calls.

Fail-safe behavior: If detection fails for any reason, assumes text is present
and allows the frame to proceed to OCR.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Module-level cached model (lazy loaded)
_east_net: Optional[cv2.dnn.Net] = None
_model_lock = threading.Lock()

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.4

# Model download URL and filename
EAST_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
EAST_MODEL_FILENAME = "frozen_east_text_detection.pb"


def _get_model_path() -> Path:
    """Get path to EAST model in the model cache directory.

    Returns:
        Path where the model should be stored.
    """
    from core.settings import load_settings

    settings = load_settings()
    model_dir = Path(settings.model_cache_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / EAST_MODEL_FILENAME


def _download_east_model(dest_path: Path) -> bool:
    """Download EAST model (~50MB).

    Args:
        dest_path: Where to save the downloaded model.

    Returns:
        True if download succeeded, False otherwise.
    """
    import urllib.request
    import urllib.error

    logger.info(f"Downloading EAST text detection model to {dest_path}")
    logger.info("This is a one-time download (~50MB)...")

    try:
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        urllib.request.urlretrieve(EAST_MODEL_URL, dest_path)

        # Verify file was downloaded
        if dest_path.exists() and dest_path.stat().st_size > 0:
            logger.info(f"EAST model downloaded successfully ({dest_path.stat().st_size / 1024 / 1024:.1f}MB)")
            return True
        else:
            logger.error("Download appeared to succeed but file is missing or empty")
            return False

    except urllib.error.URLError as e:
        logger.error(f"Failed to download EAST model (network error): {e}")
        return False
    except OSError as e:
        logger.error(f"Failed to save EAST model (disk error): {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to download EAST model: {e}")
        return False


def _load_east_model() -> Optional[cv2.dnn.Net]:
    """Lazy load EAST model (thread-safe).

    Returns:
        Loaded OpenCV DNN network, or None if loading failed.
    """
    global _east_net

    # Fast path: already loaded
    if _east_net is not None:
        return _east_net

    with _model_lock:
        # Double-check after acquiring lock
        if _east_net is not None:
            return _east_net

        model_path = _get_model_path()

        # Download if not present
        if not model_path.exists():
            if not _download_east_model(model_path):
                logger.warning("EAST model not available - text detection will be skipped")
                return None

        # Load the model
        try:
            _east_net = cv2.dnn.readNet(str(model_path))
            logger.info("EAST text detection model loaded successfully")
            return _east_net
        except cv2.error as e:
            logger.error(f"Failed to load EAST model (OpenCV error): {e}")
            # Try to remove corrupted file
            try:
                model_path.unlink()
            except OSError:
                pass
            return None
        except Exception as e:
            logger.error(f"Failed to load EAST model: {e}")
            return None


def has_text_regions(
    image_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Quick check if image contains text regions.

    This is a fast pre-filter (~50-100ms) to avoid running expensive OCR/VLM
    on frames that have no text.

    Args:
        image_path: Path to image file (JPEG, PNG, etc.)
        confidence_threshold: Minimum confidence score (0.0-1.0) for text detection.
            Lower values detect more text but may have false positives.

    Returns:
        True if text was detected or if detection failed (fail-safe behavior).
        False only if we're confident no text is present.
    """
    net = _load_east_model()
    if net is None:
        logger.debug("EAST model not available, assuming text present (fail-safe)")
        return True  # Fail-safe: assume text present

    # Load image
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return True  # Fail-safe
    except Exception as e:
        logger.warning(f"Error reading image {image_path}: {e}")
        return True  # Fail-safe

    orig_h, orig_w = image.shape[:2]

    # EAST requires dimensions divisible by 32
    # Use 320x320 for speed (larger = more accurate but slower)
    new_w, new_h = 320, 320

    # Create blob from image
    try:
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(new_w, new_h),
            mean=(123.68, 116.78, 103.94),  # ImageNet mean values
            swapRB=True,
            crop=False,
        )
    except Exception as e:
        logger.warning(f"Error creating blob from image: {e}")
        return True  # Fail-safe

    # Run forward pass
    try:
        net.setInput(blob)
        # EAST outputs two layers:
        # - feature_fusion/Conv_7/Sigmoid: confidence scores
        # - feature_fusion/concat_3: geometry (bounding boxes)
        scores, geometry = net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ])
    except Exception as e:
        logger.warning(f"Error running EAST forward pass: {e}")
        return True  # Fail-safe

    # Check if any region has confidence above threshold
    # scores shape: (1, 1, H/4, W/4)
    score_map = scores[0, 0]

    # Simple check: is any pixel above threshold?
    has_text = bool(np.any(score_map > confidence_threshold))

    logger.debug(
        f"Text detection for {image_path.name}: "
        f"{'found text' if has_text else 'no text'} "
        f"(max confidence: {score_map.max():.3f})"
    )

    return has_text


def detect_text_regions(
    image_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
) -> list[dict]:
    """Detect all text regions with bounding boxes.

    More detailed than has_text_regions() - returns actual bounding boxes
    which could be used for targeted OCR or visualization.

    Args:
        image_path: Path to image file.
        confidence_threshold: Minimum confidence for detection.
        nms_threshold: Non-maximum suppression threshold.

    Returns:
        List of dicts with keys: x, y, w, h, confidence.
        Returns empty list if no text found or on error.
    """
    net = _load_east_model()
    if net is None:
        return []

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
    except Exception:
        return []

    orig_h, orig_w = image.shape[:2]
    new_w, new_h = 320, 320
    ratio_w = orig_w / new_w
    ratio_h = orig_h / new_h

    try:
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (new_w, new_h),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        net.setInput(blob)
        scores, geometry = net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ])
    except Exception:
        return []

    # Decode predictions
    boxes, confidences = _decode_predictions(scores, geometry, confidence_threshold)

    if len(boxes) == 0:
        return []

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxesRotated(
        boxes, confidences, confidence_threshold, nms_threshold
    )

    results = []
    for i in indices:
        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[idx]
        conf = confidences[idx]

        # Get bounding rect from rotated box
        vertices = cv2.boxPoints(box)
        vertices = vertices.astype(np.int32)

        # Scale back to original image size
        vertices[:, 0] = (vertices[:, 0] * ratio_w).astype(np.int32)
        vertices[:, 1] = (vertices[:, 1] * ratio_h).astype(np.int32)

        x_min = max(0, vertices[:, 0].min())
        y_min = max(0, vertices[:, 1].min())
        x_max = min(orig_w, vertices[:, 0].max())
        y_max = min(orig_h, vertices[:, 1].max())

        results.append({
            "x": int(x_min),
            "y": int(y_min),
            "w": int(x_max - x_min),
            "h": int(y_max - y_min),
            "confidence": float(conf),
        })

    return results


def _decode_predictions(
    scores: np.ndarray,
    geometry: np.ndarray,
    min_confidence: float,
) -> tuple[list, list]:
    """Decode EAST model predictions into bounding boxes.

    Args:
        scores: Confidence scores from EAST (1, 1, H/4, W/4).
        geometry: Geometry data from EAST (1, 5, H/4, W/4).
        min_confidence: Minimum confidence threshold.

    Returns:
        Tuple of (boxes, confidences) for NMS processing.
    """
    num_rows, num_cols = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]
            if score < min_confidence:
                continue

            # Compute offset (EAST uses 4x downsampling)
            offset_x = x * 4.0
            offset_y = y * 4.0

            # Extract angle and compute sin/cos
            angle = angles_data[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Get dimensions
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            # Compute box center
            end_x = offset_x + (cos_a * x_data1[x]) + (sin_a * x_data2[x])
            end_y = offset_y - (sin_a * x_data1[x]) + (cos_a * x_data2[x])

            # Create rotated rect (center, size, angle)
            center = (end_x, end_y)
            size = (w, h)
            angle_deg = -angle * 180.0 / np.pi

            boxes.append((center, size, angle_deg))
            confidences.append(float(score))

    return boxes, confidences


def is_text_detection_available() -> bool:
    """Check if EAST text detection is available.

    Returns:
        True if the model can be loaded (or downloaded), False otherwise.
    """
    model_path = _get_model_path()
    if model_path.exists():
        return True

    # Check if we can likely download (don't actually download here)
    # Just verify the parent directory is writable
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False
