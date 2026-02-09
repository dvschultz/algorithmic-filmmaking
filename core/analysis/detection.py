"""Object detection using YOLO26 and YOLOE-26 (open-vocabulary).

Supports two detection modes:
- Fixed (default): YOLO26n with 80 COCO classes — fast, zero overhead
- Open-vocabulary: YOLOE-26s with user-defined text-prompted classes
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# COCO class names (80 classes used by YOLO)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Person class index in COCO
PERSON_CLASS_ID = 0

# YOLOE-26 model for open-vocabulary detection
_YOLOE_MODEL_NAME = "yoloe-26s.pt"

# Lazy load models
_model = None
_model_lock = threading.Lock()

_ov_model = None
_ov_model_classes: Optional[list[str]] = None
_ov_model_lock = threading.Lock()


def _get_model_cache_dir() -> Path:
    """Get the model cache directory from settings."""
    try:
        from core.settings import load_settings
        settings = load_settings()
        cache_dir = settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except Exception:
        # Fallback to default
        default = Path.home() / ".cache" / "scene-ripper" / "models"
        default.mkdir(parents=True, exist_ok=True)
        return default


def _load_yolo(model_size: str = "n"):
    """
    Lazy load YOLO26 model (thread-safe).

    Args:
        model_size: Model size variant - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large)
                   Default is 'n' for fastest CPU inference.
    """
    global _model

    # Fast path: already loaded
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is None:
            logger.info(f"Loading YOLO26{model_size} model...")

            # Set ultralytics cache directory
            cache_dir = _get_model_cache_dir()
            os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))

            from ultralytics import YOLO

            # YOLO26 will download the model to cache on first use (~6MB for nano)
            model_name = f"yolo26{model_size}.pt"
            _model = YOLO(model_name)

            logger.info(f"YOLO26{model_size} model loaded")

    return _model


def _load_yoloe(custom_classes: list[str]):
    """Lazy load YOLOE-26 open-vocabulary model (thread-safe).

    Args:
        custom_classes: List of class names for text-prompted detection.
    """
    global _ov_model, _ov_model_classes

    # Fast path: already loaded with matching classes
    if _ov_model is not None and _ov_model_classes == custom_classes:
        return _ov_model

    with _ov_model_lock:
        if _ov_model is None or _ov_model_classes != custom_classes:
            logger.info(f"Loading YOLOE-26s with {len(custom_classes)} custom classes...")

            cache_dir = _get_model_cache_dir()
            os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))

            from ultralytics import YOLO

            _ov_model = YOLO(_YOLOE_MODEL_NAME)
            _ov_model.set_classes(custom_classes)
            _ov_model_classes = list(custom_classes)

            logger.info(f"YOLOE-26s loaded with classes: {custom_classes}")

    return _ov_model


def detect_objects_open_vocab(
    image_path: Path,
    custom_classes: list[str],
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """Detect objects using open-vocabulary text-prompted detection.

    Uses YOLOE-26 to detect user-specified object classes that are not
    limited to the fixed 80 COCO categories.

    Args:
        image_path: Path to image file
        custom_classes: List of class names to detect (e.g., ["camera", "microphone"])
        confidence_threshold: Minimum detection confidence (0.0-1.0).
            Lower default than fixed detection since open-vocab is less precise.

    Returns:
        List of detections, each with:
        - label: Class name from custom_classes
        - confidence: Detection confidence (0.0-1.0)
        - bbox: Bounding box as [x1, y1, x2, y2] in pixels
    """
    if not custom_classes:
        logger.warning("No custom classes provided for open-vocab detection")
        return []

    model = _load_yoloe(custom_classes)

    try:
        results = model(
            str(image_path),
            verbose=False,
            conf=confidence_threshold,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                label = custom_classes[cls_id] if cls_id < len(custom_classes) else f"class_{cls_id}"

                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [int(x) for x in bbox],
                })

        logger.debug(
            f"Open-vocab detected {len(detections)} objects in {image_path.name} "
            f"(classes: {custom_classes})"
        )
        return detections

    except Exception as e:
        logger.error(f"Open-vocab detection failed for {image_path}: {e}")
        return []


def detect_objects(
    image_path: Path,
    confidence_threshold: float = 0.5,
    classes: Optional[list[int]] = None,
    model_size: str = "n",
) -> list[dict]:
    """
    Detect objects in an image using YOLOv8.

    Args:
        image_path: Path to image file
        confidence_threshold: Minimum detection confidence (0.0-1.0)
        classes: Optional list of class IDs to filter (None = detect all classes)
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')

    Returns:
        List of detections, each with:
        - label: Class name (e.g., "person", "car")
        - confidence: Detection confidence (0.0-1.0)
        - bbox: Bounding box as [x1, y1, x2, y2] in pixels
    """
    model = _load_yolo(model_size)

    try:
        # Run inference
        results = model(
            str(image_path),
            verbose=False,
            conf=confidence_threshold,
            classes=classes,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # Get class name
                label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"

                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [int(x) for x in bbox],
                })

        logger.debug(f"Detected {len(detections)} objects in {image_path.name}")
        return detections

    except Exception as e:
        logger.error(f"Object detection failed for {image_path}: {e}")
        return []


def count_people(
    image_path: Path,
    confidence_threshold: float = 0.5,
    model_size: str = "n",
) -> int:
    """
    Count people in an image using YOLOv8.

    Args:
        image_path: Path to image file
        confidence_threshold: Minimum detection confidence
        model_size: YOLO model size

    Returns:
        Number of people detected
    """
    # Filter to person class only (class 0 in COCO)
    detections = detect_objects(
        image_path,
        confidence_threshold=confidence_threshold,
        classes=[PERSON_CLASS_ID],
        model_size=model_size,
    )
    return len(detections)


def get_object_counts(
    image_path: Path,
    confidence_threshold: float = 0.5,
    model_size: str = "n",
) -> dict[str, int]:
    """
    Get counts of each detected object type.

    Args:
        image_path: Path to image file
        confidence_threshold: Minimum detection confidence
        model_size: YOLO model size

    Returns:
        Dictionary mapping object labels to counts, e.g., {"person": 3, "car": 2}
    """
    detections = detect_objects(
        image_path,
        confidence_threshold=confidence_threshold,
        model_size=model_size,
    )

    counts: dict[str, int] = {}
    for det in detections:
        label = det["label"]
        counts[label] = counts.get(label, 0) + 1

    return counts


def get_unique_labels(
    image_path: Path,
    confidence_threshold: float = 0.5,
    model_size: str = "n",
) -> list[str]:
    """
    Get list of unique object labels detected in an image.

    Args:
        image_path: Path to image file
        confidence_threshold: Minimum detection confidence
        model_size: YOLO model size

    Returns:
        List of unique object labels sorted alphabetically
    """
    detections = detect_objects(
        image_path,
        confidence_threshold=confidence_threshold,
        model_size=model_size,
    )

    labels = set(det["label"] for det in detections)
    return sorted(labels)


def is_model_loaded() -> bool:
    """Check if the YOLO model is already loaded."""
    return _model is not None


def unload_model():
    """Unload all YOLO models to free memory."""
    global _model, _ov_model, _ov_model_classes

    with _model_lock:
        _model = None

    with _ov_model_lock:
        _ov_model = None
        _ov_model_classes = None

    logger.info("YOLO models unloaded")
