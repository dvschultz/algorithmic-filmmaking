"""Frame classification using MobileNetV3 for ImageNet labels."""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet class labels URL
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Lazy load model and labels
_model = None
_model_lock = threading.Lock()
_labels: Optional[list[str]] = None
_preprocess = None


def _get_model_cache_dir() -> Path:
    """Get the model cache directory from settings."""
    try:
        from core.settings import load_settings
        settings = load_settings()
        cache_dir = settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except Exception:
        # Fallback to platform-appropriate default
        if sys.platform == "win32":
            import os as _os
            base = Path(_os.environ.get("LOCALAPPDATA", str(Path.home())))
            default = base / "scene-ripper" / "cache" / "models"
        else:
            default = Path.home() / ".cache" / "scene-ripper" / "models"
        default.mkdir(parents=True, exist_ok=True)
        return default


def _load_model():
    """Lazy load MobileNetV3-Small model and ImageNet labels (thread-safe)."""
    global _model, _labels, _preprocess

    # Fast path: already loaded
    if _model is not None:
        return _model, _labels, _preprocess

    with _model_lock:
        # Double-check after acquiring lock
        if _model is None:
            logger.info("Loading MobileNetV3-Small model...")

            # Set torch hub cache to our model directory
            cache_dir = _get_model_cache_dir()
            os.environ.setdefault("TORCH_HOME", str(cache_dir))

            import torch
            from torchvision import models, transforms

            # Use MobileNetV3-Small with ImageNet weights
            _model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            _model.eval()

            # Create preprocessing pipeline
            _preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            # Load ImageNet labels
            labels_path = cache_dir / "imagenet_classes.txt"
            if labels_path.exists():
                with open(labels_path, "r") as f:
                    _labels = [line.strip() for line in f.readlines()]
            else:
                # Download labels
                import urllib.request
                logger.info("Downloading ImageNet class labels...")
                try:
                    with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=30) as response:
                        content = response.read().decode("utf-8")
                        _labels = [line.strip() for line in content.strip().split("\n")]
                        # Cache locally
                        with open(labels_path, "w") as f:
                            f.write(content)
                except Exception as e:
                    logger.warning(f"Could not download ImageNet labels: {e}")
                    # Use numeric labels as fallback
                    _labels = [f"class_{i}" for i in range(1000)]

            logger.info("MobileNetV3-Small model loaded")

    return _model, _labels, _preprocess


def classify_frame(
    image_path: Path,
    top_k: int = 5,
    threshold: float = 0.1,
) -> list[tuple[str, float]]:
    """
    Classify objects in a frame using MobileNetV3.

    Args:
        image_path: Path to image file
        top_k: Number of top predictions to return
        threshold: Minimum confidence threshold

    Returns:
        List of (label, confidence) tuples sorted by confidence descending
    """
    import torch

    model, labels, preprocess = _load_model()

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs, top_indices):
            confidence = prob.item()
            if confidence >= threshold:
                label = labels[idx.item()] if idx.item() < len(labels) else f"class_{idx.item()}"
                results.append((label, round(confidence, 4)))

        logger.debug(f"Classification for {image_path.name}: {results[:3]}")
        return results

    except Exception as e:
        logger.error(f"Classification failed for {image_path}: {e}")
        return []


def get_top_labels(
    image_path: Path,
    top_k: int = 5,
    threshold: float = 0.1,
) -> list[str]:
    """
    Get top classification labels for an image (labels only, no confidence).

    Args:
        image_path: Path to image file
        top_k: Number of top labels to return
        threshold: Minimum confidence threshold

    Returns:
        List of label strings
    """
    results = classify_frame(image_path, top_k=top_k, threshold=threshold)
    return [label for label, _ in results]


def is_model_loaded() -> bool:
    """Check if the classification model is already loaded."""
    return _model is not None


def unload_model():
    """Unload the classification model to free memory."""
    global _model, _labels, _preprocess

    with _model_lock:
        _model = None
        _labels = None
        _preprocess = None
        logger.info("Classification model unloaded")
