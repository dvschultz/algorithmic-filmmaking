"""Shot type classification using CLIP zero-shot."""

import logging
import threading
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Shot type categories for zero-shot classification
SHOT_TYPES = [
    "wide shot",
    "medium shot",
    "close-up",
    "extreme close-up",
]

# Human-readable display names
SHOT_TYPE_DISPLAY = {
    "wide shot": "Wide",
    "medium shot": "Medium",
    "close-up": "Close-up",
    "extreme close-up": "Extreme CU",
}

# Lazy load models to avoid slow import
_clip_model = None
_clip_processor = None
_clip_model_lock = threading.Lock()

# Pin model revision for supply chain security
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_CLIP_MODEL_REVISION = "e6a30b603a447e251fdaca1c3056b2a16cdfebeb"


def _load_clip_model():
    """Lazy load CLIP model and processor (thread-safe)."""
    global _clip_model, _clip_processor

    # Fast path: already loaded
    if _clip_model is not None:
        return _clip_model, _clip_processor

    with _clip_model_lock:
        # Double-check after acquiring lock
        if _clip_model is None:
            logger.info("Loading CLIP model...")
            from transformers import CLIPProcessor, CLIPModel

            # Use base CLIP model - good balance of speed and accuracy
            # Pin revision for reproducibility and supply chain security
            _clip_processor = CLIPProcessor.from_pretrained(
                _CLIP_MODEL_NAME, revision=_CLIP_MODEL_REVISION
            )
            _clip_model = CLIPModel.from_pretrained(
                _CLIP_MODEL_NAME, revision=_CLIP_MODEL_REVISION
            )
            logger.info("CLIP model loaded")

    return _clip_model, _clip_processor


def classify_shot_type(
    image_path: Path,
    threshold: float = 0.0,
) -> tuple[str, float]:
    """
    Classify the shot type of an image using CLIP zero-shot classification.

    Args:
        image_path: Path to the image file (thumbnail)
        threshold: Minimum confidence threshold (0.0 to accept any)

    Returns:
        Tuple of (shot_type, confidence) where shot_type is one of SHOT_TYPES
        and confidence is a float between 0 and 1
    """
    try:
        import torch

        model, processor = _load_clip_model()

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")

        # Prepare text prompts - format as descriptions for CLIP
        text_prompts = [f"a {shot_type} of a scene" for shot_type in SHOT_TYPES]

        # Process inputs
        inputs = processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Get best match
        best_idx = probs.argmax().item()
        confidence = probs[0, best_idx].item()
        shot_type = SHOT_TYPES[best_idx]

        logger.debug(f"Shot type for {image_path.name}: {shot_type} ({confidence:.2f})")

        if confidence < threshold:
            return ("unknown", confidence)

        return (shot_type, confidence)

    except Exception as e:
        logger.error(f"Error classifying shot type: {e}")
        return ("unknown", 0.0)


def get_display_name(shot_type: str) -> str:
    """Get human-readable display name for shot type."""
    return SHOT_TYPE_DISPLAY.get(shot_type, shot_type.title())
