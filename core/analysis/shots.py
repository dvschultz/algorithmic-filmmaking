"""Shot type classification using SigLIP 2 zero-shot or cloud inference.

Supports two tiers:
- Local (default): SigLIP 2-based zero-shot classification on thumbnails (free, local)
- Cloud: Gemini Flash Lite or Replicate VideoMAE (paid, more accurate)
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Shot type categories for zero-shot classification
# Matches VideoMAE categories: LS, FS, MS, CS, ECS
SHOT_TYPES = [
    "wide shot",       # LS - Long Shot
    "full shot",       # FS - Full Shot (full body visible)
    "medium shot",     # MS - Medium Shot (waist up)
    "close-up",        # CS - Close-up (head and shoulders)
    "extreme close-up", # ECS - Extreme Close-up (face detail)
]

# Detailed prompts for SigLIP 2 zero-shot classification
# SigLIP uses sigmoid per-label (not softmax), so prompts should be
# self-contained descriptions that work independently
SHOT_TYPE_PROMPTS = {
    "wide shot": [
        "This is a photo of an establishing shot showing a vast landscape or cityscape.",
        "This is a photo of a long shot where people appear very small in the environment.",
        "This is a photo of a wide angle shot of a large space with tiny distant figures.",
        "This is a photo of a panoramic view showing the entire location.",
    ],
    "full shot": [
        "This is a photo of a shot showing one person's entire body from head to feet.",
        "This is a photo of a single person standing with their full body visible in frame.",
        "This is a photo of a full length portrait of someone from head to toe.",
        "This is a photo of a shot framing one standing figure completely.",
    ],
    "medium shot": [
        "This is a photo of a medium shot showing a person from the waist up to their head.",
        "This is a photo of two or three people shown from the waist up in conversation.",
        "This is a photo of a shot of people sitting at a table showing their upper bodies.",
        "This is a photo of a cowboy shot showing someone from mid-thigh to head.",
    ],
    "close-up": [
        "This is a photo of a close-up of a person's face filling most of the frame.",
        "This is a photo of a head and shoulders shot focusing on facial expression.",
        "This is a photo of a tight shot of someone's face showing emotion.",
        "This is a photo of a portrait shot from the neck up.",
    ],
    "extreme close-up": [
        "This is a photo of an extreme close-up showing only eyes filling the screen.",
        "This is a photo of a shot of just lips or mouth in extreme detail.",
        "This is a photo of a macro shot of a single facial feature like an eye.",
        "This is a photo of an intense close-up where only part of a face is visible.",
    ],
}

# Human-readable display names
SHOT_TYPE_DISPLAY = {
    "wide shot": "Wide",
    "full shot": "Full",
    "medium shot": "Medium",
    "close-up": "Close-up",
    "extreme close-up": "Extreme CU",
}

# Lazy load models to avoid slow import
_model = None
_processor = None
_model_lock = threading.Lock()

# SigLIP 2 model for zero-shot shot type classification
_SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"


def load_classification_model():
    """Lazy load SigLIP 2 model and processor (thread-safe)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    with _model_lock:
        if _model is None:
            logger.info("Loading SigLIP 2 model for shot classification...")
            from transformers import AutoProcessor, AutoModel

            _processor = AutoProcessor.from_pretrained(_SIGLIP_MODEL_NAME)
            _model = AutoModel.from_pretrained(_SIGLIP_MODEL_NAME)
            logger.info("SigLIP 2 model loaded")

    return _model, _processor


def is_model_loaded() -> bool:
    """Check if the classification model is currently loaded."""
    return _model is not None


def unload_model():
    """Unload the classification model to free memory."""
    global _model, _processor
    with _model_lock:
        _model = None
        _processor = None
    logger.info("SigLIP 2 classification model unloaded")


# Keep backward-compatible alias for any code that imports load_clip_model
load_clip_model = load_classification_model


def classify_shot_type(
    image_path: Path,
    threshold: float = 0.0,
    use_ensemble: bool = True,
) -> tuple[str, float]:
    """
    Classify the shot type of an image using SigLIP 2 zero-shot classification.

    SigLIP 2 uses sigmoid scoring (independent per-label probability) rather
    than softmax. Scores are averaged across ensemble prompts per category.

    Args:
        image_path: Path to the image file (thumbnail)
        threshold: Minimum confidence threshold (0.0 to accept any)
        use_ensemble: If True, use multiple prompts per category for better accuracy

    Returns:
        Tuple of (shot_type, confidence) where shot_type is one of SHOT_TYPES
        and confidence is a float between 0 and 1
    """
    try:
        import torch

        model, processor = load_classification_model()

        image = Image.open(image_path).convert("RGB")

        if use_ensemble and SHOT_TYPE_PROMPTS:
            # Ensemble approach: average sigmoid scores across multiple prompts per category
            all_prompts = []
            prompt_to_category = []
            for shot_type in SHOT_TYPES:
                prompts = SHOT_TYPE_PROMPTS.get(shot_type, [f"This is a photo of {shot_type}."])
                all_prompts.extend(prompts)
                prompt_to_category.extend([shot_type] * len(prompts))

            inputs = processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                # SigLIP uses sigmoid, not softmax
                probs = torch.sigmoid(logits_per_image)[0]

            # Average probabilities per category
            category_scores = {st: [] for st in SHOT_TYPES}
            for i, prob in enumerate(probs):
                category_scores[prompt_to_category[i]].append(prob.item())

            avg_scores = {st: sum(scores) / len(scores) for st, scores in category_scores.items()}

            best_type = max(avg_scores, key=avg_scores.get)
            confidence = avg_scores[best_type]
        else:
            # Simple approach: one prompt per category
            text_prompts = [f"This is a photo of {shot_type}." for shot_type in SHOT_TYPES]

            inputs = processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.sigmoid(logits_per_image)

            best_idx = probs.argmax().item()
            confidence = probs[0, best_idx].item()
            best_type = SHOT_TYPES[best_idx]

        logger.debug(f"Shot type for {image_path.name}: {best_type} ({confidence:.2f})")

        if confidence < threshold:
            return ("unknown", confidence)

        return (best_type, confidence)

    except Exception as e:
        logger.error(f"Error classifying shot type: {e}")
        return ("unknown", 0.0)


def get_display_name(shot_type: str) -> str:
    """Get human-readable display name for shot type."""
    return SHOT_TYPE_DISPLAY.get(shot_type, shot_type.title())


def classify_shot_type_tiered(
    image_path: Path,
    source_path: Optional[Path] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
    threshold: float = 0.0,
    use_ensemble: bool = True,
) -> tuple[str, float]:
    """Classify shot type using the configured tier (local or cloud).

    Routes to either local SigLIP 2 classification or cloud classification
    (Gemini Flash Lite or Replicate VideoMAE) based on the
    shot_classifier_tier setting.

    Args:
        image_path: Path to the thumbnail image (used for local tier)
        source_path: Path to source video (used for cloud Replicate tier)
        start_frame: Clip start frame (used for cloud Replicate tier)
        end_frame: Clip end frame (used for cloud Replicate tier)
        fps: Video frame rate (used for cloud Replicate tier)
        threshold: Minimum confidence threshold
        use_ensemble: Whether to use ensemble prompts (local tier only)

    Returns:
        Tuple of (shot_type, confidence)
    """
    from core.settings import load_settings

    settings = load_settings()
    tier = settings.shot_classifier_tier

    if tier == "cloud":
        try:
            from core.analysis.shots_cloud import classify_shot_cloud

            shot_type, confidence = classify_shot_cloud(
                image_path=image_path,
                model=getattr(settings, "shot_classifier_cloud_model", None),
            )

            if confidence < threshold:
                return ("unknown", confidence)

            return (shot_type, confidence)

        except ValueError as e:
            logger.warning(f"Cloud classification unavailable: {e}. Falling back to local.")
            return classify_shot_type(image_path, threshold, use_ensemble)
        except RuntimeError as e:
            logger.error(f"Cloud classification failed: {e}. Falling back to local.")
            return classify_shot_type(image_path, threshold, use_ensemble)

    # Local tier (default): use SigLIP 2
    return classify_shot_type(image_path, threshold, use_ensemble)
