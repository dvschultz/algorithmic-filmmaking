"""Shot type classification using CLIP zero-shot."""

import logging
import threading
from pathlib import Path

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

# Detailed prompts for better CLIP zero-shot classification
# Multiple prompts per category improve accuracy through ensemble
# Note: Prompts optimized for human subjects (most common in film)
# For non-human subjects, accuracy may vary
SHOT_TYPE_PROMPTS = {
    "wide shot": [
        "an establishing shot showing a vast landscape or cityscape",
        "a long shot where people appear very small in the environment",
        "a wide angle shot of a large space with tiny distant figures",
        "a panoramic view showing the entire location",
    ],
    "full shot": [
        "a shot showing one person's entire body from head to feet",
        "a single person standing with their full body visible in frame",
        "a full length portrait of someone from head to toe",
        "a shot framing one standing figure completely",
    ],
    "medium shot": [
        "a medium shot showing a person from the waist up to their head",
        "two or three people shown from the waist up in conversation",
        "a shot of people sitting at a table showing their upper bodies",
        "a cowboy shot showing someone from mid-thigh to head",
    ],
    "close-up": [
        "a close-up of a person's face filling most of the frame",
        "a head and shoulders shot focusing on facial expression",
        "a tight shot of someone's face showing emotion",
        "a portrait shot from the neck up",
    ],
    "extreme close-up": [
        "an extreme close-up showing only eyes filling the screen",
        "a shot of just lips or mouth in extreme detail",
        "a macro shot of a single facial feature like an eye",
        "an intense close-up where only part of a face is visible",
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
    use_ensemble: bool = True,
) -> tuple[str, float]:
    """
    Classify the shot type of an image using CLIP zero-shot classification.

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

        model, processor = _load_clip_model()

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")

        if use_ensemble and SHOT_TYPE_PROMPTS:
            # Ensemble approach: average scores across multiple prompts per category
            all_prompts = []
            prompt_to_category = []
            for shot_type in SHOT_TYPES:
                prompts = SHOT_TYPE_PROMPTS.get(shot_type, [f"a {shot_type} of a scene"])
                all_prompts.extend(prompts)
                prompt_to_category.extend([shot_type] * len(prompts))

            # Process all prompts at once
            inputs = processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]

            # Average probabilities per category
            category_scores = {st: [] for st in SHOT_TYPES}
            for i, prob in enumerate(probs):
                category_scores[prompt_to_category[i]].append(prob.item())

            avg_scores = {st: sum(scores) / len(scores) for st, scores in category_scores.items()}

            # Find best category
            best_type = max(avg_scores, key=avg_scores.get)
            confidence = avg_scores[best_type]
        else:
            # Simple approach: one prompt per category
            text_prompts = [f"a {shot_type} of a scene" for shot_type in SHOT_TYPES]

            inputs = processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

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
