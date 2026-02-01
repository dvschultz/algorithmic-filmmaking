"""Cloud-based shot type classification using Replicate.

Uses VideoMAE model fine-tuned for cinematographic shot types.
Provides significantly better accuracy than CLIP-based classification.

Shot types:
- LS: Long Shot (wide, establishing)
- FS: Full Shot (full body visible)
- MS: Medium Shot (waist up)
- CS: Close-up Shot (head and shoulders)
- ECS: Extreme Close-up Shot (face detail only)
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Shot type display names
SHOT_TYPE_DISPLAY = {
    "LS": "Long Shot",
    "FS": "Full Shot",
    "MS": "Medium Shot",
    "CS": "Close-up",
    "ECS": "Extreme Close-up",
}

# Map VideoMAE labels to simplified categories for UI consistency
SHOT_TYPE_SIMPLIFIED = {
    "LS": "wide shot",
    "FS": "full shot",
    "MS": "medium shot",
    "CS": "close-up",
    "ECS": "extreme close-up",
}


def get_replicate_api_key() -> Optional[str]:
    """Get Replicate API key from settings or environment."""
    import os
    from core.settings import load_settings

    # Check environment first
    env_key = os.environ.get("REPLICATE_API_TOKEN")
    if env_key:
        return env_key

    # Check keyring
    try:
        import keyring
        key = keyring.get_password("scene-ripper", "replicate_api_key")
        if key:
            return key
    except Exception:
        pass

    return None


def set_replicate_api_key(api_key: str) -> bool:
    """Store Replicate API key in keyring."""
    try:
        import keyring
        keyring.set_password("scene-ripper", "replicate_api_key", api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to store Replicate API key: {e}")
        return False


def classify_shot_replicate(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    model_version: Optional[str] = None,
) -> tuple[str, float, dict]:
    """Classify shot type using VideoMAE on Replicate.

    Args:
        source_path: Path to source video
        start_frame: Clip start frame
        end_frame: Clip end frame
        fps: Video frame rate
        model_version: Replicate model version (default from settings)

    Returns:
        Tuple of (shot_type, confidence, all_scores)

    Raises:
        ValueError: If API key not configured
        RuntimeError: If classification fails
    """
    import replicate

    from core.analysis.description import extract_clip_segment
    from core.settings import load_settings

    api_key = get_replicate_api_key()
    if not api_key:
        raise ValueError(
            "Replicate API key not configured. "
            "Set REPLICATE_API_TOKEN environment variable or configure in Settings."
        )

    settings = load_settings()
    model_version = model_version or settings.shot_classifier_replicate_model

    if not model_version:
        raise ValueError(
            "Replicate model version not configured. "
            "Set shot_classifier_replicate_model in settings."
        )

    # Extract clip segment
    logger.info(f"Extracting clip: frames {start_frame}-{end_frame}")
    temp_video = extract_clip_segment(source_path, start_frame, end_frame, fps)

    try:
        logger.info(f"Calling Replicate model: {model_version}")

        # Run prediction
        output = replicate.run(
            model_version,
            input={
                "video": open(temp_video, "rb"),
                "return_all_scores": True,
            },
        )

        shot_type = output.get("shot_type", "unknown")
        confidence = output.get("confidence", 0.0)
        all_scores = output.get("all_scores", {})

        logger.info(f"Classification result: {shot_type} ({confidence:.2%})")

        return shot_type, confidence, all_scores

    except Exception as e:
        logger.error(f"Replicate classification failed: {e}")
        raise RuntimeError(f"Shot classification failed: {e}") from e

    finally:
        # Cleanup temp file
        if temp_video.exists():
            temp_video.unlink()
            logger.debug(f"Cleaned up temp video: {temp_video}")


def classify_shot_from_thumbnail(
    thumbnail_path: Path,
    model_version: Optional[str] = None,
) -> tuple[str, float]:
    """Classify shot type from a single thumbnail image.

    Note: This is less accurate than video-based classification.
    Consider using classify_shot_replicate with video input when possible.

    Args:
        thumbnail_path: Path to thumbnail image
        model_version: Replicate model version

    Returns:
        Tuple of (shot_type, confidence)
    """
    # For single images, fall back to CLIP or VLM
    # VideoMAE requires video input (16 frames)
    logger.warning(
        "VideoMAE requires video input. "
        "Falling back to CLIP for single thumbnail classification."
    )

    from core.analysis.shots import classify_shot_type
    return classify_shot_type(thumbnail_path)


def get_display_name(shot_type: str) -> str:
    """Get human-readable display name for shot type."""
    return SHOT_TYPE_DISPLAY.get(shot_type, shot_type)


def get_simplified_type(shot_type: str) -> str:
    """Get simplified shot type label for consistency."""
    return SHOT_TYPE_SIMPLIFIED.get(shot_type, shot_type.lower())
