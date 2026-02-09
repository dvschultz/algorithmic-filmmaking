"""Cloud-based shot type classification using Gemini Flash Lite or Replicate.

Default: Gemini Flash Lite via LiteLLM (19x cheaper than Replicate VideoMAE).
Legacy: Replicate VideoMAE kept for backward compatibility.

Shot types:
- LS: Long Shot (wide, establishing)
- FS: Full Shot (full body visible)
- MS: Medium Shot (waist up)
- CS: Close-up Shot (head and shoulders)
- ECS: Extreme Close-up Shot (face detail only)
"""

import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cloud model for shot classification
_DEFAULT_CLOUD_MODEL = "gemini-2.5-flash-lite"

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


def classify_shot_cloud(
    image_path: Path,
    model: Optional[str] = None,
) -> tuple[str, float]:
    """Classify shot type using a cloud VLM (Gemini Flash Lite by default).

    Sends the thumbnail to a cloud VLM with a structured prompt requesting
    one of the standard SHOT_TYPES. Much cheaper than Replicate VideoMAE
    (~$0.00026/clip vs $0.005/clip).

    Args:
        image_path: Path to the thumbnail image
        model: LiteLLM model string (default: gemini-2.5-flash-lite)

    Returns:
        Tuple of (shot_type, confidence)

    Raises:
        ValueError: If no API key is configured for the model's provider
        RuntimeError: If classification fails
    """
    import litellm

    from core.analysis.shots import SHOT_TYPES
    from core.settings import get_gemini_api_key, load_settings

    model = model or _DEFAULT_CLOUD_MODEL

    # Get API key based on model provider
    api_key = None
    if "gemini" in model.lower():
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY environment variable or configure in Settings."
            )
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"

    # Encode image as base64
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # Determine mime type
    suffix = image_path.suffix.lower()
    mime_type = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

    shot_types_str = ", ".join(f'"{st}"' for st in SHOT_TYPES)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Classify this film frame into exactly one shot type. "
                        f"Valid types: {shot_types_str}.\n\n"
                        f"Return ONLY a JSON object: "
                        f'{{\"shot_type\": \"<type>\", \"confidence\": <0.0-1.0>}}'
                    ),
                },
            ],
        }
    ]

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=0.0,
            max_tokens=100,
        )

        text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]

        result = json.loads(text)
        shot_type = result.get("shot_type", "unknown")
        confidence = float(result.get("confidence", 0.5))

        # Validate shot type
        if shot_type not in SHOT_TYPES:
            # Try fuzzy matching
            shot_type_lower = shot_type.lower().strip()
            for st in SHOT_TYPES:
                if st in shot_type_lower or shot_type_lower in st:
                    shot_type = st
                    break
            else:
                logger.warning(f"Cloud returned unknown shot type: {shot_type}")
                shot_type = "unknown"

        logger.debug(f"Cloud shot classification: {shot_type} ({confidence:.2f})")
        return (shot_type, confidence)

    except json.JSONDecodeError:
        logger.error(f"Cloud shot classification returned non-JSON: {text}")
        raise RuntimeError("Cloud shot classification returned invalid response")
    except Exception as e:
        logger.error(f"Cloud shot classification failed: {e}")
        raise RuntimeError(f"Cloud shot classification failed: {e}") from e


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
