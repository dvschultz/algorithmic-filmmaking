"""Vision-Language Model (VLM) integration for video frame description.

Supports multiple tiers:
- CPU: Moondream 2B (via transformers) - optimized for standard hardware
- Cloud: GPT-4o, Claude, Gemini (via LiteLLM) - high quality, requires API key
- GPU: (Future) LLaVA/Qwen - high quality local inference
"""

import base64
import logging
import threading
from pathlib import Path
from typing import Optional, Literal

from PIL import Image
import litellm

from core.settings import load_settings

logger = logging.getLogger(__name__)

# Thread-safe model loading
_model_lock = threading.Lock()

# Global model cache to avoid reloading heavy weights
_CPU_MODEL = None
_CPU_TOKENIZER = None


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_cpu_model(model_id: str):
    """Load CPU-optimized model (Moondream).

    Thread-safe lazy loading with singleton pattern.
    """
    global _CPU_MODEL, _CPU_TOKENIZER

    # Fast path: model already loaded
    if _CPU_MODEL is not None:
        return _CPU_MODEL, _CPU_TOKENIZER

    # Thread-safe loading
    with _model_lock:
        # Double-check after acquiring lock
        if _CPU_MODEL is not None:
            return _CPU_MODEL, _CPU_TOKENIZER

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading CPU vision model: {model_id}...")

            # Load tokenizer and model
            # trust_remote_code=True is required for Moondream
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-08-26")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision="2024-08-26"
            )

            _CPU_MODEL = model
            _CPU_TOKENIZER = tokenizer
            return model, tokenizer

        except ImportError:
            logger.error("transformers, einops, or torchvision not installed. Cannot use CPU vision tier.")
            raise RuntimeError(
                "Missing dependencies for CPU vision. "
                "Please install: pip install transformers einops torchvision"
            )
        except Exception as e:
            logger.error(f"Failed to load CPU model: {e}")
            raise


def describe_frame_cpu(image_path: Path, prompt: str = "Describe this image.") -> str:
    """Generate description using local CPU model (Moondream)."""
    model_id = load_settings().description_model_cpu
    model, tokenizer = _load_cpu_model(model_id)
    
    try:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        return model.answer_question(enc_image, prompt, tokenizer)
    except Exception as e:
        logger.error(f"CPU inference failed: {e}")
        return f"Error generating description: {e}"


def describe_frame_cloud(image_path: Path, prompt: str = "Describe this image.") -> str:
    """Generate description using Cloud API (via LiteLLM)."""
    from core.settings import (
        get_openai_api_key, 
        get_anthropic_api_key, 
        get_gemini_api_key
    )
    
    settings = load_settings()
    model = settings.description_model_cloud
    
    # Ensure API keys are available
    # LiteLLM usually handles environment variables, but we can helper set them if needed
    api_key = None
    if "gpt" in model:
        api_key = get_openai_api_key()
    elif "claude" in model:
        api_key = get_anthropic_api_key()
    elif "gemini" in model:
        api_key = get_gemini_api_key()
        
    if not api_key:
        logger.warning(f"No API key found for cloud model {model}. Inference may fail.")

    base64_image = encode_image_base64(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Cloud inference failed: {e}")
        return f"Error generating description: {e}"


def describe_frame(
    image_path: Path, 
    tier: Optional[str] = None,
    prompt: str = "Describe this video frame in detail. Focus on main subjects, action, setting, and mood."
) -> tuple[str, str]:
    """Generate description for a video frame.
    
    Args:
        image_path: Path to the image file
        tier: 'cpu', 'gpu', or 'cloud'. If None, uses settings default.
        prompt: Instruction for the model
        
    Returns:
        Tuple of (description, model_name)
    """
    settings = load_settings()
    tier = tier or settings.description_model_tier
    
    logger.info(f"Describing frame {image_path.name} using {tier} tier")
    
    if tier == "cpu":
        desc = describe_frame_cpu(image_path, prompt)
        return desc, settings.description_model_cpu
        
    elif tier == "cloud":
        desc = describe_frame_cloud(image_path, prompt)
        return desc, settings.description_model_cloud
        
    elif tier == "gpu":
        # Placeholder for Phase 5
        logger.warning("GPU tier not implemented yet, falling back to CPU")
        desc = describe_frame_cpu(image_path, prompt)
        return desc, settings.description_model_cpu

    else:
        return f"Unknown tier: {tier}", "unknown"


def is_model_loaded() -> bool:
    """Check if a CPU model is currently loaded."""
    return _CPU_MODEL is not None


def unload_model():
    """Unload the CPU model to free memory."""
    global _CPU_MODEL, _CPU_TOKENIZER

    with _model_lock:
        _CPU_MODEL = None
        _CPU_TOKENIZER = None
        logger.info("VLM model unloaded")
