"""Vision-Language Model (VLM) integration for video frame description.

Supports multiple tiers:
- CPU: Moondream 2B (via transformers) - optimized for standard hardware
- Cloud: GPT-4o, Claude, Gemini (via LiteLLM) - high quality, requires API key
- GPU: (Future) LLaVA/Qwen - high quality local inference

Gemini models support video input natively, providing richer temporal understanding.
"""

import base64
import logging
import shutil
import subprocess
import tempfile
import threading
import uuid
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


# Default Moondream revision - updated to fix GenerationMixin compatibility
# with transformers v4.50+. See: https://huggingface.co/vikhyatk/moondream2/discussions/39
MOONDREAM_REVISION = "2025-06-21"


def _get_ffmpeg_path() -> str:
    """Find FFmpeg executable."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
        )
    return path


def is_video_capable_model(model: str) -> bool:
    """Check if model supports video input.

    Currently only Gemini models support native video understanding.

    Args:
        model: Model name/identifier

    Returns:
        True if the model supports video input
    """
    return "gemini" in model.lower()


def extract_clip_segment(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_dir: Optional[Path] = None,
) -> Path:
    """Extract clip segment from source video using FFmpeg.

    Args:
        source_path: Path to source video file
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Video frame rate
        output_dir: Directory for temp file (default: system temp)

    Returns:
        Path to extracted video segment (MP4 format)

    Raises:
        RuntimeError: If FFmpeg extraction fails
    """
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())

    # Generate unique filename
    output_path = output_dir / f"clip_{uuid.uuid4().hex[:8]}.mp4"

    cmd = [
        _get_ffmpeg_path(),
        "-y",
        "-ss", str(start_time),
        "-i", str(source_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"Extracting clip: {start_time:.2f}s - {start_time + duration:.2f}s")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extraction failed: {result.stderr}")

    logger.info(f"Extracted clip to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def encode_video_base64(video_path: Path) -> str:
    """Encode video to base64 string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_video_cloud(
    video_path: Path,
    prompt: str = "Describe this video clip in 3 sentences or less. Focus on the main subjects, action, and setting.",
) -> tuple[str, str]:
    """Send video to Gemini for description via LiteLLM.

    Args:
        video_path: Path to video file
        prompt: Description prompt

    Returns:
        Tuple of (description, model_name_with_suffix)

    Raises:
        ValueError: If API key is not configured
        RuntimeError: If video description fails
    """
    from core.settings import get_gemini_api_key

    settings = load_settings()
    model = settings.description_model_cloud
    original_model = model

    logger.info(f"Video description requested with model: {model}")

    # Normalize model name for LiteLLM
    if "gemini" in model.lower() and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        model = f"gemini/{model}"

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key not configured. Please add it in Settings.")

    # Encode video to base64
    base64_video = encode_video_base64(video_path)
    file_size_mb = video_path.stat().st_size / 1024 / 1024
    logger.info(f"Encoded video: {file_size_mb:.1f} MB")

    # LiteLLM message format for video (using file type)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "file",
                    "file": {
                        "file_id": f"data:video/mp4;base64,{base64_video}",
                        "format": "video/mp4",
                    },
                },
            ],
        }
    ]

    try:
        logger.info(f"Calling LiteLLM with video, model={model}")
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
        )
        description = response.choices[0].message.content

        # Validate response content
        if description is None:
            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
            logger.warning(f"Video API returned None content, finish_reason={finish_reason}")
            if finish_reason == 'content_filter':
                raise RuntimeError(f"Video content filtered by {original_model} safety policy")
            raise RuntimeError(f"Video API returned empty response (finish_reason={finish_reason})")

        if not description.strip():
            raise RuntimeError(f"Video API returned blank response from {original_model}")

        return description, f"{original_model} (video)"
    except Exception as e:
        logger.error(f"Video description failed for model {model}: {e}")
        raise RuntimeError(f"Video description failed ({original_model}): {e}") from e


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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading CPU vision model: {model_id} (revision: {MOONDREAM_REVISION})...")

            # Determine best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            logger.info(f"Using device: {device}")

            # Load tokenizer and model
            # trust_remote_code=True is required for Moondream
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=MOONDREAM_REVISION,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision=MOONDREAM_REVISION,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            )

            # Move model to device (ensures all components on same device)
            model = model.to(device)

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
        raise RuntimeError(f"CPU inference failed: {e}") from e


def describe_frame_cloud(image_path: Path, prompt: str = "Describe this image.") -> str:
    """Generate description using Cloud API (via LiteLLM)."""
    from core.settings import (
        get_openai_api_key,
        get_anthropic_api_key,
        get_gemini_api_key
    )

    settings = load_settings()
    model = settings.description_model_cloud
    original_model = model  # Keep for logging

    logger.info(f"Cloud description requested with model: {model}")

    # Normalize model name for LiteLLM
    # This ensures "gemini-..." routes to AI Studio (via API key) instead of Vertex AI
    if "gemini" in model.lower() and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        model = f"gemini/{model}"
    elif "claude" in model.lower() and not any(model.startswith(p) for p in ["anthropic/", "bedrock/"]):
        model = f"anthropic/{model}"

    logger.info(f"Normalized model name: {model}")

    # Ensure API keys are available
    # LiteLLM usually handles environment variables, but we can helper set them if needed
    api_key = None
    if "gpt" in model.lower() or "openai" in model.lower():
        api_key = get_openai_api_key()
        logger.info("Using OpenAI API key")
    elif "claude" in model.lower() or "anthropic" in model.lower():
        api_key = get_anthropic_api_key()
        logger.info("Using Anthropic API key")
    elif "gemini" in model.lower():
        api_key = get_gemini_api_key()
        logger.info("Using Gemini API key")

    if not api_key:
        raise ValueError(f"No API key found for cloud model {original_model}. Please configure the API key in Settings.")

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
        logger.info(f"Calling LiteLLM with model={model}")
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
        )
        content = response.choices[0].message.content

        # Validate response content
        if content is None:
            # Check for content filtering
            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
            logger.warning(f"API returned None content, finish_reason={finish_reason}")
            if finish_reason == 'content_filter':
                raise RuntimeError(f"Content filtered by {original_model} safety policy")
            raise RuntimeError(f"API returned empty response (finish_reason={finish_reason})")

        if not content.strip():
            raise RuntimeError(f"API returned blank response from {original_model}")

        return content
    except Exception as e:
        logger.error(f"Cloud inference failed for model {model}: {e}")
        raise RuntimeError(f"Cloud inference failed ({original_model}): {e}") from e


def describe_frame(
    image_path: Path,
    tier: Optional[str] = None,
    prompt: str = "Describe this video frame in 3 sentences or less. Focus on the main subjects, action, and setting.",
    source_path: Optional[Path] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> tuple[str, str]:
    """Generate description for a video frame or clip.

    If source video info is provided and the model supports video input,
    extracts and sends the clip for richer temporal understanding.
    Otherwise uses single frame analysis.

    Args:
        image_path: Path to the image file (fallback thumbnail)
        tier: 'cpu', 'gpu', or 'cloud'. If None, uses settings default.
        prompt: Instruction for the model
        source_path: Path to source video file (for video extraction)
        start_frame: Starting frame number of clip
        end_frame: Ending frame number of clip
        fps: Video frame rate

    Returns:
        Tuple of (description, model_name)

    Raises:
        RuntimeError: If description generation fails
        ValueError: If tier is unknown or API key is missing
    """
    settings = load_settings()
    tier = tier or settings.description_model_tier

    logger.info(f"Describing frame {image_path.name} using {tier} tier")

    if tier == "cpu":
        desc = describe_frame_cpu(image_path, prompt)
        return desc, settings.description_model_cpu

    elif tier == "cloud":
        model = settings.description_model_cloud
        input_mode = settings.description_input_mode

        # Check if we should use video input (only for Gemini when mode is "video")
        if (
            input_mode == "video"
            and is_video_capable_model(model)
            and source_path is not None
            and start_frame is not None
            and end_frame is not None
            and fps is not None
        ):
            try:
                # Extract and describe video clip
                logger.info(f"Using video mode for Gemini (frames {start_frame}-{end_frame})")
                temp_video = extract_clip_segment(
                    source_path, start_frame, end_frame, fps
                )
                try:
                    return describe_video_cloud(temp_video, prompt)
                finally:
                    # Cleanup temp file
                    if temp_video.exists():
                        temp_video.unlink()
                        logger.debug(f"Cleaned up temp video: {temp_video}")
            except Exception as e:
                logger.warning(f"Video description failed, falling back to frame: {e}")
                # Fall through to frame-based description

        # Frame-based description (default or when video mode not selected)
        logger.info(f"Using frame mode for description")
        desc = describe_frame_cloud(image_path, prompt)
        return desc, settings.description_model_cloud

    elif tier == "gpu":
        # Placeholder for Phase 5
        logger.warning("GPU tier not implemented yet, falling back to CPU")
        desc = describe_frame_cpu(image_path, prompt)
        return desc, settings.description_model_cpu

    else:
        raise ValueError(f"Unknown tier: {tier}")


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


def clear_model_cache(model_id: str = "vikhyatk/moondream2") -> bool:
    """Clear the HuggingFace cache for a specific model.

    This is useful when updating to a new model revision.

    Returns:
        True if cache was cleared, False if cache dir not found.
    """
    import shutil
    from huggingface_hub import HfFolder

    try:
        from huggingface_hub import cached_assets_path
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)

        # HuggingFace stores models in: cache_dir/models--{org}--{model}
        model_cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = cache_dir / model_cache_name

        if model_cache_path.exists():
            shutil.rmtree(model_cache_path)
            logger.info(f"Cleared model cache: {model_cache_path}")
            return True
        else:
            logger.info(f"No cache found for {model_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to clear model cache: {e}")
        return False
