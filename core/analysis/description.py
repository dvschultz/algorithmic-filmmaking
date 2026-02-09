"""Vision-Language Model (VLM) integration for video frame description.

Supports multiple tiers:
- Local: Qwen3-VL 4B (via mlx-vlm) - high quality on Apple Silicon
- Cloud: GPT-4o, Claude, Gemini (via LiteLLM) - high quality, requires API key

Gemini models support video input natively, providing richer temporal understanding.
"""

import base64
import logging
import platform
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
_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None

# Local VLM model
_LOCAL_VLM_NAME = "mlx-community/Qwen3-VL-4B-4bit"
_LOCAL_VLM_FALLBACK = "vikhyatk/moondream2"  # Fallback for non-Apple-Silicon
MOONDREAM_REVISION = "2025-06-21"

# Backward-compatible aliases
_CPU_MODEL = None
_CPU_TOKENIZER = None


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_ffmpeg_path() -> str:
    """Find FFmpeg executable."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
        )
    return path


def is_mlx_vlm_available() -> bool:
    """Check if mlx-vlm is available (Apple Silicon only)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def is_video_capable_model(model: str) -> bool:
    """Check if model supports video input.

    Gemini models and Qwen3-VL support native video understanding.

    Args:
        model: Model name/identifier

    Returns:
        True if the model supports video input
    """
    model_lower = model.lower()
    return "gemini" in model_lower or "qwen" in model_lower


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


def _load_local_model():
    """Load local VLM model (thread-safe).

    Uses Qwen3-VL via mlx-vlm on Apple Silicon, falls back to Moondream otherwise.
    """
    global _LOCAL_MODEL, _LOCAL_PROCESSOR

    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR

    with _model_lock:
        if _LOCAL_MODEL is not None:
            return _LOCAL_MODEL, _LOCAL_PROCESSOR

        if is_mlx_vlm_available():
            _load_qwen3_vlm()
        else:
            _load_moondream_fallback()

    return _LOCAL_MODEL, _LOCAL_PROCESSOR


def _load_qwen3_vlm():
    """Load Qwen3-VL via mlx-vlm."""
    global _LOCAL_MODEL, _LOCAL_PROCESSOR

    from mlx_vlm import load

    settings = load_settings()
    model_id = settings.description_model_local

    logger.info(f"Loading local VLM via mlx-vlm: {model_id}")
    _LOCAL_MODEL, _LOCAL_PROCESSOR = load(model_id)
    logger.info(f"Local VLM loaded: {model_id}")


def _load_moondream_fallback():
    """Load Moondream as fallback for non-Apple-Silicon."""
    global _LOCAL_MODEL, _LOCAL_PROCESSOR, _CPU_MODEL, _CPU_TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = load_settings()
    model_id = settings.description_model_local

    # If the setting points to a Qwen mlx model but we can't use mlx, use fallback
    if "mlx" in model_id.lower() or "qwen" in model_id.lower():
        model_id = _LOCAL_VLM_FALLBACK
        logger.warning(f"mlx-vlm not available, falling back to {model_id}")

    logger.info(f"Loading CPU vision model: {model_id} (revision: {MOONDREAM_REVISION})...")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

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
    model = model.to(device)

    _LOCAL_MODEL = model
    _LOCAL_PROCESSOR = tokenizer
    # Backward-compatible aliases
    _CPU_MODEL = model
    _CPU_TOKENIZER = tokenizer
    logger.info(f"CPU vision model loaded: {model_id}")


def describe_frame_local(image_path: Path, prompt: str = "Describe this image.") -> str:
    """Generate description using local VLM (Qwen3-VL or Moondream fallback)."""
    model, processor = _load_local_model()

    if is_mlx_vlm_available():
        return _describe_with_mlx_vlm(model, processor, str(image_path), prompt)
    else:
        return _describe_with_moondream(model, processor, image_path, prompt)


def _describe_with_mlx_vlm(model, processor, image_path: str, prompt: str) -> str:
    """Generate description using mlx-vlm."""
    from mlx_vlm import generate

    try:
        return generate(model, processor, image_path, prompt, max_tokens=256)
    except Exception as e:
        logger.error(f"mlx-vlm inference failed: {e}")
        raise RuntimeError(f"Local VLM inference failed: {e}") from e


def _describe_with_moondream(model, tokenizer, image_path: Path, prompt: str) -> str:
    """Generate description using Moondream (fallback)."""
    try:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        return model.answer_question(enc_image, prompt, tokenizer)
    except Exception as e:
        logger.error(f"Moondream inference failed: {e}")
        raise RuntimeError(f"CPU inference failed: {e}") from e


# Backward-compatible alias
def describe_frame_cpu(image_path: Path, prompt: str = "Describe this image.") -> str:
    """Generate description using local model. Alias for describe_frame_local."""
    return describe_frame_local(image_path, prompt)


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
        tier: 'local', 'cloud' (also accepts legacy 'cpu'/'gpu'). If None, uses settings default.
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

    # Normalize legacy tier names
    if tier in ("cpu", "gpu"):
        tier = "local"

    logger.info(f"Describing frame {image_path.name} using {tier} tier")

    if tier == "local":
        desc = describe_frame_local(image_path, prompt)
        return desc, settings.description_model_local

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
                logger.info(f"Using video mode for cloud VLM (frames {start_frame}-{end_frame})")
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

    else:
        raise ValueError(f"Unknown tier: {tier}")


def is_model_loaded() -> bool:
    """Check if a local model is currently loaded."""
    return _LOCAL_MODEL is not None


def unload_model():
    """Unload the local model to free memory."""
    global _LOCAL_MODEL, _LOCAL_PROCESSOR, _CPU_MODEL, _CPU_TOKENIZER

    with _model_lock:
        _LOCAL_MODEL = None
        _LOCAL_PROCESSOR = None
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
