"""DINOv2 embedding extraction for visual similarity algorithms.

Owns its own DINOv2 model instance (decoupled from shots module which uses
SigLIP 2 for classification). Provides functions for:
- Single thumbnail embedding extraction
- Batch thumbnail embedding extraction
- Boundary frame (first/last) embedding extraction from video
"""

import logging
import subprocess
import tempfile
import threading
from pathlib import Path

from core.binary_resolver import find_binary, get_subprocess_kwargs

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Own DINOv2 model instance (separate from classification model in shots.py)
_model = None
_processor = None
_model_lock = threading.Lock()

# DINOv2 ViT-B/14 — self-supervised vision transformer (768-dim embeddings)
_DINOV2_MODEL_NAME = "facebook/dinov2-base"
_EMBEDDING_DIM = 768
_EMBEDDING_MODEL_TAG = "dinov2-vit-b-14"  # Tag stored on Clip.embedding_model


def _get_model():
    """Lazy load DINOv2 model and processor (thread-safe)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    with _model_lock:
        if _model is None:
            logger.info("Loading DINOv2 model for embeddings...")
            from transformers import AutoImageProcessor, AutoModel

            _processor = AutoImageProcessor.from_pretrained(_DINOV2_MODEL_NAME)
            _model = AutoModel.from_pretrained(_DINOV2_MODEL_NAME)
            logger.info("DINOv2 embedding model loaded")

    return _model, _processor


def is_model_loaded() -> bool:
    """Check if the embedding model is currently loaded."""
    return _model is not None


def unload_model():
    """Unload the embedding model to free memory."""
    global _model, _processor
    with _model_lock:
        _model = None
        _processor = None
    logger.info("DINOv2 embedding model unloaded")


def _image_to_embedding(image: Image.Image) -> list[float]:
    """Compute DINOv2 embedding for a PIL Image.

    Uses the CLS token from the last hidden state as the image representation.

    Args:
        image: PIL Image in RGB mode

    Returns:
        Normalized embedding vector as list of floats (768 dimensions)
    """
    import torch

    model, processor = _get_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token is the first token in the last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # L2 normalize for cosine similarity
        cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)

    return cls_embedding[0].cpu().numpy().tolist()


def extract_clip_embeddings_batch(
    thumbnail_paths: list[Path],
) -> list[list[float]]:
    """Extract DINOv2 embeddings for multiple thumbnails efficiently.

    Processes images in batches to leverage GPU parallelism (if available)
    or reduce per-image overhead on CPU.

    Args:
        thumbnail_paths: List of paths to thumbnail images

    Returns:
        List of normalized embedding vectors (one per input path).
        Missing/unreadable images get a zero vector.
    """
    import torch

    model, processor = _get_model()

    # Load all images
    images = []
    valid_indices = []
    for i, path in enumerate(thumbnail_paths):
        try:
            if path and path.exists():
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")

    if not images:
        # Return zero vectors for all
        return [[0.0] * _EMBEDDING_DIM for _ in thumbnail_paths]

    # Process in batches of 32
    batch_size = 32
    all_embeddings = {}

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start:batch_start + batch_size]
        batch_indices = valid_indices[batch_start:batch_start + batch_size]

        inputs = processor(images=batch_images, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)

        for j, idx in enumerate(batch_indices):
            all_embeddings[idx] = cls_embeddings[j].cpu().numpy().tolist()

    # Build result, filling missing with zero vectors
    zero_vec = [0.0] * _EMBEDDING_DIM
    return [all_embeddings.get(i, zero_vec) for i in range(len(thumbnail_paths))]


def extract_boundary_embeddings(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> tuple[list[float], list[float]]:
    """Extract DINOv2 embeddings of the first and last frames of a clip.

    Uses FFmpeg to extract the actual frames from the source video,
    then computes DINOv2 embeddings for each.

    Args:
        source_path: Path to the source video file
        start_frame: Clip start frame number
        end_frame: Clip end frame number
        fps: Video frame rate

    Returns:
        Tuple of (first_frame_embedding, last_frame_embedding),
        each a normalized embedding vector (768 dimensions)

    Raises:
        ValueError: If fps is not positive
        RuntimeError: If frame extraction or embedding fails
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    start_time = start_frame / fps
    # Last frame: one frame before end
    last_frame_time = max(start_time, (end_frame - 1) / fps)

    first_image = _extract_frame_image(source_path, start_time)
    last_image = _extract_frame_image(source_path, last_frame_time)

    first_emb = _image_to_embedding(first_image)
    last_emb = _image_to_embedding(last_image)

    return first_emb, last_emb


def _extract_frame_image(source_path: Path, time_seconds: float) -> Image.Image:
    """Extract a single frame from video at the given time as a PIL Image.

    Args:
        source_path: Path to source video
        time_seconds: Time position in seconds

    Returns:
        PIL Image in RGB mode

    Raises:
        RuntimeError: If frame extraction fails
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _ffmpeg = find_binary("ffmpeg") or "ffmpeg"
        cmd = [
            _ffmpeg, "-y",
            "-ss", str(time_seconds),
            "-i", str(source_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(tmp_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            **get_subprocess_kwargs(),
        )

        if result.returncode != 0 or not tmp_path.exists():
            raise RuntimeError(
                f"FFmpeg frame extraction failed at {time_seconds}s: {result.stderr}"
            )

        image = Image.open(tmp_path).convert("RGB")
        return image
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
