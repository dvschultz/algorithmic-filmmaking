"""CLIP embedding extraction for visual similarity algorithms.

Reuses the CLIP model from core/analysis/shots.py (single model instance).
Provides functions for:
- Single thumbnail embedding extraction
- Batch thumbnail embedding extraction
- Boundary frame (first/last) embedding extraction from video
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _get_clip_model():
    """Get the shared CLIP model and processor from shots module."""
    from core.analysis.shots import load_clip_model
    return load_clip_model()


def _image_to_embedding(image: Image.Image) -> list[float]:
    """Compute CLIP embedding for a PIL Image.

    Args:
        image: PIL Image in RGB mode

    Returns:
        Normalized embedding vector as list of floats (512 dimensions)
    """
    import torch

    model, processor = _get_clip_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # L2 normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features[0].cpu().numpy().tolist()


def extract_clip_embeddings_batch(
    thumbnail_paths: list[Path],
) -> list[list[float]]:
    """Extract CLIP embeddings for multiple thumbnails efficiently.

    Processes images in batches to leverage GPU parallelism (if available)
    or reduce per-image overhead on CPU.

    Args:
        thumbnail_paths: List of paths to thumbnail images

    Returns:
        List of normalized embedding vectors (one per input path).
        Missing/unreadable images get a zero vector.
    """
    import torch

    model, processor = _get_clip_model()

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
        return [[0.0] * 512 for _ in thumbnail_paths]

    # Process in batches of 32
    batch_size = 32
    all_embeddings = {}

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start:batch_start + batch_size]
        batch_indices = valid_indices[batch_start:batch_start + batch_size]

        inputs = processor(images=batch_images, return_tensors="pt", padding=True)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for j, idx in enumerate(batch_indices):
            all_embeddings[idx] = image_features[j].cpu().numpy().tolist()

    # Build result, filling missing with zero vectors
    zero_vec = [0.0] * 512
    return [all_embeddings.get(i, zero_vec) for i in range(len(thumbnail_paths))]


def extract_boundary_embeddings(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> tuple[list[float], list[float]]:
    """Extract CLIP embeddings of the first and last frames of a clip.

    Uses FFmpeg to extract the actual frames from the source video,
    then computes CLIP embeddings for each.

    Args:
        source_path: Path to the source video file
        start_frame: Clip start frame number
        end_frame: Clip end frame number
        fps: Video frame rate

    Returns:
        Tuple of (first_frame_embedding, last_frame_embedding),
        each a normalized CLIP embedding vector (512 dimensions)

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
        cmd = [
            "ffmpeg", "-y",
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
