"""Face detection and embedding extraction using InsightFace/ArcFace.

Provides face detection, embedding extraction, and comparison for
person identification. Used by the Rose Hobart sequencer and the
standalone "Detect Faces" analysis operation.
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_lock = threading.Lock()

# Sensitivity presets (cosine similarity thresholds for ArcFace)
SENSITIVITY_PRESETS = {
    "strict": 0.50,   # Very high confidence, frontal faces only
    "balanced": 0.35,  # Good accuracy, allows angled faces
    "loose": 0.25,     # Permissive, may include ambiguous matches
}


def _get_model_cache_dir() -> Path:
    """Get the model cache directory from settings."""
    try:
        from core.settings import load_settings
        settings = load_settings()
        cache_dir = settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except Exception:
        if sys.platform == "win32":
            import os as _os
            base = Path(_os.environ.get("LOCALAPPDATA", str(Path.home())))
            default = base / "scene-ripper" / "cache" / "models"
        else:
            default = Path.home() / ".cache" / "scene-ripper" / "models"
        default.mkdir(parents=True, exist_ok=True)
        return default


def _load_insightface():
    """Lazy load InsightFace model (thread-safe).

    Uses double-check locking pattern matching detection.py.
    Auto-downloads model on first use.
    """
    global _model

    # Fast path: already loaded
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is None:
            logger.info("Loading InsightFace model...")

            import insightface
            from insightface.app import FaceAnalysis

            cache_dir = _get_model_cache_dir()
            insightface_dir = cache_dir / "insightface"
            insightface_dir.mkdir(parents=True, exist_ok=True)

            # Detect execution providers
            providers = ["CPUExecutionProvider"]
            if sys.platform == "darwin":
                try:
                    import platform
                    if platform.processor() == "arm" or "arm64" in platform.machine():
                        # Apple Silicon - try CoreML
                        try:
                            import onnxruntime
                            available = onnxruntime.get_available_providers()
                            if "CoreMLExecutionProvider" in available:
                                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                        except ImportError:
                            pass
                except Exception:
                    pass

            _model = FaceAnalysis(
                name="buffalo_l",
                root=str(insightface_dir),
                providers=providers,
            )
            _model.prepare(ctx_id=0, det_size=(640, 640))

            logger.info("InsightFace model loaded")

    return _model


def extract_faces_from_image(image_path: Path) -> list[dict]:
    """Detect faces in a single image.

    Args:
        image_path: Path to image file.

    Returns:
        List of face dicts with keys:
        - bbox: [x, y, w, h] bounding box
        - embedding: list of 512 floats (ArcFace)
        - confidence: detection confidence (0.0-1.0)
    """
    import cv2

    model = _load_insightface()

    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return []

    faces = model.get(img)
    results = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        # Convert from [x1, y1, x2, y2] to [x, y, w, h]
        x, y, x2, y2 = bbox
        results.append({
            "bbox": [x, y, x2 - x, y2 - y],
            "embedding": face.embedding.tolist(),
            "confidence": float(face.det_score),
        })

    logger.debug(f"Found {len(results)} faces in {image_path.name}")
    return results


def extract_faces_from_clip(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    sample_interval: float = 1.0,
) -> list[dict]:
    """Extract face embeddings from a video clip by sampling frames.

    Samples frames at the given interval. For clips shorter than the
    interval, samples a single frame at the midpoint.

    Args:
        source_path: Path to source video file.
        start_frame: Clip start frame.
        end_frame: Clip end frame (exclusive).
        fps: Video framerate.
        sample_interval: Seconds between frame samples.

    Returns:
        List of face dicts (same as extract_faces_from_image) with
        additional 'frame_number' key.
    """
    import cv2

    model = _load_insightface()

    duration_frames = end_frame - start_frame
    if duration_frames <= 0:
        return []

    duration_seconds = duration_frames / fps
    interval_frames = int(sample_interval * fps)

    # Calculate sample positions
    if duration_seconds < sample_interval:
        # Short clip: sample midpoint only
        sample_positions = [start_frame + duration_frames // 2]
    else:
        sample_positions = list(range(start_frame, end_frame, max(1, interval_frames)))

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {source_path}")
        return []

    results = []
    try:
        for frame_pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue

            faces = model.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                x, y, x2, y2 = bbox
                results.append({
                    "bbox": [x, y, x2 - x, y2 - y],
                    "embedding": face.embedding.tolist(),
                    "confidence": float(face.det_score),
                    "frame_number": frame_pos,
                })
    finally:
        cap.release()

    logger.debug(
        f"Extracted {len(results)} faces from {len(sample_positions)} frames "
        f"in {source_path.name}"
    )
    return results


def compare_faces(
    reference_embeddings: list[list[float]],
    clip_faces: list[dict],
    threshold: float,
) -> tuple[bool, float]:
    """Compare clip faces against reference embeddings.

    Args:
        reference_embeddings: List of reference face embeddings.
        clip_faces: List of face dicts from extract_faces_from_clip.
        threshold: Cosine similarity threshold for a match.

    Returns:
        Tuple of (is_match, max_similarity).
        is_match is True if any face pair exceeds threshold.
        max_similarity is the highest cosine similarity found (0.0 if no faces).
    """
    if not reference_embeddings or not clip_faces:
        return False, 0.0

    ref_arrays = [np.array(e, dtype=np.float32) for e in reference_embeddings]
    max_sim = 0.0

    for face in clip_faces:
        face_emb = np.array(face["embedding"], dtype=np.float32)
        face_norm = np.linalg.norm(face_emb)
        if face_norm == 0:
            continue

        for ref in ref_arrays:
            ref_norm = np.linalg.norm(ref)
            if ref_norm == 0:
                continue
            similarity = float(np.dot(ref, face_emb) / (ref_norm * face_norm))
            if similarity > max_sim:
                max_sim = similarity

    return max_sim >= threshold, max_sim


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Average multiple face embeddings for robust identity representation.

    Used when user provides 2-3 reference images of the same person.

    Args:
        embeddings: List of face embedding vectors (each 512 floats).

    Returns:
        Averaged embedding vector as list of floats.
    """
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return list(embeddings[0])

    arr = np.array(embeddings, dtype=np.float32)
    mean = np.mean(arr, axis=0)
    # L2-normalize the averaged embedding
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.tolist()


def is_model_loaded() -> bool:
    """Check if the InsightFace model is already loaded."""
    return _model is not None


def unload_model():
    """Unload the InsightFace model to free memory."""
    global _model

    with _model_lock:
        _model = None

    logger.info("InsightFace model unloaded")
