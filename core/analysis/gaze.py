"""Gaze direction estimation using MediaPipe Face Mesh iris landmarks.

Estimates where subjects are looking in video clips by analyzing iris
positions relative to eye corners. Stores continuous yaw/pitch angles
and categorical labels (at_camera, looking_left, looking_right,
looking_up, looking_down) for use by gaze-based sequencer algorithms.
"""

import logging
import threading
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Angle thresholds for categorization (degrees from center)
GAZE_YAW_THRESHOLD = 10.0   # degrees from center for left/right
GAZE_PITCH_THRESHOLD = 8.0  # degrees for up/down (tighter: vertical is noisier)

# Scaling factors for iris ratio -> angle conversion
MAX_YAW_ANGLE = 30.0   # reliable yaw estimation range
MAX_PITCH_ANGLE = 20.0  # reliable pitch estimation range

# Canonical gaze category values and display names (single source of truth)
GAZE_CATEGORIES = ("at_camera", "looking_left", "looking_right", "looking_up", "looking_down")
GAZE_CATEGORY_DISPLAY = {
    "at_camera": "At Camera",
    "looking_left": "Looking Left",
    "looking_right": "Looking Right",
    "looking_up": "Looking Up",
    "looking_down": "Looking Down",
}
GAZE_CATEGORY_SHORT = {
    "at_camera": "C",
    "looking_left": "\u2190",
    "looking_right": "\u2192",
    "looking_up": "\u2191",
    "looking_down": "\u2193",
}

# MediaPipe Face Mesh iris landmark indices (refine_landmarks=True required)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corner landmark indices from Face Mesh topology
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Vertical eye boundary landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def ensure_gaze_runtime_available():
    """Validate that the gaze detection runtime imports cleanly.

    Performs a narrow import check for feature registry validation.
    Does not load the model — just confirms mediapipe is importable.
    """
    try:
        import mediapipe  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"gaze detection runtime is incomplete: {e}") from e


def _get_model_path() -> str:
    """Get the path to the FaceLandmarker model file, downloading if needed.

    Downloads the model to the scene-ripper model cache directory on first use.
    """
    import os
    import urllib.request

    from core.settings import load_settings

    settings = load_settings()
    model_dir = os.path.join(settings.model_cache_dir, "mediapipe")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "face_landmarker.task")

    if not os.path.exists(model_path):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        logger.info("Downloading MediaPipe FaceLandmarker model...")
        tmp_path = model_path + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, model_path)
        except Exception:
            # Clean up partial download so next attempt retries
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        logger.info("Downloaded FaceLandmarker model to %s", model_path)

    return model_path


def _load_face_mesh():
    """Lazy load MediaPipe FaceLandmarker model (thread-safe).

    Uses the Tasks API (mp.tasks.vision.FaceLandmarker) which replaced
    the legacy mp.solutions.face_mesh API in mediapipe >= 0.10.14.
    Downloads the model file on first use.

    Returns:
        Loaded FaceLandmarker instance.
    """
    global _model

    # Fast path: already loaded
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is None:
            logger.info("Loading MediaPipe FaceLandmarker for gaze estimation...")

            if mp is None:
                raise RuntimeError("mediapipe is not installed")

            model_path = _get_model_path()
            base_options = mp.tasks.BaseOptions(
                model_asset_path=model_path,
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=5,
                min_face_detection_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _model = mp.tasks.vision.FaceLandmarker.create_from_options(options)

            logger.info("MediaPipe FaceLandmarker loaded for gaze estimation")

    return _model


def unload_model():
    """Unload the MediaPipe FaceMesh model to free memory."""
    global _model

    with _model_lock:
        if _model is not None:
            try:
                _model.close()
            except Exception:
                pass
            _model = None

    logger.info("MediaPipe FaceMesh (gaze) unloaded")


def is_model_loaded() -> bool:
    """Check if the FaceMesh model is already loaded."""
    return _model is not None


# ---------------------------------------------------------------------------
# Gaze categorization
# ---------------------------------------------------------------------------

def categorize_gaze(yaw_deg: float, pitch_deg: float) -> str:
    """Categorize gaze direction from continuous yaw/pitch angles.

    When both yaw and pitch exceed their thresholds, yaw takes priority
    because horizontal gaze is more editorially significant and more
    reliably estimated than vertical gaze.

    Sign convention:
        - Positive yaw = looking right, negative = looking left
        - Positive pitch = looking down, negative = looking up

    Args:
        yaw_deg: Horizontal gaze angle in degrees.
        pitch_deg: Vertical gaze angle in degrees.

    Returns:
        One of: "at_camera", "looking_left", "looking_right",
        "looking_up", "looking_down".
    """
    # Check yaw first (priority)
    if abs(yaw_deg) > GAZE_YAW_THRESHOLD:
        return "looking_right" if yaw_deg > 0 else "looking_left"

    # Then check pitch
    if abs(pitch_deg) > GAZE_PITCH_THRESHOLD:
        return "looking_down" if pitch_deg > 0 else "looking_up"

    return "at_camera"


# ---------------------------------------------------------------------------
# Per-frame gaze extraction
# ---------------------------------------------------------------------------

def _select_largest_face(face_landmarks_list, img_w: int, img_h: int):
    """Select the largest face by bounding box area from detected faces.

    Args:
        face_landmarks_list: List of face landmark lists from FaceLandmarker.
            Each entry is a list of NormalizedLandmark objects.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        The landmark list for the largest face, or None.
    """
    if not face_landmarks_list:
        return None

    best_face = None
    best_area = -1.0

    for landmarks in face_landmarks_list:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        bbox_w = (max(xs) - min(xs)) * img_w
        bbox_h = (max(ys) - min(ys)) * img_h
        area = bbox_w * bbox_h

        if area > best_area:
            best_area = area
            best_face = landmarks

    return best_face


def _compute_iris_ratios(landmarks, img_w: int, img_h: int):
    """Compute horizontal and vertical iris-to-eye-corner ratios.

    For each eye, computes:
    - Horizontal ratio: iris_center position between inner and outer corners (0=outer, 1=inner)
    - Vertical ratio: iris_center position between top and bottom eyelid (0=top, 1=bottom)

    Returns the average ratios across both eyes.

    Args:
        landmarks: Face landmark list from FaceLandmarker (list of NormalizedLandmark).
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Tuple of (horizontal_ratio, vertical_ratio) where 0.5 = centered.
    """
    lm = landmarks

    # Left eye iris center (pixel coords)
    left_iris_x = np.mean([lm[i].x * img_w for i in LEFT_IRIS])
    left_iris_y = np.mean([lm[i].y * img_h for i in LEFT_IRIS])

    # Right eye iris center (pixel coords)
    right_iris_x = np.mean([lm[i].x * img_w for i in RIGHT_IRIS])
    right_iris_y = np.mean([lm[i].y * img_h for i in RIGHT_IRIS])

    # Left eye horizontal ratio: position between outer and inner corners
    left_outer_x = lm[LEFT_EYE_OUTER].x * img_w
    left_inner_x = lm[LEFT_EYE_INNER].x * img_w
    left_h_range = left_inner_x - left_outer_x
    if abs(left_h_range) < 1e-6:
        left_h_ratio = 0.5
    else:
        left_h_ratio = (left_iris_x - left_outer_x) / left_h_range

    # Right eye horizontal ratio: position between inner and outer corners
    right_inner_x = lm[RIGHT_EYE_INNER].x * img_w
    right_outer_x = lm[RIGHT_EYE_OUTER].x * img_w
    right_h_range = right_outer_x - right_inner_x
    if abs(right_h_range) < 1e-6:
        right_h_ratio = 0.5
    else:
        right_h_ratio = (right_iris_x - right_inner_x) / right_h_range

    h_ratio = (left_h_ratio + right_h_ratio) / 2.0

    # Left eye vertical ratio: position between top and bottom eyelid
    left_top_y = lm[LEFT_EYE_TOP].y * img_h
    left_bottom_y = lm[LEFT_EYE_BOTTOM].y * img_h
    left_v_range = left_bottom_y - left_top_y
    if abs(left_v_range) < 1e-6:
        left_v_ratio = 0.5
    else:
        left_v_ratio = (left_iris_y - left_top_y) / left_v_range

    # Right eye vertical ratio: position between top and bottom eyelid
    right_top_y = lm[RIGHT_EYE_TOP].y * img_h
    right_bottom_y = lm[RIGHT_EYE_BOTTOM].y * img_h
    right_v_range = right_bottom_y - right_top_y
    if abs(right_v_range) < 1e-6:
        right_v_ratio = 0.5
    else:
        right_v_ratio = (right_iris_y - right_top_y) / right_v_range

    v_ratio = (left_v_ratio + right_v_ratio) / 2.0

    return h_ratio, v_ratio


def extract_gaze_from_frame(
    landmarker,
    frame_bgr,
    img_w: int,
    img_h: int,
) -> Optional[tuple[float, float, str]]:
    """Process a single BGR frame and extract gaze direction.

    Selects the largest face by bounding box area, computes iris-to-eye-corner
    ratios, and converts to angular gaze estimate.

    Args:
        landmarker: Loaded MediaPipe FaceLandmarker instance.
        frame_bgr: BGR image (numpy array from cv2).
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Tuple of (yaw_deg, pitch_deg, category) or None if no face detected.
        - yaw_deg: positive = looking right, negative = looking left
        - pitch_deg: positive = looking down, negative = looking up
        - category: one of "at_camera", "looking_left", "looking_right",
          "looking_up", "looking_down"
    """
    if mp is None:
        raise RuntimeError("mediapipe is not installed")

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Use mediapipe Image wrapper for the Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None

    # Select largest face by bounding box area
    # results.face_landmarks is list[list[NormalizedLandmark]]
    face = _select_largest_face(results.face_landmarks, img_w, img_h)
    if face is None:
        return None

    # Check we have enough landmarks for iris (need 478, indices up to 477)
    if len(face) < 478:
        logger.debug("Face has %d landmarks (need 478 for iris), skipping", len(face))
        return None

    # Compute iris-to-eye-corner ratios
    h_ratio, v_ratio = _compute_iris_ratios(face, img_w, img_h)

    # Scale ratios to angles
    # h_ratio: 0.5 = center, >0.5 = looking right, <0.5 = looking left
    yaw_deg = (h_ratio - 0.5) * 2 * MAX_YAW_ANGLE
    # v_ratio: 0.5 = center, >0.5 = looking down, <0.5 = looking up
    pitch_deg = (v_ratio - 0.5) * 2 * MAX_PITCH_ANGLE

    category = categorize_gaze(yaw_deg, pitch_deg)

    return (yaw_deg, pitch_deg, category)


# ---------------------------------------------------------------------------
# Per-clip gaze extraction
# ---------------------------------------------------------------------------

def extract_gaze_from_clip(
    source_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    sample_interval: float = 1.0,
) -> Optional[dict]:
    """Extract gaze direction from a video clip by sampling frames.

    Samples frames at the given interval, collects per-frame gaze
    estimates, and returns the dominant category with median angles
    from frames matching that category.

    For clips shorter than the sample interval, samples a single frame
    at the midpoint.

    Args:
        source_path: Path to source video file.
        start_frame: Clip start frame.
        end_frame: Clip end frame (exclusive).
        fps: Video framerate.
        sample_interval: Seconds between frame samples (default 1.0).

    Returns:
        Dict with keys:
        - "gaze_yaw": float — median yaw from dominant category frames
        - "gaze_pitch": float — median pitch from dominant category frames
        - "gaze_category": str — most frequent category across sampled frames
        Returns None if no faces were detected in any sampled frame.
    """
    face_mesh = _load_face_mesh()

    duration_frames = end_frame - start_frame
    if duration_frames <= 0:
        return None

    duration_seconds = duration_frames / fps
    interval_frames = int(sample_interval * fps)

    # Calculate sample positions
    if duration_seconds < sample_interval:
        # Short clip: sample midpoint only
        sample_positions = [start_frame + duration_frames // 2]
    else:
        sample_positions = list(
            range(start_frame, end_frame, max(1, interval_frames))
        )

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.warning("Could not open video: %s", source_path)
        return None

    # Collect per-frame gaze results
    frame_results: list[tuple[float, float, str]] = []

    try:
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for frame_pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue

            result = extract_gaze_from_frame(face_mesh, frame, img_w, img_h)
            if result is not None:
                frame_results.append(result)
    finally:
        cap.release()

    if not frame_results:
        return None

    # Find dominant category (most frequent)
    categories = [r[2] for r in frame_results]
    category_counts: dict[str, int] = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    dominant_category = max(category_counts, key=category_counts.get)

    # Compute median yaw and pitch from frames matching dominant category
    matching = [r for r in frame_results if r[2] == dominant_category]
    median_yaw = float(np.median([r[0] for r in matching]))
    median_pitch = float(np.median([r[1] for r in matching]))

    logger.debug(
        "Gaze from %d/%d frames: category=%s yaw=%.1f pitch=%.1f (%s)",
        len(frame_results),
        len(sample_positions),
        dominant_category,
        median_yaw,
        median_pitch,
        source_path,
    )

    return {
        "gaze_yaw": median_yaw,
        "gaze_pitch": median_pitch,
        "gaze_category": dominant_category,
    }
