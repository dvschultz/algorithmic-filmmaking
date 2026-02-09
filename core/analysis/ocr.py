"""On-screen text extraction using PaddleOCR with VLM fallback.

This module provides text extraction from video frames using a hybrid approach:
1. PaddleOCR PP-OCRv5 (local, fast, free) for text detection + recognition
2. VLM fallback (GPT-4o, Claude, Gemini) for stylized or hard-to-read text

PaddleOCR handles both text detection and recognition in one pass,
replacing the previous EAST + Tesseract pipeline.
"""

import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Thread-safe PaddleOCR availability check
_paddleocr_available: Optional[bool] = None
_paddleocr_lock = threading.Lock()

# Lazy-loaded PaddleOCR instance (heavy init, reuse across calls)
_ocr_engine = None
_ocr_engine_lock = threading.Lock()


def _check_paddleocr() -> bool:
    """Check if PaddleOCR is installed and available.

    Returns:
        True if PaddleOCR is available, False otherwise.
    """
    global _paddleocr_available
    if _paddleocr_available is not None:
        return _paddleocr_available

    with _paddleocr_lock:
        if _paddleocr_available is None:
            try:
                from paddleocr import PaddleOCR  # noqa: F401
                _paddleocr_available = True
                logger.info("PaddleOCR available")
            except ImportError:
                _paddleocr_available = False
                logger.info("PaddleOCR not available")

    return _paddleocr_available


def _get_ocr_engine():
    """Get or create PaddleOCR engine (thread-safe singleton)."""
    global _ocr_engine

    if _ocr_engine is not None:
        return _ocr_engine

    with _ocr_engine_lock:
        if _ocr_engine is None:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR engine...")
            _ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
            logger.info("PaddleOCR engine ready")

    return _ocr_engine


def is_paddleocr_available() -> bool:
    """Public function to check PaddleOCR availability."""
    return _check_paddleocr()


# Legacy alias
def is_tesseract_available() -> bool:
    """Check if local OCR is available. Now backed by PaddleOCR."""
    return _check_paddleocr()


def extract_text_from_frame(
    frame_path: Path,
    use_vlm_fallback: bool = True,
    vlm_model: Optional[str] = None,
    vlm_only: bool = False,
    confidence_threshold: float = 0.6,
    skip_detection: bool = False,
) -> tuple[str, float, str]:
    """Extract text from a single video frame.

    Uses PaddleOCR first (if available), then falls back to VLM
    if PaddleOCR fails or returns low-confidence results.

    Args:
        frame_path: Path to the frame image file
        use_vlm_fallback: Whether to use VLM if PaddleOCR fails or has low confidence
        vlm_model: VLM model to use for fallback (default: from settings)
        vlm_only: If True, skip PaddleOCR and only use VLM
        confidence_threshold: Minimum confidence to accept PaddleOCR result (0.0-1.0)
        skip_detection: Ignored (PaddleOCR handles detection internally)

    Returns:
        Tuple of (text, confidence, source) where:
        - text: The extracted text content
        - confidence: Confidence score from 0.0 to 1.0
        - source: "paddleocr", "vlm", or "none"
    """
    # VLM-only mode: skip PaddleOCR entirely
    if vlm_only:
        if not use_vlm_fallback:
            logger.warning(
                "extract_text_from_frame called with vlm_only=True but use_vlm_fallback=False. "
                "This is a contradictory configuration; returning empty result."
            )
            return ("", 0.0, "none")
        try:
            text, conf = _vlm_text_extraction(frame_path, vlm_model)
            if text:
                logger.debug(f"VLM extracted (VLM-only): '{text[:50]}...' (confidence: {conf:.2f})")
                return (text, conf, "vlm")
        except Exception as e:
            logger.warning(f"VLM text extraction failed: {e}")
        return ("", 0.0, "none")

    text = ""
    confidence = 0.0
    source = "paddleocr"

    # Try PaddleOCR first
    if _check_paddleocr():
        try:
            ocr = _get_ocr_engine()
            result = ocr.ocr(str(frame_path), cls=True)

            if result and result[0]:
                words = []
                confidences = []
                for line in result[0]:
                    # Each line: [bbox, (text, confidence)]
                    line_text = line[1][0]
                    line_conf = line[1][1]
                    if line_text.strip():
                        words.append(line_text.strip())
                        confidences.append(line_conf)

                if words:
                    text = " ".join(words)
                    confidence = sum(confidences) / len(confidences)
                    source = "paddleocr"
                    logger.debug(f"PaddleOCR extracted: '{text[:50]}...' (confidence: {confidence:.2f})")

        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed: {e}")

    # VLM fallback if PaddleOCR unavailable or low confidence
    if use_vlm_fallback and (not text or confidence < confidence_threshold):
        try:
            vlm_text, vlm_conf = _vlm_text_extraction(frame_path, vlm_model)
            if vlm_text and (not text or vlm_conf > confidence):
                text = vlm_text
                confidence = vlm_conf
                source = "vlm"
                logger.debug(f"VLM extracted: '{text[:50]}...' (confidence: {confidence:.2f})")
        except Exception as e:
            logger.warning(f"VLM text extraction failed: {e}")

    return text, confidence, source


def _vlm_text_extraction(
    frame_path: Path,
    model: Optional[str] = None,
) -> tuple[str, float]:
    """Extract text using Vision Language Model.

    Args:
        frame_path: Path to the frame image
        model: VLM model to use (default: from settings)

    Returns:
        Tuple of (text, confidence) where confidence is estimated.
    """
    import litellm
    from core.analysis.description import encode_image_base64
    from core.settings import load_settings, get_gemini_api_key, get_openai_api_key, get_anthropic_api_key

    settings = load_settings()
    model = model or settings.description_model_cloud or "gemini-3-flash-preview"
    original_model = model

    # Normalize model name for LiteLLM
    if "gemini" in model.lower() and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
        model = f"gemini/{model}"
    elif "claude" in model.lower() and not any(model.startswith(p) for p in ["anthropic/", "bedrock/"]):
        model = f"anthropic/{model}"

    # Get appropriate API key
    api_key = None
    if "gpt" in model.lower() or "openai" in model.lower():
        api_key = get_openai_api_key()
    elif "claude" in model.lower() or "anthropic" in model.lower():
        api_key = get_anthropic_api_key()
    elif "gemini" in model.lower():
        api_key = get_gemini_api_key()

    if not api_key:
        raise ValueError(f"No API key found for VLM model {original_model}")

    base64_image = encode_image_base64(frame_path)

    prompt = """Extract ALL visible text from this image. Include:
- Signs, labels, titles
- Subtitles or captions
- Text on documents or screens
- Any other readable text

Return ONLY the extracted text, one phrase per line. If no text is visible, return "NO_TEXT_FOUND".
Do not add any commentary or descriptions."""

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

    logger.debug(f"Calling VLM for text extraction with model={model}")
    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
    )

    text = response.choices[0].message.content.strip()

    if text == "NO_TEXT_FOUND":
        return "", 0.0

    # VLM confidence is estimated - we trust VLM output fairly highly
    confidence = 0.85 if text else 0.0
    return text, confidence


def extract_text_from_clip(
    clip,
    source,
    num_keyframes: int = 3,
    use_vlm_fallback: bool = True,
    vlm_model: Optional[str] = None,
    vlm_only: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_text_detection: bool = True,
) -> list:
    """Extract text from multiple keyframes of a clip.

    Samples keyframes evenly across the clip and runs OCR on each.
    Results are returned as a list of ExtractedText objects.

    Args:
        clip: Clip object with start_frame and end_frame
        source: Source object containing the video file path and fps
        num_keyframes: Number of frames to sample (default 3: start, middle, end)
        use_vlm_fallback: Whether to use VLM for low-confidence results
        vlm_model: VLM model to use (default: from settings)
        vlm_only: If True, skip PaddleOCR and only use VLM
        progress_callback: Optional callback(current, total) for progress updates
        use_text_detection: Ignored (PaddleOCR handles detection internally)

    Returns:
        List of ExtractedText objects from models/clip.py
    """
    from core.ffmpeg import extract_frame
    from models.clip import ExtractedText

    results = []

    # Calculate keyframe positions
    total_frames = clip.end_frame - clip.start_frame
    if total_frames <= 0:
        logger.warning(f"Clip {clip.id} has no frames")
        return results

    # Constrain num_keyframes
    num_keyframes = min(max(1, num_keyframes), 5)

    # Distribute keyframes evenly
    if num_keyframes >= total_frames:
        frame_positions = list(range(clip.start_frame, clip.end_frame + 1))
    elif num_keyframes == 1:
        # Just the middle frame
        frame_positions = [clip.start_frame + total_frames // 2]
    else:
        # Evenly distribute: first, middle points, last
        step = total_frames / (num_keyframes - 1)
        frame_positions = [
            int(clip.start_frame + i * step)
            for i in range(num_keyframes)
        ]

    logger.debug(f"Extracting text from clip {clip.id} at frames: {frame_positions}")

    # Extract and OCR each keyframe
    for i, frame_num in enumerate(frame_positions):
        if progress_callback:
            progress_callback(i + 1, len(frame_positions))

        # Create temp file for frame
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            frame_path = Path(tmp.name)

        try:
            # Extract frame from video
            extract_frame(source.file_path, frame_num, frame_path, source.fps)

            if not frame_path.exists() or frame_path.stat().st_size == 0:
                logger.warning(f"Failed to extract frame {frame_num} from {source.file_path}")
                continue

            # Run OCR
            text, confidence, ocr_source = extract_text_from_frame(
                frame_path,
                use_vlm_fallback=use_vlm_fallback,
                vlm_model=vlm_model,
                vlm_only=vlm_only,
                skip_detection=True,  # PaddleOCR handles detection internally
            )

            if text:
                results.append(ExtractedText(
                    frame_number=frame_num,
                    text=text,
                    confidence=confidence,
                    source=ocr_source,
                ))

        except Exception as e:
            logger.error(f"Error extracting text from frame {frame_num}: {e}")

        finally:
            # Clean up temp file
            try:
                if frame_path.exists():
                    frame_path.unlink()
            except Exception:
                pass

    logger.info(f"Extracted {len(results)} text segments from clip {clip.id}")
    return results
