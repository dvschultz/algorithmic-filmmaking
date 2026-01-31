"""On-screen text extraction using Tesseract OCR with VLM fallback.

This module provides text extraction from video frames using a hybrid approach:
1. Tesseract OCR (local, fast, free) for clear printed text
2. VLM fallback (GPT-4o, Claude, Gemini) for stylized or hard-to-read text

Tesseract is optional - if not installed, falls back to VLM-only mode.
"""

import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Thread-safe Tesseract availability check
_tesseract_available: Optional[bool] = None
_tesseract_lock = threading.Lock()


def _check_tesseract() -> bool:
    """Check if Tesseract is installed and available.

    Returns:
        True if Tesseract is available, False otherwise.
    """
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available

    with _tesseract_lock:
        if _tesseract_available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                _tesseract_available = True
                logger.info("Tesseract OCR available")
            except Exception as e:
                _tesseract_available = False
                logger.info(f"Tesseract OCR not available: {e}")

    return _tesseract_available


def is_tesseract_available() -> bool:
    """Public function to check Tesseract availability."""
    return _check_tesseract()


def extract_text_from_frame(
    frame_path: Path,
    use_vlm_fallback: bool = True,
    vlm_model: Optional[str] = None,
    vlm_only: bool = False,
    confidence_threshold: float = 0.6,
    skip_detection: bool = False,
) -> tuple[str, float, str]:
    """Extract text from a single video frame.

    Uses Tesseract OCR first (if available), then falls back to VLM
    if Tesseract fails or returns low-confidence results.

    Args:
        frame_path: Path to the frame image file
        use_vlm_fallback: Whether to use VLM if Tesseract fails or has low confidence
        vlm_model: VLM model to use for fallback (default: from settings)
        vlm_only: If True, skip Tesseract and only use VLM
        confidence_threshold: Minimum confidence to accept Tesseract result (0.0-1.0)
        skip_detection: If True, bypass EAST pre-filter (default: False)

    Returns:
        Tuple of (text, confidence, source) where:
        - text: The extracted text content
        - confidence: Confidence score from 0.0 to 1.0
        - source: "tesseract", "vlm", "skipped", or "none"
    """
    # Pre-filter with EAST detection if enabled
    if not skip_detection:
        try:
            from core.settings import load_settings
            settings = load_settings()
            if settings.text_detection_enabled:
                from core.analysis.text_detection import has_text_regions
                if not has_text_regions(
                    frame_path,
                    confidence_threshold=settings.text_detection_confidence,
                ):
                    logger.debug(f"No text detected in {frame_path.name}, skipping OCR")
                    return ("", 0.0, "skipped")
        except Exception as e:
            logger.warning(f"Text detection pre-filter failed, proceeding with OCR: {e}")
    # VLM-only mode: skip Tesseract entirely
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
    source = "tesseract"

    # Try Tesseract first
    if _check_tesseract():
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(frame_path)
            # Get detailed data including confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Filter by confidence and extract text
            words = []
            confidences = []
            for i, word in enumerate(data['text']):
                conf = int(data['conf'][i])
                if conf > 0 and word.strip():
                    words.append(word)
                    confidences.append(conf / 100.0)

            if words:
                text = " ".join(words)
                confidence = sum(confidences) / len(confidences)
                source = "tesseract"
                logger.debug(f"Tesseract extracted: '{text[:50]}...' (confidence: {confidence:.2f})")

        except Exception as e:
            logger.warning(f"Tesseract extraction failed: {e}")

    # VLM fallback if Tesseract unavailable or low confidence
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
    model = model or settings.description_model_cloud or "gpt-4o"
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
        vlm_only: If True, skip Tesseract and only use VLM
        progress_callback: Optional callback(current, total) for progress updates
        use_text_detection: Whether to use EAST pre-filter (default: True)

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
                skip_detection=not use_text_detection,
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
