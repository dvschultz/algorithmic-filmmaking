"""Speech transcription using faster-whisper."""

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Lazy load to avoid startup delay
_model = None
_model_name = None
_faster_whisper_available = None

# Available model configurations
WHISPER_MODELS = {
    "tiny.en": {"size": "39MB", "speed": "~32x", "accuracy": "Basic", "vram": "<1GB"},
    "small.en": {"size": "244MB", "speed": "~15x", "accuracy": "Good", "vram": "~1GB"},
    "medium.en": {"size": "769MB", "speed": "~5x", "accuracy": "Better", "vram": "~2GB"},
    "large-v3": {"size": "1.5GB", "speed": "~2x", "accuracy": "Best", "vram": "~4GB"},
    "large-v3-turbo": {"size": "~800MB", "speed": "~4x", "accuracy": "Best", "vram": "~2GB"},
}


class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class FasterWhisperNotInstalledError(TranscriptionError):
    """Raised when faster-whisper is not installed."""
    def __init__(self):
        super().__init__(
            "faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        )


class ModelDownloadError(TranscriptionError):
    """Raised when model download fails."""
    pass


def is_faster_whisper_available() -> bool:
    """Check if faster-whisper is installed."""
    global _faster_whisper_available
    if _faster_whisper_available is None:
        try:
            import faster_whisper
            _faster_whisper_available = True
        except ImportError:
            _faster_whisper_available = False
    return _faster_whisper_available


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    start_time: float  # seconds from clip start
    end_time: float
    text: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        """Deserialize from dictionary."""
        return cls(
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
        )


def get_model(model_name: str = "small.en"):
    """Get or load the Whisper model (lazy loading).

    Args:
        model_name: Name of the Whisper model to load

    Returns:
        Loaded WhisperModel instance

    Raises:
        FasterWhisperNotInstalledError: If faster-whisper is not installed
        ModelDownloadError: If model download fails
    """
    global _model, _model_name

    if not is_faster_whisper_available():
        raise FasterWhisperNotInstalledError()

    if _model is None or _model_name != model_name:
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {model_name}")
        model_info = WHISPER_MODELS.get(model_name, {})
        logger.info(f"Model size: {model_info.get('size', 'unknown')} - this may take a moment to download on first use")

        try:
            # Use int8 quantization for CPU, auto-detect GPU
            _model = WhisperModel(
                model_name,
                device="auto",  # auto-detect CPU/GPU
                compute_type="int8",  # Use int8 for CPU efficiency
            )
            _model_name = model_name
            logger.info(f"Whisper model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ModelDownloadError(f"Failed to load model '{model_name}': {e}") from e

    return _model


def transcribe_video(
    video_path: Path,
    model_name: str = "small.en",
    language: str = "en",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[TranscriptSegment]:
    """Transcribe audio from a video file.

    Args:
        video_path: Path to video file
        model_name: Whisper model to use
        language: Language code (e.g., "en", "es", "auto")
        progress_callback: Optional callback(progress, message)

    Returns:
        List of TranscriptSegment objects
    """
    model = get_model(model_name)

    if progress_callback:
        progress_callback(0.1, "Extracting audio...")

    # faster-whisper can process video files directly
    segments, info = model.transcribe(
        str(video_path),
        language=language if language != "auto" else None,
        word_timestamps=True,
        vad_filter=True,  # Filter out non-speech
    )

    if progress_callback:
        progress_callback(0.5, "Processing segments...")

    results = []
    for segment in segments:
        results.append(
            TranscriptSegment(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text.strip(),
                confidence=segment.avg_logprob,
            )
        )

    if progress_callback:
        progress_callback(1.0, f"Transcribed {len(results)} segments")

    return results


def transcribe_clip(
    source_path: Path,
    start_time: float,
    end_time: float,
    model_name: str = "small.en",
    language: str = "en",
) -> list[TranscriptSegment]:
    """Transcribe a specific clip range from a video.

    For efficiency, extracts the audio segment first.

    Args:
        source_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        model_name: Whisper model to use
        language: Language code

    Returns:
        List of TranscriptSegment objects with times relative to clip start
    """
    # Extract audio segment to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Extract audio segment with FFmpeg
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-i",
                str(source_path),
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",  # Whisper expects 16kHz
                "-ac",
                "1",  # Mono
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
        )

        # Check if audio was actually extracted (file size > 0)
        if tmp_path.stat().st_size == 0:
            logger.warning(f"No audio extracted from {source_path} ({start_time}-{end_time})")
            return []

        # Transcribe the segment
        model = get_model(model_name)
        segments, info = model.transcribe(
            str(tmp_path),
            language=language if language != "auto" else None,
            word_timestamps=True,
            vad_filter=True,
        )

        # Build results - timestamps are already relative to clip start
        # since we extracted the segment
        results = []
        for segment in segments:
            results.append(
                TranscriptSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob,
                )
            )

        return results

    except subprocess.CalledProcessError as e:
        logger.warning(f"FFmpeg failed to extract audio: {e.stderr.decode() if e.stderr else e}")
        return []

    finally:
        tmp_path.unlink(missing_ok=True)


def get_transcript_text(segments: list[TranscriptSegment]) -> str:
    """Get full transcript text from segments.

    Args:
        segments: List of TranscriptSegment objects

    Returns:
        Combined transcript text
    """
    return " ".join(seg.text for seg in segments)
