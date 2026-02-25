"""Speech transcription using faster-whisper, lightning-whisper-mlx, or Groq cloud.

Supports three backends:
- faster-whisper (default): CPU/CUDA, cross-platform, int8 quantisation
- mlx-whisper: Apple Silicon GPU via MLX (4-10x faster on M-series Macs)
- groq: Cloud transcription via Groq API (fast, low-cost, no local compute)

Backend selection:
- "auto" (default): prefers mlx-whisper on Apple Silicon, falls back to faster-whisper
- "faster-whisper": force CPU/CUDA backend
- "mlx-whisper": force MLX backend (Apple Silicon only)
- "groq": force cloud transcription via Groq API
"""

import logging
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.binary_resolver import find_binary, get_subprocess_kwargs

logger = logging.getLogger(__name__)

# Lazy load to avoid startup delay
_model = None
_model_name = None
_faster_whisper_available = None

_mlx_model = None
_mlx_model_name = None
_mlx_whisper_available = None

# Available model configurations
WHISPER_MODELS = {
    "tiny.en": {"size": "39MB", "speed": "~32x", "accuracy": "Basic", "vram": "<1GB"},
    "small.en": {"size": "244MB", "speed": "~15x", "accuracy": "Good", "vram": "~1GB"},
    "medium.en": {"size": "769MB", "speed": "~5x", "accuracy": "Better", "vram": "~2GB"},
    "large-v3": {"size": "1.5GB", "speed": "~2x", "accuracy": "Best", "vram": "~4GB"},
    "large-v3-turbo": {"size": "~800MB", "speed": "~4x", "accuracy": "Best", "vram": "~2GB"},
}

# Model name mapping: faster-whisper name → lightning-whisper-mlx name
# lightning-whisper-mlx uses slightly different naming conventions
_MLX_MODEL_MAP = {
    "tiny.en": "tiny.en",
    "small.en": "small.en",
    "medium.en": "medium.en",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3",  # mlx-whisper doesn't have turbo; fall back to large-v3
}

# Distilled variants available in mlx-whisper (faster, slightly less accurate)
MLX_DISTIL_MODELS = {
    "distil-small.en": {"size": "~166MB", "speed": "~20x", "accuracy": "Good", "vram": "<1GB"},
    "distil-medium.en": {"size": "~400MB", "speed": "~10x", "accuracy": "Better", "vram": "~1GB"},
    "distil-large-v3": {"size": "~800MB", "speed": "~6x", "accuracy": "Best", "vram": "~2GB"},
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


def is_mlx_whisper_available() -> bool:
    """Check if lightning-whisper-mlx is installed and running on Apple Silicon."""
    global _mlx_whisper_available
    if _mlx_whisper_available is None:
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            _mlx_whisper_available = False
        else:
            try:
                import lightning_whisper_mlx  # noqa: F401
                _mlx_whisper_available = True
            except ImportError:
                _mlx_whisper_available = False
    return _mlx_whisper_available


def _resolve_backend(backend: str = "auto") -> str:
    """Resolve 'auto' backend to a concrete choice.

    Args:
        backend: "auto", "faster-whisper", "mlx-whisper", or "groq"

    Returns:
        "faster-whisper", "mlx-whisper", or "groq"
    """
    if backend == "groq":
        return "groq"
    if backend == "mlx-whisper":
        if not is_mlx_whisper_available():
            logger.warning("mlx-whisper requested but not available; falling back to faster-whisper")
            return "faster-whisper"
        return "mlx-whisper"
    if backend == "faster-whisper":
        return "faster-whisper"
    # auto: prefer mlx-whisper on Apple Silicon
    if is_mlx_whisper_available():
        return "mlx-whisper"
    return "faster-whisper"


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
    """Get or load the faster-whisper model (lazy loading).

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


def get_mlx_model(model_name: str = "small.en"):
    """Get or load the lightning-whisper-mlx model (lazy loading).

    Args:
        model_name: Name of the Whisper model to load (faster-whisper naming)

    Returns:
        Loaded LightningWhisperMLX instance

    Raises:
        TranscriptionError: If mlx-whisper is not available
        ModelDownloadError: If model loading fails
    """
    global _mlx_model, _mlx_model_name

    if not is_mlx_whisper_available():
        raise TranscriptionError(
            "lightning-whisper-mlx is not installed or not on Apple Silicon. "
            "Install it with: pip install lightning-whisper-mlx"
        )

    # Map to mlx model name
    mlx_name = _MLX_MODEL_MAP.get(model_name, model_name)

    if _mlx_model is None or _mlx_model_name != mlx_name:
        from lightning_whisper_mlx import LightningWhisperMLX

        logger.info(f"Loading MLX Whisper model: {mlx_name}")

        try:
            _mlx_model = LightningWhisperMLX(model=mlx_name, batch_size=12)
            _mlx_model_name = mlx_name
            logger.info(f"MLX Whisper model loaded: {mlx_name}")
        except Exception as e:
            logger.error(f"Failed to load MLX Whisper model: {e}")
            raise ModelDownloadError(f"Failed to load MLX model '{mlx_name}': {e}") from e

    return _mlx_model


def transcribe_video(
    video_path: Path,
    model_name: str = "small.en",
    language: str = "en",
    backend: str = "auto",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[TranscriptSegment]:
    """Transcribe audio from a video file.

    Args:
        video_path: Path to video file
        model_name: Whisper model to use (ignored for groq backend)
        language: Language code (e.g., "en", "es", "auto")
        backend: "auto", "faster-whisper", "mlx-whisper", or "groq"
        progress_callback: Optional callback(progress, message)

    Returns:
        List of TranscriptSegment objects
    """
    resolved = _resolve_backend(backend)

    if resolved == "groq":
        return _transcribe_cloud_groq(video_path, language, progress_callback)

    if resolved == "mlx-whisper":
        return _transcribe_video_mlx(video_path, model_name, language, progress_callback)

    return _transcribe_video_faster_whisper(video_path, model_name, language, progress_callback)


def _transcribe_video_faster_whisper(
    video_path: Path,
    model_name: str,
    language: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[TranscriptSegment]:
    """Transcribe using faster-whisper backend."""
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


def _transcribe_video_mlx(
    video_path: Path,
    model_name: str,
    language: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[TranscriptSegment]:
    """Transcribe using lightning-whisper-mlx backend.

    mlx-whisper requires audio input, so we extract audio via FFmpeg first.
    """
    mlx_model = get_mlx_model(model_name)

    if progress_callback:
        progress_callback(0.1, "Extracting audio for MLX Whisper...")

    # Extract audio to temp WAV — mlx-whisper needs an audio file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _ffmpeg = find_binary("ffmpeg") or "ffmpeg"
        subprocess.run(
            [
                _ffmpeg, "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
            **get_subprocess_kwargs(),
        )

        if progress_callback:
            progress_callback(0.3, "Transcribing with MLX Whisper...")

        result = mlx_model.transcribe(audio_path=str(tmp_path))

        if progress_callback:
            progress_callback(0.8, "Processing segments...")

        return _parse_mlx_result(result)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode() if e.stderr else e}")
        return []
    finally:
        tmp_path.unlink(missing_ok=True)


def transcribe_clip(
    source_path: Path,
    start_time: float,
    end_time: float,
    model_name: str = "small.en",
    language: str = "en",
    backend: str = "auto",
) -> list[TranscriptSegment]:
    """Transcribe a specific clip range from a video.

    For efficiency, extracts the audio segment first.

    Args:
        source_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        model_name: Whisper model to use (ignored for groq backend)
        language: Language code
        backend: "auto", "faster-whisper", "mlx-whisper", or "groq"

    Returns:
        List of TranscriptSegment objects with times relative to clip start
    """
    resolved = _resolve_backend(backend)

    # Extract audio segment to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Extract audio segment with FFmpeg
        _ffmpeg = find_binary("ffmpeg") or "ffmpeg"
        subprocess.run(
            [
                _ffmpeg, "-y",
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", str(source_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
            **get_subprocess_kwargs(),
        )

        # Check if audio was actually extracted (file size > 0)
        if tmp_path.stat().st_size == 0:
            logger.warning(f"No audio extracted from {source_path} ({start_time}-{end_time})")
            return []

        if resolved == "groq":
            return _transcribe_cloud_groq(tmp_path, language)

        if resolved == "mlx-whisper":
            mlx_model = get_mlx_model(model_name)
            result = mlx_model.transcribe(audio_path=str(tmp_path))
            return _parse_mlx_result(result)

        # faster-whisper backend
        model = get_model(model_name)
        segments, info = model.transcribe(
            str(tmp_path),
            language=language if language != "auto" else None,
            word_timestamps=True,
            vad_filter=True,
        )

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


def _parse_mlx_result(result: dict) -> list[TranscriptSegment]:
    """Convert lightning-whisper-mlx output to TranscriptSegment list.

    Args:
        result: Dict with 'text', 'segments', 'language' keys

    Returns:
        List of TranscriptSegment objects
    """
    segments_out = []
    for seg in result.get("segments", []):
        segments_out.append(
            TranscriptSegment(
                start_time=seg.get("start", 0.0),
                end_time=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                confidence=0.0,  # mlx-whisper doesn't provide logprobs
            )
        )
    return segments_out


# Groq cloud transcription models
GROQ_MODELS = {
    "whisper-large-v3-turbo": {"speed": "fastest", "cost": "$0.04/hr"},
    "distil-whisper-large-v3-en": {"speed": "fast", "cost": "$0.02/hr"},
    "whisper-large-v3": {"speed": "standard", "cost": "$0.111/hr"},
}

_DEFAULT_GROQ_MODEL = "whisper-large-v3-turbo"


def _transcribe_cloud_groq(
    audio_path: Path,
    language: str = "en",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[TranscriptSegment]:
    """Transcribe using Groq cloud API via LiteLLM.

    Requires GROQ_API_KEY environment variable.

    Args:
        audio_path: Path to audio/video file
        language: Language code
        progress_callback: Optional callback(progress, message)

    Returns:
        List of TranscriptSegment objects
    """
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set; cannot use Groq transcription")
        raise TranscriptionError(
            "GROQ_API_KEY environment variable not set. "
            "Get a key at https://console.groq.com"
        )

    if progress_callback:
        progress_callback(0.1, "Preparing audio for Groq cloud transcription...")

    # Extract audio to temp file if input is video
    suffix = audio_path.suffix.lower()
    needs_extraction = suffix not in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")

    if needs_extraction:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _ffmpeg = find_binary("ffmpeg") or "ffmpeg"
            subprocess.run(
                [
                    _ffmpeg, "-y",
                    "-i", str(audio_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    str(tmp_path),
                ],
                capture_output=True,
                check=True,
                **get_subprocess_kwargs(),
            )
            audio_file_path = tmp_path
        except subprocess.CalledProcessError as e:
            tmp_path.unlink(missing_ok=True)
            logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode() if e.stderr else e}")
            return []
    else:
        audio_file_path = audio_path
        tmp_path = None

    try:
        if progress_callback:
            progress_callback(0.3, "Sending to Groq cloud...")

        # Load cloud model from settings
        try:
            from core.settings import load_settings
            settings = load_settings()
            groq_model = getattr(settings, "transcription_cloud_model", _DEFAULT_GROQ_MODEL)
        except Exception:
            groq_model = _DEFAULT_GROQ_MODEL

        import litellm

        with open(audio_file_path, "rb") as f:
            response = litellm.transcription(
                model=f"groq/{groq_model}",
                file=f,
                language=language if language != "auto" else None,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        if progress_callback:
            progress_callback(0.8, "Processing Groq response...")

        segments = []
        # LiteLLM returns a TranscriptionResponse with segments
        resp_segments = getattr(response, "segments", None) or []
        for seg in resp_segments:
            segments.append(
                TranscriptSegment(
                    start_time=seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0),
                    end_time=seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0),
                    text=(seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip(),
                    confidence=0.0,
                )
            )

        # Fallback: if no segments but we got text, return as single segment
        if not segments:
            text = getattr(response, "text", "")
            if text:
                segments.append(
                    TranscriptSegment(
                        start_time=0.0,
                        end_time=0.0,
                        text=text.strip(),
                        confidence=0.0,
                    )
                )

        if progress_callback:
            progress_callback(1.0, f"Groq transcribed {len(segments)} segments")

        logger.info(f"Groq cloud transcription: {len(segments)} segments from {audio_path.name}")
        return segments

    except Exception as e:
        logger.error(f"Groq cloud transcription failed: {e}")
        raise TranscriptionError(f"Groq transcription failed: {e}") from e

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def get_transcript_text(segments: list[TranscriptSegment]) -> str:
    """Get full transcript text from segments.

    Args:
        segments: List of TranscriptSegment objects

    Returns:
        Combined transcript text
    """
    return " ".join(seg.text for seg in segments)
