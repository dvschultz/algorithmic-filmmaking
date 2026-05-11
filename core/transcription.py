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

import contextlib
import importlib.util
import logging
import os
import platform
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.binary_resolver import find_binary, get_subprocess_env, get_subprocess_kwargs

logger = logging.getLogger(__name__)

# Avoid Hugging Face tokenizer thread-pool warnings when FFmpeg subprocesses
# are spawned after tokenizer initialization.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Lazy load to avoid startup delay
_model = None
_model_name = None
_faster_whisper_available = None

_mlx_model = None
_mlx_model_name = None
_mlx_whisper_available = None
_mlx_model_lock = threading.Lock()


def _mlx_workdir() -> Path:
    """Writable parent directory for lightning-whisper-mlx's ``./mlx_models/`` cache.

    The library hardcodes a relative path, so the process CWD must be a
    writable directory at both load and transcribe time. In frozen macOS
    builds CWD defaults to ``/`` (read-only), so we redirect to the model
    cache dir under Application Support.
    """
    from core.settings import load_settings
    workdir = load_settings().model_cache_dir / "mlx_whisper"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


@contextlib.contextmanager
def _chdir_for_mlx():
    """Temporarily chdir to the MLX workdir so ``./mlx_models/`` resolves to a writable path."""
    previous = os.getcwd()
    try:
        os.chdir(_mlx_workdir())
        yield
    finally:
        try:
            os.chdir(previous)
        except OSError:
            # Previous CWD may have disappeared (e.g., temp dir cleanup);
            # fall back to the workdir rather than leaking CWD errors.
            os.chdir(_mlx_workdir())


def _has_audio_stream(path: Path) -> Optional[bool]:
    """Return whether a media file has an audio stream.

    Returns ``False`` when ffprobe can positively determine that no audio stream
    exists. Returns ``None`` when probing is unavailable or inconclusive so
    transcription can continue with the existing backend-specific behavior.
    """
    ffprobe_path = find_binary("ffprobe")
    if ffprobe_path is None:
        logger.debug("ffprobe not available; skipping audio-stream preflight for %s", path)
        return None

    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            **get_subprocess_kwargs(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("ffprobe audio-stream preflight failed for %s: %s", path, exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "ffprobe audio-stream preflight returned non-zero for %s: %s",
            path,
            (result.stderr or "").strip(),
        )
        return None

    return bool(result.stdout.strip())


def _require_ffmpeg() -> str:
    """Return the resolved FFmpeg path or raise a user-facing transcription error."""
    ffmpeg_path = find_binary("ffmpeg")
    if ffmpeg_path is None:
        raise FFmpegNotFoundError()
    return ffmpeg_path


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
    "tiny.en": "tiny",
    "small.en": "small",
    "medium.en": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3",  # PINNED: lightning-whisper-mlx lacks turbo variant; silently maps to large-v3
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


class FFmpegNotFoundError(TranscriptionError):
    """Raised when FFmpeg is required but unavailable."""

    def __init__(self):
        super().__init__(
            "FFmpeg is required for transcription but was not found. "
            "Install FFmpeg from Settings > Dependencies and try again."
        )


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
    """Check if faster-whisper is installed without importing PyAV."""
    global _faster_whisper_available
    if _faster_whisper_available is None:
        _faster_whisper_available = (
            importlib.util.find_spec("faster_whisper") is not None
        )
    return _faster_whisper_available


def ensure_faster_whisper_runtime_available():
    """Validate that the faster-whisper runtime imports cleanly."""
    try:
        from faster_whisper import WhisperModel

        return WhisperModel
    except Exception as e:
        raise RuntimeError(f"transcription runtime is incomplete: {e}") from e


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


def ensure_mlx_whisper_runtime_available():
    """Validate that the MLX transcription runtime imports cleanly."""
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        raise RuntimeError("mlx-whisper is only available on Apple Silicon")

    try:
        from lightning_whisper_mlx import LightningWhisperMLX

        return LightningWhisperMLX
    except Exception as e:
        raise RuntimeError(f"mlx transcription runtime is incomplete: {e}") from e


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
class WordTimestamp:
    """A single word with start/end timestamps from forced alignment or word-level ASR.

    Times are in the same frame of reference as the parent ``TranscriptSegment``
    (clip-relative seconds when produced from per-clip transcription/alignment).

    ``probability`` is optional because different backends surface different
    confidence metrics: ``faster-whisper`` reports a per-word probability,
    while forced aligners (e.g. ``ctc-forced-aligner``) emit a different
    signal that may or may not be normalized into this field.
    """

    start: float  # clip-relative seconds
    end: float
    text: str
    probability: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data: dict = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.probability is not None:
            data["probability"] = self.probability
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WordTimestamp":
        """Deserialize from dictionary."""
        probability = data.get("probability")
        return cls(
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            text=data.get("text", ""),
            probability=probability if probability is None else float(probability),
        )


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    start_time: float  # seconds from clip start
    end_time: float
    text: str
    confidence: float = 0.0
    # Word-level timestamps. ``None`` means "no word data surfaced yet" (legacy
    # transcripts, or MLX transcripts before forced alignment runs). ``[]`` means
    # "alignment ran and produced no words" (silence / instrumental). The two
    # states are deliberately distinct and must not be conflated.
    words: Optional[list[WordTimestamp]] = None
    # ISO 639-1 detected language for this segment (e.g. ``"en"``). Populated
    # from ``faster-whisper``'s ``info.language`` or MLX's ``result["language"]``.
    # ``None`` only for transcripts produced before U1 (legacy project files).
    language: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data: dict = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
        }
        # Only emit the new keys when they carry information so old project
        # files stay byte-comparable after a round trip through code that
        # didn't populate them.
        if self.words is not None:
            data["words"] = [w.to_dict() for w in self.words]
        if self.language is not None:
            data["language"] = self.language
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        """Deserialize from dictionary.

        Back-compat: legacy segments lacking ``words`` and ``language`` keys
        deserialize with ``words=None`` and ``language=None``. The distinction
        between ``words=None`` (no data surfaced) and ``words=[]`` (alignment
        ran, found nothing) is preserved.
        """
        if "words" in data:
            raw_words = data["words"]
            words: Optional[list[WordTimestamp]] = (
                [WordTimestamp.from_dict(w) for w in raw_words]
                if raw_words is not None
                else None
            )
        else:
            words = None

        return cls(
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            words=words,
            language=data.get("language"),
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
        try:
            import certifi
            import os as _os
            _os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            _os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass

        WhisperModel = ensure_faster_whisper_runtime_available()

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

    if _mlx_model is not None and _mlx_model_name == mlx_name:
        return _mlx_model

    with _mlx_model_lock:
        if _mlx_model is not None and _mlx_model_name == mlx_name:
            return _mlx_model

        LightningWhisperMLX = ensure_mlx_whisper_runtime_available()

        try:
            import certifi
            import os as _os
            _os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            _os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass

        logger.info(f"Loading MLX Whisper model: {mlx_name}")

        try:
            # lightning-whisper-mlx reads/writes ``./mlx_models/`` relative to
            # CWD — chdir into a writable dir so downloads don't fail with
            # "Read-only file system" when the frozen app is launched from /.
            with _chdir_for_mlx():
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
    has_audio = _has_audio_stream(video_path)
    if has_audio is False:
        logger.info("Skipping transcription for %s: no audio track found", video_path)
        if progress_callback:
            progress_callback(1.0, "No audio track found")
        return []

    resolved = _resolve_backend(backend)

    if resolved == "groq":
        return _transcribe_cloud_groq(video_path, language, progress_callback)

    if resolved == "mlx-whisper":
        return _transcribe_video_mlx(video_path, model_name, language, progress_callback)

    return _transcribe_video_faster_whisper(video_path, model_name, language, progress_callback)


def _build_word_timestamps(raw_words) -> Optional[list[WordTimestamp]]:
    """Convert faster-whisper's per-segment .words iterable to WordTimestamp objects.

    Returns ``None`` when the upstream segment did not surface word data
    (defensive — ``word_timestamps=True`` should always populate this, but
    callers should not crash when it's missing). Returns ``[]`` when the
    segment surfaced an explicit empty list.
    """
    if raw_words is None:
        return None
    result: list[WordTimestamp] = []
    for w in raw_words:
        text = getattr(w, "word", None)
        if text is None:
            text = getattr(w, "text", "")
        probability = getattr(w, "probability", None)
        result.append(
            WordTimestamp(
                start=float(getattr(w, "start", 0.0)),
                end=float(getattr(w, "end", 0.0)),
                text=str(text).strip(),
                probability=float(probability) if probability is not None else None,
            )
        )
    return result


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

    detected_language = getattr(info, "language", None)

    results = []
    for segment in segments:
        results.append(
            TranscriptSegment(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text.strip(),
                confidence=segment.avg_logprob,
                words=_build_word_timestamps(getattr(segment, "words", None)),
                language=detected_language,
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
        _ffmpeg = _require_ffmpeg()
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
            env=get_subprocess_env(),
            **get_subprocess_kwargs(),
        )

        if progress_callback:
            progress_callback(0.3, "Transcribing with MLX Whisper...")

        # MLX transcribe also resolves ``./mlx_models/<name>`` relative to CWD.
        with _chdir_for_mlx():
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
    has_audio = _has_audio_stream(source_path)
    if has_audio is False:
        logger.info("Skipping clip transcription for %s: no audio track found", source_path)
        return []

    resolved = _resolve_backend(backend)

    # Extract audio segment to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Extract audio segment with FFmpeg
        _ffmpeg = _require_ffmpeg()
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
            env=get_subprocess_env(),
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
            with _chdir_for_mlx():
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

        detected_language = getattr(info, "language", None)

        results = []
        for segment in segments:
            results.append(
                TranscriptSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob,
                    words=_build_word_timestamps(getattr(segment, "words", None)),
                    language=detected_language,
                )
            )

        return results

    except FileNotFoundError as e:
        raise FFmpegNotFoundError() from e
    except subprocess.CalledProcessError as e:
        logger.warning(f"FFmpeg failed to extract audio: {e.stderr.decode() if e.stderr else e}")
        return []

    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_mlx_result(result: dict) -> list[TranscriptSegment]:
    """Convert lightning-whisper-mlx output to TranscriptSegment list.

    Args:
        result: Dict with 'text', 'segments', 'language' keys.
            Segments are lists of [start_frames, end_frames, text] where
            frames are mel spectrogram frames (100 frames/second for
            standard Whisper HOP_LENGTH=160, SAMPLE_RATE=16000).

    Returns:
        List of TranscriptSegment objects. ``language`` is populated from
        ``result["language"]`` (previously discarded); ``words`` is left as
        ``None`` — U2's forced-alignment pass fills it in.
    """
    # Whisper mel frames → seconds: frames / (SAMPLE_RATE / HOP_LENGTH)
    _FRAMES_PER_SECOND = 100.0  # 16000 / 160

    detected_language = result.get("language")

    segments_out = []
    for seg in result.get("segments", []):
        if isinstance(seg, dict):
            # Older or alternative format: dict with 'start', 'end', 'text' keys
            segments_out.append(
                TranscriptSegment(
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    confidence=0.0,
                    words=None,
                    language=detected_language,
                )
            )
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            # Current format: [start_frames, end_frames, text]
            segments_out.append(
                TranscriptSegment(
                    start_time=float(seg[0]) / _FRAMES_PER_SECOND,
                    end_time=float(seg[1]) / _FRAMES_PER_SECOND,
                    text=str(seg[2]).strip(),
                    confidence=0.0,
                    words=None,
                    language=detected_language,
                )
            )
        else:
            logger.warning(f"Unexpected segment format: {seg!r}")
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
            _ffmpeg = _require_ffmpeg()
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
                env=get_subprocess_env(),
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
