"""Application settings management.

This module provides Qt-free settings management with:
- JSON file storage (~/.config/scene-ripper/config.json)
- Environment variable overrides
- Secure keyring storage for API keys
- QSettings migration support (for existing users)

Priority order: Environment variables > JSON config > Defaults
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Keyring service name for secure credential storage
KEYRING_SERVICE = "com.scene-ripper.app"
KEYRING_YOUTUBE_KEY = "youtube_api_key"
KEYRING_LLM_API_KEY = "llm_api_key"  # Legacy, kept for compatibility

# Provider-specific keyring keys
KEYRING_ANTHROPIC_API_KEY = "anthropic_api_key"
KEYRING_OPENAI_API_KEY = "openai_api_key"
KEYRING_GEMINI_API_KEY = "gemini_api_key"
KEYRING_OPENROUTER_API_KEY = "openrouter_api_key"
KEYRING_REPLICATE_API_KEY = "replicate_api_key"

# Config schema version
CONFIG_VERSION = "1.0"

# Environment variable names
ENV_YOUTUBE_API_KEY = "YOUTUBE_API_KEY"
ENV_CACHE_DIR = "SCENE_RIPPER_CACHE_DIR"
ENV_DOWNLOAD_DIR = "SCENE_RIPPER_DOWNLOAD_DIR"
ENV_EXPORT_DIR = "SCENE_RIPPER_EXPORT_DIR"
ENV_CONFIG_PATH = "SCENE_RIPPER_CONFIG"
ENV_SENSITIVITY = "SCENE_RIPPER_SENSITIVITY"
ENV_WHISPER_MODEL = "SCENE_RIPPER_WHISPER_MODEL"

# LLM environment variables
ENV_LLM_PROVIDER = "SCENE_RIPPER_LLM_PROVIDER"
ENV_LLM_MODEL = "SCENE_RIPPER_LLM_MODEL"
ENV_LLM_API_KEY = "SCENE_RIPPER_LLM_API_KEY"  # Legacy, kept for compatibility
ENV_LLM_API_BASE = "SCENE_RIPPER_LLM_API_BASE"
ENV_LLM_TEMPERATURE = "SCENE_RIPPER_LLM_TEMPERATURE"

# Provider-specific API key environment variables
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
ENV_REPLICATE_API_KEY = "REPLICATE_API_TOKEN"


def _get_api_key_from_keyring() -> str:
    """Retrieve API key from system keyring."""
    try:
        import keyring
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_YOUTUBE_KEY)
        return key or ""
    except Exception as e:
        logger.debug(f"Could not read from keyring: {e}")
        return ""


def _set_api_key_in_keyring(api_key: str) -> bool:
    """Store API key in system keyring."""
    try:
        import keyring
        if api_key:
            keyring.set_password(KEYRING_SERVICE, KEYRING_YOUTUBE_KEY, api_key)
        else:
            # Delete the key if empty
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_YOUTUBE_KEY)
            except keyring.errors.PasswordDeleteError:
                pass  # Key didn't exist
        return True
    except Exception as e:
        logger.warning(f"Could not write to keyring: {e}")
        return False


def _get_llm_api_key_from_keyring() -> str:
    """Retrieve LLM API key from system keyring."""
    try:
        import keyring
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_LLM_API_KEY)
        return key or ""
    except Exception as e:
        logger.debug(f"Could not read LLM API key from keyring: {e}")
        return ""


def _set_llm_api_key_in_keyring(api_key: str) -> bool:
    """Store LLM API key in system keyring."""
    try:
        import keyring
        if api_key:
            keyring.set_password(KEYRING_SERVICE, KEYRING_LLM_API_KEY, api_key)
        else:
            # Delete the key if empty
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_LLM_API_KEY)
            except keyring.errors.PasswordDeleteError:
                pass  # Key didn't exist
        return True
    except Exception as e:
        logger.warning(f"Could not write LLM API key to keyring: {e}")
        return False


# Provider-specific API key getters/setters

def _get_provider_api_key_from_keyring(keyring_key: str) -> str:
    """Retrieve a provider-specific API key from system keyring."""
    try:
        import keyring
        key = keyring.get_password(KEYRING_SERVICE, keyring_key)
        return key or ""
    except Exception as e:
        logger.debug(f"Could not read {keyring_key} from keyring: {e}")
        return ""


def _set_provider_api_key_in_keyring(keyring_key: str, api_key: str) -> bool:
    """Store a provider-specific API key in system keyring."""
    try:
        import keyring
        if api_key:
            keyring.set_password(KEYRING_SERVICE, keyring_key, api_key)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, keyring_key)
            except keyring.errors.PasswordDeleteError:
                pass
        return True
    except Exception as e:
        logger.warning(f"Could not write {keyring_key} to keyring: {e}")
        return False


def get_anthropic_api_key() -> str:
    """Get Anthropic API key with priority: env var > keyring."""
    if api_key := os.environ.get(ENV_ANTHROPIC_API_KEY):
        return api_key
    return _get_provider_api_key_from_keyring(KEYRING_ANTHROPIC_API_KEY)


def set_anthropic_api_key(api_key: str) -> bool:
    """Store Anthropic API key in system keyring."""
    return _set_provider_api_key_in_keyring(KEYRING_ANTHROPIC_API_KEY, api_key)


def get_openai_api_key() -> str:
    """Get OpenAI API key with priority: env var > keyring."""
    if api_key := os.environ.get(ENV_OPENAI_API_KEY):
        return api_key
    return _get_provider_api_key_from_keyring(KEYRING_OPENAI_API_KEY)


def set_openai_api_key(api_key: str) -> bool:
    """Store OpenAI API key in system keyring."""
    return _set_provider_api_key_in_keyring(KEYRING_OPENAI_API_KEY, api_key)


def get_gemini_api_key() -> str:
    """Get Gemini API key with priority: env var > keyring."""
    if api_key := os.environ.get(ENV_GEMINI_API_KEY):
        return api_key
    return _get_provider_api_key_from_keyring(KEYRING_GEMINI_API_KEY)


def set_gemini_api_key(api_key: str) -> bool:
    """Store Gemini API key in system keyring."""
    return _set_provider_api_key_in_keyring(KEYRING_GEMINI_API_KEY, api_key)


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key with priority: env var > keyring."""
    if api_key := os.environ.get(ENV_OPENROUTER_API_KEY):
        return api_key
    return _get_provider_api_key_from_keyring(KEYRING_OPENROUTER_API_KEY)


def set_openrouter_api_key(api_key: str) -> bool:
    """Store OpenRouter API key in system keyring."""
    return _set_provider_api_key_in_keyring(KEYRING_OPENROUTER_API_KEY, api_key)


def get_replicate_api_key() -> str:
    """Get Replicate API key with priority: env var > keyring."""
    if api_key := os.environ.get(ENV_REPLICATE_API_KEY):
        return api_key
    return _get_provider_api_key_from_keyring(KEYRING_REPLICATE_API_KEY)


def set_replicate_api_key(api_key: str) -> bool:
    """Store Replicate API key in system keyring."""
    return _set_provider_api_key_in_keyring(KEYRING_REPLICATE_API_KEY, api_key)


def is_api_key_from_env(provider: str) -> bool:
    """Check if a provider's API key is set via environment variable.

    Args:
        provider: Provider name (anthropic, openai, gemini, openrouter)

    Returns:
        True if the API key is set via environment variable
    """
    env_map = {
        "anthropic": ENV_ANTHROPIC_API_KEY,
        "openai": ENV_OPENAI_API_KEY,
        "gemini": ENV_GEMINI_API_KEY,
        "openrouter": ENV_OPENROUTER_API_KEY,
        "replicate": ENV_REPLICATE_API_KEY,
    }
    env_var = env_map.get(provider)
    return bool(env_var and os.environ.get(env_var))


def get_llm_api_key() -> str:
    """Get LLM API key for the currently selected provider.

    Priority: provider-specific env var > provider-specific keyring > legacy env var > legacy keyring

    Returns:
        API key string, or empty string if not configured
    """
    settings = load_settings()
    provider = settings.llm_provider

    # Route to provider-specific getter
    if provider == "anthropic":
        return get_anthropic_api_key()
    elif provider == "openai":
        return get_openai_api_key()
    elif provider == "gemini":
        return get_gemini_api_key()
    elif provider == "openrouter":
        return get_openrouter_api_key()

    # For local/ollama, no key needed
    # Fall back to legacy key for backwards compatibility
    if api_key := os.environ.get(ENV_LLM_API_KEY):
        return api_key
    return _get_llm_api_key_from_keyring()


def set_llm_api_key(api_key: str) -> bool:
    """Store LLM API key in system keyring (legacy function).

    Args:
        api_key: The API key to store

    Returns:
        True if save succeeded
    """
    return _set_llm_api_key_in_keyring(api_key)


def _get_videos_dir() -> Path:
    """Get platform-appropriate videos directory.

    - Linux: Uses XDG_VIDEOS_DIR or ~/Videos
    - macOS: Uses ~/Movies
    - Windows: Uses ~/Videos
    """
    if sys.platform == "linux":
        # Check XDG_VIDEOS_DIR first (set by user-dirs.dirs)
        xdg_videos = os.environ.get("XDG_VIDEOS_DIR")
        if xdg_videos:
            return Path(xdg_videos)
        # Fallback to ~/Videos (common on most Linux distros)
        return Path.home() / "Videos"
    elif sys.platform == "darwin":
        return Path.home() / "Movies"
    else:
        # Windows and others
        return Path.home() / "Videos"


def _get_default_download_dir() -> Path:
    """Get platform-appropriate download directory for videos."""
    return _get_videos_dir() / "Scene Ripper Downloads"


def _get_default_export_dir() -> Path:
    """Get platform-appropriate export directory."""
    return _get_videos_dir()


def _get_config_dir() -> Path:
    """Get platform-appropriate config directory (XDG-compliant)."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "scene-ripper"
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "scene-ripper"


def _get_config_path() -> Path:
    """Get config file path, respecting SCENE_RIPPER_CONFIG env var."""
    if custom_path := os.environ.get(ENV_CONFIG_PATH):
        return Path(custom_path)
    return _get_config_dir() / "config.json"


def _get_cache_dir() -> Path:
    """Get platform-appropriate cache directory."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "scene-ripper" / "cache"
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "scene-ripper"


# Quality preset definitions
QUALITY_PRESETS = {
    "high": {"crf": 18, "preset": "slow", "bitrate": "10M"},
    "medium": {"crf": 23, "preset": "medium", "bitrate": "5M"},
    "low": {"crf": 28, "preset": "fast", "bitrate": "2M"},
}

# Resolution presets (max width, max height)
RESOLUTION_PRESETS = {
    "original": (None, None),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
}

# FPS presets
FPS_PRESETS = {
    "original": None,
    "24": 24.0,
    "30": 30.0,
    "60": 60.0,
}


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    # Paths (use platform-appropriate defaults)
    thumbnail_cache_dir: Path = field(
        default_factory=lambda: _get_cache_dir() / "thumbnails"
    )
    model_cache_dir: Path = field(
        default_factory=lambda: _get_cache_dir() / "models"
    )
    download_dir: Path = field(default_factory=_get_default_download_dir)
    export_dir: Path = field(default_factory=_get_default_export_dir)

    # Detection defaults
    default_sensitivity: float = 3.0
    min_scene_length_seconds: float = 0.5

    # Export defaults
    export_quality: str = "medium"  # high, medium, low
    export_resolution: str = "original"  # original, 1080p, 720p, 480p
    export_fps: str = "original"  # original, 24, 30, 60

    # Transcription settings
    transcription_model: str = "small.en"  # tiny.en, small.en, medium.en, large-v3
    transcription_language: str = "en"  # en, auto, or specific language code
    transcription_backend: str = "auto"  # auto, faster-whisper, mlx-whisper

    # Appearance
    theme_preference: str = "system"  # system, light, dark

    # YouTube API
    youtube_api_key: str = ""
    youtube_results_count: int = 25  # 10-50
    youtube_parallel_downloads: int = 2  # 1-3

    # LLM Settings (API key stored in keyring, not here)
    llm_provider: str = "local"  # local, openai, anthropic, gemini, openrouter
    llm_model: str = "qwen3:8b"  # Default for local Ollama
    llm_api_base: str = ""  # For local/custom endpoints (default: http://localhost:11434 for Ollama)
    llm_temperature: float = 0.7

    # Per-provider model preferences
    ollama_model: str = "qwen3:8b"
    openai_model: str = "gpt-5.2"
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    gemini_model: str = "gemini-2.5-flash"
    openrouter_model: str = "anthropic/claude-sonnet-4"

    # Vision Description Settings
    description_model_tier: str = "local"  # local, cloud (legacy: cpu, gpu)
    description_model_local: str = "mlx-community/Qwen3-VL-4B-4bit"
    description_model_cpu: str = "vikhyatk/moondream2"  # Legacy — kept for backward compat
    description_model_gpu: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf"  # Legacy
    description_model_cloud: str = "gemini-3-flash-preview"
    description_temporal_frames: int = 4
    description_input_mode: str = "frame"  # "frame" = single frame, "video" = video clip (Gemini only)

    # Text Extraction (OCR) Settings
    text_extraction_method: str = "hybrid"  # paddleocr, vlm, hybrid
    text_extraction_vlm_model: str = "gemini-3-flash-preview"
    text_detection_enabled: bool = True  # Deprecated: PaddleOCR handles detection internally
    text_detection_confidence: float = 0.5  # Deprecated: PaddleOCR has its own thresholds

    # Exquisite Corpus (Poetry Generation) Settings
    exquisite_corpus_model: str = "gemini-3-flash-preview"  # Model for poem generation
    exquisite_corpus_temperature: float = 0.8  # Creativity level (0.0-1.0)

    # Shot Classification Settings
    shot_classifier_tier: str = "cpu"  # cpu, cloud
    shot_classifier_cloud_model: str = "gemini-2.5-flash-lite"  # Cloud VLM model for shot classification
    shot_classifier_replicate_model: str = "dvschultz/shot-type-classifier"  # Legacy VideoMAE model

    # Analysis Picker - remembered selection
    analysis_selected_operations: list = field(
        default_factory=lambda: ["colors", "shots", "transcribe"]
    )

    # Analysis Parallelism Settings
    color_analysis_parallelism: int = 4  # 1-8, I/O-bound image loading
    description_parallelism: int = 3  # 1-5, cloud API I/O-bound
    transcription_parallelism: int = 2  # 1-4, FFmpeg + Whisper memory-limited
    local_model_parallelism: int = 1  # 1-4, CLIP/MobileNet/YOLO not thread-safe

    # Rich Cinematography Analysis Settings
    cinematography_input_mode: str = "frame"  # "frame" = single keyframe, "video" = video clip (Gemini only)
    cinematography_model: str = "gemini-3-flash-preview"  # VLM model for rich analysis
    cinematography_batch_parallelism: int = 2  # Number of concurrent VLM requests (1-5)

    def get_quality_preset(self) -> dict:
        """Get FFmpeg parameters for current quality setting."""
        return QUALITY_PRESETS.get(self.export_quality, QUALITY_PRESETS["medium"])

    def get_resolution(self) -> tuple[Optional[int], Optional[int]]:
        """Get max width/height for current resolution setting."""
        return RESOLUTION_PRESETS.get(self.export_resolution, (None, None))

    def get_fps(self) -> Optional[float]:
        """Get FPS value for current setting (None = use source)."""
        return FPS_PRESETS.get(self.export_fps, None)

    def min_scene_length_frames(self, fps: float = 30.0) -> int:
        """Convert min scene length to frames."""
        return int(self.min_scene_length_seconds * fps)

    def get_model_for_provider(self, provider: str) -> str:
        """Get the configured model for a specific provider.

        Args:
            provider: Provider key (local, openai, anthropic, gemini, openrouter)

        Returns:
            The model string configured for that provider
        """
        model_map = {
            "local": self.ollama_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "gemini": self.gemini_model,
            "openrouter": self.openrouter_model,
        }
        return model_map.get(provider, self.llm_model)


def get_default_settings() -> Settings:
    """Create a Settings instance with all defaults."""
    return Settings()


# Track which settings are from environment variables
_env_overridden: Set[str] = set()


def get_env_overridden_settings() -> Set[str]:
    """Get the set of settings names that are overridden by environment variables.

    Returns:
        Set of setting field names that were loaded from env vars
    """
    return _env_overridden.copy()


def is_from_environment(setting_name: str) -> bool:
    """Check if a specific setting was loaded from an environment variable.

    Args:
        setting_name: Name of the setting field (e.g., "youtube_api_key")

    Returns:
        True if the setting value came from an environment variable
    """
    return setting_name in _env_overridden


def _apply_env_overrides(settings: Settings) -> Settings:
    """Apply environment variable overrides to settings.

    Args:
        settings: Settings instance to modify

    Returns:
        Modified settings (same instance)
    """
    global _env_overridden
    _env_overridden = set()

    # YOUTUBE_API_KEY
    if api_key := os.environ.get(ENV_YOUTUBE_API_KEY):
        settings.youtube_api_key = api_key
        _env_overridden.add("youtube_api_key")

    # SCENE_RIPPER_CACHE_DIR
    if cache_dir := os.environ.get(ENV_CACHE_DIR):
        settings.thumbnail_cache_dir = Path(cache_dir)
        _env_overridden.add("thumbnail_cache_dir")

    # SCENE_RIPPER_DOWNLOAD_DIR
    if download_dir := os.environ.get(ENV_DOWNLOAD_DIR):
        settings.download_dir = Path(download_dir)
        _env_overridden.add("download_dir")

    # SCENE_RIPPER_EXPORT_DIR
    if export_dir := os.environ.get(ENV_EXPORT_DIR):
        settings.export_dir = Path(export_dir)
        _env_overridden.add("export_dir")

    # SCENE_RIPPER_SENSITIVITY
    if sensitivity := os.environ.get(ENV_SENSITIVITY):
        try:
            settings.default_sensitivity = float(sensitivity)
            _env_overridden.add("default_sensitivity")
        except ValueError:
            logger.warning(f"Invalid {ENV_SENSITIVITY}: {sensitivity}")

    # SCENE_RIPPER_WHISPER_MODEL
    if model := os.environ.get(ENV_WHISPER_MODEL):
        settings.transcription_model = model
        _env_overridden.add("transcription_model")

    # LLM environment variables
    if llm_provider := os.environ.get(ENV_LLM_PROVIDER):
        settings.llm_provider = llm_provider
        _env_overridden.add("llm_provider")

    if llm_model := os.environ.get(ENV_LLM_MODEL):
        settings.llm_model = llm_model
        _env_overridden.add("llm_model")

    if llm_api_base := os.environ.get(ENV_LLM_API_BASE):
        settings.llm_api_base = llm_api_base
        _env_overridden.add("llm_api_base")

    if llm_temperature := os.environ.get(ENV_LLM_TEMPERATURE):
        try:
            settings.llm_temperature = float(llm_temperature)
            _env_overridden.add("llm_temperature")
        except ValueError:
            logger.warning(f"Invalid {ENV_LLM_TEMPERATURE}: {llm_temperature}")

    return settings


def _load_from_json(config_path: Path, settings: Settings) -> Settings:
    """Load settings from JSON config file.

    Args:
        config_path: Path to the JSON config file
        settings: Settings instance to modify

    Returns:
        Modified settings (same instance)
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return settings

    # Paths section
    if paths := data.get("paths"):
        if val := paths.get("thumbnail_cache_dir"):
            settings.thumbnail_cache_dir = Path(val).expanduser()
        if val := paths.get("model_cache_dir"):
            settings.model_cache_dir = Path(val).expanduser()
        if val := paths.get("download_dir"):
            settings.download_dir = Path(val).expanduser()
        if val := paths.get("export_dir"):
            settings.export_dir = Path(val).expanduser()

    # Detection section
    # Note: auto_analyze_colors, auto_classify_shots are deprecated and ignored
    if detection := data.get("detection"):
        if "default_sensitivity" in detection:
            settings.default_sensitivity = float(detection["default_sensitivity"])
        if "min_scene_length_seconds" in detection:
            settings.min_scene_length_seconds = float(detection["min_scene_length_seconds"])

    # Transcription section
    # Note: auto_transcribe is deprecated and ignored
    if transcription := data.get("transcription"):
        if val := transcription.get("model"):
            settings.transcription_model = val
        if val := transcription.get("language"):
            settings.transcription_language = val
        if val := transcription.get("backend"):
            settings.transcription_backend = val

    # Export section
    if export := data.get("export"):
        if val := export.get("quality"):
            settings.export_quality = val
        if val := export.get("resolution"):
            settings.export_resolution = val
        if val := export.get("fps"):
            settings.export_fps = val

    # Appearance section
    if appearance := data.get("appearance"):
        if val := appearance.get("theme_preference"):
            settings.theme_preference = val

    # YouTube section (API key comes from keyring, not JSON)
    if youtube := data.get("youtube"):
        if "results_count" in youtube:
            settings.youtube_results_count = int(youtube["results_count"])
        if "parallel_downloads" in youtube:
            settings.youtube_parallel_downloads = int(youtube["parallel_downloads"])

    # LLM section (API key comes from keyring, not JSON)
    if llm := data.get("llm"):
        if val := llm.get("provider"):
            settings.llm_provider = val
        if val := llm.get("model"):
            settings.llm_model = val
        if val := llm.get("api_base"):
            settings.llm_api_base = val
        if "temperature" in llm:
            settings.llm_temperature = float(llm["temperature"])
        # Per-provider model preferences
        if val := llm.get("ollama_model"):
            settings.ollama_model = val
        if val := llm.get("openai_model"):
            settings.openai_model = val
        if val := llm.get("anthropic_model"):
            settings.anthropic_model = val
        if val := llm.get("gemini_model"):
            settings.gemini_model = val
        if val := llm.get("openrouter_model"):
            settings.openrouter_model = val

    # Description section
    if description := data.get("description"):
        if val := description.get("model_tier"):
            # Migrate legacy tier names
            if val in ("cpu", "gpu"):
                val = "local"
            settings.description_model_tier = val
        if val := description.get("model_local"):
            settings.description_model_local = val
        if val := description.get("model_cpu"):
            settings.description_model_cpu = val
        if val := description.get("model_gpu"):
            settings.description_model_gpu = val
        if val := description.get("model_cloud"):
            settings.description_model_cloud = val
        if "temporal_frames" in description:
            settings.description_temporal_frames = int(description["temporal_frames"])
        if val := description.get("input_mode"):
            settings.description_input_mode = val
        # Migration: convert old use_video_for_gemini to new input_mode
        elif "use_video_for_gemini" in description:
            settings.description_input_mode = "video" if description["use_video_for_gemini"] else "frame"

    # Text Extraction section
    if text_extraction := data.get("text_extraction"):
        if val := text_extraction.get("method"):
            # Migrate legacy method name
            if val == "tesseract":
                val = "paddleocr"
            settings.text_extraction_method = val
        if val := text_extraction.get("vlm_model"):
            settings.text_extraction_vlm_model = val
        if "detection_enabled" in text_extraction:
            settings.text_detection_enabled = bool(text_extraction["detection_enabled"])
        if "detection_confidence" in text_extraction:
            settings.text_detection_confidence = float(text_extraction["detection_confidence"])

    # Exquisite Corpus section
    if exquisite_corpus := data.get("exquisite_corpus"):
        if val := exquisite_corpus.get("model"):
            settings.exquisite_corpus_model = val
        if "temperature" in exquisite_corpus:
            settings.exquisite_corpus_temperature = float(exquisite_corpus["temperature"])

    # Shot Classifier section
    if shot_classifier := data.get("shot_classifier"):
        if val := shot_classifier.get("tier"):
            settings.shot_classifier_tier = val
        if val := shot_classifier.get("cloud_model"):
            settings.shot_classifier_cloud_model = val
        if val := shot_classifier.get("replicate_model"):
            settings.shot_classifier_replicate_model = val

    # Analysis parallelism section
    if analysis := data.get("analysis"):
        if "color_analysis_parallelism" in analysis:
            settings.color_analysis_parallelism = int(analysis["color_analysis_parallelism"])
        if "description_parallelism" in analysis:
            settings.description_parallelism = int(analysis["description_parallelism"])
        if "transcription_parallelism" in analysis:
            settings.transcription_parallelism = int(analysis["transcription_parallelism"])
        if "local_model_parallelism" in analysis:
            settings.local_model_parallelism = int(analysis["local_model_parallelism"])
        if "selected_operations" in analysis:
            val = analysis["selected_operations"]
            if isinstance(val, list):
                settings.analysis_selected_operations = val

    # Cinematography section
    if cinematography := data.get("cinematography"):
        if val := cinematography.get("input_mode"):
            settings.cinematography_input_mode = val
        if val := cinematography.get("model"):
            settings.cinematography_model = val
        if "batch_parallelism" in cinematography:
            settings.cinematography_batch_parallelism = int(cinematography["batch_parallelism"])

    return settings


def _settings_to_json(settings: Settings) -> dict:
    """Convert settings to JSON-serializable dict.

    Args:
        settings: Settings instance

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "version": CONFIG_VERSION,
        "paths": {
            "thumbnail_cache_dir": str(settings.thumbnail_cache_dir),
            "model_cache_dir": str(settings.model_cache_dir),
            "download_dir": str(settings.download_dir),
            "export_dir": str(settings.export_dir),
        },
        "detection": {
            "default_sensitivity": settings.default_sensitivity,
            "min_scene_length_seconds": settings.min_scene_length_seconds,
        },
        "transcription": {
            "model": settings.transcription_model,
            "language": settings.transcription_language,
            "backend": settings.transcription_backend,
        },
        "export": {
            "quality": settings.export_quality,
            "resolution": settings.export_resolution,
            "fps": settings.export_fps,
        },
        "appearance": {
            "theme_preference": settings.theme_preference,
        },
        "youtube": {
            "results_count": settings.youtube_results_count,
            "parallel_downloads": settings.youtube_parallel_downloads,
            # Note: API key is NOT stored here - it goes to keyring
        },
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "api_base": settings.llm_api_base,
            "temperature": settings.llm_temperature,
            # Per-provider model preferences
            "ollama_model": settings.ollama_model,
            "openai_model": settings.openai_model,
            "anthropic_model": settings.anthropic_model,
            "gemini_model": settings.gemini_model,
            "openrouter_model": settings.openrouter_model,
            # Note: API key is NOT stored here - it goes to keyring
        },
        "description": {
            "model_tier": settings.description_model_tier,
            "model_local": settings.description_model_local,
            "model_cpu": settings.description_model_cpu,
            "model_gpu": settings.description_model_gpu,
            "model_cloud": settings.description_model_cloud,
            "temporal_frames": settings.description_temporal_frames,
            "input_mode": settings.description_input_mode,
        },
        "text_extraction": {
            "method": settings.text_extraction_method,
            "vlm_model": settings.text_extraction_vlm_model,
            "detection_enabled": settings.text_detection_enabled,
            "detection_confidence": settings.text_detection_confidence,
        },
        "exquisite_corpus": {
            "model": settings.exquisite_corpus_model,
            "temperature": settings.exquisite_corpus_temperature,
        },
        "shot_classifier": {
            "tier": settings.shot_classifier_tier,
            "cloud_model": settings.shot_classifier_cloud_model,
            "replicate_model": settings.shot_classifier_replicate_model,
            # Note: API key is NOT stored here - it goes to keyring
        },
        "analysis": {
            "color_analysis_parallelism": settings.color_analysis_parallelism,
            "description_parallelism": settings.description_parallelism,
            "transcription_parallelism": settings.transcription_parallelism,
            "local_model_parallelism": settings.local_model_parallelism,
            "selected_operations": settings.analysis_selected_operations,
        },
        "cinematography": {
            "input_mode": settings.cinematography_input_mode,
            "model": settings.cinematography_model,
            "batch_parallelism": settings.cinematography_batch_parallelism,
        },
    }


def load_settings() -> Settings:
    """Load settings with priority: env vars > JSON config > defaults.

    This function is Qt-free and works in headless environments.

    Returns:
        Settings instance populated from available sources
    """
    settings = Settings()

    # 1. Load from JSON file if it exists
    config_path = _get_config_path()
    if config_path.exists():
        settings = _load_from_json(config_path, settings)

    # 2. Load API keys from keyring (if not already set from JSON, which doesn't store them)
    if not settings.youtube_api_key:
        settings.youtube_api_key = _get_api_key_from_keyring()

    # 3. Apply environment variable overrides (highest priority)
    settings = _apply_env_overrides(settings)

    logger.debug(f"Settings loaded (env overrides: {_env_overridden})")
    return settings


def save_settings(settings: Settings) -> bool:
    """Save settings to JSON file.

    This function is Qt-free and works in headless environments.
    API key is stored in keyring, not in the JSON file.

    Args:
        settings: Settings instance to save

    Returns:
        True if save succeeded
    """
    config_path = _get_config_path()

    try:
        # Create config directory with restrictive permissions
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(config_path.parent, 0o700)

        # Convert settings to JSON
        data = _settings_to_json(settings)

        # Atomic write: write to temp file then rename
        temp_path = config_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Set restrictive permissions on Unix
        if os.name != "nt":
            os.chmod(temp_path, 0o600)

        # Atomic rename
        os.replace(temp_path, config_path)

        # Save API key to keyring (secure storage)
        if settings.youtube_api_key:
            _set_api_key_in_keyring(settings.youtube_api_key)

        logger.info(f"Settings saved to {config_path}")
        return True

    except (OSError, IOError) as e:
        logger.error(f"Failed to save settings: {e}")
        return False


def migrate_from_qsettings() -> bool:
    """Migrate settings from QSettings to JSON (one-time operation).

    Called by GUI on first launch if JSON doesn't exist but QSettings does.
    This function requires Qt to be available.

    Returns:
        True if migration was performed, False otherwise
    """
    config_path = _get_config_path()

    # Skip if JSON already exists
    if config_path.exists():
        return False

    try:
        # Import Qt only for migration
        from PySide6.QtCore import QSettings

        qsettings = QSettings()
        if not qsettings.allKeys():
            return False  # No QSettings to migrate

        logger.info("Migrating settings from QSettings to JSON...")

        settings = Settings()

        # Load paths
        if qsettings.contains("paths/thumbnail_cache_dir"):
            settings.thumbnail_cache_dir = Path(qsettings.value("paths/thumbnail_cache_dir"))
        if qsettings.contains("paths/download_dir"):
            settings.download_dir = Path(qsettings.value("paths/download_dir"))
        if qsettings.contains("paths/export_dir"):
            settings.export_dir = Path(qsettings.value("paths/export_dir"))

        # Load detection settings
        # Note: auto_analyze_colors, auto_classify_shots are deprecated - ignored if present
        if qsettings.contains("detection/default_sensitivity"):
            settings.default_sensitivity = float(qsettings.value("detection/default_sensitivity"))
        if qsettings.contains("detection/min_scene_length_seconds"):
            settings.min_scene_length_seconds = float(qsettings.value("detection/min_scene_length_seconds"))

        # Load export settings
        if qsettings.contains("export/quality"):
            settings.export_quality = qsettings.value("export/quality")
        if qsettings.contains("export/resolution"):
            settings.export_resolution = qsettings.value("export/resolution")
        if qsettings.contains("export/fps"):
            settings.export_fps = qsettings.value("export/fps")

        # Load transcription settings
        # Note: auto_transcribe is deprecated - ignored if present
        if qsettings.contains("transcription/model"):
            settings.transcription_model = qsettings.value("transcription/model")
        if qsettings.contains("transcription/language"):
            settings.transcription_language = qsettings.value("transcription/language")

        # Load appearance settings
        if qsettings.contains("appearance/theme_preference"):
            settings.theme_preference = qsettings.value("appearance/theme_preference")

        # Load YouTube settings (API key already in keyring)
        settings.youtube_api_key = _get_api_key_from_keyring()
        if qsettings.contains("youtube/results_count"):
            settings.youtube_results_count = int(qsettings.value("youtube/results_count"))
        if qsettings.contains("youtube/parallel_downloads"):
            settings.youtube_parallel_downloads = int(qsettings.value("youtube/parallel_downloads"))

        # Save to JSON
        save_settings(settings)

        logger.info("Migration complete. Settings saved to JSON.")
        return True

    except ImportError:
        # Qt not available (headless), skip migration
        return False
    except Exception as e:
        logger.warning(f"Migration from QSettings failed: {e}")
        return False


def validate_download_dir(path: Path) -> tuple[bool, str]:
    """Validate and attempt to create download directory.

    Tests that the directory exists (or can be created) and is writable.

    Args:
        path: Path to the download directory

    Returns:
        Tuple of (success, error_message). On success, error_message is empty.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Verify writable by creating and removing a test file
        test_file = path / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True, ""
    except PermissionError:
        return False, f"Permission denied: Cannot write to {path}"
    except OSError as e:
        return False, f"Cannot create directory: {e}"


def get_default_download_dir() -> Path:
    """Get the platform-appropriate default download directory.

    Returns:
        Path to the default download directory (~/Movies/Scene Ripper Downloads on macOS)
    """
    return _get_default_download_dir()


def is_download_dir_from_env() -> bool:
    """Check if the download directory setting comes from an environment variable.

    Returns:
        True if SCENE_RIPPER_DOWNLOAD_DIR environment variable is set
    """
    return bool(os.environ.get(ENV_DOWNLOAD_DIR))


def get_cache_size(cache_dir: Path) -> int:
    """Calculate total size of cache directory in bytes.

    Args:
        cache_dir: Path to cache directory

    Returns:
        Total size in bytes
    """
    if not cache_dir.exists():
        return 0

    total = 0
    try:
        for file in cache_dir.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
    except OSError:
        pass

    return total


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "245 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
