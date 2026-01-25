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
    download_dir: Path = field(default_factory=_get_default_download_dir)
    export_dir: Path = field(default_factory=_get_default_export_dir)

    # Detection defaults
    default_sensitivity: float = 3.0
    min_scene_length_seconds: float = 0.5
    auto_analyze_colors: bool = True
    auto_classify_shots: bool = True

    # Export defaults
    export_quality: str = "medium"  # high, medium, low
    export_resolution: str = "original"  # original, 1080p, 720p, 480p
    export_fps: str = "original"  # original, 24, 30, 60

    # Transcription settings
    transcription_model: str = "small.en"  # tiny.en, small.en, medium.en, large-v3
    transcription_language: str = "en"  # en, auto, or specific language code
    auto_transcribe: bool = True  # Auto-transcribe on detection

    # Appearance
    theme_preference: str = "system"  # system, light, dark

    # YouTube API
    youtube_api_key: str = ""
    youtube_results_count: int = 25  # 10-50
    youtube_parallel_downloads: int = 2  # 1-3

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
        if val := paths.get("download_dir"):
            settings.download_dir = Path(val).expanduser()
        if val := paths.get("export_dir"):
            settings.export_dir = Path(val).expanduser()

    # Detection section
    if detection := data.get("detection"):
        if "default_sensitivity" in detection:
            settings.default_sensitivity = float(detection["default_sensitivity"])
        if "min_scene_length_seconds" in detection:
            settings.min_scene_length_seconds = float(detection["min_scene_length_seconds"])
        if "auto_analyze_colors" in detection:
            settings.auto_analyze_colors = bool(detection["auto_analyze_colors"])
        if "auto_classify_shots" in detection:
            settings.auto_classify_shots = bool(detection["auto_classify_shots"])

    # Transcription section
    if transcription := data.get("transcription"):
        if val := transcription.get("model"):
            settings.transcription_model = val
        if val := transcription.get("language"):
            settings.transcription_language = val
        if "auto_transcribe" in transcription:
            settings.auto_transcribe = bool(transcription["auto_transcribe"])

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
            "download_dir": str(settings.download_dir),
            "export_dir": str(settings.export_dir),
        },
        "detection": {
            "default_sensitivity": settings.default_sensitivity,
            "min_scene_length_seconds": settings.min_scene_length_seconds,
            "auto_analyze_colors": settings.auto_analyze_colors,
            "auto_classify_shots": settings.auto_classify_shots,
        },
        "transcription": {
            "model": settings.transcription_model,
            "language": settings.transcription_language,
            "auto_transcribe": settings.auto_transcribe,
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

    # 2. Load API key from keyring (if not already set from JSON, which doesn't store it)
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
        if qsettings.contains("detection/default_sensitivity"):
            settings.default_sensitivity = float(qsettings.value("detection/default_sensitivity"))
        if qsettings.contains("detection/min_scene_length_seconds"):
            settings.min_scene_length_seconds = float(qsettings.value("detection/min_scene_length_seconds"))
        if qsettings.contains("detection/auto_analyze_colors"):
            settings.auto_analyze_colors = qsettings.value("detection/auto_analyze_colors") == "true"
        if qsettings.contains("detection/auto_classify_shots"):
            settings.auto_classify_shots = qsettings.value("detection/auto_classify_shots") == "true"

        # Load export settings
        if qsettings.contains("export/quality"):
            settings.export_quality = qsettings.value("export/quality")
        if qsettings.contains("export/resolution"):
            settings.export_resolution = qsettings.value("export/resolution")
        if qsettings.contains("export/fps"):
            settings.export_fps = qsettings.value("export/fps")

        # Load transcription settings
        if qsettings.contains("transcription/model"):
            settings.transcription_model = qsettings.value("transcription/model")
        if qsettings.contains("transcription/language"):
            settings.transcription_language = qsettings.value("transcription/language")
        if qsettings.contains("transcription/auto_transcribe"):
            settings.auto_transcribe = qsettings.value("transcription/auto_transcribe") == "true"

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
