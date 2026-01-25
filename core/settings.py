"""Application settings management using QSettings."""

import logging
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings

logger = logging.getLogger(__name__)

# Keyring service name for secure credential storage
KEYRING_SERVICE = "com.scene-ripper.app"
KEYRING_YOUTUBE_KEY = "youtube_api_key"


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
        default_factory=lambda: Path.home() / ".cache" / "scene-ripper" / "thumbnails"
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


def load_settings() -> Settings:
    """
    Load settings from QSettings storage.

    Returns:
        Settings instance populated from stored values or defaults
    """
    qsettings = QSettings()
    settings = Settings()

    try:
        # Load paths
        if qsettings.contains("paths/thumbnail_cache_dir"):
            settings.thumbnail_cache_dir = Path(
                qsettings.value("paths/thumbnail_cache_dir")
            )
        if qsettings.contains("paths/download_dir"):
            settings.download_dir = Path(qsettings.value("paths/download_dir"))
        if qsettings.contains("paths/export_dir"):
            settings.export_dir = Path(qsettings.value("paths/export_dir"))

        # Load detection settings
        if qsettings.contains("detection/default_sensitivity"):
            settings.default_sensitivity = float(
                qsettings.value("detection/default_sensitivity")
            )
        if qsettings.contains("detection/min_scene_length_seconds"):
            settings.min_scene_length_seconds = float(
                qsettings.value("detection/min_scene_length_seconds")
            )
        if qsettings.contains("detection/auto_analyze_colors"):
            settings.auto_analyze_colors = (
                qsettings.value("detection/auto_analyze_colors") == "true"
            )
        if qsettings.contains("detection/auto_classify_shots"):
            settings.auto_classify_shots = (
                qsettings.value("detection/auto_classify_shots") == "true"
            )

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
            settings.auto_transcribe = (
                qsettings.value("transcription/auto_transcribe") == "true"
            )

        # Load appearance settings
        if qsettings.contains("appearance/theme_preference"):
            settings.theme_preference = qsettings.value("appearance/theme_preference")

        # Load YouTube settings
        # API key is stored securely in keyring, not QSettings
        settings.youtube_api_key = _get_api_key_from_keyring()
        # Migrate from old QSettings storage if present
        if not settings.youtube_api_key and qsettings.contains("youtube/api_key"):
            old_key = qsettings.value("youtube/api_key")
            if old_key:
                settings.youtube_api_key = old_key
                _set_api_key_in_keyring(old_key)
                qsettings.remove("youtube/api_key")  # Remove from insecure storage
                logger.info("Migrated YouTube API key to secure keyring storage")
        if qsettings.contains("youtube/results_count"):
            settings.youtube_results_count = int(
                qsettings.value("youtube/results_count")
            )
        if qsettings.contains("youtube/parallel_downloads"):
            settings.youtube_parallel_downloads = int(
                qsettings.value("youtube/parallel_downloads")
            )

        logger.info("Settings loaded successfully")

    except Exception as e:
        logger.warning(f"Error loading settings, using defaults: {e}")
        settings = Settings()

    return settings


def save_settings(settings: Settings) -> bool:
    """
    Save settings to QSettings storage.

    Args:
        settings: Settings instance to save

    Returns:
        True if save succeeded
    """
    try:
        qsettings = QSettings()

        # Save paths
        qsettings.setValue("paths/thumbnail_cache_dir", str(settings.thumbnail_cache_dir))
        qsettings.setValue("paths/download_dir", str(settings.download_dir))
        qsettings.setValue("paths/export_dir", str(settings.export_dir))

        # Save detection settings
        qsettings.setValue("detection/default_sensitivity", settings.default_sensitivity)
        qsettings.setValue(
            "detection/min_scene_length_seconds", settings.min_scene_length_seconds
        )
        qsettings.setValue(
            "detection/auto_analyze_colors",
            "true" if settings.auto_analyze_colors else "false",
        )
        qsettings.setValue(
            "detection/auto_classify_shots",
            "true" if settings.auto_classify_shots else "false",
        )

        # Save export settings
        qsettings.setValue("export/quality", settings.export_quality)
        qsettings.setValue("export/resolution", settings.export_resolution)
        qsettings.setValue("export/fps", settings.export_fps)

        # Save transcription settings
        qsettings.setValue("transcription/model", settings.transcription_model)
        qsettings.setValue("transcription/language", settings.transcription_language)
        qsettings.setValue(
            "transcription/auto_transcribe",
            "true" if settings.auto_transcribe else "false",
        )

        # Save appearance settings
        qsettings.setValue("appearance/theme_preference", settings.theme_preference)

        # Save YouTube settings
        # API key goes to secure keyring, not QSettings
        _set_api_key_in_keyring(settings.youtube_api_key)
        qsettings.setValue("youtube/results_count", settings.youtube_results_count)
        qsettings.setValue("youtube/parallel_downloads", settings.youtube_parallel_downloads)

        qsettings.sync()
        logger.info("Settings saved successfully")
        return True

    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False


def get_cache_size(cache_dir: Path) -> int:
    """
    Calculate total size of cache directory in bytes.

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
    """
    Format byte size as human-readable string.

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
