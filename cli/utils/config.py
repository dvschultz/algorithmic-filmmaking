"""Qt-free configuration for the CLI.

DEPRECATED: This module is deprecated. Use core.settings instead.

This module now wraps core.settings for backward compatibility.
All new code should import from core.settings directly:

    from core.settings import load_settings, save_settings, Settings
"""

import warnings
from pathlib import Path
from typing import Optional

from core.settings import (
    load_settings,
    save_settings,
    Settings,
    _get_config_dir,
    _get_config_path,
    _get_cache_dir,
)

# Emit deprecation warning when module is imported
warnings.warn(
    "cli.utils.config is deprecated. Use core.settings instead.",
    DeprecationWarning,
    stacklevel=2,
)


def get_config_dir() -> Path:
    """Get config directory following XDG spec on Linux.

    DEPRECATED: Use core.settings._get_config_dir() instead.

    Returns:
        Path to the configuration directory
    """
    return _get_config_dir()


def get_config_path() -> Path:
    """Get path to the configuration file.

    DEPRECATED: Use core.settings._get_config_path() instead.

    Returns:
        Path to config.json
    """
    return _get_config_path()


def get_cache_dir() -> Path:
    """Get cache directory following XDG spec.

    DEPRECATED: Use core.settings._get_cache_dir() instead.

    Returns:
        Path to the cache directory
    """
    return _get_cache_dir()


class CLIConfig:
    """CLI configuration, independent of Qt.

    DEPRECATED: This class is deprecated. Use core.settings.Settings instead.

    This class wraps core.settings.Settings for backward compatibility.
    Settings are loaded with priority: env vars > config file > defaults.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize from a Settings object.

        Args:
            settings: Settings instance (uses load_settings() if None)
        """
        self._settings = settings or load_settings()

    @property
    def download_dir(self) -> Optional[Path]:
        return self._settings.download_dir

    @download_dir.setter
    def download_dir(self, value: Optional[Path]):
        self._settings.download_dir = value

    @property
    def export_dir(self) -> Optional[Path]:
        return self._settings.export_dir

    @export_dir.setter
    def export_dir(self, value: Optional[Path]):
        self._settings.export_dir = value

    @property
    def cache_dir(self) -> Optional[Path]:
        return self._settings.thumbnail_cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Optional[Path]):
        self._settings.thumbnail_cache_dir = value

    @property
    def default_sensitivity(self) -> float:
        return self._settings.default_sensitivity

    @default_sensitivity.setter
    def default_sensitivity(self, value: float):
        self._settings.default_sensitivity = value

    @property
    def min_scene_length_seconds(self) -> float:
        return self._settings.min_scene_length_seconds

    @min_scene_length_seconds.setter
    def min_scene_length_seconds(self, value: float):
        self._settings.min_scene_length_seconds = value

    @property
    def transcription_model(self) -> str:
        return self._settings.transcription_model

    @transcription_model.setter
    def transcription_model(self, value: str):
        self._settings.transcription_model = value

    @property
    def transcription_language(self) -> str:
        return self._settings.transcription_language

    @transcription_language.setter
    def transcription_language(self, value: str):
        self._settings.transcription_language = value

    @property
    def export_quality(self) -> str:
        return self._settings.export_quality

    @export_quality.setter
    def export_quality(self, value: str):
        self._settings.export_quality = value

    @property
    def export_resolution(self) -> str:
        return self._settings.export_resolution

    @export_resolution.setter
    def export_resolution(self, value: str):
        self._settings.export_resolution = value

    @property
    def youtube_api_key(self) -> Optional[str]:
        return self._settings.youtube_api_key or None

    @youtube_api_key.setter
    def youtube_api_key(self, value: Optional[str]):
        self._settings.youtube_api_key = value or ""

    @property
    def youtube_results_count(self) -> int:
        return self._settings.youtube_results_count

    @youtube_results_count.setter
    def youtube_results_count(self, value: int):
        self._settings.youtube_results_count = value

    @classmethod
    def load(cls) -> "CLIConfig":
        """Load config with priority: env vars > config file > defaults.

        DEPRECATED: Use core.settings.load_settings() instead.

        Returns:
            CLIConfig instance with settings loaded
        """
        return cls(load_settings())

    def save(self) -> bool:
        """Save current config to file.

        DEPRECATED: Use core.settings.save_settings() instead.

        Returns:
            True if save succeeded
        """
        return save_settings(self._settings)
