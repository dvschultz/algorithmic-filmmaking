"""Qt-free configuration for the CLI."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def get_config_dir() -> Path:
    """Get config directory following XDG spec on Linux.

    Returns:
        Path to the configuration directory
    """
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "scene-ripper"


def get_config_path() -> Path:
    """Get path to the configuration file.

    Returns:
        Path to config.json
    """
    return get_config_dir() / "config.json"


def get_cache_dir() -> Path:
    """Get cache directory following XDG spec.

    Returns:
        Path to the cache directory
    """
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:  # macOS/Linux
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "scene-ripper"


@dataclass
class CLIConfig:
    """CLI configuration, independent of Qt.

    Settings are loaded with priority: env vars > config file > defaults.
    """

    # Paths
    download_dir: Optional[Path] = None
    export_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    # Detection defaults
    default_sensitivity: float = 3.0
    min_scene_length_seconds: float = 0.5

    # Transcription defaults
    transcription_model: str = "small.en"
    transcription_language: str = "en"

    # Export defaults
    export_quality: str = "medium"  # high, medium, low
    export_resolution: str = "original"  # original, 1080p, 720p, 480p

    # YouTube
    youtube_api_key: Optional[str] = None
    youtube_results_count: int = 25

    @classmethod
    def load(cls) -> "CLIConfig":
        """Load config with priority: env vars > config file > defaults.

        Returns:
            CLIConfig instance with settings loaded
        """
        config = cls()

        # Set default paths
        config.cache_dir = get_cache_dir()
        config.download_dir = Path.home() / "Movies" / "Scene Ripper Downloads"
        config.export_dir = Path.home() / "Movies" / "Scene Ripper Exports"

        # Load from config file if exists
        config_path = get_config_path()
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(config, key):
                            # Convert path strings to Path objects
                            if key.endswith("_dir") and value is not None:
                                value = Path(value)
                            setattr(config, key, value)
            except (json.JSONDecodeError, OSError):
                pass  # Use defaults if config file is invalid

        # Override with environment variables (highest priority)
        if api_key := os.environ.get("YOUTUBE_API_KEY"):
            config.youtube_api_key = api_key
        if cache_dir := os.environ.get("SCENE_RIPPER_CACHE_DIR"):
            config.cache_dir = Path(cache_dir)
        if download_dir := os.environ.get("SCENE_RIPPER_DOWNLOAD_DIR"):
            config.download_dir = Path(download_dir)
        if export_dir := os.environ.get("SCENE_RIPPER_EXPORT_DIR"):
            config.export_dir = Path(export_dir)
        if sensitivity := os.environ.get("SCENE_RIPPER_SENSITIVITY"):
            try:
                config.default_sensitivity = float(sensitivity)
            except ValueError:
                pass
        if model := os.environ.get("SCENE_RIPPER_WHISPER_MODEL"):
            config.transcription_model = model

        return config

    def save(self) -> bool:
        """Save current config to file.

        Returns:
            True if save succeeded
        """
        config_path = get_config_path()
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "download_dir": str(self.download_dir) if self.download_dir else None,
                "export_dir": str(self.export_dir) if self.export_dir else None,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "default_sensitivity": self.default_sensitivity,
                "min_scene_length_seconds": self.min_scene_length_seconds,
                "transcription_model": self.transcription_model,
                "transcription_language": self.transcription_language,
                "export_quality": self.export_quality,
                "export_resolution": self.export_resolution,
                "youtube_api_key": self.youtube_api_key,
                "youtube_results_count": self.youtube_results_count,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except OSError:
            return False
