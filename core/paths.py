"""Centralized path resolution for frozen (PyInstaller) and source modes.

When running as a PyInstaller .app bundle, sys.frozen is set and
sys._MEIPASS points to the bundled resources. All path resolution
for managed binaries, on-demand packages, and app data should go
through this module.
"""

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """Return True if running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_base_path() -> Path:
    """Return the base path for bundled resources.

    When frozen: Contents/Frameworks/ inside the .app bundle.
    When running from source: the project root directory.
    """
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def get_app_support_dir() -> Path:
    """Return the Application Support directory for Scene Ripper.

    On macOS: ~/Library/Application Support/Scene Ripper/
    On Windows: %LOCALAPPDATA%/Scene Ripper/
    On Linux: ~/.local/share/scene-ripper/
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Scene Ripper"
    if sys.platform == "win32":
        return Path(os.environ.get(
            "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
        )) / "Scene Ripper"
    return Path.home() / ".local" / "share" / "scene-ripper"


def get_managed_bin_dir() -> Path:
    """Return the directory for downloaded binaries (ffmpeg, yt-dlp, etc.)."""
    return get_app_support_dir() / "bin"


def get_managed_packages_dir() -> Path:
    """Return the directory for on-demand Python packages."""
    return get_app_support_dir() / "packages"


def get_managed_python_dir() -> Path:
    """Return the directory for the standalone Python used by pip."""
    return get_app_support_dir() / "python"


def get_resource_path(relative_path: str) -> Path:
    """Get the absolute path to a bundled resource file.

    Works in both frozen and source modes.
    """
    return get_base_path() / relative_path


def get_log_dir() -> Path:
    """Return the directory for application log files.

    On macOS: ~/Library/Logs/Scene Ripper/
    On Windows/Linux: <app_support>/logs/
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "Scene Ripper"
    return get_app_support_dir() / "logs"


def ensure_app_dirs() -> None:
    """Create essential application directories if they don't exist.

    Called at startup when running in frozen mode.
    """
    for dir_path in [
        get_app_support_dir(),
        get_managed_bin_dir(),
        get_log_dir(),
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
