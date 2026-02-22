"""Centralized binary resolution for external tools (FFmpeg, yt-dlp, etc.).

Replaces scattered shutil.which() calls with a single lookup chain:
  1. Managed bin dir (~/Library/Application Support/Scene Ripper/bin/)
  2. Common Homebrew / user paths (for macOS GUI apps that lack shell PATH)
  3. Standard PATH via shutil.which()
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from core.paths import is_frozen, get_managed_bin_dir

logger = logging.getLogger(__name__)

# Paths that macOS GUI apps often miss because they don't inherit shell env
_EXTRA_SEARCH_PATHS = [
    "/opt/homebrew/bin",       # Homebrew on Apple Silicon
    "/opt/homebrew/sbin",
    "/usr/local/bin",          # Homebrew on Intel / manual installs
    "/usr/local/sbin",
]

_USER_SEARCH_PATHS = [
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / ".deno" / "bin"),
]


def find_binary(name: str) -> Optional[str]:
    """Find an external binary by name.

    Search order:
      1. Managed bin dir (when running as frozen app or always if dir exists)
      2. Extra search paths (Homebrew, user local)
      3. Standard PATH via shutil.which()

    Args:
        name: Binary name (e.g., "ffmpeg", "ffprobe", "yt-dlp")

    Returns:
        Absolute path to the binary, or None if not found.
    """
    # 1. Check managed bin directory
    managed_dir = get_managed_bin_dir()
    managed_path = managed_dir / name
    if managed_path.is_file() and os.access(managed_path, os.X_OK):
        logger.debug(f"Found {name} in managed bin dir: {managed_path}")
        return str(managed_path)

    # 2. Check extra search paths (especially important for macOS GUI apps)
    for search_dir in _EXTRA_SEARCH_PATHS + _USER_SEARCH_PATHS:
        candidate = os.path.join(search_dir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            logger.debug(f"Found {name} in extra path: {candidate}")
            return candidate

    # 3. Standard PATH lookup
    result = shutil.which(name)
    if result:
        logger.debug(f"Found {name} via PATH: {result}")
    else:
        logger.debug(f"{name} not found in any search location")
    return result


def get_subprocess_env() -> dict:
    """Get an environment dict with augmented PATH for subprocess calls.

    Ensures that managed binaries and common tool paths are findable
    by child processes (FFmpeg, yt-dlp, Deno, etc.).
    """
    env = os.environ.copy()
    path = env.get("PATH", "")
    path_parts = path.split(os.pathsep)

    # Prepend managed bin dir
    managed_bin = str(get_managed_bin_dir())
    if managed_bin not in path_parts:
        path_parts.insert(0, managed_bin)

    # Add common paths that GUI apps miss
    for p in _EXTRA_SEARCH_PATHS + _USER_SEARCH_PATHS:
        if p not in path_parts:
            path_parts.insert(1, p)

    env["PATH"] = os.pathsep.join(path_parts)
    return env
