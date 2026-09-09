"""Centralized binary resolution for external tools (FFmpeg, yt-dlp, etc.).

Replaces scattered shutil.which() calls with a single lookup chain. Frozen apps
prefer their bundled runtime binaries first so old managed downloads cannot
shadow the signed FFmpeg/FFprobe shipped with the app.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from core.paths import get_base_path, get_bundled_bin_dir, get_managed_bin_dir, is_frozen

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

# Windows-specific search paths for common install locations
if sys.platform == "win32":
    _EXTRA_SEARCH_PATHS.extend([
        r"C:\Program Files\FFmpeg\bin",
        r"C:\ffmpeg\bin",
        str(Path.home() / "scoop" / "shims"),
    ])

# Linux-specific search paths (Snap packages, explicit /usr/bin for desktop-launched apps)
if sys.platform == "linux":
    _EXTRA_SEARCH_PATHS.extend([
        "/snap/bin",
        "/usr/bin",
    ])


def find_binary(name: str) -> Optional[str]:
    """Find an external binary by name.

    Search order:
      1. Bundled app runtime directories in frozen builds
      2. Managed bin dir
      3. Extra search paths (Homebrew, user local)
      4. Standard PATH via shutil.which()

    Args:
        name: Binary name (e.g., "ffmpeg", "ffprobe", "yt-dlp")

    Returns:
        Absolute path to the binary, or None if not found.
    """
    # On Windows, check both the bare name and with .exe suffix
    suffixes = [".exe", ""] if sys.platform == "win32" else [""]

    # 0. An explicit hand-off from the host (isolated workers get the host's
    #    resolved ffmpeg/ffprobe, bundled or managed, via the environment).
    explicit = os.environ.get(f"SCENE_RIPPER_{name.upper().replace('-', '_')}")
    if explicit and Path(explicit).is_file():
        return str(Path(explicit))

    # 1. Check bundled runtime directories in frozen apps
    if is_frozen():
        base_path = get_base_path()
        executable_dir = Path(sys.executable).resolve().parent
        bundled_dirs = [
            get_bundled_bin_dir(),
            executable_dir / "bin",
            base_path,
            executable_dir,
        ]
        seen_dirs: set[Path] = set()
        for search_dir in bundled_dirs:
            if search_dir in seen_dirs:
                continue
            seen_dirs.add(search_dir)
            for suffix in suffixes:
                candidate = search_dir / (name + suffix)
                if candidate.is_file():
                    logger.debug(f"Found {name} in bundled runtime: {candidate}")
                    return str(candidate)

    # 2. Check managed bin directory
    managed_dir = get_managed_bin_dir()
    for suffix in suffixes:
        managed_path = managed_dir / (name + suffix)
        if managed_path.is_file():
            logger.debug(f"Found {name} in managed bin dir: {managed_path}")
            return str(managed_path)

    # 3. Check extra search paths (especially important for macOS GUI apps)
    for search_dir in _EXTRA_SEARCH_PATHS + _USER_SEARCH_PATHS:
        for suffix in suffixes:
            candidate = os.path.join(search_dir, name + suffix)
            if os.path.isfile(candidate):
                logger.debug(f"Found {name} in extra path: {candidate}")
                return candidate

    # 4. Standard PATH lookup (shutil.which handles .exe on Windows natively)
    result = shutil.which(name)
    if result:
        logger.debug(f"Found {name} via PATH: {result}")
    else:
        logger.debug(f"{name} not found in any search location")
    return result


def is_bundled_binary_path(path: str | os.PathLike[str] | None) -> bool:
    """Return True when a binary path points into the frozen app bundle."""
    if not path or not is_frozen():
        return False

    try:
        candidate = Path(path).resolve(strict=False)
    except OSError:
        return False

    executable_dir = Path(sys.executable).resolve().parent
    candidate_roots = [
        get_bundled_bin_dir(),
        executable_dir / "bin",
        get_base_path(),
        executable_dir,
    ]
    seen_roots: set[Path] = set()
    for root in candidate_roots:
        try:
            resolved_root = root.resolve(strict=False)
        except OSError:
            continue
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        if candidate == resolved_root or candidate.is_relative_to(resolved_root):
            return True
    return False


def get_subprocess_kwargs() -> dict:
    """Return platform-appropriate kwargs for subprocess calls.

    On Windows, adds CREATE_NO_WINDOW flag to prevent console windows
    from flashing behind the GUI during subprocess operations.
    """
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000 — only defined on Windows
        create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return {"creationflags": create_no_window}
    return {}


def get_subprocess_env() -> dict:
    """Get an environment dict with augmented PATH for subprocess calls.

    Ensures that managed binaries and common tool paths are findable
    by child processes (FFmpeg, yt-dlp, Deno, etc.).
    """
    env = os.environ.copy()
    path = env.get("PATH", "")
    path_parts = path.split(os.pathsep)

    extra_insert_index = 0
    if is_frozen():
        bundled_bins = [
            str(get_bundled_bin_dir()),
            str(Path(sys.executable).resolve().parent / "bin"),
        ]
        for bundled_bin in bundled_bins:
            if bundled_bin not in path_parts:
                path_parts.insert(extra_insert_index, bundled_bin)
                extra_insert_index += 1

    managed_bin = str(get_managed_bin_dir())
    if managed_bin not in path_parts:
        path_parts.insert(extra_insert_index, managed_bin)
        extra_insert_index += 1

    # Add common paths that GUI apps miss
    for p in _EXTRA_SEARCH_PATHS + _USER_SEARCH_PATHS:
        if p not in path_parts:
            path_parts.insert(extra_insert_index, p)
            extra_insert_index += 1

    env["PATH"] = os.pathsep.join(path_parts)
    return env


def log_resolved_binary(name: str) -> str | None:
    """Log the resolved binary path and first version line for diagnostics."""
    path = find_binary(name)
    if path is None:
        logger.warning("%s binary not resolved", name)
        return None

    version = "version unavailable"
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            **get_subprocess_kwargs(),
        )
        output_lines = (result.stdout or result.stderr or "").splitlines()
        if output_lines:
            version = output_lines[0].strip() or version
    except Exception as exc:
        version = f"version check failed: {exc}"

    location = "bundled" if is_bundled_binary_path(path) else "external"
    logger.info("Resolved %s binary: path=%s location=%s %s", name, path, location, version)
    return path
