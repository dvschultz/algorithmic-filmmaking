"""Path validation — single source of truth for both GUI and MCP surfaces.

This module replaces the duplicated implementations previously in
``core/chat_tools.py:_validate_path`` and ``scene_ripper_mcp/security.py``.
Both consumers now import from here.

The signature follows the stricter MCP convention:
- absolute paths only (``~`` is expanded; relative paths rejected)
- raw ``..`` rejected before any path normalization
- safe-roots whitelist (home, temp, /Volumes on macOS, drive roots on Windows)
- optional ``must_exist`` / ``must_be_file`` / ``must_be_dir`` checks
"""

from __future__ import annotations

import string
import sys
import tempfile
from pathlib import Path
from typing import Tuple

# Safe root directories. The GUI and MCP surfaces use the same whitelist.
SAFE_ROOTS: list[Path] = [
    Path.home(),
    Path("/tmp").resolve(),  # macOS: /tmp -> /private/tmp
    Path(tempfile.gettempdir()).resolve(),
]

if sys.platform == "darwin":
    SAFE_ROOTS.extend(
        [
            Path("/var/folders"),
            Path("/Volumes"),
            Path("/private/tmp"),
        ]
    )
elif sys.platform == "win32":
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:\\")
        if drive.exists():
            SAFE_ROOTS.append(drive)


def _is_path_under(path: Path, root: Path) -> bool:
    """Check if ``path`` is contained within ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_path(
    path_str: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Tuple[bool, str, Path]:
    """Validate a filesystem path.

    Returns ``(is_valid, error_message, resolved_path)``. The resolved path is
    ``Path()`` (empty) when validation fails before resolution succeeds, so
    callers should always check ``is_valid`` first.

    Path traversal (``..``) is rejected by inspecting the raw input string —
    ``Path(...).resolve()`` would normalize the traversal away and silently
    accept it.
    """
    if not path_str:
        return False, "Path cannot be empty", Path()

    if ".." in path_str:
        return False, "Path traversal not allowed", Path()

    raw = Path(path_str)
    if not raw.is_absolute() and not path_str.startswith("~"):
        return False, f"Only absolute paths are allowed: {path_str}", Path()

    try:
        path = raw.expanduser().resolve()
    except OSError as exc:
        return False, f"Invalid path: {exc}", Path()

    is_safe = any(_is_path_under(path, root.resolve()) for root in SAFE_ROOTS)
    if not is_safe:
        return False, f"Path must be under home directory or temp: {path}", path

    if must_exist and not path.exists():
        return False, f"Path does not exist: {path}", path

    if must_be_file and path.exists() and not path.is_file():
        return False, f"Path is not a file: {path}", path

    if must_be_dir and path.exists() and not path.is_dir():
        return False, f"Path is not a directory: {path}", path

    return True, "", path


# Common video file extensions used by ``validate_video_path``.
_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".wmv",
        ".flv",
        ".ts",
        ".mts",
    }
)


def validate_video_path(path_str: str) -> Tuple[bool, str, Path]:
    """Validate a video file path. Requires the file to exist."""
    valid, error, path = validate_path(path_str, must_exist=True, must_be_file=True)
    if not valid:
        return valid, error, path

    if path.suffix.lower() not in _VIDEO_EXTENSIONS:
        return (
            False,
            f"File does not appear to be a video: {path.suffix}",
            path,
        )

    return True, "", path


def validate_project_path(path_str: str, must_exist: bool = True) -> Tuple[bool, str, Path]:
    """Validate a ``.sceneripper`` project file path."""
    if not path_str.endswith(".sceneripper"):
        return False, "Project path must end with .sceneripper", Path()

    return validate_path(path_str, must_exist=must_exist, must_be_file=must_exist)
