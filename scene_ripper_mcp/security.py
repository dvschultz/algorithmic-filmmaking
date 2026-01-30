"""Security utilities for MCP server."""

import tempfile
from pathlib import Path
from typing import Tuple

# Safe root directories (reuse pattern from chat_tools.py)
SAFE_ROOTS = [
    Path.home(),
    Path("/tmp").resolve(),  # Resolve symlinks (macOS: /tmp -> /private/tmp)
    Path(tempfile.gettempdir()).resolve(),
]

# On macOS, also allow /var/folders (temp), /Volumes (external drives), /private/tmp
import sys

if sys.platform == "darwin":
    SAFE_ROOTS.extend(
        [
            Path("/var/folders"),
            Path("/Volumes"),
            Path("/private/tmp"),
        ]
    )


def _is_path_under(path: Path, root: Path) -> bool:
    """Check if path is under root directory."""
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
    """Validate a path for security and existence.

    Args:
        path_str: Path string to validate
        must_exist: Require path to exist
        must_be_file: Require path to be a file
        must_be_dir: Require path to be a directory

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    if not path_str:
        return False, "Path cannot be empty", Path()

    try:
        path = Path(path_str).expanduser().resolve()
    except Exception as e:
        return False, f"Invalid path: {e}", Path()

    # Check for path traversal (in original string, before resolution)
    if ".." in path_str:
        return False, "Path traversal not allowed", path

    # Check if path is absolute (required for MCP tools)
    if not Path(path_str).is_absolute() and not path_str.startswith("~"):
        return False, f"Only absolute paths are allowed: {path_str}", path

    # Verify path is under safe root
    is_safe = any(_is_path_under(path, root.resolve()) for root in SAFE_ROOTS)
    if not is_safe:
        return False, f"Path must be under home directory or temp: {path}", path

    # Existence checks
    if must_exist and not path.exists():
        return False, f"Path does not exist: {path}", path

    if must_be_file and path.exists() and not path.is_file():
        return False, f"Path is not a file: {path}", path

    if must_be_dir and path.exists() and not path.is_dir():
        return False, f"Path is not a directory: {path}", path

    return True, "", path


def validate_video_path(path_str: str) -> Tuple[bool, str, Path]:
    """Validate a video file path.

    Args:
        path_str: Path string to validate

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    valid, error, path = validate_path(path_str, must_exist=True, must_be_file=True)
    if not valid:
        return valid, error, path

    # Check for common video extensions
    video_extensions = {
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
    if path.suffix.lower() not in video_extensions:
        return (
            False,
            f"File does not appear to be a video: {path.suffix}",
            path,
        )

    return True, "", path


def validate_project_path(path_str: str, must_exist: bool = True) -> Tuple[bool, str, Path]:
    """Validate a project file path.

    Args:
        path_str: Path string to validate
        must_exist: Whether the project file must exist

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    if not path_str.endswith(".sceneripper"):
        return False, "Project path must end with .sceneripper", Path()

    return validate_path(path_str, must_exist=must_exist, must_be_file=must_exist)
