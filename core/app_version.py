"""Helpers for determining the running application version."""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
from pathlib import Path

from core.paths import get_resource_path, is_frozen

logger = logging.getLogger(__name__)

_DEFAULT_VERSION = "0.0.0"
_VERSION_RESOURCE = "core/app_version.txt"
_BUILD_VERSION_RESOURCE = "core/app_build_version.txt"
_UPDATE_CHANNEL_ENV = "APP_UPDATE_CHANNEL"


def get_app_version() -> str:
    """Return the best available application version string.

    Resolution order:
    1. `APP_VERSION` environment variable
    2. bundled `core/app_version.txt` resource
    3. macOS Info.plist bundle version
    4. nearest Git tag when running from source
    5. `0.0.0` fallback
    """
    for candidate in (
        _version_from_env(),
        _version_from_resource(),
        _version_from_macos_bundle(),
        _version_from_git(),
    ):
        if candidate:
            return candidate
    return _DEFAULT_VERSION


def get_display_version() -> str:
    """Return the human-facing version string."""
    return get_app_version().lstrip("v")


def get_machine_version() -> str:
    """Return the machine-ordered version used for updater comparisons."""
    if build_version := os.environ.get("APP_BUILD_VERSION", "").strip():
        return build_version

    try:
        path = get_resource_path(_BUILD_VERSION_RESOURCE)
    except Exception as exc:
        logger.debug("Could not resolve bundled build version resource: %s", exc)
        path = None

    if path and path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.debug("Could not read bundled build version resource %s: %s", path, exc)

    return get_display_version()


def get_release_channel() -> str:
    """Return the configured update channel for the running build."""
    return os.environ.get(_UPDATE_CHANNEL_ENV, "stable").strip() or "stable"


def _version_from_env() -> str:
    return os.environ.get("APP_VERSION", "").strip()


def _version_from_resource() -> str:
    try:
        path = get_resource_path(_VERSION_RESOURCE)
    except Exception as exc:
        logger.debug("Could not resolve bundled app version resource: %s", exc)
        return ""

    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.debug("Could not read bundled app version resource %s: %s", path, exc)
    return ""


def _version_from_macos_bundle() -> str:
    if not is_frozen():
        return ""

    executable_path = Path(os.environ.get("PYINSTALLER_EXE_PATH", "")).resolve() if os.environ.get("PYINSTALLER_EXE_PATH") else None
    if not executable_path:
        try:
            import sys
            executable_path = Path(sys.executable).resolve()
        except Exception:
            return ""

    info_plist = executable_path.parent.parent / "Info.plist"
    if not info_plist.exists():
        return ""

    try:
        with info_plist.open("rb") as fh:
            plist = plistlib.load(fh)
        return str(plist.get("CFBundleShortVersionString", "")).strip()
    except Exception as exc:
        logger.debug("Could not read macOS bundle version from %s: %s", info_plist, exc)
        return ""


def _version_from_git() -> str:
    if is_frozen():
        return ""

    project_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        logger.debug("Could not derive app version from git tags: %s", exc)
        return ""

    return result.stdout.strip()
