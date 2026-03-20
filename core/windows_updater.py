"""Windows native updater detection and WinSparkle integration."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from core.app_version import get_display_version, get_machine_version
from core.paths import get_base_path, get_resource_path, is_frozen
from core.update_models import UpdateCapability

WINSPARKLE_DLL_ENV = "SCENE_RIPPER_WINSPARKLE_DLL"
WINSPARKLE_FEED_ENV = "WINSPARKLE_APPCAST_URL"
WINSPARKLE_BETA_FEED_ENV = "WINSPARKLE_APPCAST_BETA_URL"
WINSPARKLE_PUBLIC_KEY_ENV = "WINSPARKLE_PUBLIC_ED_KEY"
WINSPARKLE_FEED_RESOURCE = "core/app_update_feed_url.txt"
WINSPARKLE_BETA_FEED_RESOURCE = "core/app_update_feed_url_beta.txt"
WINSPARKLE_PUBLIC_KEY_RESOURCE = "core/app_update_public_key.txt"
WINSPARKLE_REGISTRY_PATH = r"Software\Algorithmic Filmmaking\Scene Ripper\Updates"
DEFAULT_UPDATE_INTERVAL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class WindowsUpdaterStatus:
    """Status for the Windows native updater integration."""

    available: bool
    capability: UpdateCapability
    reason: str = ""
    dll_path: Path | None = None
    feed_url: str = ""
    public_key: str = ""
    initialized: bool = False
    launched: bool = False
    error_message: str = ""


_winsparkle_library = None
_winsparkle_initialized = False
_configured_feed_url = ""
_configured_channel = ""
_configured_auto_check = False


def _read_resource_text(resource_name: str) -> str:
    """Read a bundled updater metadata file if present."""
    try:
        path = get_resource_path(resource_name)
    except Exception:
        return ""

    if not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def get_feed_url(update_channel: str = "stable") -> str:
    """Return the configured WinSparkle appcast URL for a channel."""
    normalized_channel = (update_channel or "stable").strip().lower()
    if normalized_channel == "beta":
        return (
            os.environ.get(WINSPARKLE_BETA_FEED_ENV, "").strip()
            or _read_resource_text(WINSPARKLE_BETA_FEED_RESOURCE)
        )

    return (
        os.environ.get(WINSPARKLE_FEED_ENV, "").strip()
        or _read_resource_text(WINSPARKLE_FEED_RESOURCE)
    )


def get_public_ed_key() -> str:
    """Return the bundled WinSparkle EdDSA public key."""
    return (
        os.environ.get(WINSPARKLE_PUBLIC_KEY_ENV, "").strip()
        or _read_resource_text(WINSPARKLE_PUBLIC_KEY_RESOURCE)
    )


def find_winsparkle_dll(base_path: Path | None = None) -> Path | None:
    """Find a bundled WinSparkle DLL."""
    if env_path := os.environ.get(WINSPARKLE_DLL_ENV, "").strip():
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    base = base_path or get_base_path()
    executable_dir = Path(sys.executable).resolve().parent
    candidates = [
        base / "WinSparkle.dll",
        base / "winsparkle.dll",
        executable_dir / "WinSparkle.dll",
        executable_dir / "winsparkle.dll",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def get_status(
    *,
    update_channel: str = "stable",
    base_path: Path | None = None,
) -> WindowsUpdaterStatus:
    """Return native updater availability for the current build."""
    if sys.platform != "win32" or not is_frozen():
        return WindowsUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Native Windows updates are only available in installed Windows builds.",
        )

    feed_url = get_feed_url(update_channel)
    public_key = get_public_ed_key()
    dll_path = find_winsparkle_dll(base_path)

    if update_channel and update_channel != "stable" and not feed_url:
        return WindowsUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason=f"No WinSparkle feed is configured for the {update_channel} channel.",
            dll_path=dll_path,
            public_key=public_key,
        )

    if not feed_url or not public_key:
        return WindowsUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Native updater metadata is not configured in this build.",
            dll_path=dll_path,
            feed_url=feed_url,
            public_key=public_key,
        )

    if dll_path is None:
        return WindowsUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="WinSparkle is not bundled with this app yet.",
            feed_url=feed_url,
            public_key=public_key,
        )

    return WindowsUpdaterStatus(
        available=True,
        capability=UpdateCapability.NATIVE_CHECK,
        dll_path=dll_path,
        feed_url=feed_url,
        public_key=public_key,
        initialized=_winsparkle_initialized,
    )


def _load_library(dll_path: Path):
    """Load and annotate the WinSparkle DLL."""
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise OSError("WinSparkle loading is only supported on Windows.")

    library = loader(str(dll_path))
    library.win_sparkle_set_appcast_url.argtypes = [ctypes.c_char_p]
    library.win_sparkle_set_appcast_url.restype = None
    library.win_sparkle_set_eddsa_public_key.argtypes = [ctypes.c_char_p]
    library.win_sparkle_set_eddsa_public_key.restype = ctypes.c_int
    library.win_sparkle_set_app_details.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
    ]
    library.win_sparkle_set_app_details.restype = None
    library.win_sparkle_set_app_build_version.argtypes = [ctypes.c_wchar_p]
    library.win_sparkle_set_app_build_version.restype = None
    library.win_sparkle_set_registry_path.argtypes = [ctypes.c_char_p]
    library.win_sparkle_set_registry_path.restype = None
    library.win_sparkle_set_automatic_check_for_updates.argtypes = [ctypes.c_int]
    library.win_sparkle_set_automatic_check_for_updates.restype = None
    library.win_sparkle_set_update_check_interval.argtypes = [ctypes.c_int]
    library.win_sparkle_set_update_check_interval.restype = None
    library.win_sparkle_init.argtypes = []
    library.win_sparkle_init.restype = None
    library.win_sparkle_cleanup.argtypes = []
    library.win_sparkle_cleanup.restype = None
    library.win_sparkle_check_update_with_ui.argtypes = []
    library.win_sparkle_check_update_with_ui.restype = None
    return library


def _configure_library(library, *, feed_url: str, public_key: str, automatically_check: bool) -> None:
    """Configure WinSparkle before initialization."""
    library.win_sparkle_set_appcast_url(feed_url.encode("utf-8"))
    if library.win_sparkle_set_eddsa_public_key(public_key.encode("utf-8")) != 1:
        raise OSError("Invalid WinSparkle EdDSA public key.")

    library.win_sparkle_set_app_details(
        "Algorithmic Filmmaking",
        "Scene Ripper",
        get_display_version(),
    )
    library.win_sparkle_set_app_build_version(get_machine_version())
    library.win_sparkle_set_registry_path(WINSPARKLE_REGISTRY_PATH.encode("utf-8"))
    library.win_sparkle_set_automatic_check_for_updates(1 if automatically_check else 0)
    library.win_sparkle_set_update_check_interval(DEFAULT_UPDATE_INTERVAL_SECONDS)


def shutdown() -> None:
    """Clean up any initialized WinSparkle instance."""
    global _winsparkle_library, _winsparkle_initialized
    global _configured_feed_url, _configured_channel, _configured_auto_check

    if _winsparkle_library is not None and _winsparkle_initialized:
        _winsparkle_library.win_sparkle_cleanup()

    _winsparkle_library = None
    _winsparkle_initialized = False
    _configured_feed_url = ""
    _configured_channel = ""
    _configured_auto_check = False


def _ensure_initialized(
    status: WindowsUpdaterStatus,
    *,
    update_channel: str,
    automatically_check: bool,
):
    """Load, configure, and initialize WinSparkle if needed."""
    global _winsparkle_library, _winsparkle_initialized
    global _configured_feed_url, _configured_channel, _configured_auto_check

    reconfigure = (
        _winsparkle_initialized
        and (
            _configured_feed_url != status.feed_url
            or _configured_channel != update_channel
            or _configured_auto_check != automatically_check
        )
    )
    if reconfigure:
        shutdown()

    if _winsparkle_library is None:
        if status.dll_path is None:
            raise OSError("WinSparkle DLL path is unavailable.")
        _winsparkle_library = _load_library(status.dll_path)

    if not _winsparkle_initialized:
        _configure_library(
            _winsparkle_library,
            feed_url=status.feed_url,
            public_key=status.public_key,
            automatically_check=automatically_check,
        )
        _winsparkle_library.win_sparkle_init()
        _winsparkle_initialized = True
        _configured_feed_url = status.feed_url
        _configured_channel = update_channel
        _configured_auto_check = automatically_check

    return _winsparkle_library


def start_interactive_update_check(
    *,
    update_channel: str = "stable",
    automatically_check: bool = False,
    base_path: Path | None = None,
) -> WindowsUpdaterStatus:
    """Launch an interactive WinSparkle update check."""
    status = get_status(update_channel=update_channel, base_path=base_path)
    if not status.available:
        return status

    try:
        library = _ensure_initialized(
            status,
            update_channel=update_channel,
            automatically_check=automatically_check,
        )
        library.win_sparkle_check_update_with_ui()
    except OSError as exc:
        return replace(status, error_message=str(exc))

    return replace(status, initialized=True, launched=True)
