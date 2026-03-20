"""macOS native updater detection and Sparkle CLI integration."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from core.update_models import UpdateCapability
from core.paths import get_base_path, is_frozen

SPARKLE_FEED_URL_INFO_KEY = "SUFeedURL"
SPARKLE_PUBLIC_KEY_INFO_KEY = "SUPublicEDKey"
SPARKLE_CLI_ENV = "SCENE_RIPPER_SPARKLE_CLI"
SPARKLE_FEED_ENV = "SPARKLE_FEED_URL"


@dataclass(frozen=True)
class MacOSUpdaterStatus:
    """Status for the macOS native updater integration."""

    available: bool
    capability: UpdateCapability
    reason: str = ""
    bundle_path: Path | None = None
    sparkle_cli_path: Path | None = None
    feed_url: str = ""
    public_key: str = ""
    launched: bool = False
    error_message: str = ""


def get_bundle_path(executable_path: str | Path | None = None) -> Path | None:
    """Return the enclosing .app bundle path for the current executable."""
    if sys.platform != "darwin":
        return None

    executable = Path(executable_path or sys.executable).resolve(strict=False)
    if executable.suffix == ".app":
        return executable

    for parent in executable.parents:
        if parent.suffix == ".app":
            return parent
    return None


def is_translocated_path(path: Path) -> bool:
    """Return True if the bundle appears to be running from App Translocation."""
    return "AppTranslocation" in path.as_posix()


def is_running_from_disk_image(path: Path) -> bool:
    """Return True if the bundle path is inside a mounted DMG volume."""
    return path.as_posix().startswith("/Volumes/")


def is_in_applications(path: Path) -> bool:
    """Return True if the app is installed in a standard Applications directory."""
    posix_path = path.as_posix()
    return posix_path.startswith("/Applications/") or posix_path.startswith(
        (Path.home() / "Applications").as_posix() + "/"
    )


def read_bundle_info(bundle_path: Path | None = None) -> dict:
    """Read the app bundle Info.plist."""
    bundle = bundle_path or get_bundle_path()
    if bundle is None:
        return {}
    info_plist = bundle / "Contents" / "Info.plist"
    if not info_plist.exists():
        return {}
    with info_plist.open("rb") as fh:
        return plistlib.load(fh)


def get_feed_url(bundle_path: Path | None = None) -> str:
    """Return the configured Sparkle feed URL, if any."""
    if env_url := os.environ.get(SPARKLE_FEED_ENV, "").strip():
        return env_url
    return str(read_bundle_info(bundle_path).get(SPARKLE_FEED_URL_INFO_KEY, "")).strip()


def get_public_ed_key(bundle_path: Path | None = None) -> str:
    """Return the bundled Sparkle public EdDSA key, if any."""
    return str(read_bundle_info(bundle_path).get(SPARKLE_PUBLIC_KEY_INFO_KEY, "")).strip()


def find_sparkle_cli(bundle_path: Path | None = None, base_path: Path | None = None) -> Path | None:
    """Find a bundled sparkle CLI binary."""
    if env_path := os.environ.get(SPARKLE_CLI_ENV, "").strip():
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    base = base_path or get_base_path()
    bundle = bundle_path or get_bundle_path()
    candidates = [
        base / "bin" / "sparkle",
        base / "Sparkle" / "bin" / "sparkle",
        base / "sparkle" / "bin" / "sparkle",
        base / "sparkle.app" / "Contents" / "MacOS" / "sparkle",
        base / "Sparkle.framework" / "Versions" / "Current" / "Resources" / "bin" / "sparkle",
        base / "Sparkle.framework" / "Versions" / "A" / "Resources" / "bin" / "sparkle",
    ]
    if bundle is not None:
        candidates.extend(
            [
                bundle / "Contents" / "MacOS" / "sparkle",
                bundle / "Contents" / "Resources" / "bin" / "sparkle",
                bundle / "Contents" / "Resources" / "sparkle.app" / "Contents" / "MacOS" / "sparkle",
                bundle / "Contents" / "Resources" / "Sparkle.framework" / "Versions" / "Current" / "Resources" / "bin" / "sparkle",
                bundle / "Contents" / "Resources" / "Sparkle.framework" / "Versions" / "A" / "Resources" / "bin" / "sparkle",
                bundle / "Contents" / "Frameworks" / "Sparkle.framework" / "Versions" / "Current" / "Resources" / "bin" / "sparkle",
                bundle / "Contents" / "Frameworks" / "Sparkle.framework" / "Versions" / "B" / "Resources" / "bin" / "sparkle",
                bundle / "Contents" / "Frameworks" / "Sparkle.framework" / "Versions" / "A" / "Resources" / "bin" / "sparkle",
                bundle / "Contents" / "Frameworks" / "sparkle.app" / "Contents" / "MacOS" / "sparkle",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def get_status(
    *,
    executable_path: str | Path | None = None,
    base_path: Path | None = None,
) -> MacOSUpdaterStatus:
    """Return native updater availability for the current build."""
    if sys.platform != "darwin" or not is_frozen():
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Native macOS updates are only available in installed macOS app bundles.",
        )

    bundle_path = get_bundle_path(executable_path)
    if bundle_path is None:
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Could not locate the Scene Ripper app bundle.",
        )

    if is_translocated_path(bundle_path):
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Move Scene Ripper to Applications to enable native updates.",
            bundle_path=bundle_path,
        )

    if is_running_from_disk_image(bundle_path):
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Copy Scene Ripper to Applications before using native updates.",
            bundle_path=bundle_path,
        )

    if not is_in_applications(bundle_path):
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Install Scene Ripper in Applications to enable native updates.",
            bundle_path=bundle_path,
        )

    feed_url = get_feed_url(bundle_path)
    public_key = get_public_ed_key(bundle_path)
    sparkle_cli_path = find_sparkle_cli(bundle_path, base_path)

    if not feed_url or not public_key:
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Native updater metadata is not configured in this build.",
            bundle_path=bundle_path,
            sparkle_cli_path=sparkle_cli_path,
            feed_url=feed_url,
            public_key=public_key,
        )

    if sparkle_cli_path is None:
        return MacOSUpdaterStatus(
            available=False,
            capability=UpdateCapability.FALLBACK_BROWSER,
            reason="Sparkle is not bundled with this app yet.",
            bundle_path=bundle_path,
            feed_url=feed_url,
            public_key=public_key,
        )

    return MacOSUpdaterStatus(
        available=True,
        capability=UpdateCapability.NATIVE_CHECK,
        bundle_path=bundle_path,
        sparkle_cli_path=sparkle_cli_path,
        feed_url=feed_url,
        public_key=public_key,
    )


def start_interactive_update_check(
    *,
    update_channel: str = "stable",
    executable_path: str | Path | None = None,
    base_path: Path | None = None,
) -> MacOSUpdaterStatus:
    """Launch an interactive native update check using Sparkle CLI."""
    status = get_status(executable_path=executable_path, base_path=base_path)
    if not status.available or status.sparkle_cli_path is None or status.bundle_path is None:
        return status

    command = [
        str(status.sparkle_cli_path),
        "bundle",
        "--application",
        str(status.bundle_path),
        "--check-immediately",
        "--interactive",
        "--feed-url",
        status.feed_url,
        "--user-agent-name",
        "Scene Ripper",
    ]
    if update_channel and update_channel != "stable":
        command.extend(["--channels", update_channel])

    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return replace(status, error_message=str(exc))

    return replace(status, launched=True)
