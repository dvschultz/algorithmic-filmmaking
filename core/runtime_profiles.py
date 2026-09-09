"""Allowlisted runtime profiles for managed native workers.

Callers (UI, chat, MCP, CLI) request installation or repair by profile id
only. A profile maps to feature names in ``core.feature_registry`` whose
package pins come from the verified ``core/package_manifest.json``; nothing
here accepts package names, URLs, or executable paths from a caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    family: str
    """Worker family sharing one warm process and serialized accelerator access."""
    features: tuple[str, ...]
    """Feature-registry names whose manifest pins this profile installs."""
    task_kinds: tuple[str, ...]
    description: str
    probe_module: str = ""
    """Top-level runtime module whose install location decides which interpreter hosts the family."""


PROFILES: dict[str, RuntimeProfile] = {
    "transcription-whisper": RuntimeProfile(
        id="transcription-whisper",
        family="transcription",
        features=("transcribe",),
        task_kinds=("transcribe",),
        description="faster-whisper transcription in an isolated worker",
        probe_module="faster_whisper",
    ),
}


def family_probe_module(family: str) -> str | None:
    for profile in PROFILES.values():
        if profile.family == family and profile.probe_module:
            return profile.probe_module
    return None


def get_profile(profile_id: Any) -> RuntimeProfile:
    if not isinstance(profile_id, str) or profile_id not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown runtime profile {profile_id!r}; known profiles: {known}")
    return PROFILES[profile_id]


def profile_status(profile_id: str) -> dict[str, Any]:
    """Whether the profile's features are installed and which packages are missing."""
    from core.feature_registry import check_feature

    profile = get_profile(profile_id)
    missing: list[str] = []
    for feature in profile.features:
        available, feature_missing = check_feature(feature)
        if not available:
            missing.extend(feature_missing)
    return {
        "profile": profile.id, "family": profile.family, "features": list(profile.features),
        "task_kinds": list(profile.task_kinds), "installed": not missing, "missing": missing,
    }


def install_profile(profile_id: str, progress_callback=None) -> dict[str, Any]:
    """Install a profile's features through the verified manifest; never arbitrary packages."""
    from core.feature_registry import install_for_feature

    profile = get_profile(profile_id)
    failed = [feature for feature in profile.features if not install_for_feature(feature, progress_callback)]
    status = profile_status(profile_id)
    status["failed_features"] = failed
    status["success"] = not failed and status["installed"]
    return status
