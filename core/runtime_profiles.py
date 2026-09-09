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
    """Install a profile's features through the verified manifest; never arbitrary packages.

    A warm worker for the profile's family is retired first so the install
    never lands under a process that already imported the old runtime, and the
    health check afterwards runs in a fresh worker.
    """
    from core.feature_registry import install_for_feature

    profile = get_profile(profile_id)
    _retire_family_worker(profile.family)
    failed = [feature for feature in profile.features if not install_for_feature(feature, progress_callback)]
    status = profile_status(profile_id)
    status["failed_features"] = failed
    status["success"] = not failed and status["installed"]
    return status


def _retire_family_worker(family: str) -> None:
    from core.runtime_supervisor import _default, default_supervisor

    if _default is None:
        return  # nothing warm yet; do not start a worker just to stop it
    default_supervisor().restart_family(family)


def probe_profile_runtime(profile_id: str) -> dict[str, Any]:
    """Import the profile's runtime inside a fresh worker and report the result.

    Raises ``RuntimeError`` when the runtime is missing or broken in the worker,
    mirroring the in-process ``ensure_*_runtime_available`` validators. The host
    never imports the native runtime itself.
    """
    from core.runtime_supervisor import WorkerError, default_supervisor

    profile = get_profile(profile_id)
    if not profile.probe_module:
        return {"ok": True, "profile": profile.id, "probed": False}
    supervisor = default_supervisor()
    supervisor.restart_family(profile.family)
    try:
        result = supervisor.run(profile.family, "probe", {"module": profile.probe_module}, timeout=120.0)
    except WorkerError as exc:
        raise RuntimeError(f"{profile.probe_module} runtime is incomplete in the {profile.family} worker: {exc}") from exc
    result["profile"] = profile.id
    result["probed"] = True
    return result
