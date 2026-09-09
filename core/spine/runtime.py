"""Native runtime profiles: status, staged install/repair, rollback (plan U14).

Every surface (chat, MCP, CLI, UI) reports the same profile status and runs
installs through the same staged path in ``core.runtime_profiles``: packages
land in a staging directory, a fresh worker health-checks them, and only
then are they promoted. Nothing here imports a native runtime into the host.
"""

from __future__ import annotations

from threading import Event
from typing import Any, Callable


def _isolation() -> dict[str, Any]:
    from core.runtime_families import enabled_families
    from core.transcription import native_worker_enabled

    return {
        "native_worker_isolation": native_worker_enabled(),
        "isolated_families": sorted(enabled_families()),
    }


def list_runtime_profiles() -> dict[str, Any]:
    """Every allowlisted profile with its install status and promoted overlays."""
    from core.runtime_profiles import PROFILES, profile_overlays, profile_status

    from core.runtime_families import FAMILIES, family_isolated
    from core.runtime_profiles import profile_can_stage

    profiles = []
    for profile_id, profile in PROFILES.items():
        status = profile_status(profile_id)
        status["description"] = profile.description
        status["overlays"] = [path.name for path in profile_overlays(profile_id)]
        status["staged_installs"] = profile_can_stage(profile)
        status["isolated"] = family_isolated(profile.family)
        profiles.append(status)
    families = [
        {"family": fid, "description": f.description, "isolated": family_isolated(fid)}
        for fid, f in FAMILIES.items()
    ]
    return {"success": True, "profiles": profiles, "families": families, **_isolation()}


def get_runtime_profile_status(profile_id: str, *, probe: bool = False) -> dict[str, Any]:
    """Status for one profile; ``probe`` also imports the runtime in a worker."""
    from core.runtime_profiles import get_profile, probe_profile_runtime, profile_overlays, profile_status

    try:
        get_profile(profile_id)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    status = profile_status(profile_id)
    status["overlays"] = [path.name for path in profile_overlays(profile_id)]
    if probe and status["installed"]:
        try:
            status["health"] = probe_profile_runtime(profile_id)
        except RuntimeError as exc:
            status["health_error"] = str(exc)
            status["installed"] = False
    status.update(success=True, **_isolation())
    return status


def install_runtime_profile(
    profile_id: str,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Stage, health-check and promote a profile install; the old runtime survives failure."""
    from core.runtime_profiles import get_profile, install_profile

    try:
        get_profile(profile_id)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    result = install_profile(profile_id, progress_callback, cancel_event=cancel_event)
    result.update(_isolation())
    return result


def rollback_runtime_profile(profile_id: str) -> dict[str, Any]:
    """Drop the newest promoted overlay for a profile and re-check the runtime."""
    from core.runtime_profiles import get_profile, rollback_profile

    try:
        get_profile(profile_id)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    result = rollback_profile(profile_id)
    result.update(_isolation())
    return result
