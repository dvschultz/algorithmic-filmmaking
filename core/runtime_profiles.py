"""Allowlisted runtime profiles for managed native workers.

Callers (UI, chat, MCP, CLI) request installation or repair by profile id
only. A profile maps to feature names in ``core.feature_registry`` whose
package pins come from the verified ``core/package_manifest.json``; nothing
here accepts package names, URLs, or executable paths from a caller.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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


def get_managed_staging_dir() -> Path:
    from core.paths import get_app_support_dir

    return get_app_support_dir() / "packages-staging"


def overlay_name(profile: RuntimeProfile, stamp: int | None = None) -> str:
    """Overlay directories carry their profile so rollback can find them."""
    stamp = int(time.time() * 1000) if stamp is None else stamp
    return f"overlay-{stamp}-{profile.id}"


def profile_overlays(profile_id: str) -> list[Path]:
    """Promoted overlays for a profile, newest first."""
    from core.paths import get_managed_package_overlays_dir

    profile = get_profile(profile_id)
    root = get_managed_package_overlays_dir()
    if not root.is_dir():
        return []
    suffix = f"-{profile.id}"
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.endswith(suffix)),
        key=lambda path: path.name, reverse=True,
    )


def install_profile(
    profile_id: str,
    progress_callback=None,
    *,
    cancel_event: "threading.Event | None" = None,
    staged: bool = True,
) -> dict[str, Any]:
    """Install a profile's features through the verified manifest; never arbitrary packages.

    Staged (default): packages are installed into a fresh staging directory,
    health-checked from a fresh worker that sees that directory first, and
    only then promoted to a package overlay (an atomic rename). A failed or
    cancelled install discards the staging directory, so the previous runtime
    stays usable; a previous overlay stays on disk for ``rollback_profile``.
    The host process never imports the native runtime.

    ``staged=False`` keeps the legacy in-place ``install_for_feature`` path.
    """
    from core.feature_registry import install_for_feature, stage_feature_packages

    profile = get_profile(profile_id)
    _retire_family_worker(profile.family)
    if not staged:
        failed = [feature for feature in profile.features if not install_for_feature(feature, progress_callback)]
        status = profile_status(profile_id)
        status["failed_features"] = failed
        status["success"] = not failed and status["installed"]
        return status

    staging_root = get_managed_staging_dir()
    staging_root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    stage_dir = staging_root / overlay_name(profile, stamp)
    outcome: dict[str, Any] = {"profile": profile.id, "family": profile.family, "staged_dir": str(stage_dir)}
    try:
        failed = [
            feature for feature in profile.features
            if not stage_feature_packages(feature, stage_dir, progress_callback, cancel_event)
        ]
        if cancel_event is not None and cancel_event.is_set():
            outcome.update(success=False, cancelled=True, error="Install cancelled; previous runtime kept")
            return outcome
        if failed:
            outcome.update(success=False, failed_features=failed, error="Package install failed; previous runtime kept")
            return outcome
        try:
            health = probe_profile_runtime(profile_id, staged_paths=(stage_dir,))
        except RuntimeError as exc:
            outcome.update(success=False, error=f"Health check failed; previous runtime kept: {exc}")
            return outcome
        promoted = _promote_stage(stage_dir, profile, stamp)
        outcome.update(success=True, promoted_dir=str(promoted), health=health)
        return outcome
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        _retire_family_worker(profile.family)
        if outcome.get("success"):
            outcome.update({k: v for k, v in profile_status(profile_id).items() if k not in outcome})
        else:
            outcome.setdefault("failed_features", [])
            outcome.update({k: v for k, v in profile_status(profile_id).items() if k not in outcome})


def _promote_stage(stage_dir: Path, profile: RuntimeProfile, stamp: int) -> Path:
    from core.paths import get_managed_package_overlays_dir

    overlays = get_managed_package_overlays_dir()
    overlays.mkdir(parents=True, exist_ok=True)
    target = overlays / overlay_name(profile, stamp)
    os.replace(stage_dir, target)  # same volume: atomic switch
    return target


def rollback_profile(profile_id: str) -> dict[str, Any]:
    """Remove the newest promoted overlay for a profile and re-check the runtime.

    Older overlays and the base package directory are untouched, so the
    runtime that was active before the last install is what remains.
    """
    profile = get_profile(profile_id)
    overlays = profile_overlays(profile_id)
    if not overlays:
        return {"success": False, "profile": profile.id, "error": "No promoted overlay to roll back"}
    _retire_family_worker(profile.family)
    newest = overlays[0]
    shutil.rmtree(newest, ignore_errors=True)
    result: dict[str, Any] = {"success": True, "profile": profile.id, "removed": str(newest)}
    try:
        result["health"] = probe_profile_runtime(profile_id)
    except RuntimeError as exc:
        result["health_error"] = str(exc)
    result.update({k: v for k, v in profile_status(profile_id).items() if k not in result})
    return result


def _retire_family_worker(family: str) -> None:
    from core.runtime_supervisor import _default, default_supervisor

    if _default is None:
        return  # nothing warm yet; do not start a worker just to stop it
    default_supervisor().restart_family(family)


def probe_profile_runtime(profile_id: str, *, staged_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Import the profile's runtime inside a fresh worker and report the result.

    Raises ``RuntimeError`` when the runtime is missing or broken in the worker,
    mirroring the in-process ``ensure_*_runtime_available`` validators. The host
    never imports the native runtime itself. ``staged_paths`` puts not-yet-
    promoted install directories ahead of the live ones for that one worker.
    """
    from core.runtime_supervisor import ManagedWorker, WorkerError, default_launch, default_supervisor

    profile = get_profile(profile_id)
    if not profile.probe_module:
        return {"ok": True, "profile": profile.id, "probed": False}
    try:
        if staged_paths:
            worker = ManagedWorker(default_launch(profile.family, ensure_interpreter=True, staged_paths=staged_paths))
            worker.start()
            try:
                result = worker.run("probe", {"module": profile.probe_module}, timeout=120.0)
            finally:
                worker.close()
        else:
            supervisor = default_supervisor()
            supervisor.restart_family(profile.family)
            result = supervisor.run(profile.family, "probe", {"module": profile.probe_module}, timeout=120.0)
    except WorkerError as exc:
        raise RuntimeError(f"{profile.probe_module} runtime is incomplete in the {profile.family} worker: {exc}") from exc
    result["profile"] = profile.id
    result["probed"] = True
    return result
