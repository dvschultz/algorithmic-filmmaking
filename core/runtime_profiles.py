"""Allowlisted runtime profiles for managed native workers.

Callers (UI, chat, MCP, CLI) request installation or repair by profile id
only. A profile maps to feature names in ``core.feature_registry`` whose
package pins come from the verified ``core/package_manifest.json``; nothing
here accepts package names, URLs, or executable paths from a caller.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


def _apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


PROFILES: dict[str, RuntimeProfile] = {
    "transcription-whisper": RuntimeProfile(
        id="transcription-whisper",
        family="transcription",
        features=("worker_engine", "transcribe"),
        task_kinds=("transcribe",),
        description="faster-whisper transcription in an isolated worker",
        probe_module="faster_whisper",
    ),
    "vision-torch": RuntimeProfile(
        id="vision-torch",
        family="vision",
        features=("worker_engine", "embeddings", "shot_classify", "object_detect", "image_classify", "face_detect", "gaze_detect"),
        task_kinds=("analysis",),
        description="torch/transformers embeddings and shots, YOLO objects, InsightFace faces, MediaPipe gaze",
        probe_module="torch",
    ),
    "ocr-paddle": RuntimeProfile(
        id="ocr-paddle",
        family="ocr",
        features=("worker_engine", "ocr"),
        task_kinds=("analysis",),
        description="PaddleOCR text extraction in an isolated worker",
        probe_module="paddleocr",
    ),
    "vlm-local": RuntimeProfile(
        id="vlm-local",
        family="vlm",
        features=("worker_engine", "describe_local") if _apple_silicon() else ("worker_engine", "describe_local_cpu"),
        task_kinds=("analysis",),
        description="local vision-language model (mlx-vlm on Apple Silicon, transformers elsewhere)",
        probe_module="mlx_vlm" if _apple_silicon() else "transformers",
    ),
    "audio-librosa": RuntimeProfile(
        id="audio-librosa",
        family="audio",
        features=("worker_engine", "audio_analysis", "stem_separation"),
        task_kinds=("analysis",),
        description="librosa audio analysis and Demucs stem separation",
        probe_module="librosa",
    ),
    "alignment-ctc": RuntimeProfile(
        id="alignment-ctc",
        family="alignment",
        features=("worker_engine", "word_alignment"),
        task_kinds=("analysis",),
        description="CTC forced word alignment",
        probe_module="ctc_forced_aligner",
    ),
}


def profile_for_feature(feature: str) -> RuntimeProfile | None:
    """The profile (and so the worker family) that owns a feature, if any."""
    for profile in PROFILES.values():
        if feature in profile.features and feature != "worker_engine":
            return profile
    return None


def profile_can_stage(profile: RuntimeProfile) -> bool:
    """Whether every feature of the profile installs with ``pip --target``.

    Features flagged ``native_install`` (torch, mlx, insightface, ...) need the
    managed interpreter's site-packages; they install in place and are still
    health-checked in a worker afterwards.
    """
    from core.feature_registry import FEATURE_DEPS

    return all(not FEATURE_DEPS[f].native_install for f in profile.features if f in FEATURE_DEPS)


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
            missing.extend(item for item in feature_missing if item not in missing)
    return {
        "profile": profile.id, "family": profile.family, "features": list(profile.features),
        "task_kinds": list(profile.task_kinds), "installed": not missing, "missing": missing,
    }


def get_managed_staging_dir() -> Path:
    from core.paths import get_app_support_dir

    return get_app_support_dir() / "packages-staging"


def _probe_timeout() -> float:
    """Seconds to wait for a worker to import a freshly installed runtime.

    Cold-cache imports of a native stack take minutes, and on Windows runners a
    package tree pip has only just written is scanned on first read, which put
    faster-whisper's import past the old 240s ceiling. The value is a ceiling,
    not a delay, so err generous; SCENE_RIPPER_PROBE_TIMEOUT overrides it.
    """
    raw = os.environ.get("SCENE_RIPPER_PROBE_TIMEOUT", "")
    try:
        value = float(raw)
    except ValueError:
        return 600.0
    return value if value > 0 else 600.0


PROBE_TIMEOUT = 600.0
"""Cold-cache imports of a freshly installed runtime can take minutes."""

STALE_STAGE_SECONDS = 6 * 3600
"""Staging directories older than this belong to a crashed install and are swept."""

_profile_locks: dict[str, threading.Lock] = {}
_profile_locks_guard = threading.Lock()
_probe_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
_probe_cache_lock = threading.Lock()


def _profile_lock(profile_id: str) -> threading.Lock:
    with _profile_locks_guard:
        return _profile_locks.setdefault(profile_id, threading.Lock())


def overlay_name(profile: RuntimeProfile, stamp: int | None = None) -> str:
    """Overlay directories carry their profile so rollback can find them."""
    stamp = int(time.time() * 1000) if stamp is None else stamp
    return f"overlay-{stamp}-{profile.id}"


def _overlay_stamp(path: Path) -> int:
    try:
        return int(path.name.split("-")[1])
    except (IndexError, ValueError):
        return 0


def profile_overlays(profile_id: str) -> list[Path]:
    """Promoted overlays for a profile, newest first (by install stamp)."""
    from core.paths import get_managed_package_overlays_dir

    profile = get_profile(profile_id)
    root = get_managed_package_overlays_dir()
    if not root.is_dir():
        return []
    suffix = f"-{profile.id}"
    return sorted(
        (path for path in root.iterdir()
         if path.is_dir() and path.name.startswith("overlay-") and path.name.endswith(suffix)),
        key=lambda path: (_overlay_stamp(path), path.name), reverse=True,
    )


def _overlay_key(profile_id: str) -> tuple[str, ...]:
    return tuple(path.name for path in profile_overlays(profile_id))


def _sweep_stale_stages(staging_root: Path) -> None:
    """Remove staging directories a crashed install left behind (never fresh ones)."""
    cutoff = time.time() - STALE_STAGE_SECONDS
    try:
        entries = list(staging_root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir() and not entry.is_symlink() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


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
    The host process never imports the native runtime. Installs of one
    profile are serialized within this process.

    ``staged=False`` keeps the legacy in-place ``install_for_feature`` path.
    """
    from core.feature_registry import install_for_feature

    profile = get_profile(profile_id)
    with _profile_lock(profile.id):
        with _probe_cache_lock:
            _probe_cache.pop((profile.id, _overlay_key(profile.id)), None)
        if staged and not profile_can_stage(profile):
            staged = False  # site-packages installs cannot be staged; still probed in a worker below
        if not staged:
            _retire_family_worker(profile.family)
            failed: list[str] = []
            errors: list[str] = []
            for feature in profile.features:
                # install_for_feature raises when its post-install runtime
                # validation fails (a version conflict in the installed set, say).
                # This function reports install failures, so surface it as a
                # failed feature instead of unwinding through the caller.
                try:
                    installed_ok = bool(install_for_feature(feature, progress_callback))
                except Exception as exc:  # noqa: BLE001 - reported in the outcome
                    logger.warning("Install of %s for profile %s failed: %s", feature, profile.id, exc)
                    installed_ok = False
                    errors.append(f"{feature}: {exc}")
                if not installed_ok:
                    failed.append(feature)
            with _probe_cache_lock:
                _probe_cache.clear()
            _retire_family_worker(profile.family)
            status = profile_status(profile_id)
            status["failed_features"] = failed
            status["staged"] = False
            status["success"] = not failed and status["installed"]
            if errors:
                status["error"] = "; ".join(errors)
            if status["success"] and profile.probe_module:
                try:
                    status["health"] = probe_profile_runtime(profile_id)
                except RuntimeError as exc:
                    status["success"] = False
                    status["error"] = f"Health check failed after install: {exc}"
            return status
        outcome = _staged_install(profile, progress_callback, cancel_event)
        with _probe_cache_lock:
            _probe_cache.clear()
        if outcome["success"]:
            # Retire the warm worker so the next task sees the promoted overlay
            # (waits for a running task under the family lock) and let an
            # isolation-off host see the new directory too.
            _emit(progress_callback, 1.0, "Switching the worker to the new runtime...")
            _retire_family_worker(profile.family)
            from core.dependency_manager import _ensure_managed_packages_importable

            _ensure_managed_packages_importable()
        outcome.update({k: v for k, v in profile_status(profile_id).items() if k not in outcome})
        return outcome


def _emit(progress_callback, fraction: float, message: str) -> None:
    if progress_callback is not None:
        try:
            progress_callback(fraction, message)
        except Exception:  # noqa: BLE001 - a UI callback must not abort an install
            pass


def _cancelled(cancel_event) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _staged_install(profile: RuntimeProfile, progress_callback, cancel_event) -> dict[str, Any]:
    """Stage -> health check -> promote. Returns the outcome; never raises for install failures."""
    import uuid

    from core.feature_registry import stage_feature_packages

    staging_root = get_managed_staging_dir()
    staging_root.mkdir(parents=True, exist_ok=True)
    _sweep_stale_stages(staging_root)
    stamp = int(time.time() * 1000)
    stage_dir = staging_root / f"{overlay_name(profile, stamp)}-{uuid.uuid4().hex[:8]}"
    outcome: dict[str, Any] = {"profile": profile.id, "family": profile.family, "staged_dir": str(stage_dir)}
    keep_stage = False
    try:
        failed = [
            feature for feature in profile.features
            if not stage_feature_packages(feature, stage_dir, progress_callback, cancel_event)
        ]
        if _cancelled(cancel_event):
            return _failure(outcome, "Install cancelled; previous runtime kept", cancelled=True)
        if failed:
            return _failure(outcome, "Package install failed; previous runtime kept", failed_features=failed)
        _emit(progress_callback, 0.95, "Checking the new runtime in an isolated worker...")
        try:
            health = probe_profile_runtime(profile.id, staged_paths=(stage_dir,))
        except RuntimeError as exc:
            return _failure(outcome, f"Health check failed; previous runtime kept: {exc}")
        imported_from = Path(str(health.get("file") or ""))
        if not _within(imported_from, stage_dir):
            return _failure(
                outcome, "Health check imported the runtime from outside the staged directory; previous runtime kept",
                health=health,
            )
        if _cancelled(cancel_event):
            return _failure(outcome, "Install cancelled; previous runtime kept", cancelled=True)
        promoted = _promote_stage(stage_dir, profile, stamp)
        keep_stage = True  # renamed away; nothing left to discard
        outcome.update(success=True, promoted_dir=str(promoted), health=health, failed_features=[])
        return outcome
    finally:
        if not keep_stage and stage_dir.exists():
            # Discarding a staged install deletes tens of thousands of files. On
            # the Windows runner the step ran its full budget after a failed
            # probe, and this is the main suspect; time it so the next run says.
            started = time.monotonic()
            logger.info("Discarding staged install at %s", stage_dir)
            shutil.rmtree(stage_dir, ignore_errors=True)
            logger.info("Discarded staged install in %.1fs", time.monotonic() - started)


def _failure(outcome: dict[str, Any], error: str, **extra: Any) -> dict[str, Any]:
    outcome.update(success=False, error=error, failed_features=list(extra.pop("failed_features", [])), **extra)
    return outcome


def _within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _promote_stage(stage_dir: Path, profile: RuntimeProfile, stamp: int) -> Path:
    from core.paths import get_managed_package_overlays_dir

    overlays = get_managed_package_overlays_dir()
    overlays.mkdir(parents=True, exist_ok=True)
    target = overlays / overlay_name(profile, stamp)
    while target.exists():
        stamp += 1
        target = overlays / overlay_name(profile, stamp)
    os.replace(stage_dir, target)  # same volume (both under app support): atomic switch
    return target


def rollback_profile(profile_id: str) -> dict[str, Any]:
    """Remove the newest promoted overlay for a profile and re-check the runtime.

    Older overlays and the base package directory are untouched, so the
    runtime that was active before the last install is what remains. The
    overlay is renamed out of the search path first, so a partially deleted
    directory can never be picked up.
    """
    profile = get_profile(profile_id)
    with _profile_lock(profile.id):
        overlays = profile_overlays(profile_id)
        if not overlays:
            return {"success": False, "profile": profile.id, "error": "No promoted overlay to roll back"}
        _retire_family_worker(profile.family)
        newest = overlays[0]
        trash = newest.with_name(".trash-" + newest.name)
        try:
            os.replace(newest, trash)
        except OSError as exc:
            return {
                "success": False, "profile": profile.id, "removed": None,
                "error": f"Could not remove overlay {newest.name}; close other Scene Ripper processes and retry: {exc}",
            }
        shutil.rmtree(trash, ignore_errors=True)
        with _probe_cache_lock:
            _probe_cache.clear()
        result: dict[str, Any] = {"success": True, "profile": profile.id, "removed": str(newest)}
        if trash.exists():
            result["warning"] = f"Overlay {newest.name} was retired but some files could not be deleted"
        try:
            result["health"] = probe_profile_runtime(profile_id)
        except RuntimeError as exc:
            result["health_error"] = str(exc)
        result.update({k: v for k, v in profile_status(profile_id).items() if k not in result})
        return result


def _retire_family_worker(family: str) -> None:
    import core.runtime_supervisor as supervisor_module

    if supervisor_module._default is None:
        return  # nothing warm yet; do not start a worker just to stop it
    supervisor_module.default_supervisor().restart_family(family)


def probe_profile_runtime(
    profile_id: str, *, staged_paths: tuple[Path, ...] = (), restart: bool = True,
) -> dict[str, Any]:
    """Import the profile's runtime inside a worker and report the result.

    Raises ``RuntimeError`` when the runtime is missing or broken in the worker,
    mirroring the in-process ``ensure_*_runtime_available`` validators. The host
    never imports the native runtime itself. ``staged_paths`` puts not-yet-
    promoted install directories ahead of the live ones for one dedicated
    worker. ``restart=False`` reuses the family's warm worker (a readiness
    check must not kill a running task); the result is cached per overlay
    set until the next install or rollback.
    """
    from core.runtime_supervisor import ManagedWorker, WorkerError, default_launch, default_supervisor

    profile = get_profile(profile_id)
    if not profile.probe_module:
        return {"ok": True, "profile": profile.id, "probed": False}
    cache_key = (profile.id, _overlay_key(profile.id))
    if not staged_paths and not restart:
        with _probe_cache_lock:
            cached = _probe_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
    try:
        if staged_paths:
            launch = default_launch(profile.family, ensure_interpreter=True, staged_paths=staged_paths)
            expected = tuple(path for path in staged_paths if path.is_dir())
            if tuple(launch.package_paths[:len(expected)]) != expected:
                raise RuntimeError(
                    "Staged health check needs the managed interpreter; the worker would run "
                    f"{launch.interpreter} without the staged directory on its path"
                )
            worker = ManagedWorker(launch)
            worker.start()
            try:
                result = worker.run("probe", {"module": profile.probe_module}, timeout=_probe_timeout())
            finally:
                worker.close()
        else:
            supervisor = default_supervisor()
            if restart:
                supervisor.restart_family(profile.family)
            result = supervisor.run(profile.family, "probe", {"module": profile.probe_module}, timeout=_probe_timeout())
    except WorkerError as exc:
        raise RuntimeError(f"{profile.probe_module} runtime is incomplete in the {profile.family} worker: {exc}") from exc
    result["profile"] = profile.id
    result["probed"] = True
    if not staged_paths:
        with _probe_cache_lock:
            _probe_cache[cache_key] = dict(result)
    return result
