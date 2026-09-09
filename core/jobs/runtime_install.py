"""Durable runtime-profile install/repair for job runtimes (plan U14).

The job is project-independent: it stages the profile's packages, health-
checks them in a fresh worker, and promotes them. Cancellation kills pip and
discards the staging directory, so the previous runtime remains usable.
"""

from __future__ import annotations

from threading import Event
from typing import Any, Callable

from core.jobs.spec import OperationSpec

OPERATION_VERSION = 1


def runtime_install_job_spec(profile_id: str) -> OperationSpec:
    from core.runtime_profiles import get_profile

    profile = get_profile(profile_id)  # ValueError for anything but an allowlisted id
    return OperationSpec.build(
        kind="install_runtime_profile", version=OPERATION_VERSION,
        arguments={"profile": profile.id},
        inputs={"profile": profile.id, "features": list(profile.features)},
        persistence="job_history",
    )


def run_runtime_install_job(
    profile_id: str, progress: Callable[[float, str], None], cancel: Event,
) -> dict[str, Any]:
    from core.spine.runtime import install_runtime_profile

    progress(0.0, f"Installing runtime profile {profile_id}...")
    result = install_runtime_profile(profile_id, progress_callback=progress, cancel_event=cancel)
    if result.get("success"):
        progress(1.0, f"Runtime profile {profile_id} ready")
    return result
