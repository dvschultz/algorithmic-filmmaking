"""Reuse OS-backed leases to distinguish live runtimes from abandoned jobs."""

from typing import BinaryIO
from uuid import UUID

from core.project_lock import acquire_lock_record


def acquire_owner(owner_id: str) -> BinaryIO:
    """Acquire a unique runtime lease; the OS releases it on process exit."""
    if str(UUID(owner_id)) != owner_id:
        raise ValueError("Invalid job owner identity")
    return acquire_lock_record(f"job-owner:{owner_id}")
