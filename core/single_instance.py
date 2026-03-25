"""Single-instance guard for packaged desktop app launches."""

from __future__ import annotations

import logging
import os
from typing import Optional

from PySide6.QtCore import QLockFile

from core.paths import get_app_support_dir

logger = logging.getLogger(__name__)

_instance_lock: Optional[QLockFile] = None


def acquire_single_instance_lock(lock_name: str = "scene-ripper.lock") -> bool:
    """Acquire the app-wide single-instance lock.

    Returns False when another packaged instance already holds the lock.
    """
    global _instance_lock

    if _instance_lock is not None:
        return True

    lock_path = get_app_support_dir() / lock_name
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock = QLockFile(str(lock_path))
    lock.setStaleLockTime(0)

    if not lock.tryLock(0):
        logger.warning(
            "Another Scene Ripper instance is already running; refusing duplicate launch (pid=%s, lock=%s)",
            os.getpid(),
            lock_path,
        )
        return False

    _instance_lock = lock
    logger.info("Acquired single-instance lock for pid=%s at %s", os.getpid(), lock_path)
    return True


def release_single_instance_lock() -> None:
    """Release the app-wide single-instance lock if held."""
    global _instance_lock

    if _instance_lock is None:
        return

    try:
        _instance_lock.unlock()
    finally:
        _instance_lock = None
