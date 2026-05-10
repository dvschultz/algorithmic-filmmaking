"""Per-project mutex registry.

Concurrent jobs targeting the same project file serialise through a
``threading.RLock`` keyed on the project's canonical resolved path
(``Path(project_path).expanduser().resolve()`` — never the raw caller string,
to defeat tilde-vs-absolute aliasing) (R17).

Different-project jobs run in parallel.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional


class ProjectLockRegistry:
    """Thread-safe map of canonical project path -> RLock.

    Uses a registry-level lock to make ``get_lock`` itself thread-safe; the
    per-project lock returned is the one callers acquire/release around the
    actual work.
    """

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[str, threading.RLock] = defaultdict(
            threading.RLock
        )
        # Track which job currently holds each project lock so the runtime
        # can populate ``blocking_job_id`` on queued rows.
        self._holders: dict[str, str] = {}

    @staticmethod
    def canonical(project_path: Optional[str | Path]) -> Optional[str]:
        """Return the canonical resolved path string for ``project_path``.

        Returns None if ``project_path`` is None — jobs that do not target
        a project (e.g. a future search-only op) bypass the per-project
        mutex entirely.
        """
        if project_path is None:
            return None
        return str(Path(project_path).expanduser().resolve())

    def get_lock(self, canonical_path: str) -> threading.RLock:
        """Return the lock for ``canonical_path``. Idempotent."""
        with self._registry_lock:
            return self._locks[canonical_path]

    def set_holder(self, canonical_path: str, job_id: str) -> None:
        with self._registry_lock:
            self._holders[canonical_path] = job_id

    def clear_holder(self, canonical_path: str, job_id: str) -> None:
        with self._registry_lock:
            current = self._holders.get(canonical_path)
            if current == job_id:
                del self._holders[canonical_path]

    def current_holder(self, canonical_path: str) -> Optional[str]:
        with self._registry_lock:
            return self._holders.get(canonical_path)
