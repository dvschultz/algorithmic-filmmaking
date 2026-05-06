"""Project file I/O helpers with mtime-based collision detection.

Used by MCP tool wrappers and the jobs framework to defend against the
GUI-vs-MCP same-project hazard described in plan R19. Best-effort, not a
guarantee — the per-project mutex (lock.py in the jobs framework) handles
intra-process races, the mtime guard catches most external collisions, and
last-writer-wins remains possible under the read-modify-write window in
``core/project.py:save_project``.

The MCP server's ``v1`` scope assumes the GUI is closed when MCP drives the
project. This guard is the safety net for the cases where it isn't.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from core.project import Project

# Mtime-comparison tolerance in seconds. Filesystem caches and network mounts
# can report sub-second drift on otherwise-untouched files; 1s is conservative
# enough to avoid false positives without missing real external writes.
MTIME_TOLERANCE_SECONDS: float = 1.0


class ProjectModifiedExternally(Exception):
    """Raised by ``save_with_mtime_check`` when the project file's mtime has
    drifted between load and save.

    The MCP tool wrappers catch this and surface a structured
    ``project_modified_externally`` error to the caller.
    """

    def __init__(
        self,
        path: Path,
        expected_mtime: float,
        current_mtime: float,
    ) -> None:
        self.path = path
        self.expected_mtime = expected_mtime
        self.current_mtime = current_mtime
        super().__init__(
            f"Project file mtime drifted: expected {expected_mtime}, "
            f"got {current_mtime} (path={path})"
        )


def load_with_mtime(path: Path | str) -> Tuple[Project, float]:
    """Load a project and capture its file mtime at load time.

    Returns ``(project, mtime)``. Caller must hold onto ``mtime`` and pass it
    to ``save_with_mtime_check`` to detect external modifications between
    load and save.

    Raises whatever ``Project.load()`` raises (``ProjectLoadError``,
    ``MissingSourceError``).
    """
    p = Path(path)
    mtime = p.stat().st_mtime
    project = Project.load(p)
    return project, mtime


def save_with_mtime_check(
    project: Project,
    path: Path | str,
    expected_mtime: float,
) -> None:
    """Save a project, aborting if its file mtime drifted from ``expected_mtime``.

    Drift is measured with ``MTIME_TOLERANCE_SECONDS`` slack — within tolerance
    counts as unchanged. If the file does not exist (e.g. new-project save
    against a fresh path), the check is skipped — drift is meaningless when
    there is no prior version.

    Raises ``ProjectModifiedExternally`` when the mtime drifted; the project
    is **not** written in that case.
    """
    p = Path(path)
    if p.exists():
        current = p.stat().st_mtime
        if abs(current - expected_mtime) > MTIME_TOLERANCE_SECONDS:
            raise ProjectModifiedExternally(p, expected_mtime, current)

    project.save(p)


def save_path_resolved(path: Path | str) -> Path:
    """Resolve ``path`` to an absolute, expanded ``Path``.

    Convenience helper for callers that need to canonicalise the project
    path before keying mutexes or recording it in the jobs database.
    Centralised here so the canonical-path rule lives next to the I/O
    helpers that depend on it.
    """
    return Path(path).expanduser().resolve()
