"""Jobs framework for the Scene Ripper MCP server.

The MCP server splits long-running ops into ``start_*`` / ``get_job_status`` /
``get_job_result`` / ``cancel_job`` / ``list_jobs`` / ``purge_old_jobs`` tools.
This package owns the persistence layer (SQLite, ``store.py``), the in-memory
``ThreadPoolExecutor`` runtime (``runtime.py``), and the per-project mutex
(``lock.py``) that serialises jobs targeting the same project file.

The framework is MCP-only (R27): the GUI continues to use
``ui/workers/CancellableWorker`` for its own long-running work and does not
consume this code.
"""

from scene_ripper_mcp.jobs.store import (
    JobNotFoundError,
    JobRow,
    JobStore,
    sanitize_traceback,
)
from scene_ripper_mcp.jobs.lock import ProjectLockRegistry
from scene_ripper_mcp.jobs.runtime import (
    InvalidIdempotencyKeyError,
    JobRuntime,
)

__all__ = [
    "InvalidIdempotencyKeyError",
    "JobNotFoundError",
    "JobRow",
    "JobRuntime",
    "JobStore",
    "ProjectLockRegistry",
    "sanitize_traceback",
]
