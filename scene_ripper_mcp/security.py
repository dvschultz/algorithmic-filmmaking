"""Security utilities for the MCP server.

The implementations live in ``core.spine.security`` so that both the GUI agent
(via ``core/chat_tools.py``) and the MCP server use a single validator. This
module re-exports the public surface so existing MCP tool imports keep working
without churning every call site.
"""

from core.spine.security import (
    SAFE_ROOTS,
    validate_path,
    validate_project_path,
    validate_video_path,
)

__all__ = [
    "SAFE_ROOTS",
    "validate_path",
    "validate_project_path",
    "validate_video_path",
]
