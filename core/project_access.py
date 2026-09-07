"""Read-only preflight shared by desktop agent dispatch paths."""

from typing import Any

READ_ONLY_MESSAGE = (
    "This project uses a newer schema and is read-only. "
    "Open it in a newer version of Scene Ripper to edit or save it."
)


def read_only_tool_error(project: Any, tool: Any) -> str | None:
    if getattr(project, "is_read_only", False) is not True:
        return None
    if tool.modifies_project_state or tool.name in {"save_project", "export_bundle"}:
        return READ_ONLY_MESSAGE
    return None
