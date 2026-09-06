"""Shared session-local undo and redo with transport-friendly results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project


def undo(project: Project) -> dict:
    try:
        label = project.session.undo()
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    if label is None:
        return {"success": False, "error": "Nothing to undo"}
    return {"success": True, "undone": label}


def redo(project: Project) -> dict:
    try:
        label = project.session.redo()
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    if label is None:
        return {"success": False, "error": "Nothing to redo"}
    return {"success": True, "redone": label}
