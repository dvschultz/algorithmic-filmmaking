"""Save-project spine impl.

Persists the in-memory ``Project`` to disk. Validates the destination path
through ``core.spine.security`` and falls back to ``settings.export_dir`` when
no path was given.
"""

from __future__ import annotations

from typing import Optional

from core.settings import load_settings
from core.spine.security import validate_path


def save_project(project, path: Optional[str] = None) -> dict:
    """Save the project to disk."""
    if path:
        valid, error, validated_path = validate_path(path)
        if not valid:
            return {"success": False, "error": f"Invalid path: {error}"}
        save_path = validated_path
    elif project.path:
        save_path = project.path
    else:
        settings = load_settings()
        project_name = project.metadata.name or "untitled"
        save_path = settings.export_dir / f"{project_name}.sceneripper"

    if save_path.suffix.lower() != ".sceneripper":
        save_path = save_path.with_suffix(".sceneripper")

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        test_file = save_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        return {
            "success": False,
            "error": (
                f"Cannot save project: No write access to '{save_path.parent}'. "
                "Please check directory permissions or choose a different location."
            ),
        }
    except OSError as e:
        return {
            "success": False,
            "error": (
                f"Cannot save project: Directory '{save_path.parent}' is not "
                f"accessible ({e}). Please check the path or choose a different "
                "location."
            ),
        }

    try:
        success = project.save(path=save_path)
    except Exception as e:  # pragma: no cover - defensive
        return {"success": False, "error": str(e)}

    if success:
        return {
            "success": True,
            "path": str(save_path),
            "message": f"Project saved to {save_path.name}",
        }
    return {"success": False, "error": "Failed to save project"}


__all__ = ["save_project"]
