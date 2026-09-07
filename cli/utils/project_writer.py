"""Keep project ownership until the invoking Click command closes."""

from pathlib import Path

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result
from core.project_lock import ProjectBusyError, project_writer


def own_project(ctx: click.Context, path: Path) -> Path:
    """Acquire before loading; Click releases ownership even on early exit."""
    try:
        return ctx.with_resource(project_writer(path)).path
    except ProjectBusyError as exc:
        if (ctx.obj or {}).get("json", False):
            output_result({"success": False, "error": exc.to_dict()}, as_json=True)
            exit_with(ExitCode.GENERAL_ERROR)
        exit_with(ExitCode.GENERAL_ERROR, str(exc))
    except OSError as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Cannot acquire project writer: {exc}")
