"""Native runtime profiles: status, staged install/repair, rollback (plan U14)."""

from __future__ import annotations

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import json_flag as _json, output_result


def _finish(ctx: click.Context, result: dict) -> None:
    if not result.get("success"):
        error = str(result.get("error") or "")
        # Bad input (unknown profile) is a validation error; a failed, cancelled
        # or unhealthy install is an operational failure.
        code = ExitCode.VALIDATION_ERROR if "Unknown runtime profile" in error else ExitCode.GENERAL_ERROR
        if _json(ctx):
            output_result(result, as_json=True)
            exit_with(code)
        exit_with(code, error)
    output_result(result, as_json=_json(ctx))


@click.group()
def runtime() -> None:
    """Manage isolated native runtime profiles (install, repair, roll back)."""


@runtime.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List runtime profiles with their install status."""
    from core.spine.runtime import list_runtime_profiles

    _finish(ctx, list_runtime_profiles())


@runtime.command("status")
@click.argument("profile")
@click.option("--probe", is_flag=True, help="Also import the runtime in a fresh worker as a health check")
@click.pass_context
def status(ctx: click.Context, profile: str, probe: bool) -> None:
    """Show one profile's status."""
    from core.spine.runtime import get_runtime_profile_status

    _finish(ctx, get_runtime_profile_status(profile, probe=probe))


@runtime.command("install")
@click.argument("profile")
@click.pass_context
def install(ctx: click.Context, profile: str) -> None:
    """Install or repair a profile: staged, health-checked in a worker, then promoted."""
    from core.spine.runtime import install_runtime_profile

    def progress(fraction: float, message: str) -> None:
        if not _json(ctx):
            click.echo(f"[{int(fraction * 100):3d}%] {message}", err=True)

    _finish(ctx, install_runtime_profile(profile, progress_callback=progress))


@runtime.command("rollback")
@click.argument("profile")
@click.pass_context
def rollback(ctx: click.Context, profile: str) -> None:
    """Remove the newest promoted overlay so the previous runtime is active again."""
    from core.spine.runtime import rollback_runtime_profile

    _finish(ctx, rollback_runtime_profile(profile))
