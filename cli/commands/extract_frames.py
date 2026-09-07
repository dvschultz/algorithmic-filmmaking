"""Extract source frames through the shared durable operation."""

from pathlib import Path
from threading import Event

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result
from cli.utils.project_writer import own_project


@click.command("extract-frames")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("source_id")
@click.option(
    "--mode",
    type=click.Choice(["interval", "all", "smart"]),
    default="interval",
    show_default=True,
)
@click.option("--interval", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--clip-id", default=None, help="Restrict extraction to this source's clip."
)
@click.pass_context
def extract_frames(
    ctx: click.Context,
    project_file: Path,
    source_id: str,
    mode: str,
    interval: int,
    clip_id: str | None,
) -> None:
    """Append extracted frames and save PROJECT_FILE. A new invocation adds a new batch."""
    from core.jobs.frame_extraction import run_frame_extraction_job
    from core.jobs.store import JobStore
    from core.settings import load_settings

    path = own_project(ctx, project_file)
    store = JobStore(load_settings().cache_dir / "jobs.db")
    try:
        result = run_frame_extraction_job(
            store,
            path,
            source_id,
            lambda *_: None,
            Event(),
            mode=mode,
            interval=interval,
            clip_id=clip_id,
        )
        output_result(result, as_json=bool((ctx.obj or {}).get("json")))
        if not result["success"]:
            exit_with(ExitCode.GENERAL_ERROR)
    except (ValueError, RuntimeError, OSError) as exc:
        exit_with(ExitCode.GENERAL_ERROR, str(exc))
    finally:
        store.close()
