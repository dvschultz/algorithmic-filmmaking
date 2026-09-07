"""Import audio through the shared durable operation."""

from pathlib import Path
from threading import Event

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result
from cli.utils.project_writer import own_project


@click.command("import-audio")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("file_path")
@click.pass_context
def import_audio(ctx: click.Context, project_file: Path, file_path: str) -> None:
    """Import FILE_PATH and save PROJECT_FILE. Relative media paths use its directory."""
    from core.jobs.audio_import import run_audio_import_job
    from core.jobs.store import JobStore
    from core.settings import load_settings

    path = own_project(ctx, project_file)
    store = JobStore(load_settings().cache_dir / "jobs.db")
    try:
        result = run_audio_import_job(store, path, file_path, lambda *_: None, Event())
        output_result(result, as_json=bool((ctx.obj or {}).get("json")))
        if not result["success"]:
            exit_with(ExitCode.GENERAL_ERROR)
    except (ValueError, RuntimeError, OSError) as exc:
        exit_with(ExitCode.GENERAL_ERROR, str(exc))
    finally:
        store.close()
