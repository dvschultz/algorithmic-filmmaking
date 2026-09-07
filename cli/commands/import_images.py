"""Import still images through the shared durable operation."""

from pathlib import Path
from threading import Event
import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result
from cli.utils.project_writer import own_project


@click.command("import-images")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("file_paths", nargs=-1, required=True)
@click.option(
    "--copy/--reference",
    "copy_files",
    default=True,
    help="Copy images into the project (default), or reference originals.",
)
@click.pass_context
def import_images(
    ctx: click.Context,
    project_file: Path,
    file_paths: tuple[str, ...],
    copy_files: bool,
) -> None:
    """Append images and save PROJECT_FILE. Relative media paths use its directory."""
    from core.jobs.image_import import run_image_import_job
    from core.jobs.store import JobStore
    from core.settings import load_settings

    path = own_project(ctx, project_file)
    store = JobStore(load_settings().cache_dir / "jobs.db")
    try:
        result = run_image_import_job(
            store,
            path,
            list(file_paths),
            lambda *_: None,
            Event(),
            copy_files=copy_files,
        )
        output_result(result, as_json=bool((ctx.obj or {}).get("json")))
        if not result["success"]:
            exit_with(ExitCode.GENERAL_ERROR)
    except (ValueError, RuntimeError, OSError) as exc:
        exit_with(ExitCode.GENERAL_ERROR, str(exc))
    finally:
        store.close()
