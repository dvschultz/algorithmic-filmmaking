"""Transcribe a standalone audio source with durable project publication."""

from pathlib import Path
from threading import Event

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result
from cli.utils.project_writer import own_project


@click.command("transcribe-audio")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("audio_source_id")
@click.option("--model", default="small.en", show_default=True)
@click.option("--language", default="en", show_default=True)
@click.option(
    "--backend",
    type=click.Choice(["auto", "faster-whisper", "mlx-whisper", "groq"]),
    default="auto",
)
@click.option("--force", is_flag=True, help="Replace an existing audio transcript.")
@click.option(
    "--segmentation-mode",
    type=click.Choice(["backend", "sentence", "phrase", "fixed"]),
    default="backend",
)
@click.option(
    "--segment-max-seconds", type=click.FloatRange(min=0, min_open=True), default=12.0
)
@click.pass_context
def transcribe_audio(
    ctx: click.Context,
    project_file: Path,
    audio_source_id: str,
    model: str,
    language: str,
    backend: str,
    force: bool,
    segmentation_mode: str,
    segment_max_seconds: float,
) -> None:
    """Transcribe an imported audio source by its exact ID and save the project."""
    from core.jobs.audio_transcription import run_audio_transcription_job
    from core.jobs.store import JobStore
    from core.operations.transcription import TranscriptionOptions
    from core.settings import load_settings

    path = own_project(ctx, project_file)
    store = JobStore(load_settings().cache_dir / "jobs.db")
    try:
        result = run_audio_transcription_job(
            store,
            path,
            audio_source_id,
            TranscriptionOptions(
                model=model,
                language=language,
                backend=backend,
                segmentation_mode=segmentation_mode,
                segment_max_seconds=segment_max_seconds,
            ),
            lambda *_: None,
            Event(),
            force=force,
        )
        output_result(result, as_json=bool((ctx.obj or {}).get("json")))
        if not result["success"]:
            exit_with(ExitCode.GENERAL_ERROR)
    except (ValueError, RuntimeError, OSError) as exc:
        exit_with(ExitCode.GENERAL_ERROR, str(exc))
    finally:
        store.close()
