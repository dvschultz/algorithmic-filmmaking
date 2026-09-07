"""Transcription command for speech-to-text."""

from pathlib import Path

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_success, output_info
from cli.utils.progress import ProgressContext
from cli.utils.project_writer import own_project


# Available models with descriptions
WHISPER_MODELS = {
    "tiny.en": "Fastest, basic accuracy (~39MB)",
    "small.en": "Good balance of speed/accuracy (~244MB)",
    "medium.en": "Better accuracy, slower (~769MB)",
    "large-v3": "Best accuracy, requires GPU (~1.5GB)",
}


@click.command()
@click.argument("project_file", type=click.Path(path_type=Path), required=False)
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to transcribe (default: all)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(list(WHISPER_MODELS.keys())),
    default=None,
    help="Whisper model to use",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Language code (e.g., 'en', 'es', 'auto' for detection)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-transcribe clips that already have transcripts",
)
@click.option(
    "--list-models",
    is_flag=True,
    help="List available Whisper models and exit",
)
@click.pass_context
def transcribe(
    ctx: click.Context,
    project_file: Path | None,
    clip_ids: tuple[str, ...],
    model: str | None,
    language: str | None,
    force: bool,
    list_models: bool,
) -> None:
    """Transcribe speech in video clips.

    Uses faster-whisper for efficient speech-to-text transcription.
    Models are downloaded automatically on first use.

    \b
    Examples:
        scene_ripper transcribe project.json
        scene_ripper transcribe project.json --model small.en
        scene_ripper transcribe project.json -c clip1 -c clip2
        scene_ripper transcribe --list-models
    """
    # Handle --list-models
    if list_models:
        click.echo("Available Whisper models:")
        for name, desc in WHISPER_MODELS.items():
            click.echo(f"  {name:12} - {desc}")
        return

    # Validate project_file is provided
    if project_file is None:
        exit_with(ExitCode.USAGE_ERROR, "Missing argument 'PROJECT_FILE'")

    if not project_file.exists():
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    project_file = own_project(ctx, project_file)

    from core.project import Project, ProjectLoadError

    config = CLIConfig.load()

    # Use config defaults if not specified
    if model is None:
        model = config.transcription_model
    if language is None:
        language = config.transcription_language

    # Load project
    try:
        project = Project.load(
            project_file, missing_source_callback=lambda path, sid: None,
        )
        clips = project.clips
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Filter clips if specific IDs provided
    clips_to_transcribe = clips
    if clip_ids:
        clip_set = set(clip_ids)
        clips_to_transcribe = [
            c for c in clips if c.id in clip_set or c.id[:8] in clip_set
        ]
        if not clips_to_transcribe:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")

    # Filter out already-transcribed clips unless force
    if not force:
        clips_to_transcribe = [c for c in clips_to_transcribe if c.transcript is None]

    if not clips_to_transcribe:
        output_info("All clips already have transcripts. Use --force to re-transcribe.")
        return

    output_info(f"Using Whisper model: {model}")
    output_info("Loading model (this may take a moment on first run)...")

    from threading import Event
    from core.jobs.store import JobStore
    from core.jobs.transcription import run_transcription_job
    from core.operations.transcription import TranscriptionOptions
    from core.settings import load_settings

    with ProgressContext("Transcribing") as progress:
        try:
            batch = run_transcription_job(
                JobStore(load_settings().cache_dir / "jobs.db"),
                project_file,
                [clip.id for clip in clips_to_transcribe],
                TranscriptionOptions(model=model, language=language),
                progress.update,
                Event(),
                force=force,
            )["result"]
        except Exception as exc:
            exit_with(ExitCode.GENERAL_ERROR, f"Transcription failed: {exc}")
        progress.update(1.0, "Complete")
    transcribed_count = len(batch["succeeded"])
    total_segments = sum(item["segment_count"] for item in batch["succeeded"])
    errors = [f"Clip {item['clip_id'][:8]}: {item.get('message') or item['code']}" for item in batch["failed"] + batch["unprocessed"]]

    if not transcribed_count and any(item["code"] == "dependency_missing" for item in batch["failed"]):
        exit_with(ExitCode.DEPENDENCY_MISSING, errors[0])

    result = {
        "transcribed_clips": transcribed_count,
        "total_segments": total_segments,
        "errors": len(errors),
        "total_clips": len(clips),
        "model": model,
        "language": language,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(
            f"Transcribed {transcribed_count} clips ({total_segments} segments)"
        )
        if errors:
            for err in errors[:5]:
                output_info(f"  Error: {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")
