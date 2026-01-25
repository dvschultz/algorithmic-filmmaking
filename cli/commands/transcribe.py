"""Transcription command for speech-to-text."""

from pathlib import Path

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_success, output_info
from cli.utils.progress import ProgressContext


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

    # Check for faster-whisper
    try:
        from core.transcription import (
            transcribe_clip,
            is_faster_whisper_available,
            FasterWhisperNotInstalledError,
            WHISPER_MODELS as CORE_MODELS,
        )
        from core.project import load_project, save_project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    if not is_faster_whisper_available():
        exit_with(
            ExitCode.DEPENDENCY_MISSING,
            "faster-whisper is not installed. Install with: pip install faster-whisper",
        )

    config = CLIConfig.load()

    # Use config defaults if not specified
    if model is None:
        model = config.transcription_model
    if language is None:
        language = config.transcription_language

    # Load project
    try:
        sources, clips, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    sources_by_id = {s.id: s for s in sources}

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

    transcribed_count = 0
    errors = []
    total_segments = 0

    with ProgressContext("Transcribing") as progress:
        total = len(clips_to_transcribe)
        for i, clip in enumerate(clips_to_transcribe):
            progress.update(i / total, f"Clip {i + 1}/{total}")

            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                errors.append(f"Clip {clip.id[:8]}: source not found")
                continue

            try:
                fps = source.fps
                start_time = clip.start_time(fps)
                end_time = clip.end_time(fps)

                # Transcribe the clip
                segments = transcribe_clip(
                    source_path=source.file_path,
                    start_time=start_time,
                    end_time=end_time,
                    model_name=model,
                    language=language,
                )

                clip.transcript = segments if segments else None
                transcribed_count += 1
                total_segments += len(segments)

            except Exception as e:
                errors.append(f"Clip {clip.id[:8]}: {e}")

        progress.update(1.0, "Complete")

    # Save updated project
    success = save_project(
        filepath=project_file,
        sources=sources,
        clips=clips,
        sequence=sequence,
        ui_state=ui_state,
        metadata=metadata,
    )

    if not success:
        exit_with(ExitCode.GENERAL_ERROR, "Failed to save project")

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
