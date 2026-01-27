"""Scene detection command."""

from pathlib import Path

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_success
from cli.utils.progress import create_progress_callback
from cli.utils.signals import (
    setup_signal_handlers,
    restore_default_handlers,
    GracefulExit,
    ProgressCheckpoint,
)


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sensitivity",
    "-s",
    type=float,
    default=None,
    help="Detection sensitivity (1.0=sensitive, 10.0=less sensitive)",
)
@click.option(
    "--min-scene-length",
    "-m",
    type=float,
    default=0.5,
    help="Minimum scene length in seconds",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output project file path (default: <video>.json)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing output file",
)
@click.pass_context
def detect(
    ctx: click.Context,
    video: Path,
    sensitivity: float | None,
    min_scene_length: float,
    output: Path | None,
    force: bool,
) -> None:
    """Detect scenes in a video file.

    Creates a project file with detected clips that can be used
    with other scene_ripper commands.

    \b
    Examples:
        scene_ripper detect movie.mp4
        scene_ripper detect movie.mp4 -s 5.0 -o my_project.json
        scene_ripper detect movie.mp4 --min-scene-length 1.0
    """
    # Load config for defaults
    config = CLIConfig.load()

    # Use default sensitivity if not specified
    if sensitivity is None:
        sensitivity = config.default_sensitivity

    # Validate sensitivity range
    if not 1.0 <= sensitivity <= 10.0:
        exit_with(
            ExitCode.VALIDATION_ERROR,
            f"Sensitivity must be between 1.0 and 10.0, got {sensitivity}",
        )

    # Determine output path
    if output is None:
        output = video.with_suffix(".json")

    # Check for existing file
    if output.exists() and not force:
        exit_with(
            ExitCode.VALIDATION_ERROR,
            f"Output file exists: {output}. Use --force to overwrite.",
        )

    # Import heavy dependencies only when needed (keeps CLI startup fast)
    try:
        from core.scene_detect import SceneDetector, DetectionConfig
        from core.project import Project
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    # Create detector and config
    fps_estimate = 30.0  # Will be updated after opening video
    detection_config = DetectionConfig(
        threshold=sensitivity,
        min_scene_length=int(min_scene_length * fps_estimate),
        use_adaptive=True,
    )

    detector = SceneDetector(config=detection_config)
    progress = create_progress_callback("Detecting scenes")

    # Set up signal handling for graceful shutdown
    checkpoint_path = output.with_suffix(".checkpoint.json")
    checkpoint = ProgressCheckpoint(checkpoint_path)
    setup_signal_handlers(checkpoint)

    try:
        # Run detection
        source, clips = detector.detect_scenes_with_progress(
            video_path=video,
            progress_callback=progress,
        )

        # Update min_scene_length based on actual FPS
        detection_config.min_scene_length = int(min_scene_length * source.fps)

        # Create project using Project class
        project = Project.new(name=output.stem)
        project.add_source(source)
        project.add_clips(clips)

        if not project.save(output):
            exit_with(ExitCode.GENERAL_ERROR, "Failed to save project file")

        # Clean up checkpoint file on successful completion
        checkpoint.clear()

        # Output result
        result = {
            "video": str(video.resolve()),
            "output": str(output.resolve()),
            "clips_detected": len(clips),
            "sensitivity": sensitivity,
            "duration_seconds": source.duration_seconds,
            "fps": source.fps,
            "resolution": f"{source.width}x{source.height}",
        }

        as_json = ctx.obj.get("json", False)
        if as_json:
            output_result(result, as_json=True)
        else:
            output_success(f"Detected {len(clips)} scenes in {video.name}")
            output_result(result, as_json=False)

    except GracefulExit:
        click.echo("\nDetection interrupted.", err=True)
        exit_with(ExitCode.GENERAL_ERROR, "Detection was interrupted by user")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Video not found: {video}")
    except Exception as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Detection failed: {e}")
    finally:
        restore_default_handlers()
