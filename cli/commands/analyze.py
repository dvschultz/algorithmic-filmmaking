"""Analysis commands for color extraction and shot classification."""

from pathlib import Path
from typing import Optional

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_success, output_info
from cli.utils.progress import ProgressContext


@click.group()
def analyze() -> None:
    """Analyze clips in a project.

    \b
    Commands:
        colors    Extract dominant colors from clips
        shots     Classify shot types (wide, medium, close-up)
    """
    pass


@analyze.command("colors")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to analyze (default: all)",
)
@click.option(
    "--num-colors",
    "-n",
    type=int,
    default=5,
    help="Number of dominant colors to extract (default: 5)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-analyze clips that already have color data",
)
@click.pass_context
def colors(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    num_colors: int,
    force: bool,
) -> None:
    """Extract dominant colors from clips.

    Uses k-means clustering to find the most prominent colors
    in each clip's representative frame.

    \b
    Examples:
        scene_ripper analyze colors project.json
        scene_ripper analyze colors project.json --num-colors 3
        scene_ripper analyze colors project.json -c clip1 -c clip2
    """
    try:
        from core.project import load_project, save_project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.analysis.color import extract_dominant_colors
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

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
    clips_to_analyze = clips
    if clip_ids:
        clip_set = set(clip_ids)
        # Also match by prefix
        clips_to_analyze = [
            c for c in clips if c.id in clip_set or c.id[:8] in clip_set
        ]
        if not clips_to_analyze:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")

    # Filter out already-analyzed clips unless force
    if not force:
        clips_to_analyze = [c for c in clips_to_analyze if c.dominant_colors is None]

    if not clips_to_analyze:
        output_info("All clips already have color data. Use --force to re-analyze.")
        return

    # Initialize thumbnail generator
    try:
        thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    analyzed_count = 0
    errors = []

    with ProgressContext("Analyzing colors") as progress:
        total = len(clips_to_analyze)
        for i, clip in enumerate(clips_to_analyze):
            progress.update(i / total, f"Clip {i + 1}/{total}")

            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                errors.append(f"Clip {clip.id[:8]}: source not found")
                continue

            try:
                # Generate thumbnail for analysis
                fps = source.fps
                start_time = clip.start_time(fps)
                end_time = clip.end_time(fps)

                thumb_path = thumb_gen.generate_clip_thumbnail(
                    video_path=source.file_path,
                    start_seconds=start_time,
                    end_seconds=end_time,
                    width=320,  # Higher resolution for color analysis
                    height=180,
                )

                # Extract colors
                clip.dominant_colors = extract_dominant_colors(
                    image_path=thumb_path,
                    n_colors=num_colors,
                )
                analyzed_count += 1

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
        "analyzed_clips": analyzed_count,
        "errors": len(errors),
        "total_clips": len(clips),
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(f"Analyzed colors for {analyzed_count} clips")
        if errors:
            for err in errors[:5]:
                output_info(f"  {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")


@analyze.command("shots")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to analyze (default: all)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-analyze clips that already have shot type data",
)
@click.pass_context
def shots(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    force: bool,
) -> None:
    """Classify shot types in clips.

    Uses CLIP zero-shot classification to identify shot types:
    wide shot, medium shot, close-up, extreme close-up.

    Note: First run will download the CLIP model (~600MB).

    \b
    Examples:
        scene_ripper analyze shots project.json
        scene_ripper analyze shots project.json --force
        scene_ripper analyze shots project.json -c clip1 -c clip2
    """
    try:
        from core.project import load_project, save_project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.analysis.shots import classify_shot_type
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

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
    clips_to_analyze = clips
    if clip_ids:
        clip_set = set(clip_ids)
        clips_to_analyze = [
            c for c in clips if c.id in clip_set or c.id[:8] in clip_set
        ]
        if not clips_to_analyze:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")

    # Filter out already-analyzed clips unless force
    if not force:
        clips_to_analyze = [c for c in clips_to_analyze if c.shot_type is None]

    if not clips_to_analyze:
        output_info("All clips already have shot type data. Use --force to re-analyze.")
        return

    # Initialize thumbnail generator
    try:
        thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    analyzed_count = 0
    errors = []
    shot_counts: dict[str, int] = {}

    output_info("Loading CLIP model (this may take a moment on first run)...")

    with ProgressContext("Classifying shots") as progress:
        total = len(clips_to_analyze)
        for i, clip in enumerate(clips_to_analyze):
            progress.update(i / total, f"Clip {i + 1}/{total}")

            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                errors.append(f"Clip {clip.id[:8]}: source not found")
                continue

            try:
                # Generate thumbnail for analysis
                fps = source.fps
                start_time = clip.start_time(fps)
                end_time = clip.end_time(fps)

                thumb_path = thumb_gen.generate_clip_thumbnail(
                    video_path=source.file_path,
                    start_seconds=start_time,
                    end_seconds=end_time,
                    width=320,
                    height=180,
                )

                # Classify shot type
                shot_type, confidence = classify_shot_type(image_path=thumb_path)
                clip.shot_type = shot_type
                analyzed_count += 1

                # Track counts
                shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1

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
        "analyzed_clips": analyzed_count,
        "errors": len(errors),
        "total_clips": len(clips),
        "shot_types": shot_counts,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(f"Classified shot types for {analyzed_count} clips")
        if shot_counts:
            for shot_type, count in sorted(shot_counts.items()):
                click.echo(f"  {shot_type}: {count}")
        if errors:
            for err in errors[:5]:
                output_info(f"  Error: {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")
