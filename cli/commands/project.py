"""Project management commands."""

from pathlib import Path
from typing import Optional

import click

from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_table, output_success


@click.group()
def project() -> None:
    """Manage Scene Ripper projects.

    \b
    Commands:
        info          Show project information
        list-clips    List all clips in a project
        add-to-sequence  Add clips to the timeline sequence
    """
    pass


@project.command("info")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def info(ctx: click.Context, project_file: Path) -> None:
    """Show project information.

    Displays metadata about a project file including source videos,
    clip counts, and sequence status.

    \b
    Examples:
        scene_ripper project info my_project.json
        scene_ripper --json project info my_project.json
    """
    try:
        from core.project import Project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        proj = Project.load(
            project_file,
            missing_source_callback=lambda path, sid: None,  # Skip missing sources
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Build result
    result = {
        "project_name": proj.metadata.name,
        "project_id": proj.metadata.id,
        "version": proj.metadata.version,
        "created_at": proj.metadata.created_at,
        "modified_at": proj.metadata.modified_at,
        "source_count": len(proj.sources),
        "clip_count": len(proj.clips),
        "sequence_clips": len(proj.sequence.get_all_clips()) if proj.sequence else 0,
        "sequence_duration_seconds": proj.sequence.duration_seconds if proj.sequence else 0,
    }

    # Add source details
    if proj.sources:
        result["sources"] = [
            {
                "id": s.id,
                "filename": s.filename,
                "duration_seconds": s.duration_seconds,
                "fps": s.fps,
                "resolution": f"{s.width}x{s.height}",
                "exists": s.file_path.exists(),
            }
            for s in proj.sources
        ]

    as_json = ctx.obj.get("json", False)
    output_result(result, as_json=as_json)


@project.command("list-clips")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--filter",
    "-f",
    "filter_expr",
    help="Filter clips (e.g., 'shot_type=close-up', 'has_transcript=true')",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Limit number of results",
)
@click.option(
    "--duration-min",
    type=float,
    default=None,
    help="Minimum duration in seconds",
)
@click.option(
    "--duration-max",
    type=float,
    default=None,
    help="Maximum duration in seconds",
)
@click.option(
    "--aspect-ratio",
    type=click.Choice(["16:9", "4:3", "9:16"]),
    default=None,
    help="Filter by aspect ratio",
)
@click.pass_context
def list_clips(
    ctx: click.Context,
    project_file: Path,
    filter_expr: Optional[str],
    limit: Optional[int],
    duration_min: Optional[float],
    duration_max: Optional[float],
    aspect_ratio: Optional[str],
) -> None:
    """List all clips in a project.

    Shows clip IDs, timecodes, and analysis results.

    \b
    Examples:
        scene_ripper project list-clips project.json
        scene_ripper project list-clips project.json --filter shot_type=close-up
        scene_ripper project list-clips project.json -n 10
        scene_ripper project list-clips project.json --duration-min 2.0 --duration-max 10.0
        scene_ripper project list-clips project.json --aspect-ratio 16:9
    """
    try:
        from core.project import Project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        proj = Project.load(
            project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Apply filter if specified
    clips = proj.clips
    filtered_clips = clips
    if filter_expr:
        filtered_clips = _apply_filter(clips, filter_expr)

    # Apply duration filter
    if duration_min is not None or duration_max is not None:
        filtered_clips = _apply_duration_filter(
            filtered_clips, proj, duration_min, duration_max
        )

    # Apply aspect ratio filter
    if aspect_ratio:
        filtered_clips = _apply_aspect_ratio_filter(filtered_clips, proj, aspect_ratio)

    # Apply limit
    if limit:
        filtered_clips = filtered_clips[:limit]

    as_json = ctx.obj.get("json", False)

    if as_json:
        # JSON output with full details
        result = {
            "total_clips": len(clips),
            "filtered_clips": len(filtered_clips),
            "clips": [],
        }
        for clip in filtered_clips:
            source = proj.sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0
            clip_data = {
                "id": clip.id,
                "source_id": clip.source_id,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "start_time": clip.start_time(fps),
                "end_time": clip.end_time(fps),
                "duration_seconds": clip.duration_seconds(fps),
                "shot_type": clip.shot_type,
                "has_colors": clip.dominant_colors is not None,
                "has_transcript": clip.transcript is not None,
            }
            if clip.dominant_colors:
                clip_data["colors"] = [
                    {"r": r, "g": g, "b": b} for r, g, b in clip.dominant_colors
                ]
            if clip.transcript:
                clip_data["transcript"] = clip.get_transcript_text()
            result["clips"].append(clip_data)
        output_result(result, as_json=True)
    else:
        # Table output
        headers = ["ID", "Start", "End", "Duration", "Shot Type", "Transcript"]
        rows = []
        for clip in filtered_clips:
            source = proj.sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0
            start = _format_time(clip.start_time(fps))
            end = _format_time(clip.end_time(fps))
            duration = f"{clip.duration_seconds(fps):.1f}s"
            shot = clip.shot_type or "-"
            transcript = "Yes" if clip.transcript else "-"
            rows.append([clip.id[:8], start, end, duration, shot, transcript])

        click.echo(f"Found {len(filtered_clips)} clips (of {len(clips)} total)")
        output_table(headers, rows)


@project.command("add-to-sequence")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.argument("clip_ids", nargs=-1)
@click.option(
    "--all",
    "add_all",
    is_flag=True,
    help="Add all clips to sequence",
)
@click.option(
    "--filter",
    "-f",
    "filter_expr",
    help="Add clips matching filter (e.g., 'shot_type=close-up')",
)
@click.option(
    "--duration-min",
    type=float,
    default=None,
    help="Minimum duration in seconds",
)
@click.option(
    "--duration-max",
    type=float,
    default=None,
    help="Maximum duration in seconds",
)
@click.option(
    "--aspect-ratio",
    type=click.Choice(["16:9", "4:3", "9:16"]),
    default=None,
    help="Filter by aspect ratio",
)
@click.pass_context
def add_to_sequence(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    add_all: bool,
    filter_expr: Optional[str],
    duration_min: Optional[float],
    duration_max: Optional[float],
    aspect_ratio: Optional[str],
) -> None:
    """Add clips to the timeline sequence.

    Clips are added in the order specified. Use --all to add all clips,
    or --filter to add clips matching criteria.

    \b
    Examples:
        scene_ripper project add-to-sequence project.json clip1 clip2
        scene_ripper project add-to-sequence project.json --all
        scene_ripper project add-to-sequence project.json --filter shot_type=wide
        scene_ripper project add-to-sequence project.json --duration-min 2.0 --aspect-ratio 16:9
    """
    has_filters = filter_expr or duration_min or duration_max or aspect_ratio
    if not clip_ids and not add_all and not has_filters:
        exit_with(
            ExitCode.USAGE_ERROR,
            "Specify clip IDs, --all, or --filter to select clips",
        )

    try:
        from core.project import Project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        proj = Project.load(
            project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Build clip lookup (also allow matching by prefix)
    clips_by_id = dict(proj.clips_by_id)
    for c in proj.clips:
        clips_by_id[c.id[:8]] = c

    # Determine which clips to add
    if add_all:
        clips_to_add = proj.clips
    elif has_filters and not clip_ids:
        # Use filters to select clips
        clips_to_add = proj.clips
        if filter_expr:
            clips_to_add = _apply_filter(clips_to_add, filter_expr)
    else:
        clips_to_add = []
        for cid in clip_ids:
            if cid in clips_by_id:
                clips_to_add.append(clips_by_id[cid])
            else:
                exit_with(ExitCode.VALIDATION_ERROR, f"Clip not found: {cid}")

    # Apply duration and aspect ratio filters
    if duration_min is not None or duration_max is not None:
        clips_to_add = _apply_duration_filter(clips_to_add, proj, duration_min, duration_max)
    if aspect_ratio:
        clips_to_add = _apply_aspect_ratio_filter(clips_to_add, proj, aspect_ratio)

    if not clips_to_add:
        exit_with(ExitCode.VALIDATION_ERROR, "No clips to add")

    # Use Project's add_to_sequence method
    clip_ids_to_add = [c.id for c in clips_to_add]
    proj.add_to_sequence(clip_ids_to_add)

    # Save updated project
    if not proj.save():
        exit_with(ExitCode.GENERAL_ERROR, "Failed to save project")

    result = {
        "added_clips": len(clips_to_add),
        "total_sequence_clips": len(proj.sequence.get_all_clips()) if proj.sequence else 0,
        "sequence_duration_seconds": proj.sequence.duration_seconds if proj.sequence else 0,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        output_result(result, as_json=True)
    else:
        output_success(f"Added {len(clips_to_add)} clips to sequence")
        output_result(result, as_json=False)


def _apply_filter(clips: list, filter_expr: str) -> list:
    """Apply a filter expression to clips.

    Supported filters:
        shot_type=<value>
        has_transcript=true|false
        has_colors=true|false

    Args:
        clips: List of Clip objects
        filter_expr: Filter expression string

    Returns:
        Filtered list of clips
    """
    if "=" not in filter_expr:
        return clips

    key, value = filter_expr.split("=", 1)
    key = key.strip().lower()
    value = value.strip().lower()

    filtered = []
    for clip in clips:
        if key == "shot_type":
            if clip.shot_type and value in clip.shot_type.lower():
                filtered.append(clip)
        elif key == "has_transcript":
            has_it = clip.transcript is not None
            if (value == "true" and has_it) or (value == "false" and not has_it):
                filtered.append(clip)
        elif key == "has_colors":
            has_it = clip.dominant_colors is not None
            if (value == "true" and has_it) or (value == "false" and not has_it):
                filtered.append(clip)

    return filtered


def _format_time(seconds: float) -> str:
    """Format time in MM:SS or HH:MM:SS format."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# Aspect ratio tolerance ranges (5% tolerance)
ASPECT_RATIO_RANGES = {
    "16:9": (1.69, 1.87),   # 1.778 ± 5%
    "4:3": (1.27, 1.40),     # 1.333 ± 5%
    "9:16": (0.53, 0.59),    # 0.5625 ± 5%
}


def _apply_duration_filter(
    clips: list,
    proj,
    min_duration: Optional[float],
    max_duration: Optional[float],
) -> list:
    """Filter clips by duration.

    Args:
        clips: List of Clip objects
        proj: Project object (for source fps lookup)
        min_duration: Minimum duration in seconds (None = no minimum)
        max_duration: Maximum duration in seconds (None = no maximum)

    Returns:
        Filtered list of clips
    """
    if min_duration is None and max_duration is None:
        return clips

    filtered = []
    for clip in clips:
        source = proj.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0
        duration = clip.duration_seconds(fps)

        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue

        filtered.append(clip)

    return filtered


def _apply_aspect_ratio_filter(
    clips: list,
    proj,
    aspect_ratio: str,
) -> list:
    """Filter clips by aspect ratio.

    Args:
        clips: List of Clip objects
        proj: Project object (for source dimension lookup)
        aspect_ratio: Aspect ratio string ('16:9', '4:3', '9:16')

    Returns:
        Filtered list of clips
    """
    if aspect_ratio not in ASPECT_RATIO_RANGES:
        return clips

    min_ratio, max_ratio = ASPECT_RATIO_RANGES[aspect_ratio]
    filtered = []

    for clip in clips:
        source = proj.sources_by_id.get(clip.source_id)
        if not source or source.width == 0 or source.height == 0:
            continue  # Skip clips without dimensions

        source_aspect = source.width / source.height
        if min_ratio <= source_aspect <= max_ratio:
            filtered.append(clip)

    return filtered
