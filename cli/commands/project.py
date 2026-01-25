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
        from core.project import load_project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        sources, clips, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,  # Skip missing sources
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Build result
    result = {
        "project_name": metadata.name,
        "project_id": metadata.id,
        "version": metadata.version,
        "created_at": metadata.created_at,
        "modified_at": metadata.modified_at,
        "source_count": len(sources),
        "clip_count": len(clips),
        "sequence_clips": len(sequence.get_all_clips()) if sequence else 0,
        "sequence_duration_seconds": sequence.duration_seconds if sequence else 0,
    }

    # Add source details
    if sources:
        result["sources"] = [
            {
                "id": s.id,
                "filename": s.filename,
                "duration_seconds": s.duration_seconds,
                "fps": s.fps,
                "resolution": f"{s.width}x{s.height}",
                "exists": s.file_path.exists(),
            }
            for s in sources
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
@click.pass_context
def list_clips(
    ctx: click.Context,
    project_file: Path,
    filter_expr: Optional[str],
    limit: Optional[int],
) -> None:
    """List all clips in a project.

    Shows clip IDs, timecodes, and analysis results.

    \b
    Examples:
        scene_ripper project list-clips project.json
        scene_ripper project list-clips project.json --filter shot_type=close-up
        scene_ripper project list-clips project.json -n 10
    """
    try:
        from core.project import load_project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        sources, clips, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Build source lookup
    sources_by_id = {s.id: s for s in sources}

    # Apply filter if specified
    filtered_clips = clips
    if filter_expr:
        filtered_clips = _apply_filter(clips, filter_expr)

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
            source = sources_by_id.get(clip.source_id)
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
            source = sources_by_id.get(clip.source_id)
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
@click.pass_context
def add_to_sequence(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    add_all: bool,
    filter_expr: Optional[str],
) -> None:
    """Add clips to the timeline sequence.

    Clips are added in the order specified. Use --all to add all clips,
    or --filter to add clips matching criteria.

    \b
    Examples:
        scene_ripper project add-to-sequence project.json clip1 clip2
        scene_ripper project add-to-sequence project.json --all
        scene_ripper project add-to-sequence project.json --filter shot_type=wide
    """
    if not clip_ids and not add_all and not filter_expr:
        exit_with(
            ExitCode.USAGE_ERROR,
            "Specify clip IDs, --all, or --filter to select clips",
        )

    try:
        from core.project import load_project, save_project, ProjectLoadError
        from models.sequence import Sequence, SequenceClip
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        sources, clips, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    # Build clip lookup
    clips_by_id = {c.id: c for c in clips}
    # Also allow matching by prefix
    for c in clips:
        clips_by_id[c.id[:8]] = c

    sources_by_id = {s.id: s for s in sources}

    # Determine which clips to add
    if add_all:
        clips_to_add = clips
    elif filter_expr:
        clips_to_add = _apply_filter(clips, filter_expr)
    else:
        clips_to_add = []
        for cid in clip_ids:
            if cid in clips_by_id:
                clips_to_add.append(clips_by_id[cid])
            else:
                exit_with(ExitCode.VALIDATION_ERROR, f"Clip not found: {cid}")

    if not clips_to_add:
        exit_with(ExitCode.VALIDATION_ERROR, "No clips to add")

    # Create or update sequence
    if sequence is None:
        # Determine FPS from first source
        first_source = sources[0] if sources else None
        fps = first_source.fps if first_source else 30.0
        sequence = Sequence(name=metadata.name, fps=fps)

    # Calculate starting position (end of current sequence)
    current_frame = sequence.duration_frames

    # Add clips to sequence
    track = sequence.tracks[0]  # Use first track
    added_count = 0

    for clip in clips_to_add:
        source = sources_by_id.get(clip.source_id)
        if not source:
            continue

        seq_clip = SequenceClip(
            source_clip_id=clip.id,
            source_id=clip.source_id,
            track_index=0,
            start_frame=current_frame,
            in_point=clip.start_frame,
            out_point=clip.end_frame,
        )
        track.add_clip(seq_clip)
        current_frame += seq_clip.duration_frames
        added_count += 1

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
        "added_clips": added_count,
        "total_sequence_clips": len(sequence.get_all_clips()),
        "sequence_duration_seconds": sequence.duration_seconds,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        output_result(result, as_json=True)
    else:
        output_success(f"Added {added_count} clips to sequence")
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
