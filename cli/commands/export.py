"""Export commands for clips, datasets, EDL, and video."""

from pathlib import Path
from typing import Optional

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_success, output_info
from cli.utils.progress import ProgressContext, create_progress_callback


@click.group()
def export() -> None:
    """Export clips and project data.

    \b
    Commands:
        clips     Export individual video clips
        dataset   Export clip metadata as JSON
        edl       Export sequence as EDL for NLE import
        video     Export sequence as a single video file
    """
    pass


@export.command("clips")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for clips",
)
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to export (default: all)",
)
@click.option(
    "--filter",
    "-f",
    "filter_expr",
    help="Filter clips (e.g., 'shot_type=close-up')",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["mp4", "mov", "mkv"]),
    default="mp4",
    help="Output format (default: mp4)",
)
@click.option(
    "--accurate/--fast",
    default=True,
    help="Frame-accurate cutting (slower) vs fast keyframe cutting",
)
@click.pass_context
def clips(
    ctx: click.Context,
    project_file: Path,
    output_dir: Optional[Path],
    clip_ids: tuple[str, ...],
    filter_expr: Optional[str],
    output_format: str,
    accurate: bool,
) -> None:
    """Export individual video clips.

    Extracts each clip as a separate video file.

    \b
    Examples:
        scene_ripper export clips project.json
        scene_ripper export clips project.json -o ./clips/
        scene_ripper export clips project.json --filter shot_type=close-up
        scene_ripper export clips project.json --fast  # Faster but less precise
    """
    try:
        from core.project import load_project, ProjectLoadError
        from core.ffmpeg import FFmpegProcessor
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    # Determine output directory
    if output_dir is None:
        output_dir = config.export_dir / project_file.stem / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load project
    try:
        sources, clips_list, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    sources_by_id = {s.id: s for s in sources}

    # Filter clips
    clips_to_export = clips_list
    if clip_ids:
        clip_set = set(clip_ids)
        clips_to_export = [
            c for c in clips_list if c.id in clip_set or c.id[:8] in clip_set
        ]
    elif filter_expr:
        clips_to_export = _apply_filter(clips_list, filter_expr)

    if not clips_to_export:
        exit_with(ExitCode.VALIDATION_ERROR, "No clips to export")

    # Initialize FFmpeg
    try:
        ffmpeg = FFmpegProcessor()
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    exported_count = 0
    errors = []
    output_paths = []

    with ProgressContext("Exporting clips") as progress:
        total = len(clips_to_export)
        for i, clip in enumerate(clips_to_export):
            progress.update(i / total, f"Clip {i + 1}/{total}")

            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                errors.append(f"Clip {clip.id[:8]}: source not found")
                continue

            try:
                fps = source.fps
                start_time = clip.start_time(fps)
                duration = clip.duration_seconds(fps)

                # Generate output filename
                clip_name = f"clip_{clip.id[:8]}.{output_format}"
                output_path = output_dir / clip_name

                # Export clip
                success = ffmpeg.extract_clip(
                    input_path=source.file_path,
                    output_path=output_path,
                    start_seconds=start_time,
                    duration_seconds=duration,
                    fps=fps,
                    accurate=accurate,
                )

                if success:
                    exported_count += 1
                    output_paths.append(str(output_path))
                else:
                    errors.append(f"Clip {clip.id[:8]}: export failed")

            except Exception as e:
                errors.append(f"Clip {clip.id[:8]}: {e}")

        progress.update(1.0, "Complete")

    result = {
        "exported_clips": exported_count,
        "output_directory": str(output_dir),
        "errors": len(errors),
        "total_clips": len(clips_list),
        "accurate_mode": accurate,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        result["output_files"] = output_paths
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(f"Exported {exported_count} clips to {output_dir}")
        if errors:
            for err in errors[:5]:
                output_info(f"  Error: {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")


@export.command("dataset")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file path",
)
@click.option(
    "--pretty/--compact",
    default=True,
    help="Pretty-print JSON (default: pretty)",
)
@click.pass_context
def dataset(
    ctx: click.Context,
    project_file: Path,
    output: Optional[Path],
    pretty: bool,
) -> None:
    """Export clip metadata as JSON dataset.

    Creates a JSON file with all clip data including timecodes,
    colors, shot types, and transcripts.

    \b
    Examples:
        scene_ripper export dataset project.json
        scene_ripper export dataset project.json -o clips_data.json
        scene_ripper export dataset project.json --compact
    """
    try:
        from core.project import load_project, ProjectLoadError
        from core.dataset_export import export_dataset as do_export, DatasetExportConfig
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    # Determine output path
    if output is None:
        output = config.export_dir / f"{project_file.stem}_dataset.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load project
    try:
        sources, clips_list, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    if not sources:
        exit_with(ExitCode.VALIDATION_ERROR, "No sources in project")

    # Use first source (dataset export currently supports single source)
    source = sources[0]
    source_clips = [c for c in clips_list if c.source_id == source.id]

    # Configure export
    export_config = DatasetExportConfig(
        output_path=output,
        include_thumbnails=False,  # CLI doesn't need thumbnail paths
        pretty_print=pretty,
    )

    progress = create_progress_callback("Exporting dataset")

    success = do_export(
        source=source,
        clips=source_clips,
        config=export_config,
        progress_callback=progress,
    )

    if not success:
        exit_with(ExitCode.GENERAL_ERROR, "Failed to export dataset")

    result = {
        "output": str(output),
        "clips_exported": len(source_clips),
        "source": source.filename,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        output_result(result, as_json=True)
    else:
        output_success(f"Exported dataset to {output}")
        output_result(result, as_json=False)


@export.command("edl")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output EDL file path",
)
@click.option(
    "--title",
    "-t",
    default=None,
    help="EDL title (default: project name)",
)
@click.pass_context
def edl(
    ctx: click.Context,
    project_file: Path,
    output: Optional[Path],
    title: Optional[str],
) -> None:
    """Export sequence as EDL for NLE import.

    Creates a CMX 3600 EDL file that can be imported into
    video editing software like DaVinci Resolve, Premiere, or Final Cut.

    Requires a sequence with clips - use 'project add-to-sequence' first.

    \b
    Examples:
        scene_ripper export edl project.json
        scene_ripper export edl project.json -o timeline.edl
        scene_ripper export edl project.json --title "My Edit"
    """
    try:
        from core.project import load_project, ProjectLoadError
        from core.edl_export import export_edl as do_export, EDLExportConfig
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    # Load project
    try:
        sources, clips_list, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    if sequence is None or not sequence.get_all_clips():
        exit_with(
            ExitCode.VALIDATION_ERROR,
            "No sequence clips to export. Use 'project add-to-sequence' first.",
        )

    # Determine output path
    if output is None:
        output = config.export_dir / f"{project_file.stem}.edl"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Configure export
    export_config = EDLExportConfig(
        output_path=output,
        title=title or metadata.name,
    )

    sources_by_id = {s.id: s for s in sources}

    success = do_export(
        sequence=sequence,
        sources=sources_by_id,
        config=export_config,
    )

    if not success:
        exit_with(ExitCode.GENERAL_ERROR, "Failed to export EDL")

    result = {
        "output": str(output),
        "sequence_clips": len(sequence.get_all_clips()),
        "duration_seconds": sequence.duration_seconds,
        "title": export_config.title,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        output_result(result, as_json=True)
    else:
        output_success(f"Exported EDL to {output}")
        output_result(result, as_json=False)


@export.command("video")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output video file path",
)
@click.option(
    "--quality",
    "-q",
    type=click.Choice(["high", "medium", "low"]),
    default="medium",
    help="Output quality (default: medium)",
)
@click.option(
    "--resolution",
    "-r",
    type=click.Choice(["original", "1080p", "720p", "480p"]),
    default="original",
    help="Output resolution (default: original)",
)
@click.pass_context
def video(
    ctx: click.Context,
    project_file: Path,
    output: Optional[Path],
    quality: str,
    resolution: str,
) -> None:
    """Export sequence as a single video file.

    Renders the timeline sequence to a video file.
    Requires a sequence with clips - use 'project add-to-sequence' first.

    \b
    Examples:
        scene_ripper export video project.json
        scene_ripper export video project.json -o final_cut.mp4
        scene_ripper export video project.json --quality high --resolution 1080p
    """
    try:
        from core.project import load_project, ProjectLoadError
        from core.sequence_export import SequenceExporter, ExportConfig
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    # Load project
    try:
        sources, clips_list, sequence, metadata, ui_state = load_project(
            filepath=project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    if sequence is None or not sequence.get_all_clips():
        exit_with(
            ExitCode.VALIDATION_ERROR,
            "No sequence clips to export. Use 'project add-to-sequence' first.",
        )

    # Determine output path
    if output is None:
        output = config.export_dir / f"{project_file.stem}_export.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Map quality to CRF
    quality_crf = {"high": 15, "medium": 18, "low": 23}
    crf = quality_crf.get(quality, 18)

    # Map resolution to dimensions
    resolution_map = {
        "original": (None, None),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "480p": (854, 480),
    }
    width, height = resolution_map.get(resolution, (None, None))

    # Configure export
    export_config = ExportConfig(
        output_path=output,
        fps=sequence.fps,
        width=width,
        height=height,
        crf=crf,
    )

    # Build clip lookup
    sources_by_id = {s.id: s for s in sources}
    clips_dict = {}
    for clip in clips_list:
        source = sources_by_id.get(clip.source_id)
        if source:
            clips_dict[clip.id] = (clip, source)

    # Initialize exporter
    try:
        exporter = SequenceExporter()
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    progress = create_progress_callback("Exporting video")

    success = exporter.export(
        sequence=sequence,
        sources=sources_by_id,
        clips=clips_dict,
        config=export_config,
        progress_callback=progress,
    )

    if not success:
        exit_with(ExitCode.GENERAL_ERROR, "Failed to export video")

    result = {
        "output": str(output),
        "sequence_clips": len(sequence.get_all_clips()),
        "duration_seconds": sequence.duration_seconds,
        "quality": quality,
        "resolution": resolution,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        output_result(result, as_json=True)
    else:
        output_success(f"Exported video to {output}")
        output_result(result, as_json=False)


def _apply_filter(clips: list, filter_expr: str) -> list:
    """Apply a filter expression to clips."""
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
