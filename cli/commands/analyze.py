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


@analyze.command("describe")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to analyze (default: all)",
)
@click.option(
    "--tier",
    type=click.Choice(["cpu", "gpu", "cloud"]),
    help="Model tier to use (overrides settings)",
)
@click.option(
    "--prompt",
    help="Custom prompt for the model",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-describe clips that already have descriptions",
)
@click.pass_context
def describe(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    tier: Optional[str],
    prompt: Optional[str],
    force: bool,
) -> None:
    """Generate descriptions for clips using Vision-Language Models.

    Generates natural language descriptions for video frames using
    either local CPU models (Moondream), GPU models, or Cloud APIs.

    \b
    Examples:
        scene_ripper analyze describe project.json
        scene_ripper analyze describe project.json --tier cloud
        scene_ripper analyze describe project.json --prompt "Describe the lighting"
    """
    try:
        from core.project import load_project, save_project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.analysis.description import describe_frame
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
        clips_to_analyze = [c for c in clips_to_analyze if c.description is None]

    if not clips_to_analyze:
        output_info("All clips already have descriptions. Use --force to re-analyze.")
        return

    # Initialize thumbnail generator
    try:
        thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    analyzed_count = 0
    errors = []

    # Prepare prompt
    default_prompt = (
        "Describe this video frame in detail. Focus on main subjects, "
        "action, setting, and mood."
    )
    final_prompt = prompt or default_prompt
    
    output_info(f"Generating descriptions using tier: {tier or 'default'}")

    with ProgressContext("Generating descriptions") as progress:
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
                    width=640,  # Higher resolution for VLMs
                    height=360,
                )

                # Generate description
                description, model_name = describe_frame(
                    image_path=thumb_path,
                    tier=tier,
                    prompt=final_prompt,
                )
                
                # Check if result is an error message
                if description.startswith("Error"):
                    errors.append(f"Clip {clip.id[:8]}: {description}")
                    continue
                    
                clip.description = description
                clip.description_model = model_name
                clip.description_frames = 1
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
        output_success(f"Generated descriptions for {analyzed_count} clips")
        if errors:
            for err in errors[:5]:
                output_info(f"  {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")


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


@analyze.command("classify")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to analyze (default: all)",
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="Number of top labels to return per clip (default: 5)",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.1,
    help="Minimum confidence threshold (default: 0.1)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-analyze clips that already have classification data",
)
@click.pass_context
def classify(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    top_k: int,
    threshold: float,
    force: bool,
) -> None:
    """Classify frame content using ImageNet labels.

    Uses MobileNetV3-Small to identify objects in each clip's
    representative frame. Labels are from ImageNet (1000 categories).

    Note: First run will download the MobileNet model (~20MB).

    \b
    Examples:
        scene_ripper analyze classify project.json
        scene_ripper analyze classify project.json --top-k 3
        scene_ripper analyze classify project.json -c clip1 -c clip2 --force
    """
    try:
        from core.project import load_project, save_project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.analysis.classification import classify_frame
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
        clips_to_analyze = [c for c in clips_to_analyze if c.object_labels is None]

    if not clips_to_analyze:
        output_info("All clips already have classification data. Use --force to re-analyze.")
        return

    # Initialize thumbnail generator
    try:
        thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    analyzed_count = 0
    errors = []
    label_counts: dict[str, int] = {}

    output_info("Loading MobileNet model (this may take a moment on first run)...")

    with ProgressContext("Classifying content") as progress:
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

                # Classify content
                results = classify_frame(
                    image_path=thumb_path,
                    top_k=top_k,
                    threshold=threshold,
                )
                clip.object_labels = [label for label, _ in results]
                analyzed_count += 1

                # Track top label counts
                if results:
                    top_label = results[0][0]
                    label_counts[top_label] = label_counts.get(top_label, 0) + 1

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
        "top_labels": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:10]),
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(f"Classified content for {analyzed_count} clips")
        if label_counts:
            click.echo("Top labels:")
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
                click.echo(f"  {label}: {count}")
        if errors:
            for err in errors[:5]:
                output_info(f"  Error: {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")


@analyze.command("objects")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to analyze (default: all)",
)
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="Detection confidence threshold (default: 0.5)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-analyze clips that already have detection data",
)
@click.pass_context
def objects(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    confidence: float,
    force: bool,
) -> None:
    """Detect objects in clips using YOLOv8.

    Uses YOLOv8-nano to detect objects from COCO dataset (80 classes).
    Provides bounding boxes and person counts.

    Note: First run will download the YOLO model (~6MB).

    \b
    Examples:
        scene_ripper analyze objects project.json
        scene_ripper analyze objects project.json --confidence 0.3
        scene_ripper analyze objects project.json -c clip1 --force
    """
    try:
        from core.project import load_project, save_project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.analysis.detection import detect_objects
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
        clips_to_analyze = [c for c in clips_to_analyze if c.detected_objects is None]

    if not clips_to_analyze:
        output_info("All clips already have detection data. Use --force to re-analyze.")
        return

    # Initialize thumbnail generator
    try:
        thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    analyzed_count = 0
    errors = []
    object_counts: dict[str, int] = {}
    total_people = 0

    output_info("Loading YOLO model (this may take a moment on first run)...")

    with ProgressContext("Detecting objects") as progress:
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

                # Detect objects
                detections = detect_objects(
                    image_path=thumb_path,
                    confidence_threshold=confidence,
                )
                clip.detected_objects = detections
                clip.person_count = sum(1 for d in detections if d["label"] == "person")
                total_people += clip.person_count
                analyzed_count += 1

                # Track object counts
                for det in detections:
                    label = det["label"]
                    object_counts[label] = object_counts.get(label, 0) + 1

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
        "total_people_detected": total_people,
        "object_counts": dict(sorted(object_counts.items(), key=lambda x: -x[1])[:15]),
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(f"Detected objects in {analyzed_count} clips")
        click.echo(f"  Total people detected: {total_people}")
        if object_counts:
            click.echo("Object counts:")
            for label, count in sorted(object_counts.items(), key=lambda x: -x[1])[:15]:
                click.echo(f"  {label}: {count}")
        if errors:
            for err in errors[:5]:
                output_info(f"  Error: {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")


@analyze.command("people")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clip",
    "-c",
    "clip_ids",
    multiple=True,
    help="Specific clip IDs to analyze (default: all)",
)
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="Detection confidence threshold (default: 0.5)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-analyze clips that already have person count data",
)
@click.pass_context
def people(
    ctx: click.Context,
    project_file: Path,
    clip_ids: tuple[str, ...],
    confidence: float,
    force: bool,
) -> None:
    """Count people in clips using YOLOv8.

    Faster than full object detection when you only need person counts.
    Uses YOLOv8-nano filtered to detect only people.

    Note: First run will download the YOLO model (~6MB).

    \b
    Examples:
        scene_ripper analyze people project.json
        scene_ripper analyze people project.json --confidence 0.3
        scene_ripper analyze people project.json -c clip1 --force
    """
    try:
        from core.project import load_project, save_project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.analysis.detection import count_people
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
        clips_to_analyze = [c for c in clips_to_analyze if c.person_count is None]

    if not clips_to_analyze:
        output_info("All clips already have person count data. Use --force to re-analyze.")
        return

    # Initialize thumbnail generator
    try:
        thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
    except RuntimeError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, str(e))

    analyzed_count = 0
    errors = []
    total_people = 0
    person_distribution: dict[int, int] = {}  # count -> number of clips

    output_info("Loading YOLO model (this may take a moment on first run)...")

    with ProgressContext("Counting people") as progress:
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

                # Count people
                person_count = count_people(
                    image_path=thumb_path,
                    confidence_threshold=confidence,
                )
                clip.person_count = person_count
                total_people += person_count
                analyzed_count += 1

                # Track distribution
                person_distribution[person_count] = person_distribution.get(person_count, 0) + 1

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
        "total_people_detected": total_people,
        "distribution": person_distribution,
    }

    as_json = ctx.obj.get("json", False)
    if as_json:
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        output_success(f"Counted people in {analyzed_count} clips")
        click.echo(f"  Total people detected: {total_people}")
        if person_distribution:
            click.echo("Distribution:")
            for count, num_clips in sorted(person_distribution.items()):
                label = "person" if count == 1 else "people"
                click.echo(f"  {count} {label}: {num_clips} clips")
        if errors:
            for err in errors[:5]:
                output_info(f"  Error: {err}")
            if len(errors) > 5:
                output_info(f"  ... and {len(errors) - 5} more errors")
