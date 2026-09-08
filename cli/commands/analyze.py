"""Analysis commands for color extraction and shot classification."""

from pathlib import Path
from typing import Optional

import click

from cli.utils.config import CLIConfig
from cli.utils.errors import ExitCode, exit_with
from cli.utils.output import output_result, output_success, output_info
from cli.utils.progress import ProgressContext
from cli.utils.project_writer import own_project
from core.operations.legacy_reuse import LEGACY_REUSE_OPERATIONS


@click.group()
def analyze() -> None:
    """Analyze clips in a project.

    \b
    Commands:
        colors    Extract dominant colors from clips
        shots     Classify shot types (wide, medium, close-up)
        align     Add word timestamps to existing transcripts
        faces     Extract face embeddings with resumable results
        gaze      Estimate gaze direction with resumable results
        extract-text Extract visible text with resumable results
        embeddings Extract thumbnail embeddings with resumable results
        boundary-embeddings Extract first/last-frame embeddings with resumable results
        scalars   Measure brightness or volume with resumable results
    """
    pass


@analyze.command("accept-legacy")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--operation", type=click.Choice(LEGACY_REUSE_OPERATIONS), required=True)
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Exact clip ID (default: all clips)")
@click.pass_context
def accept_legacy(ctx: click.Context, project_file: Path, operation: str, clip_ids: tuple[str, ...]) -> None:
    """Explicitly reuse old values for current inputs without recomputing them.

    Provenance stays unknown: this decision does not verify how the old values
    were computed. Changed inputs invalidate the decision. Embeddings require
    a compatible recorded model; otherwise recompute them.
    """
    from core.project import Project
    from core.spine.analysis_reuse import accept_legacy_analysis

    path = own_project(ctx, project_file)
    project = None
    try:
        project = Project.load(path)
        result = accept_legacy_analysis(project, operation, list(clip_ids) or None)
        if result["accepted"] and not project.save():
            raise RuntimeError("Failed to save legacy reuse decisions")
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, str(exc))
    finally:
        if project is not None:
            project.close_writer()
    output_result(result, as_json=(ctx.obj or {}).get("json", False))


@analyze.command("scalars")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--operation", "scalar_kind", type=click.Choice(["brightness", "volume"]), required=True)
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Exact clip ID (default: all clips)")
@click.option("--num-samples", type=click.IntRange(min=1), default=5, show_default=True, help="Brightness sample count; volume analyzes the full clip")
@click.option("--force", "-f", is_flag=True, help="Recompute existing scalar results")
@click.pass_context
def scalars(ctx: click.Context, project_file: Path, scalar_kind: str, clip_ids: tuple[str, ...], num_samples: int, force: bool) -> None:
    """Measure brightness or volume and recover interrupted computations."""
    from threading import Event
    from core.jobs.scalars import run_scalar_job
    from core.jobs.store import JobStore

    project_file = own_project(ctx, project_file)
    try:
        store = JobStore(CLIConfig.load().cache_dir / "jobs.db")
        try:
            with ProgressContext(f"Measuring {scalar_kind}") as progress:
                result = run_scalar_job(
                    store, project_file, list(clip_ids) if clip_ids else None,
                    progress.update, Event(), kind="brightness" if scalar_kind == "brightness" else "volume",
                    num_samples=num_samples, force=force,
                )
        finally:
            store.close()
    except ValueError as exc:
        exit_with(ExitCode.VALIDATION_ERROR, str(exc))
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Scalar analysis failed: {exc}")
    output_result(result, as_json=ctx.obj.get("json", False))


@analyze.command("extract-text")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Exact clip ID (default: all clips)")
@click.option("--num-keyframes", type=click.IntRange(1, 5), default=3, show_default=True)
@click.option("--method", type=click.Choice(["paddleocr", "hybrid", "vlm"]), default="hybrid", show_default=True)
@click.option("--model", default=None, help="VLM model override")
@click.option("--force", "-f", is_flag=True, help="Replace existing OCR results")
@click.pass_context
def extract_text(ctx: click.Context, project_file: Path, clip_ids: tuple[str, ...], num_keyframes: int, method: str, model: str | None, force: bool) -> None:
    """Extract visible text and recover completed inference after interruptions."""
    from threading import Event
    from core.jobs.ocr import run_ocr_job
    from core.jobs.store import JobStore
    from core.operations.ocr import OcrOptions

    project_file = own_project(ctx, project_file)
    try:
        store = JobStore(CLIConfig.load().cache_dir / "jobs.db")
        try:
            with ProgressContext("Extracting text") as progress:
                result = run_ocr_job(
                    store, project_file, list(clip_ids) if clip_ids else None,
                    progress.update, Event(), force=force,
                    options=OcrOptions(num_keyframes, method != "paddleocr", model, method == "vlm"),
                )
        finally:
            store.close()
    except ValueError as exc:
        exit_with(ExitCode.VALIDATION_ERROR, str(exc))
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Text extraction failed: {exc}")
    output_result(result, as_json=ctx.obj.get("json", False))


@analyze.command("boundary-embeddings")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Exact clip ID (default: all clips)")
@click.option("--force", "-f", is_flag=True, help="Recompute existing boundary pairs")
@click.pass_context
def boundary_embeddings(ctx: click.Context, project_file: Path, clip_ids: tuple[str, ...], force: bool) -> None:
    """Extract first/last-frame DINOv2 vectors with durable recovery."""
    from threading import Event
    from core.jobs.boundary_embeddings import run_boundary_embedding_job
    from core.jobs.store import JobStore

    project_file = own_project(ctx, project_file)
    try:
        store = JobStore(CLIConfig.load().cache_dir / "jobs.db")
        try:
            with ProgressContext("Extracting boundary embeddings") as progress:
                result = run_boundary_embedding_job(
                    store, project_file, list(clip_ids) if clip_ids else None,
                    progress.update, Event(), force=force,
                )
        finally:
            store.close()
    except ValueError as exc:
        exit_with(ExitCode.VALIDATION_ERROR, str(exc))
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Boundary embedding analysis failed: {exc}")
    output_result(result, as_json=ctx.obj.get("json", False))


@analyze.command("embeddings")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Clip ID or eight-character prefix")
@click.option("--chunk-size", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--force", "-f", is_flag=True, help="Recompute existing embeddings")
@click.pass_context
def embeddings(ctx: click.Context, project_file: Path, clip_ids: tuple[str, ...], chunk_size: int, force: bool) -> None:
    """Extract DINOv2 thumbnail vectors with resumable batch computation."""
    from threading import Event
    from core.jobs.embeddings import run_embedding_job
    from core.jobs.store import JobStore
    from core.operations.embeddings import EmbeddingOptions
    from core.project import Project

    project_file = own_project(ctx, project_file)
    try:
        project = Project.load(project_file, missing_source_callback=lambda path, sid: None)
        selected = [c.id for c in project.clips if not clip_ids or c.id in clip_ids or c.id[:8] in clip_ids]
        if clip_ids and not selected:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")
        store = JobStore(CLIConfig.load().cache_dir / "jobs.db")
        try:
            with ProgressContext("Extracting embeddings") as progress:
                result = run_embedding_job(store, project_file, selected, progress.update, Event(), options=EmbeddingOptions(chunk_size), force=force)
        finally:
            store.close()
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Embedding analysis failed: {exc}")
    output_result(result, as_json=ctx.obj.get("json", False))


@analyze.command("gaze")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Clip ID or eight-character prefix")
@click.option("--sample-interval", type=float, default=1.0, show_default=True, help="Seconds between sampled frames")
@click.option("--force", "-f", is_flag=True, help="Re-analyze existing gaze results")
@click.pass_context
def gaze(ctx: click.Context, project_file: Path, clip_ids: tuple[str, ...], sample_interval: float, force: bool) -> None:
    """Estimate gaze direction and safely reuse interrupted computation."""
    from math import isfinite
    from threading import Event
    from core.jobs.gaze import run_gaze_job
    from core.jobs.store import JobStore
    from core.operations.gaze import GazeOptions
    from core.project import Project

    if not isfinite(sample_interval) or sample_interval <= 0:
        exit_with(ExitCode.VALIDATION_ERROR, "Sample interval must be a positive finite number")
    project_file = own_project(ctx, project_file)
    try:
        project = Project.load(project_file, missing_source_callback=lambda path, sid: None)
        selected = [c.id for c in project.clips if not clip_ids or c.id in clip_ids or c.id[:8] in clip_ids]
        if clip_ids and not selected:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")
        store = JobStore(CLIConfig.load().cache_dir / "jobs.db")
        try:
            with ProgressContext("Detecting gaze") as progress:
                result = run_gaze_job(store, project_file, selected, progress.update, Event(), options=GazeOptions(sample_interval), force=force)
        finally:
            store.close()
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Gaze analysis failed: {exc}")
    output_result(result, as_json=ctx.obj.get("json", False))


@analyze.command("faces")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip-id", "-c", "clip_ids", multiple=True, help="Clip ID or eight-character prefix")
@click.option("--sample-interval", type=float, default=1.0, show_default=True, help="Seconds between sampled frames")
@click.option("--force", "-f", is_flag=True, help="Re-analyze existing face results")
@click.pass_context
def faces(ctx: click.Context, project_file: Path, clip_ids: tuple[str, ...], sample_interval: float, force: bool) -> None:
    """Extract face embeddings and safely reuse interrupted computation."""
    from math import isfinite
    from threading import Event
    from core.jobs.faces import run_face_job
    from core.jobs.store import JobStore
    from core.operations.faces import FaceOptions
    from core.project import Project

    if not isfinite(sample_interval) or sample_interval <= 0:
        exit_with(ExitCode.VALIDATION_ERROR, "Sample interval must be a positive finite number")
    project_file = own_project(ctx, project_file)
    try:
        project = Project.load(project_file, missing_source_callback=lambda path, sid: None)
        selected = [c.id for c in project.clips if not clip_ids or c.id in clip_ids or c.id[:8] in clip_ids]
        if clip_ids and not selected:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")
        store = JobStore(CLIConfig.load().cache_dir / "jobs.db")
        try:
            with ProgressContext("Detecting faces") as progress:
                result = run_face_job(store, project_file, selected, progress.update, Event(), options=FaceOptions(sample_interval), force=force)
        finally:
            store.close()
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Face analysis failed: {exc}")
    output_result(result, as_json=ctx.obj.get("json", False))


@analyze.command("align")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip", "-c", "clip_ids", multiple=True, help="Exact clip IDs to align (repeatable; default: all)")
@click.option("--force", "-f", is_flag=True, help="Replace existing word timestamps")
@click.pass_context
def align(ctx: click.Context, project_file: Path, clip_ids: tuple[str, ...], force: bool) -> None:
    """Add word timestamps to existing transcripts.

    Requires the optional word-alignment runtime to be installed beforehand.
    """
    from threading import Event
    from core.jobs.alignment import run_alignment_job
    from core.jobs.store import JobStore
    from core.settings import load_settings

    path = own_project(ctx, project_file)
    try:
        with ProgressContext("Aligning words") as progress:
            result = run_alignment_job(
                JobStore(load_settings().cache_dir / "jobs.db"),
                path, list(clip_ids) if clip_ids else None,
                progress.update, Event(), force=force,
            )
        batch = result["result"]
    except ValueError as exc:
        exit_with(ExitCode.VALIDATION_ERROR, str(exc))
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Word alignment failed: {exc}")

    if (ctx.obj or {}).get("json", False):
        output_result(result, as_json=True)
    else:
        output_success(f"Aligned {len(batch['succeeded'])} clips; skipped {len(batch['skipped'])}; failed {len(batch['failed'])}")
        for failure in batch["failed"][:5]:
            output_info(f"  {failure['clip_id']}: {failure['message']}")
    if batch["failed"] and not batch["succeeded"]:
        code = ExitCode.DEPENDENCY_MISSING if all(item["code"] == "dependency_missing" for item in batch["failed"]) else ExitCode.GENERAL_ERROR
        exit_with(code)


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
        from dataclasses import replace
        from threading import Event
        from core.jobs.description import run_description_job
        from core.jobs.store import JobStore
        from core.operations.description import resolve_options
        from core.project import Project
        from core.thumbnail import ThumbnailGenerator
    except ImportError as exc:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {exc}")

    path = own_project(ctx, project_file)
    config = CLIConfig.load()
    try:
        project = Project.load(path, missing_source_callback=lambda path, sid: None)
        clips = project.clips
        selected = clips
        if clip_ids:
            requested = set(clip_ids)
            selected = [c for c in clips if c.id in requested or c.id[:8] in requested]
            if not selected:
                exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")
        prepared = {}
        errors = []
        ready = []
        generator = None
        for clip in selected:
            if clip.description is not None and not force:
                ready.append(clip.id)
                continue
            source = project.sources_by_id.get(clip.source_id)
            if source is None or not source.file_path.exists():
                errors.append(f"Clip {clip.id[:8]}: source not found")
                continue
            if generator is None:
                try:
                    generator = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
                except RuntimeError as exc:
                    exit_with(ExitCode.DEPENDENCY_MISSING, str(exc))
            try:
                prepared[clip.id] = generator.generate_clip_thumbnail(
                    video_path=source.file_path,
                    start_seconds=clip.start_time(source.fps),
                    end_seconds=clip.end_time(source.fps), width=640, height=360,
                )
                ready.append(clip.id)
            except Exception as exc:
                errors.append(f"Clip {clip.id[:8]}: {exc}")
        # CLI descriptions historically use frame input even when GUI video mode is enabled.
        options = replace(resolve_options(tier, prompt), input_mode="frame")
        with ProgressContext("Generating descriptions") as progress:
            batch = run_description_job(
                JobStore(config.cache_dir / "jobs.db"), path, ready,
                progress.update, Event(), options=options, force=force,
                thumbnail_paths=prepared,
            )["result"]
        analyzed_count = len(batch["succeeded"])
        errors.extend(f"Clip {item['clip_id'][:8]}: {item.get('message') or item['code']}" for item in batch["failed"])
    except FileNotFoundError as exc:
        exit_with(ExitCode.FILE_NOT_FOUND, str(exc))
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Description failed: {exc}")

    result = {"analyzed_clips": analyzed_count, "errors": len(errors), "total_clips": len(clips)}
    if (ctx.obj or {}).get("json", False):
        if errors:
            result["error_details"] = errors
        output_result(result, as_json=True)
    else:
        if not analyzed_count and not errors:
            output_info("All clips already have descriptions. Use --force to re-analyze.")
        else:
            output_success(f"Generated descriptions for {analyzed_count} clips")
        for error in errors[:5]:
            output_info(f"  {error}")
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
    project_file = own_project(ctx, project_file)
    try:
        from core.project import Project, ProjectLoadError
        from core.operations.colors import (
            ColorApplication, ColorDependencyError, color_request, compute_colors,
        )
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    try:
        project = Project.load(
            project_file,
            missing_source_callback=lambda path, sid: None,
        )
    except ProjectLoadError as e:
        exit_with(ExitCode.GENERAL_ERROR, f"Failed to load project: {e}")
    except FileNotFoundError:
        exit_with(ExitCode.FILE_NOT_FOUND, f"Project file not found: {project_file}")

    clips = project.clips
    selected = clips
    if clip_ids:
        clip_set = set(clip_ids)
        selected = [c for c in clips if c.id in clip_set or c.id[:8] in clip_set]
        if not selected:
            exit_with(ExitCode.VALIDATION_ERROR, "No matching clips found")

    try:
        request = color_request(
            project, [c.id for c in selected], num_colors,
            skip_existing=not force, skip_empty=True,
        )
    except ValueError as e:
        exit_with(ExitCode.VALIDATION_ERROR, str(e))
    application = ColorApplication(project, request)
    try:
        with ProgressContext("Analyzing colors") as progress:
            result = application.apply(compute_colors(
                request,
                progress_callback=lambda done, total, outcome: progress.update(
                    done / total, f"Clip {done}/{total}",
                ),
            ))
            progress.update(1.0, "Complete")
    except ColorDependencyError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")
    if all(o.status == "skipped" for o in result.outcomes):
        output_info("All clips already have color data. Use --force to re-analyze.")
        return
    analyzed_count = sum(o.status == "succeeded" for o in result.outcomes)
    errors = [
        f"Clip {o.target_id[:8]}: {o.message or o.code}"
        for o in result.outcomes if o.status == "failed"
    ]
    if not project.save():
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
    project_file = own_project(ctx, project_file)
    try:
        from core.project import Project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.operations.shots import ShotTypeOptions
        from core.jobs.shots import run_shot_job
        from core.jobs.store import JobStore
        from threading import Event
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    try:
        project = Project.load(
            project_file, missing_source_callback=lambda path, sid: None,
        )
        sources, clips = project.sources, project.clips
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

    errors = []
    ready = []
    prepared = {}
    thumb_gen = None
    for clip in clips_to_analyze:
        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            errors.append(f"Clip {clip.id[:8]}: source not found")
            continue
        from core.analysis_records import recorded_image_path

        prior_image = recorded_image_path(clip, source, "shots") if not force else None
        if prior_image is not None:
            prepared[clip.id] = prior_image
            ready.append(clip.id)
            continue
        if thumb_gen is None:
            try:
                thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
            except RuntimeError as exc:
                exit_with(ExitCode.DEPENDENCY_MISSING, str(exc))
        try:
            prepared[clip.id] = thumb_gen.generate_clip_thumbnail(
                video_path=source.file_path,
                start_seconds=clip.start_time(source.fps),
                end_seconds=clip.end_time(source.fps),
                width=320, height=180,
            )
            ready.append(clip.id)
        except Exception as exc:
            errors.append(f"Clip {clip.id[:8]}: {exc}")

    try:
        store = JobStore(config.cache_dir / "jobs.db")
        try:
            with ProgressContext("Classifying shots") as progress:
                batch = run_shot_job(
                    store, project_file, ready, progress.update, Event(),
                    options=ShotTypeOptions(), force=force, thumbnail_paths=prepared,
                )["result"]
        finally:
            store.close()
        analyzed_count = len(batch["succeeded"])
        errors.extend(
            f"Clip {item['clip_id'][:8]}: {item.get('message') or item['code']}"
            for item in batch["failed"]
        )
        shot_counts: dict[str, int] = {}
        for item in batch["succeeded"]:
            label = item["shot_type"]
            shot_counts[label] = shot_counts.get(label, 0) + 1
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Shot classification failed: {exc}")

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
    project_file = own_project(ctx, project_file)
    try:
        from core.project import Project, ProjectLoadError
        from core.thumbnail import ThumbnailGenerator
        from core.operations.classification import ClassificationOptions
        from core.jobs.classification import run_classification_job
        from core.jobs.store import JobStore
        from threading import Event
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    try:
        project = Project.load(
            project_file, missing_source_callback=lambda path, sid: None,
        )
        sources, clips = project.sources, project.clips
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

    errors = []
    ready = []
    prepared = {}
    thumb_gen = None
    for clip in clips_to_analyze:
        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            errors.append(f"Clip {clip.id[:8]}: source not found")
            continue
        from core.analysis_records import recorded_image_path

        prior_image = recorded_image_path(clip, source, "classify") if not force else None
        if prior_image is not None:
            prepared[clip.id] = prior_image
            ready.append(clip.id)
            continue
        if thumb_gen is None:
            try:
                thumb_gen = ThumbnailGenerator(cache_dir=config.cache_dir / "thumbnails")
            except RuntimeError as exc:
                exit_with(ExitCode.DEPENDENCY_MISSING, str(exc))
        try:
            prepared[clip.id] = thumb_gen.generate_clip_thumbnail(
                video_path=source.file_path,
                start_seconds=clip.start_time(source.fps), end_seconds=clip.end_time(source.fps),
                width=320, height=180,
            )
            ready.append(clip.id)
        except Exception as exc:
            errors.append(f"Clip {clip.id[:8]}: {exc}")

    try:
        store = JobStore(config.cache_dir / "jobs.db")
        try:
            with ProgressContext("Classifying content") as progress:
                batch = run_classification_job(
                    store, project_file, ready,
                    progress.update, Event(), options=ClassificationOptions(top_k, threshold),
                    force=force, thumbnail_paths=prepared,
                )["result"]
        finally:
            store.close()
        analyzed_count = len(batch["succeeded"])
        errors.extend(f"Clip {item['clip_id'][:8]}: {item.get('message') or item['code']}" for item in batch["failed"])
        saved = Project.load(project_file)
        label_counts: dict[str, int] = {}
        for item in batch["succeeded"]:
            labels = saved.clips_by_id[item["clip_id"]].object_labels
            if labels:
                label_counts[labels[0]] = label_counts.get(labels[0], 0) + 1
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Classification failed: {exc}")

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


def _run_object_detection_cli(
    project_file: Path,
    clips: list,
    sources_by_id: dict,
    cache_dir: Path,
    *,
    confidence: float,
    force: bool,
    detect_all: bool,
) -> tuple[dict, list[str]]:
    """Prepare CLI-sized images, then delegate execution and saving to the job."""
    from threading import Event
    from core.jobs.object_detection import run_object_detection_job
    from core.jobs.store import JobStore
    from core.operations.object_detection import ObjectDetectionOptions
    from core.project import Project
    from core.thumbnail import ThumbnailGenerator

    errors = []
    ready = []
    prepared = {}
    generator = None
    for clip in clips:
        from core.analysis_records import recorded_image_path

        source = sources_by_id.get(clip.source_id)
        prior_image = recorded_image_path(clip, source, "detect_objects") if not force else None
        if prior_image is not None:
            prepared[clip.id] = prior_image
            ready.append(clip.id)
            continue
        if source is None or not source.file_path.exists():
            errors.append(f"Clip {clip.id[:8]}: source not found")
            continue
        if generator is None:
            try:
                generator = ThumbnailGenerator(cache_dir=cache_dir / "thumbnails")
            except RuntimeError as exc:
                exit_with(ExitCode.DEPENDENCY_MISSING, str(exc))
        try:
            prepared[clip.id] = generator.generate_clip_thumbnail(
                video_path=source.file_path,
                start_seconds=clip.start_time(source.fps),
                end_seconds=clip.end_time(source.fps),
                width=320, height=180,
            )
            ready.append(clip.id)
        except Exception as exc:
            errors.append(f"Clip {clip.id[:8]}: {exc}")
    store = JobStore(cache_dir / "jobs.db")
    try:
        with ProgressContext("Detecting objects" if detect_all else "Counting people") as progress:
            batch = run_object_detection_job(
                store, project_file, ready, progress.update, Event(),
                options=ObjectDetectionOptions(confidence, detect_all),
                force=force, thumbnail_paths=prepared,
            )["result"]
    finally:
        store.close()
    errors.extend(
        f"Clip {item['clip_id'][:8]}: {item.get('message') or item['code']}"
        for item in batch["failed"] + batch["unprocessed"]
    )
    saved = Project.load(project_file)
    objects: dict[str, int] = {}
    distribution: dict[int, int] = {}
    total_people = 0
    for item in batch["succeeded"]:
        count = item["person_count"]
        total_people += count
        distribution[count] = distribution.get(count, 0) + 1
        if detect_all:
            for detection in saved.clips_by_id[item["clip_id"]].detected_objects or []:
                label = detection["label"]
                objects[label] = objects.get(label, 0) + 1
    return {
        "analyzed_clips": len(batch["succeeded"]),
        "total_people": total_people,
        "object_counts": objects,
        "distribution": distribution,
    }, errors


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
    project_file = own_project(ctx, project_file)
    try:
        from core.project import Project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    try:
        project = Project.load(
            project_file, missing_source_callback=lambda path, sid: None,
        )
        sources, clips = project.sources, project.clips
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

    try:
        summary, errors = _run_object_detection_cli(
            project_file, clips_to_analyze, sources_by_id, config.cache_dir,
            confidence=confidence, force=force, detect_all=True,
        )
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Object detection failed: {exc}")
    analyzed_count = summary["analyzed_clips"]
    total_people = summary["total_people"]
    object_counts = summary["object_counts"]

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
    project_file = own_project(ctx, project_file)
    try:
        from core.project import Project, ProjectLoadError
    except ImportError as e:
        exit_with(ExitCode.DEPENDENCY_MISSING, f"Missing dependency: {e}")

    config = CLIConfig.load()

    try:
        project = Project.load(
            project_file, missing_source_callback=lambda path, sid: None,
        )
        sources, clips = project.sources, project.clips
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

    try:
        summary, errors = _run_object_detection_cli(
            project_file, clips_to_analyze, sources_by_id, config.cache_dir,
            confidence=confidence, force=force, detect_all=False,
        )
    except Exception as exc:
        exit_with(ExitCode.GENERAL_ERROR, f"Object detection failed: {exc}")
    analyzed_count = summary["analyzed_clips"]
    total_people = summary["total_people"]
    person_distribution = summary["distribution"]

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
