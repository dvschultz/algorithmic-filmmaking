"""Sequence/timeline manipulation MCP tools."""

import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_project_path

from scene_ripper_mcp.editorial import editorial_call as _editorial_call

logger = logging.getLogger(__name__)




@mcp.tool()
async def get_sequence(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """Get the current sequence/timeline state.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with sequence structure and clips
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import MissingSourceError
        from core.spine.project_io import load_with_mtime

        try:
            project, _mtime = load_with_mtime(path)
        except MissingSourceError as e:
            return json.dumps({
                "success": False,
                "error": {"code": "source_files_missing", "message": str(e)},
            })

        sequence = project.sequence

        # An empty sequence has no clips on any track. Treat that as
        # "no sequence yet" for the agent's purposes.
        if not sequence or sum(len(t.clips) for t in sequence.tracks) == 0:
            return json.dumps(
                {
                    "success": True,
                    "has_sequence": False,
                    "sequence": None,
                }
            )

        sources_by_id = project.sources_by_id

        # Build track data
        tracks_data = []
        for track in sequence.tracks:
            track_clips = []
            for seq_clip in track.clips:
                source = sources_by_id.get(seq_clip.source_id)

                track_clips.append(
                    {
                        "id": seq_clip.id,
                        "source_clip_id": seq_clip.source_clip_id,
                        "source_name": source.filename if source else "Unknown",
                        "timeline_start": seq_clip.start_time(sequence.fps),
                        "timeline_start_frame": seq_clip.start_frame,
                        "duration": seq_clip.duration_seconds(sequence.fps),
                        "duration_frames": seq_clip.duration_frames,
                        "in_point": seq_clip.in_point,
                        "out_point": seq_clip.out_point,
                        "source_rate": seq_clip.source_rate,
                        "timeline_rate": seq_clip.timeline_rate,
                        "legacy_timing": seq_clip.legacy_timing,
                    }
                )

            tracks_data.append(
                {
                    "id": track.id,
                    "name": track.name,
                    "clip_count": len(track.clips),
                    "clips": track_clips,
                }
            )

        return json.dumps(
            {
                "success": True,
                "has_sequence": True,
                "sequence": {
                    "id": sequence.id,
                    "name": sequence.name,
                    "fps": sequence.fps,
                    "duration_frames": sequence.duration_frames,
                    "duration_seconds": sequence.duration_seconds,
                    "track_count": len(sequence.tracks),
                    "total_clips": sum(len(t.clips) for t in sequence.tracks),
                    "tracks": tracks_data,
                },
            }
        )
    except Exception as e:
        logger.exception("Failed to get sequence")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def add_to_sequence(
    project_path: Annotated[str, "Path to project file"],
    clip_ids: Annotated[list[str], "List of clip IDs to add"],
    track_index: Annotated[int, "Track index (0 = first track)"] = 0,
    position: Annotated[Optional[str], "Position: 'end' (default), 'start', or frame number"] = "end",
    ctx: Context = None,
) -> str:
    """Add clips to the sequence timeline.

    Args:
        project_path: Path to the project file
        clip_ids: List of clip IDs to add (in order)
        track_index: Which track to add to (default: 0)
        position: Where to add - 'end', 'start', or specific frame number

    Returns:
        JSON with updated sequence state
    """
    def operation(project):
        from core.spine.timeline import insert_legacy_clips
        return insert_legacy_clips(project, clip_ids, track_index=track_index, position=position)
    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def resolve_sequence_timing(
    project_path: Annotated[str, "Path to project file"],
    sequence_id: Annotated[str, "Sequence ID"],
    entry_id: Annotated[str, "Unresolved timeline entry ID"],
    convention: Annotated[str, "source or clip-relative"],
    ctx: Context = None,
) -> str:
    """Resolve preserved legacy trim coordinates; retain the original and undo."""
    def operation(project):
        from core.spine.timeline import resolve_legacy_timing
        return resolve_legacy_timing(project, sequence_id, entry_id, convention)
    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def remove_from_sequence(
    project_path: Annotated[str, "Path to project file"],
    sequence_clip_ids: Annotated[list[str], "List of sequence clip IDs to remove"],
    ctx: Context = None,
) -> str:
    """Remove clips from the sequence timeline.

    Args:
        project_path: Path to the project file
        sequence_clip_ids: List of sequence clip IDs to remove

    Returns:
        JSON with removal result
    """

    def operation(project):
        removed = project.remove_from_sequence(sequence_clip_ids, ripple=False)
        return {
            "success": True,
            "clips_removed": len(removed),
            "sequence_duration": project.sequence.duration_seconds,
        }

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def reorder_sequence(
    project_path: Annotated[str, "Path to project file"],
    clip_order: Annotated[list[str], "Sequence clip IDs in desired order"],
    track_index: Annotated[int, "Track index to reorder"] = 0,
    ctx: Context = None,
) -> str:
    """Reorder clips in the sequence.

    Rearranges clips on a track to match the specified order.
    Clips are placed sequentially with no gaps.

    Args:
        project_path: Path to the project file
        clip_order: List of sequence clip IDs in the desired order
        track_index: Which track to reorder

    Returns:
        JSON with reorder result
    """

    def operation(project):
        from core.spine.timeline import reorder_clips

        known = {
            c.id
            for i, track in enumerate(project.sequence.tracks)
            if i == track_index
            for c in track.clips
        }
        requested = [clip_id for clip_id in clip_order if clip_id in known]
        result = reorder_clips(
            project, project.sequence.id, requested, track_index=track_index
        )
        return {
            "success": True,
            "clips_reordered": len(result["clip_order"]),
            "new_order": result["clip_order"],
            "sequence_duration": project.sequence.duration_seconds,
        }

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def clear_sequence(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """Clear all clips from the sequence.

    Removes all clips from all tracks but preserves the track structure.

    Args:
        project_path: Path to the project file

    Returns:
        JSON with clear result
    """

    def operation(project):
        removed = project.clear_sequence()
        return {
            "success": True,
            "clips_removed": removed,
            "tracks_preserved": len(project.sequence.tracks),
        }

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def shuffle_sequence(
    project_path: Annotated[str, "Path to project file"],
    method: Annotated[
        str, "Shuffle method: random, reverse, by_color, by_shot_type"
    ] = "random",
    track_index: Annotated[int, "Track index to shuffle"] = 0,
    ctx: Context = None,
) -> str:
    """Shuffle clips in the sequence using various algorithms.

    Args:
        project_path: Path to the project file
        method: Shuffle algorithm to use
        track_index: Which track to shuffle

    Returns:
        JSON with shuffle result
    """

    def operation(project):
        from core.spine.timeline import shuffle_clips

        result = shuffle_clips(
            project, project.sequence.id, method, track_index=track_index
        )
        if not result["clips_shuffled"]:
            return {
                "success": True,
                "message": "No clips to shuffle",
                "clips_shuffled": 0,
            }
        return {
            "success": True,
            "method": method,
            "clips_shuffled": result["clips_shuffled"],
            "new_order": result["clip_order"],
            "sequence_duration": project.sequence.duration_seconds,
        }

    return await _editorial_call(project_path, ctx, operation)


# --- Registry-backed generation and recipes -------------------------------


@mcp.tool()
async def list_sequence_algorithms(ctx: Context = None) -> str:
    """List sequencer algorithms available through the registry with their parameter schemas.

    Returns:
        JSON with one entry per algorithm: key, version, kind, seeded,
        prerequisites, and parameters (name, type, default, choices, bounds).
    """
    from core.spine.sequences import list_algorithms

    return json.dumps(list_algorithms())


@mcp.tool()
async def generate_sequence(
    project_path: Annotated[str, "Path to project file"],
    algorithm: Annotated[str, "Registry algorithm key (see list_sequence_algorithms)"],
    clip_ids: Annotated[
        Optional[list[str]], "Clip IDs to sequence, in candidate order; omit for all enabled clips"
    ] = None,
    parameters: Annotated[Optional[dict], "Algorithm parameters; unknown keys are rejected"] = None,
    seed: Annotated[Optional[int], "Explicit seed for seeded algorithms (0 is a valid seed)"] = None,
    name: Annotated[Optional[str], "Sequence name; defaults to the algorithm label"] = None,
    show_chromatic_color_bar: Annotated[bool, "Chromatics only: render the color bar in exports"] = False,
    ctx: Context = None,
) -> str:
    """Generate a new sequence with a registry algorithm and store its recipe.

    The result is published as one reversible edit and becomes the active
    sequence. The stored recipe records normalized parameters, the seed,
    the ordered inputs, and every realized placement so the sequence can be
    inspected and reconstructed without running the algorithm again.

    Args:
        project_path: Path to the project file
        algorithm: Registry algorithm key
        clip_ids: Optional clip IDs in candidate order
        parameters: Optional algorithm parameters
        seed: Optional explicit seed; omitted seeds are drawn and recorded
        name: Optional sequence name

    Returns:
        JSON with sequence_id, recipe_id, seed, parameters, clip order and notes
    """
    from core.spine.sequences import generate_sequence as _generate

    refusal = _refuse_long_running(algorithm)
    if refusal is not None:
        return refusal

    def operation(project):
        return _generate(
            project, algorithm, clip_ids=clip_ids, parameters=parameters, seed=seed, name=name,
            show_chromatic_color_bar=show_chromatic_color_bar,
        )

    return await _editorial_call(project_path, ctx, operation)


def _refuse_long_running(algorithm: str) -> str | None:
    """Keep provider and model-prerequisite algorithms off the serial editorial path."""
    from core.remix.registry import registry

    definition = registry.get(algorithm) if isinstance(algorithm, str) else None
    if definition is not None and definition.long_running:
        return json.dumps({
            "success": False,
            "error": {
                "code": "use_job",
                "message": (
                    f"Algorithm {algorithm!r} runs provider or model work; start it with "
                    "start_generate_sequence (or start_regenerate_sequence) and poll get_job_status."
                ),
            },
        })
    return None


@mcp.tool()
async def get_sequence_recipe(
    project_path: Annotated[str, "Path to project file"],
    sequence_id: Annotated[Optional[str], "Sequence ID; omit for the active sequence"] = None,
    ctx: Context = None,
) -> str:
    """Inspect the stored recipe of a generated sequence.

    Returns:
        JSON with the recipe document, its generation fingerprint, whether it
        can be reconstructed in the current project, any input problems, and
        whether the timeline still matches the realized entries.
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        from core.spine.project_io import load_with_mtime
        from core.spine.sequences import get_sequence_recipe as _inspect

        project, _mtime = load_with_mtime(path)
        return json.dumps(_inspect(project, sequence_id))
    except Exception as exc:
        from core.spine.project_io import project_error

        return json.dumps({"success": False, "error": project_error(exc)})


@mcp.tool()
async def reconstruct_sequence(
    project_path: Annotated[str, "Path to project file"],
    sequence_id: Annotated[Optional[str], "Sequence ID; omit for the active sequence"] = None,
    name: Annotated[Optional[str], "Name for the rebuilt sequence"] = None,
    ctx: Context = None,
) -> str:
    """Rebuild a sequence from its stored recipe without re-running the algorithm.

    Replays the realized clip choices, trims and transforms as a new sequence.
    Performs no provider calls. Fails with a list of changed inputs when the
    project no longer matches the recipe, leaving existing sequences untouched.

    Returns:
        JSON with the new sequence's id, name, recipe_id and clip order
    """
    from core.spine.sequences import reconstruct_sequence as _reconstruct

    def operation(project):
        return _reconstruct(project, sequence_id, name=name)

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def list_sequences(
    project_path: Annotated[str, "Path to project file"],
    ctx: Context = None,
) -> str:
    """List sequences with ids, names, algorithm, clip counts, recipe ids and the active flag."""
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        from core.spine.project_io import load_with_mtime
        from core.spine.sequences import list_sequences as _impl

        project, _mtime = load_with_mtime(path)
        return json.dumps(_impl(project))
    except Exception as exc:
        from core.spine.project_io import project_error

        return json.dumps({"success": False, "error": project_error(exc)})


@mcp.tool()
async def activate_sequence(
    project_path: Annotated[str, "Path to project file"],
    sequence_id: Annotated[str, "Sequence ID to make active"],
    ctx: Context = None,
) -> str:
    """Make a sequence active. Saved as project state; not an undoable edit."""
    from core.spine.sequences import activate_sequence as _impl

    def operation(project):
        result = _impl(project, sequence_id)
        if result.get("success"):
            project.mark_dirty()
        return result

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def compare_sequences(
    project_path: Annotated[str, "Path to project file"],
    sequence_a: Annotated[str, "First sequence ID (A)"],
    sequence_b: Annotated[str, "Second sequence ID (B)"],
    ctx: Context = None,
) -> str:
    """Compare two sequences: clip counts, durations, seeds, changed recipe parameters, lineage.

    Returns:
        JSON with ``a`` and ``b`` summaries, ``parameter_differences``
        (``key``/``a``/``b``), ``seed_changed``, ``inputs_equal``, ``related``,
        ``timelines_identical`` and the deltas. Read-only.
    """
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        from core.spine.project_io import load_with_mtime
        from core.spine.sequences import compare_sequences as _impl

        project, _mtime = load_with_mtime(path)
        return json.dumps(_impl(project, sequence_a, sequence_b))
    except Exception as exc:
        from core.spine.project_io import project_error

        return json.dumps({"success": False, "error": project_error(exc)})


@mcp.tool()
async def duplicate_sequence(
    project_path: Annotated[str, "Path to project file"],
    sequence_id: Annotated[Optional[str], "Sequence ID; omit for the active sequence"] = None,
    name: Annotated[Optional[str], "Name for the copy"] = None,
    ctx: Context = None,
) -> str:
    """Copy a sequence's timeline and recipe as a new active sequence; nothing is recomputed."""
    from core.spine.sequences import duplicate_sequence as _impl

    def operation(project):
        return _impl(project, sequence_id, name=name)

    return await _editorial_call(project_path, ctx, operation)


@mcp.tool()
async def regenerate_sequence(
    project_path: Annotated[str, "Path to project file"],
    sequence_id: Annotated[Optional[str], "Sequence ID; omit for the active sequence"] = None,
    parameters: Annotated[Optional[dict], "Parameter overrides merged over the recipe"] = None,
    seed: Annotated[Optional[int], "Explicit seed; omitted seeds are drawn fresh"] = None,
    keep_seed: Annotated[bool, "Reuse the recipe's seed instead of drawing a new one"] = False,
    name: Annotated[Optional[str], "Name for the variation"] = None,
    ctx: Context = None,
) -> str:
    """Run a recipe's algorithm again as a new variation without touching the original.

    Fails without editing when the recipe's inputs changed, the algorithm is
    unavailable, or its version differs from the recipe's. Provider-assisted
    algorithms make new provider calls; use reconstruct_sequence to replay
    offline.
    """
    from core.spine.sequences import regenerate_sequence as _impl

    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})
    try:
        from core.spine.project_io import load_with_mtime

        project, _mtime = load_with_mtime(path)
        target = project.sequence if sequence_id is None else next((s for s in project.sequences if s.id == sequence_id), None)
        recipe = target.readable_recipe if target is not None else None
        if recipe is not None:
            refusal = _refuse_long_running(recipe.algorithm)
            if refusal is not None:
                return refusal
    except Exception as exc:
        from core.spine.project_io import project_error

        return json.dumps({"success": False, "error": project_error(exc)})

    def operation(project):
        return _impl(project, sequence_id, parameters=parameters, seed=seed, keep_seed=keep_seed, name=name)

    return await _editorial_call(project_path, ctx, operation)
