"""Sequence/timeline manipulation MCP tools."""

import json
import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import Context

from scene_ripper_mcp.server import mcp
from scene_ripper_mcp.security import validate_project_path

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
        from core.project import load_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        if not sequence:
            return json.dumps(
                {
                    "success": True,
                    "has_sequence": False,
                    "sequence": None,
                }
            )

        # Build source and clip lookups
        sources_by_id = {s.id: s for s in sources}
        clips_by_id = {c.id: c for c in clips}

        # Build track data
        tracks_data = []
        for track in sequence.tracks:
            track_clips = []
            for seq_clip in track.clips:
                source = sources_by_id.get(seq_clip.source_id)
                orig_clip = clips_by_id.get(seq_clip.source_clip_id)
                fps = source.fps if source else 30.0

                track_clips.append(
                    {
                        "id": seq_clip.id,
                        "source_clip_id": seq_clip.source_clip_id,
                        "source_name": source.filename if source else "Unknown",
                        "timeline_start": seq_clip.start_time(fps),
                        "timeline_start_frame": seq_clip.start_frame,
                        "duration": seq_clip.duration_seconds(fps),
                        "duration_frames": seq_clip.duration_frames,
                        "in_point": seq_clip.in_point,
                        "out_point": seq_clip.out_point,
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
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project
        from models.sequence import Sequence, SequenceClip, Track

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        # Build lookups
        sources_by_id = {s.id: s for s in sources}
        clips_by_id = {c.id: c for c in clips}

        # Create sequence if it doesn't exist
        if not sequence:
            # Determine FPS from sources
            fps = 30.0
            if sources:
                fps = sources[0].fps
            sequence = Sequence(name=metadata.name, fps=fps)

        # Ensure track exists
        while len(sequence.tracks) <= track_index:
            sequence.add_track()

        track = sequence.tracks[track_index]

        # Determine starting position
        if position == "end":
            start_frame = track.clips[-1].end_frame() if track.clips else 0
        elif position == "start":
            start_frame = 0
        else:
            try:
                start_frame = int(position)
            except ValueError:
                return json.dumps({"success": False, "error": f"Invalid position: {position}"})

        # Add clips
        added_clips = []
        current_frame = start_frame

        for clip_id in clip_ids:
            orig_clip = clips_by_id.get(clip_id)
            if not orig_clip:
                logger.warning(f"Clip not found: {clip_id}")
                continue

            source = sources_by_id.get(orig_clip.source_id)
            if not source:
                logger.warning(f"Source not found for clip: {clip_id}")
                continue

            # Create sequence clip
            seq_clip = SequenceClip(
                source_clip_id=orig_clip.id,
                source_id=orig_clip.source_id,
                track_index=track_index,
                start_frame=current_frame,
                in_point=0,
                out_point=orig_clip.duration_frames,
            )

            track.add_clip(seq_clip)
            added_clips.append(
                {
                    "clip_id": clip_id,
                    "sequence_clip_id": seq_clip.id,
                    "start_frame": current_frame,
                }
            )

            current_frame += orig_clip.duration_frames

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clips_added": len(added_clips),
                "added": added_clips,
                "sequence_duration": sequence.duration_seconds,
            }
        )
    except Exception as e:
        logger.exception("Failed to add to sequence")
        return json.dumps({"success": False, "error": str(e)})


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
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        if not sequence:
            return json.dumps({"success": False, "error": "No sequence in project"})

        # Remove clips
        removed_count = 0
        ids_to_remove = set(sequence_clip_ids)

        for track in sequence.tracks:
            original_count = len(track.clips)
            track.clips = [c for c in track.clips if c.id not in ids_to_remove]
            removed_count += original_count - len(track.clips)

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clips_removed": removed_count,
                "sequence_duration": sequence.duration_seconds,
            }
        )
    except Exception as e:
        logger.exception("Failed to remove from sequence")
        return json.dumps({"success": False, "error": str(e)})


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
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        if not sequence:
            return json.dumps({"success": False, "error": "No sequence in project"})

        if track_index >= len(sequence.tracks):
            return json.dumps({"success": False, "error": f"Track {track_index} does not exist"})

        track = sequence.tracks[track_index]

        # Build lookup of existing clips
        existing = {c.id: c for c in track.clips}

        # Reorder based on provided order
        new_clips = []
        current_frame = 0

        for clip_id in clip_order:
            if clip_id not in existing:
                logger.warning(f"Clip not found in sequence: {clip_id}")
                continue

            clip = existing[clip_id]
            clip.start_frame = current_frame
            new_clips.append(clip)
            current_frame += clip.duration_frames

        # Add any clips not in the order list at the end
        for clip_id, clip in existing.items():
            if clip not in new_clips:
                clip.start_frame = current_frame
                new_clips.append(clip)
                current_frame += clip.duration_frames

        track.clips = new_clips

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clips_reordered": len(new_clips),
                "new_order": [c.id for c in new_clips],
                "sequence_duration": sequence.duration_seconds,
            }
        )
    except Exception as e:
        logger.exception("Failed to reorder sequence")
        return json.dumps({"success": False, "error": str(e)})


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
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        if not sequence:
            return json.dumps(
                {
                    "success": True,
                    "message": "No sequence to clear",
                    "clips_removed": 0,
                }
            )

        # Count and clear clips
        total_removed = sum(len(track.clips) for track in sequence.tracks)

        for track in sequence.tracks:
            track.clips = []

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "clips_removed": total_removed,
                "tracks_preserved": len(sequence.tracks),
            }
        )
    except Exception as e:
        logger.exception("Failed to clear sequence")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def shuffle_sequence(
    project_path: Annotated[str, "Path to project file"],
    method: Annotated[str, "Shuffle method: random, reverse, by_color, by_shot_type"] = "random",
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
    valid, error, path = validate_project_path(project_path)
    if not valid:
        return json.dumps({"success": False, "error": error})

    try:
        import random
        from core.project import load_project, save_project

        sources, clips, sequence, metadata, ui_state, _ = load_project(path)

        if not sequence:
            return json.dumps({"success": False, "error": "No sequence in project"})

        if track_index >= len(sequence.tracks):
            return json.dumps({"success": False, "error": f"Track {track_index} does not exist"})

        track = sequence.tracks[track_index]

        if not track.clips:
            return json.dumps({"success": True, "message": "No clips to shuffle", "clips_shuffled": 0})

        # Build lookup for source clips
        clips_by_id = {c.id: c for c in clips}

        # Get list of sequence clips
        seq_clips = list(track.clips)

        if method == "random":
            random.shuffle(seq_clips)
        elif method == "reverse":
            seq_clips.reverse()
        elif method == "by_color":
            # Sort by primary color hue
            def get_hue(seq_clip):
                orig = clips_by_id.get(seq_clip.source_clip_id)
                if orig and orig.dominant_colors:
                    from core.analysis.color import rgb_to_hsv

                    return rgb_to_hsv(orig.dominant_colors[0])[0]
                return 0

            seq_clips.sort(key=get_hue)
        elif method == "by_shot_type":
            # Group by shot type
            shot_order = ["wide shot", "medium shot", "close-up", "extreme close-up", None]

            def get_shot_index(seq_clip):
                orig = clips_by_id.get(seq_clip.source_clip_id)
                shot = orig.shot_type if orig else None
                return shot_order.index(shot) if shot in shot_order else len(shot_order)

            seq_clips.sort(key=get_shot_index)
        else:
            return json.dumps({"success": False, "error": f"Unknown shuffle method: {method}"})

        # Update positions
        current_frame = 0
        for clip in seq_clips:
            clip.start_frame = current_frame
            current_frame += clip.duration_frames

        track.clips = seq_clips

        # Save project
        success = save_project(
            filepath=path,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            metadata=metadata,
        )

        if not success:
            return json.dumps({"success": False, "error": "Failed to save project"})

        return json.dumps(
            {
                "success": True,
                "method": method,
                "clips_shuffled": len(seq_clips),
                "new_order": [c.id for c in seq_clips],
                "sequence_duration": sequence.duration_seconds,
            }
        )
    except Exception as e:
        logger.exception("Failed to shuffle sequence")
        return json.dumps({"success": False, "error": str(e)})
