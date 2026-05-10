"""Read-only project-state spine queries.

Catch-all module for project queries that simply read from a ``Project``
instance and shape the data for an LLM consumer. Each function is pure and
side-effect-free.
"""

from __future__ import annotations


def get_project_state(project) -> dict:
    """Return current project information."""
    return {
        "success": True,
        "name": project.metadata.name,
        "path": str(project.path) if project.path else "Unsaved",
        "sources": [
            {
                "id": s.id,
                "name": s.file_path.name if s.file_path else "Unknown",
                "duration": s.duration_seconds,
                "fps": s.fps,
                "analyzed": s.analyzed,
                "clips": len(project.clips_by_source.get(s.id, [])),
            }
            for s in project.sources
        ],
        "clip_count": len(project.clips),
        "sequence_length": (
            len(project.sequence.tracks[0].clips) if project.sequence else 0
        ),
        "is_dirty": project.is_dirty,
    }


def get_sequence_state(project) -> dict:
    """Return detailed sequence state."""
    if project.sequence is None:
        return {
            "has_sequence": False,
            "clips": [],
            "total_duration_seconds": 0,
            "clip_count": 0,
        }

    sequence = project.sequence
    fps = sequence.fps
    clips_data = []

    for track in sequence.tracks:
        for seq_clip in track.clips:
            source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
            source = project.sources_by_id.get(seq_clip.source_id)

            clip_data = {
                "id": seq_clip.id,
                "source_clip_id": seq_clip.source_clip_id,
                "source_id": seq_clip.source_id,
                "source_name": source.file_path.name if source else "Unknown",
                "track_index": seq_clip.track_index,
                "start_frame": seq_clip.start_frame,
                "start_time_seconds": round(seq_clip.start_time(fps), 2),
                "duration_frames": seq_clip.duration_frames,
                "duration_seconds": round(seq_clip.duration_seconds(fps), 2),
                "in_point": seq_clip.in_point,
                "out_point": seq_clip.out_point,
                "hflip": seq_clip.hflip,
                "vflip": seq_clip.vflip,
                "reverse": seq_clip.reverse,
                "prerendered_path": seq_clip.prerendered_path,
            }

            if source_clip:
                if source_clip.dominant_colors:
                    clip_data["dominant_colors"] = [
                        f"#{r:02x}{g:02x}{b:02x}"
                        for r, g, b in source_clip.dominant_colors
                    ]
                if source_clip.shot_type:
                    clip_data["shot_type"] = source_clip.shot_type
                if source_clip.description:
                    clip_data["description"] = source_clip.description

            clips_data.append(clip_data)

    return {
        "has_sequence": True,
        "name": sequence.name,
        "fps": fps,
        "clips": clips_data,
        "total_duration_frames": sequence.duration_frames,
        "total_duration_seconds": round(sequence.duration_seconds, 2),
        "clip_count": len(clips_data),
    }


def get_project_summary(project) -> dict:
    """Generate a human-readable project summary in markdown."""
    lines: list[str] = []
    lines.append(f"# {project.metadata.name}")
    lines.append("")

    lines.append("## Project Info")
    lines.append(f"- **Path**: {project.path or 'Unsaved'}")
    lines.append(f"- **Created**: {project.metadata.created_at[:10]}")
    lines.append(f"- **Modified**: {project.metadata.modified_at[:10]}")
    lines.append(f"- **Unsaved changes**: {'Yes' if project.is_dirty else 'No'}")
    lines.append("")

    lines.append(f"## Sources ({len(project.sources)} videos)")
    if project.sources:
        total_duration = sum(s.duration_seconds for s in project.sources)
        lines.append(
            f"- **Total duration**: {total_duration:.1f}s ({total_duration / 60:.1f} min)"
        )
        lines.append("")
        for source in project.sources:
            clip_count = len(project.clips_by_source.get(source.id, []))
            analyzed = "✓" if source.analyzed else "✗"
            lines.append(
                f"- {source.filename} ({source.duration_seconds:.1f}s, "
                f"{clip_count} clips) [{analyzed}]"
            )
    else:
        lines.append("- No sources imported yet")
    lines.append("")

    disabled_count = sum(1 for c in project.clips if c.disabled)
    enabled_count = len(project.clips) - disabled_count
    clip_header = f"## Clips ({len(project.clips)} total"
    if disabled_count:
        clip_header += f", {enabled_count} enabled, {disabled_count} disabled"
    clip_header += ")"
    lines.append(clip_header)
    if project.clips:
        with_colors = sum(1 for c in project.clips if c.dominant_colors)
        with_shots = sum(1 for c in project.clips if c.shot_type)
        with_transcript = sum(1 for c in project.clips if c.transcript)
        with_tags = sum(1 for c in project.clips if c.tags)
        with_notes = sum(1 for c in project.clips if c.notes)
        with_gaze = sum(
            1 for c in project.clips if c.gaze_category is not None
        )

        lines.append(f"- **Color analyzed**: {with_colors}/{len(project.clips)}")
        lines.append(f"- **Shot classified**: {with_shots}/{len(project.clips)}")
        lines.append(f"- **Transcribed**: {with_transcript}/{len(project.clips)}")
        lines.append(f"- **Gaze analyzed**: {with_gaze}/{len(project.clips)}")
        lines.append(f"- **Tagged**: {with_tags}/{len(project.clips)}")
        lines.append(f"- **With notes**: {with_notes}/{len(project.clips)}")

        all_tags: set[str] = set()
        for clip in project.clips:
            all_tags.update(clip.tags)
        if all_tags:
            lines.append(f"- **Tags used**: {', '.join(sorted(all_tags))}")
    else:
        lines.append("- No clips detected yet")
    lines.append("")

    lines.append("## Sequence")
    if project.sequence and project.sequence.get_all_clips():
        seq_clips = project.sequence.get_all_clips()
        lines.append(f"- **Clips in sequence**: {len(seq_clips)}")
        lines.append(
            f"- **Total duration**: {project.sequence.duration_seconds:.1f}s"
        )
        lines.append(f"- **FPS**: {project.sequence.fps}")
    else:
        lines.append("- No sequence built yet")

    summary_text = "\n".join(lines)

    return {
        "success": True,
        "summary": summary_text,
        "stats": {
            "sources": len(project.sources),
            "clips": len(project.clips),
            "sequence_clips": (
                len(project.sequence.get_all_clips()) if project.sequence else 0
            ),
            "is_dirty": project.is_dirty,
        },
    }


def search_transcripts(
    project,
    query: str,
    case_sensitive: bool = False,
    context_chars: int = 50,
) -> dict:
    """Search clip transcripts for the given text and return match contexts."""
    if not query:
        return {"success": False, "error": "Query cannot be empty"}

    search_query = query if case_sensitive else query.lower()
    results = []

    for clip in project.clips:
        if not clip.transcript:
            continue

        full_text = clip.get_transcript_text()
        if not full_text:
            continue

        search_text = full_text if case_sensitive else full_text.lower()

        if search_query in search_text:
            pos = search_text.find(search_query)
            start = max(0, pos - context_chars)
            end = min(len(full_text), pos + len(query) + context_chars)
            context = full_text[start:end]

            if start > 0:
                context = "..." + context
            if end < len(full_text):
                context = context + "..."

            source = project.sources_by_id.get(clip.source_id)
            fps = source.fps if source else 30.0

            results.append(
                {
                    "clip_id": clip.id,
                    "source_name": source.file_path.name if source else "Unknown",
                    "match_context": context,
                    "duration_seconds": round(clip.duration_seconds(fps), 2),
                    "start_time": round(clip.start_time(fps), 2),
                    "shot_type": clip.shot_type,
                }
            )

    return {
        "success": True,
        "query": query,
        "match_count": len(results),
        "matches": results,
    }


def list_frames(
    project,
    source_id: str | None = None,
    clip_id: str | None = None,
    shot_type: str | None = None,
    has_description: bool | None = None,
) -> dict:
    """List frames in the project with optional filters."""
    frames = list(project.frames)

    if source_id:
        frames = [f for f in frames if f.source_id == source_id]
    if clip_id:
        frames = [f for f in frames if f.clip_id == clip_id]
    if shot_type:
        frames = [f for f in frames if f.shot_type == shot_type]
    if has_description is not None:
        if has_description:
            frames = [f for f in frames if f.description]
        else:
            frames = [f for f in frames if not f.description]

    results = []
    for frame in frames:
        results.append(
            {
                "id": frame.id,
                "display_name": frame.display_name(),
                "source_id": frame.source_id,
                "clip_id": frame.clip_id,
                "frame_number": frame.frame_number,
                "analyzed": frame.analyzed,
                "shot_type": frame.shot_type,
                "has_description": bool(frame.description),
                "tags": frame.tags,
            }
        )

    return {
        "success": True,
        "frames": results,
        "count": len(results),
        "total_in_project": len(project.frames),
    }


__all__ = [
    "get_project_state",
    "get_project_summary",
    "get_sequence_state",
    "list_frames",
    "search_transcripts",
]
