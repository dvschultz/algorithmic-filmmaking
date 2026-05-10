"""Helpers shared between chat-tools and spine for shaping LLM-facing payloads.

Lives in the spine package so spine modules don't have to reach back into
``core/chat_tools.py`` for these helpers. ``core/chat_tools.py`` re-exports
them so legacy imports keep working during the migration.
"""

from __future__ import annotations

from typing import Optional


def truncate_for_agent(text: str | None, limit: int = 1200) -> str | None:
    """Keep tool payloads concise enough for reliable agent summaries."""
    if not text:
        return None
    value = str(text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def clip_summary_for_agent(project, clip, source=None, index: int | None = None) -> dict:
    """Compact, factual clip context for LLM-facing tool results."""
    if source is None:
        source = project.sources_by_id.get(clip.source_id)
    fps = source.fps if source else 30.0
    row = {
        "clip_id": clip.id,
        "source_id": clip.source_id,
        "source_name": source.filename if source else None,
        "start_seconds": round(clip.start_time(fps), 3),
        "duration_seconds": round(clip.duration_seconds(fps), 3),
    }
    if index is not None:
        row["sequence_index"] = index
    if getattr(clip, "description", None):
        row["description"] = truncate_for_agent(clip.description, 240)
    if getattr(clip, "shot_type", None):
        row["shot_type"] = clip.shot_type
    return row


def summarize_clip_sequence_for_agent(
    project,
    clip_entries: list,
    *,
    limit: int = 20,
) -> dict:
    """Summarize ordered clip entries returned by sequence/remix tools."""
    clips = []
    total_duration = 0.0
    for index, entry in enumerate(clip_entries):
        if isinstance(entry, dict):
            clip = project.clips_by_id.get(entry.get("id") or entry.get("clip_id"))
            source = project.sources_by_id.get(entry.get("source_id") or "")
            if source is None and clip is not None:
                source = project.sources_by_id.get(clip.source_id)
            duration = entry.get("duration") or entry.get("duration_seconds")
        else:
            clip = entry[0] if entry else None
            source = entry[1] if len(entry) > 1 else None
            if (
                len(entry) > 3
                and isinstance(entry[2], (int, float))
                and isinstance(entry[3], (int, float))
            ):
                fps = source.fps if source else 30.0
                duration = (entry[3] - entry[2]) / fps
            else:
                duration = (
                    entry[2]
                    if len(entry) > 2 and isinstance(entry[2], (int, float))
                    else None
                )

        if clip is None:
            continue
        if duration is None:
            fps = source.fps if source else 30.0
            duration = clip.duration_seconds(fps)
        total_duration += float(duration)
        if len(clips) < limit:
            row = clip_summary_for_agent(project, clip, source, index=index + 1)
            row["sequence_duration_seconds"] = round(float(duration), 3)
            clips.append(row)

    return {
        "ordered_clip_count": len(clip_entries),
        "summarized_clip_count": len(clips),
        "total_duration_seconds": round(total_duration, 3),
        "clips": clips,
        "response_guidance": (
            "Summarize this generated sequence using only these ordered clips, "
            "durations, source names, and stated tool parameters. Do not invent "
            "visual details or rationale that is not present in the payload."
        ),
    }


def add_sequence_summary_for_agent(
    project, result: dict, clip_entries: list | None = None
) -> dict:
    """Attach a compact ordered sequence summary to a successful tool result."""
    if not result.get("success"):
        return result
    entries = clip_entries
    if entries is None:
        entries = result.get("clips", [])
    if entries:
        result["sequence_summary"] = summarize_clip_sequence_for_agent(project, entries)
    return result


def summarize_report_for_agent(
    report: str, sections: list[str], output_format: str
) -> dict:
    """Return report metadata and a bounded excerpt for chat responses."""
    excerpt_limit = 6000 if output_format == "markdown" else 3000
    return {
        "format": output_format,
        "sections_included": sections,
        "character_count": len(report),
        "word_count": len(report.split()) if output_format == "markdown" else 0,
        "is_truncated": len(report) > excerpt_limit,
        "report_excerpt": truncate_for_agent(report, excerpt_limit),
        "response_guidance": (
            "Use report_excerpt for the chat response. Do not paste the full report "
            "unless the user explicitly asks for it; mention when the report is truncated."
        ),
    }


# Aspect ratio tolerance ranges (5% tolerance) for filter_clips
ASPECT_RATIO_RANGES: dict[str, tuple[float, float]] = {
    "16:9": (1.69, 1.87),  # 1.778 ± 5%
    "4:3": (1.27, 1.40),   # 1.333 ± 5%
    "9:16": (0.53, 0.59),  # 0.5625 ± 5%
}


def append_gaze_fields(clip, clip_data: dict) -> None:
    """Append gaze fields to ``clip_data`` if present on the clip."""
    if clip.gaze_yaw is not None:
        clip_data["gaze_yaw"] = round(clip.gaze_yaw, 2)
    if clip.gaze_pitch is not None:
        clip_data["gaze_pitch"] = round(clip.gaze_pitch, 2)
    if clip.gaze_category is not None:
        clip_data["gaze_category"] = clip.gaze_category


__all__ = [
    "ASPECT_RATIO_RANGES",
    "add_sequence_summary_for_agent",
    "append_gaze_fields",
    "clip_summary_for_agent",
    "summarize_clip_sequence_for_agent",
    "summarize_report_for_agent",
    "truncate_for_agent",
]
