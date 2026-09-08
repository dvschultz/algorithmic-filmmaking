"""The conversion seam from library media to explicit sequence coordinates."""

from fractions import Fraction
from typing import TYPE_CHECKING
from collections.abc import Iterable

from models.media_time import StillHold, VideoRange, frame_boundary, frame_rate, rational
from models.sequence import SequenceClip

if TYPE_CHECKING:
    from models.clip import Clip, Source
    from models.frame import Frame
    from models.sequence import Sequence
    from core.project import Project


TIMING_FIELDS = ("start_frame", "timeline_rate", "timeline_start", "hold_frames", "hold_duration")


def sequence_source_input(project: "Project", entry: SequenceClip) -> tuple:
    """Capture the media context used to interpret a stored frame range."""
    from core.operations.analysis_inputs import clip_input
    clip = project.clips_by_id.get(entry.source_clip_id)
    source = project.sources_by_id.get(entry.source_id)
    return (
        clip_input(project, clip) if clip is not None else None,
        id(source), source.variable_frame_rate if source is not None else None,
        source.frame_timestamps if source is not None else None,
    )


def require_resolved_entry(entry: SequenceClip) -> None:
    diagnostic = getattr(entry, "legacy_timing", None)
    if diagnostic and diagnostic.get("status") == "unresolved":
        raise ValueError(
            f"Resolve legacy timing for entry {entry.id} in the sequence menu, "
            "or use project resolve-timing / resolve_sequence_timing before rendering."
        )


def require_resolved_sequence(sequence: "Sequence") -> None:
    for entry in sequence.get_all_clips():
        require_resolved_entry(entry)


def playback_range(entry: SequenceClip, source_fps: Fraction | float | str) -> VideoRange:
    require_resolved_entry(entry)
    if getattr(entry, "source_rate", None) is not None:
        media = entry.source_range
        if not isinstance(media, VideoRange):
            raise ValueError("Still holds do not have source-video playback coordinates")
        return media
    return VideoRange(entry.in_point, entry.out_point, frame_rate(source_fps))


def source_video_range(source: "Source", start: int, end: int) -> VideoRange:
    media = VideoRange(start, end, frame_rate(source.fps))
    timestamps = getattr(source, "presentation_boundaries", None)
    if getattr(source, "variable_frame_rate", False) and timestamps is None:
        raise ValueError("Variable-rate video requires verified presentation timestamps")
    if timestamps is not None:
        if end >= len(timestamps):
            raise ValueError("Source range exceeds the verified presentation map")
        return VideoRange(start, end, media.rate, presentation_range=(timestamps[start], timestamps[end]))
    return media


def playback_start(entry: SequenceClip, timeline_fps: Fraction | float | str) -> Fraction:
    return (
        entry.timeline_start_time if getattr(entry, "timeline_rate", None) is not None
        else Fraction(entry.start_frame) / frame_rate(timeline_fps)
    )


def retimed_values(sequence: "Sequence", rate: Fraction | float | str) -> list[dict]:
    """Requantize display frames without changing source ranges or elapsed time."""
    rate = frame_rate(rate)
    values = []
    for entry in sequence.get_all_clips():
        span = entry.timeline_range
        start, end = span.frames(rate)
        value = {
            "start_frame": start, "timeline_rate": str(rate),
            "timeline_start": str(span.start), "hold_frames": entry.hold_frames,
            "hold_duration": entry.hold_duration,
        }
        if entry.is_frame_entry:
            value.update(hold_frames=end - start, hold_duration=str(span.duration))
        values.append(value)
    return values


def entry_duration(entry: SequenceClip, rate: Fraction | float | str) -> Fraction:
    if (entry.source_rate is not None or entry.is_frame_entry) and not (
        entry.legacy_timing and entry.legacy_timing.get("status") == "unresolved"
    ):
        return entry.source_range.duration
    return Fraction(entry.duration_frames) / frame_rate(rate)


def timeline_end(entries: Iterable[SequenceClip], rate: Fraction | float | str) -> Fraction:
    """Preserve exact append positions, including across separate insert edits."""
    ends = []
    for entry in entries:
        if entry.timeline_rate is not None and not (
            entry.legacy_timing and entry.legacy_timing.get("status") == "unresolved"
        ):
            ends.append(entry.timeline_range.end)
        else:
            ends.append(Fraction(entry.end_frame()) / frame_rate(rate))
    return max(ends, default=Fraction(0))


def video_entry(
    clip: "Clip", source: "Source", *, timeline_fps: Fraction | float | str,
    start: Fraction, relative_range: tuple[int, int] | None = None,
    track_index: int = 0,
) -> SequenceClip:
    """Convert clip-relative selection once; persisted trims are source-absolute."""
    duration = clip.end_frame - clip.start_frame
    lo, hi = relative_range if relative_range is not None else (0, duration)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (lo, hi, track_index)):
        raise ValueError("Sequence frame offsets and track index must be integers")
    if clip.source_id != source.id or lo < 0 or hi > duration or track_index < 0:
        raise ValueError("Sequence selection falls outside its library clip")
    rate = frame_rate(timeline_fps)
    media = source_video_range(source, clip.start_frame + lo, clip.start_frame + hi)
    return SequenceClip(
        source_clip_id=clip.id, source_id=source.id, track_index=track_index,
        start_frame=frame_boundary(start, rate), in_point=media.start_frame,
        out_point=media.end_frame, source_rate=str(media.rate), timeline_rate=str(rate),
        timeline_start=str(rational(start)),
        source_presentation=(str(media.start), str(media.end)) if media.presentation_range is not None else None,
    )


def still_entry(
    frame: "Frame", *, timeline_fps: Fraction | float | str, start: Fraction,
    hold_frames: int, track_index: int = 0,
) -> SequenceClip:
    rate = frame_rate(timeline_fps)
    if isinstance(hold_frames, bool) or not isinstance(hold_frames, int):
        raise ValueError("Still hold frames must be an integer")
    StillHold(Fraction(hold_frames) / rate)
    return SequenceClip(
        frame_id=frame.id, source_id=frame.source_id or "", track_index=track_index,
        start_frame=frame_boundary(start, rate), hold_frames=hold_frames,
        timeline_rate=str(rate), timeline_start=str(rational(start)),
        hold_duration=str(Fraction(hold_frames) / rate),
    )
