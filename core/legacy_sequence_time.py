"""Lossless legacy coordinate classification; ambiguity requires a decision."""

from copy import deepcopy
from fractions import Fraction
from typing import Literal

from models.media_time import VideoRange, frame_boundary, frame_rate, rational

CoordinateConvention = Literal["source", "clip-relative"]


def convert_legacy_entry(
    entry: dict, clip: dict | None, source: dict | None, timeline_fps: float | str,
    convention: CoordinateConvention | None = None,
) -> dict:
    result = deepcopy(entry)
    explicit = convention is not None
    original = deepcopy(entry.get("legacy_timing", {}).get("original", entry))
    diagnostic = {"status": "unresolved", "original": original}
    result.update(media_time_version=1, legacy_timing=diagnostic)
    try:
        rate = frame_rate(timeline_fps)
        result["timeline_rate"] = str(rate)
        result["timeline_start"] = str(Fraction(entry.get("start_frame", 0)) / rate)
        if entry.get("media_time_version") == 1 and entry.get("timeline_start") is not None:
            exact_start = rational(entry["timeline_start"])
            if frame_boundary(exact_start, rate) == entry.get("start_frame", 0):
                result["timeline_start"] = str(exact_start)
        if entry.get("frame_id") is not None:
            hold = entry.get("hold_frames", 1)
            if isinstance(hold, bool) or not isinstance(hold, int) or hold <= 0:
                raise ValueError("invalid_still_hold")
            if entry.get("in_point", 0) or entry.get("out_point", 0):
                raise ValueError("still_has_video_coordinates")
            result["hold_duration"] = str(Fraction(hold) / rate)
            diagnostic.update(status="resolved", coordinate_space="still")
            return result
        if clip is None or source is None or clip.get("source_id") != source.get("id"):
            raise ValueError("missing_source_or_clip")
        result["source_rate"] = str(frame_rate(source["fps"]))
        lo, hi = original.get("in_point", 0), original.get("out_point", 0)
        begin, end = clip["start_frame"], clip["end_frame"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (lo, hi, begin, end)):
            raise ValueError("invalid_frame_indices")
        if begin < 0 or end <= begin:
            raise ValueError("invalid_library_clip_range")
        absolute = begin <= lo < hi <= end
        relative = 0 <= lo < hi <= end - begin
        if convention is None:
            if absolute and (not relative or begin == 0):
                convention = "source"
            elif relative and not absolute:
                convention = "clip-relative"
            else:
                raise ValueError("ambiguous_coordinates" if absolute and relative else "range_outside_clip")
        if convention not in ("source", "clip-relative"):
            raise ValueError("Unknown coordinate convention")
        if not (absolute if convention == "source" else relative):
            raise ValueError("Selected coordinate convention falls outside the clip")
        offset = begin if convention == "clip-relative" else 0
        result.update(in_point=lo + offset, out_point=hi + offset)
        timestamps = source.get("frame_timestamps")
        if source.get("variable_frame_rate") and timestamps is None:
            raise ValueError("missing_presentation_timestamps")
        if timestamps is not None:
            media = VideoRange(lo + offset, hi + offset, frame_rate(source["fps"]), tuple(rational(t) for t in timestamps))
            result["source_presentation"] = [str(media.start), str(media.end)]
        diagnostic.update(status="resolved", coordinate_space=convention)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        if explicit:
            raise ValueError(f"Cannot resolve legacy sequence entry: {exc}") from exc
        diagnostic["reason"] = str(exc)
        # Keep malformed legacy coordinates inspectable in timeline widgets.
        # The unmodified values remain in diagnostic.original for resolution.
        for key in ("start_frame", "in_point", "out_point"):
            value = result.get(key, 0)
            result[key] = max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0
        result["out_point"] = max(result["in_point"], result["out_point"])
        if entry.get("frame_id") is not None:
            hold = result.get("hold_frames", 0)
            result["hold_frames"] = max(0, hold) if isinstance(hold, int) and not isinstance(hold, bool) else 0
            result["hold_duration"] = None
    return result


def migrate_sequence_time(data: dict) -> None:
    """Upgrade both timeline keys without changing the input's raw entries."""
    clips = {clip["id"]: clip for clip in data.get("clips", [])}
    sources = {source["id"]: source for source in data.get("sources", [])}
    sequences = list(data.get("sequences", []))
    if data.get("sequence") is not None:
        sequences.append(data["sequence"])
    for sequence in sequences:
        for track in sequence.get("tracks", []):
            for index, entry in enumerate(track.get("clips", [])):
                if entry.get("media_time_version") == 1:
                    continue
                track["clips"][index] = convert_legacy_entry(
                    entry, clips.get(entry.get("source_clip_id")),
                    sources.get(entry.get("source_id")), sequence.get("fps", 30),
                )
