"""Immutable edit decisions shared by playback and render adapters.

All intervals are half-open. Timeline boundaries are rounded globally, so a
short edit may have no raster frames at a particular output rate. It remains
in the plan and regains frames when the output rate increases.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Mapping

from core.sequence_time import playback_range, playback_start, source_video_range
from models.clip import Clip, Source
from models.frame import Frame
from models.media_time import StillHold, TimelineRange, VideoRange, frame_boundary, frame_rate
from models.sequence import Sequence, SequenceClip


class RenderPlanError(ValueError):
    """An edit cannot be represented by the selected output adapter."""


@dataclass(frozen=True)
class RenderSegment:
    kind: Literal["video", "still", "gap"]
    timeline: TimelineRange
    start_frame: int
    end_frame: int
    entry_id: str | None = None
    source_id: str | None = None
    source_clip_id: str | None = None
    path: Path | None = None
    media: VideoRange | StillHold | None = None
    hflip: bool = False
    vflip: bool = False
    reverse: bool = False
    color: tuple[int, int, int] | None = None
    source_boundaries: tuple[Fraction, ...] | None = None

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class RenderPlan:
    sequence_id: str
    timeline_rate: Fraction
    output_rate: Fraction
    segments: tuple[RenderSegment, ...]
    music_path: Path | None = None
    media_stamps: tuple[tuple[Path, tuple[int, int, int, int]], ...] = ()

    def validate_media_unchanged(self) -> None:
        for path, stamp in self.media_stamps:
            if _media_stamp(path) != stamp:
                raise RenderPlanError(f"Media changed during rendering: {path}")

    @property
    def frame_count(self) -> int:
        return self.segments[-1].end_frame if self.segments else 0

    @property
    def duration(self) -> Fraction:
        return Fraction(self.frame_count) / self.output_rate

    def segment_at_frame(self, frame: int) -> RenderSegment | None:
        return next((segment for segment in self.segments if segment.start_frame <= frame < segment.end_frame), None)

    def source_frame_at(self, frame: int) -> int | None:
        """Select the held source image after globally rounding its boundaries.

        This matches the encoder's fps filter: a new source image replaces the
        previous image at its rounded output boundary. Reverse reverses the
        selected output images, including any repeated frames.
        """
        segment = self.segment_at_frame(frame)
        if segment is None or not isinstance(segment.media, VideoRange):
            return None
        if segment.reverse:
            frame = segment.end_frame - 1 - (frame - segment.start_frame)
        media = segment.media
        low, high = media.start_frame, media.end_frame
        while low < high:
            middle = (low + high) // 2
            source_time = (
                segment.source_boundaries[middle - media.start_frame]
                if segment.source_boundaries is not None else Fraction(middle) / media.rate
            )
            boundary = frame_boundary(segment.timeline.start + source_time - media.start, self.output_rate)
            if boundary <= frame:
                low = middle + 1
            else:
                high = middle
        return max(media.start_frame, low - 1)

    def source_seconds_at(self, frame: int) -> Fraction | None:
        index = self.source_frame_at(frame)
        segment = self.segment_at_frame(frame)
        if index is None or segment is None or not isinstance(segment.media, VideoRange):
            return None
        if segment.source_boundaries is not None:
            return segment.source_boundaries[index - segment.media.start_frame]
        return Fraction(index) / segment.media.rate

    def validate_edl(self) -> None:
        """Reject effects CMX 3600 cannot faithfully describe."""
        if self.music_path is not None:
            raise RenderPlanError("CMX 3600 export cannot represent the sequence music track")
        for segment in self.segments:
            if segment.kind == "gap":
                continue
            if segment.hflip or segment.vflip or segment.reverse:
                raise RenderPlanError(f"EDL cannot represent transforms on entry {segment.entry_id}")
            if segment.kind == "still":
                raise RenderPlanError(f"EDL cannot represent still hold {segment.entry_id}")
            if segment.frame_count == 0:
                raise RenderPlanError(f"EDL cannot represent subframe entry {segment.entry_id}")
            if isinstance(segment.media, VideoRange) and segment.media.presentation_range is not None:
                raise RenderPlanError(f"EDL requires a constant-rate source for entry {segment.entry_id}")
            if isinstance(segment.media, VideoRange) and segment.media.rate != self.output_rate:
                raise RenderPlanError(f"EDL requires source and timeline rates to match for entry {segment.entry_id}; conform the source first")


def entry_playback_range(entry: SequenceClip, source_fps: float) -> VideoRange:
    """Resolve the source coordinate contract used by all playback adapters."""
    return playback_range(entry, source_fps)


def entry_timeline_range(entry: SequenceClip, sequence_fps: float, source: Source | None) -> TimelineRange:
    start = playback_start(entry, sequence_fps)
    if entry.is_frame_entry:
        duration = entry.source_range.duration
    elif source is not None:
        duration = entry_playback_range(entry, source.fps).duration
    else:
        raise RenderPlanError(f"Source missing for entry {entry.id}")
    return TimelineRange(start, start + duration)


def _color(value) -> tuple[int, int, int] | None:
    colors = getattr(value, "dominant_colors", None)
    if not colors:
        return None
    if len(colors[0]) < 3:
        return None
    r, g, b = (max(0, min(255, int(component))) for component in colors[0][:3])
    return r, g, b


def _media_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def compile_render_plan(
    sequence: Sequence,
    sources: Mapping[str, Source],
    clips: Mapping[str, tuple[Clip, Source]] | None = None,
    *,
    frames: Mapping[str, Frame] | None = None,
    output_fps: float | Fraction | str | None = None,
    music_path: Path | None = None,
    check_media: bool = True,
) -> RenderPlan:
    """Validate edits once and detach their values from mutable project models.

    Source-only callers (such as the legacy EDL API) may omit the library clip
    lookup. When supplied, its references and source identities are also checked.
    GUI mapping may disable filesystem checks; encoder workers must enable them.
    """
    timeline_rate = frame_rate(sequence.fps)
    output_rate = frame_rate(sequence.fps if output_fps is None else output_fps)
    checked: set[Path] = set()

    def media_path(value: Path | str, label: str) -> Path:
        path = Path(value)
        if check_media and path not in checked:
            if not path.is_file():
                raise RenderPlanError(f"{label} media is missing: {path}")
            checked.add(path)
        return path

    segments = []
    seen: set[str] = set()
    for index, track in enumerate(sequence.tracks):
        for entry in track.clips:
            if entry.id in seen:
                raise RenderPlanError(f"Duplicate sequence entry ID: {entry.id}")
            seen.add(entry.id)
            if type(entry.track_index) is not int or entry.track_index != index:
                raise RenderPlanError(f"Track membership is inconsistent for entry {entry.id}")
            if any(type(getattr(entry, key)) is not bool for key in ("hflip", "vflip", "reverse")):
                raise RenderPlanError(f"Invalid transform flags for entry {entry.id}")
            if entry.timeline_rate is not None and frame_rate(entry.timeline_rate) != timeline_rate:
                raise RenderPlanError(f"Timeline rate changed without retiming entry {entry.id}")
            source = sources.get(entry.source_id)
            boundaries = None
            span = entry_timeline_range(entry, sequence.fps, source)
            start, end = span.frames(output_rate)
            kind: Literal["video", "still", "gap"]
            if entry.is_frame_entry:
                frame = (frames or {}).get(entry.frame_id or "")
                if frame is None:
                    raise RenderPlanError(f"Still image reference missing for entry {entry.id}")
                path = media_path(frame.file_path, "Still image")
                media = entry.source_range
                color = _color(frame)
                kind = "still"
            else:
                if source is None:
                    raise RenderPlanError(f"Source missing for entry {entry.id}")
                original = None
                if clips is not None:
                    pair = clips.get(entry.source_clip_id)
                    if pair is None:
                        raise RenderPlanError(f"Library clip reference missing for entry {entry.id}")
                    original, clip_source = pair
                    if original.source_id != source.id or clip_source.id != source.id:
                        raise RenderPlanError(f"Source reference mismatch for entry {entry.id}")
                media = entry_playback_range(entry, source.fps)
                current = source_video_range(source, entry.in_point, entry.out_point)
                if (media.start, media.end, media.rate) != (current.start, current.end, current.rate):
                    raise RenderPlanError(f"Source timing changed for entry {entry.id}; update its timing before rendering")
                path = media_path(source.file_path, "Source video")
                if source.presentation_boundaries is not None:
                    boundaries = source.presentation_boundaries[entry.in_point:entry.out_point]
                color = _color(original)
                kind = "video"
            segments.append(RenderSegment(
                kind, span, start, end, entry.id, entry.source_id, entry.source_clip_id,
                path, media, entry.hflip, entry.vflip, entry.reverse, color, boundaries,
            ))
    if not segments:
        raise RenderPlanError("Sequence has no entries")
    segments.sort(key=lambda segment: segment.timeline.start)
    result = []
    cursor = Fraction(0)
    for segment in segments:
        if segment.timeline.start < cursor:
            raise RenderPlanError(f"Overlapping tracks or clips require compositing: entry {segment.entry_id}")
        if segment.timeline.start > cursor:
            gap = TimelineRange(cursor, segment.timeline.start)
            result.append(RenderSegment("gap", gap, *gap.frames(output_rate)))
        result.append(segment)
        cursor = segment.timeline.end
    music = music_path if music_path is not None else sequence.music_path
    music_file = media_path(music, "Music") if music else None
    plan = RenderPlan(
        sequence.id, timeline_rate, output_rate, tuple(result),
        music_file,
        tuple((path, _media_stamp(path)) for path in sorted(checked)),
    )
    if plan.frame_count == 0:
        raise RenderPlanError("Sequence is shorter than one output frame at this frame rate")
    return plan
