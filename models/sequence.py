"""Data models for timeline sequences."""

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class SequenceClip:
    """A clip placed on the timeline."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_clip_id: str = ""  # Reference to original Clip
    source_id: str = ""  # Reference to Source video
    track_index: int = 0
    start_frame: int = 0  # Position on timeline (in frames)
    in_point: int = 0  # Trim start (frames into source clip)
    out_point: int = 0  # Trim end (frames into source clip)

    @property
    def duration_frames(self) -> int:
        """Duration of this clip on the timeline."""
        return self.out_point - self.in_point

    def start_time(self, fps: float) -> float:
        """Get start time on timeline in seconds."""
        return self.start_frame / fps

    def duration_seconds(self, fps: float) -> float:
        """Get duration in seconds."""
        return self.duration_frames / fps

    def end_frame(self) -> int:
        """Get end frame on timeline."""
        return self.start_frame + self.duration_frames


@dataclass
class Track:
    """A horizontal track containing clips."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Video 1"
    clips: list[SequenceClip] = field(default_factory=list)
    muted: bool = False
    locked: bool = False

    def add_clip(self, clip: SequenceClip) -> None:
        """Add a clip to this track."""
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.start_frame)

    def remove_clip(self, clip_id: str) -> Optional[SequenceClip]:
        """Remove a clip by ID. Returns the removed clip or None."""
        for i, clip in enumerate(self.clips):
            if clip.id == clip_id:
                return self.clips.pop(i)
        return None

    def get_clip_at_frame(self, frame: int) -> Optional[SequenceClip]:
        """Get clip at a specific frame, or None."""
        for clip in self.clips:
            if clip.start_frame <= frame < clip.end_frame():
                return clip
        return None


@dataclass
class Sequence:
    """A timeline composition."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Sequence"
    fps: float = 30.0
    tracks: list[Track] = field(default_factory=list)

    def __post_init__(self):
        """Ensure at least one track exists."""
        if not self.tracks:
            self.tracks.append(Track(name="Video 1"))

    @property
    def duration_frames(self) -> int:
        """Total duration in frames (end of last clip)."""
        max_frame = 0
        for track in self.tracks:
            for clip in track.clips:
                end = clip.end_frame()
                if end > max_frame:
                    max_frame = end
        return max_frame

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.duration_frames / self.fps

    def add_track(self, name: Optional[str] = None) -> Track:
        """Add a new track."""
        track_num = len(self.tracks) + 1
        track = Track(name=name or f"Video {track_num}")
        self.tracks.append(track)
        return track

    def remove_track(self, track_id: str) -> Optional[Track]:
        """Remove a track by ID. Returns removed track or None."""
        for i, track in enumerate(self.tracks):
            if track.id == track_id:
                return self.tracks.pop(i)
        return None

    def get_clip_at(self, track_index: int, frame: int) -> Optional[SequenceClip]:
        """Get clip at specific track and frame."""
        if 0 <= track_index < len(self.tracks):
            return self.tracks[track_index].get_clip_at_frame(frame)
        return None

    def get_all_clips(self) -> list[SequenceClip]:
        """Get all clips in the sequence, sorted by start frame."""
        all_clips = []
        for track in self.tracks:
            all_clips.extend(track.clips)
        return sorted(all_clips, key=lambda c: c.start_frame)
