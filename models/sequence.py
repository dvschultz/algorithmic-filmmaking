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

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.id,
            "source_clip_id": self.source_clip_id,
            "source_id": self.source_id,
            "track_index": self.track_index,
            "start_frame": self.start_frame,
            "in_point": self.in_point,
            "out_point": self.out_point,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceClip":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_clip_id=data.get("source_clip_id", ""),
            source_id=data.get("source_id", ""),
            track_index=data.get("track_index", 0),
            start_frame=data.get("start_frame", 0),
            in_point=data.get("in_point", 0),
            out_point=data.get("out_point", 0),
        )


@dataclass
class Track:
    """A horizontal track containing clips."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Video 1"
    clips: list[SequenceClip] = field(default_factory=list)

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

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.id,
            "name": self.name,
            "clips": [clip.to_dict() for clip in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        """Deserialize from dictionary."""
        clips = [
            SequenceClip.from_dict(clip_data)
            for clip_data in data.get("clips", [])
        ]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Video 1"),
            clips=clips,
        )


@dataclass
class Sequence:
    """A timeline composition."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Sequence"
    fps: float = 30.0
    tracks: list[Track] = field(default_factory=list)
    algorithm: Optional[str] = None  # e.g., "storyteller", "color", "shot_type"

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

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data = {
            "id": self.id,
            "name": self.name,
            "fps": self.fps,
            "tracks": [track.to_dict() for track in self.tracks],
        }
        if self.algorithm:
            data["algorithm"] = self.algorithm
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Sequence":
        """Deserialize from dictionary."""
        tracks = [
            Track.from_dict(track_data)
            for track_data in data.get("tracks", [])
        ]
        seq = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Sequence"),
            fps=data.get("fps", 30.0),
            tracks=tracks if tracks else [],  # Empty list to skip __post_init__ default
            algorithm=data.get("algorithm"),
        )
        # If no tracks were loaded, ensure at least one exists
        if not seq.tracks:
            seq.tracks.append(Track(name="Video 1"))
        return seq
