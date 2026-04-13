"""Data models for timeline sequences."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SequenceClip:
    """A clip or frame placed on the timeline.

    For clip-based entries: source_clip_id and source_id are set, frame_id is None.
    For frame-based entries: frame_id is set, source_clip_id may be empty.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_clip_id: str = ""  # Reference to original Clip
    source_id: str = ""  # Reference to Source video
    track_index: int = 0
    start_frame: int = 0  # Position on timeline (in frames)
    in_point: int = 0  # Trim start (frames into source clip)
    out_point: int = 0  # Trim end (frames into source clip)
    frame_id: Optional[str] = None  # Reference to Frame (if frame-based)
    hold_frames: int = 1  # Number of timeline frames to hold (for frame entries)
    hflip: bool = False  # Random horizontal flip
    vflip: bool = False  # Random vertical flip
    reverse: bool = False  # Random reverse playback
    prerendered_path: Optional[str] = None  # Path to pre-rendered clip with baked transforms
    rationale: Optional[str] = None  # LLM-generated rationale for why this clip follows the previous (Free Association sequencer)

    @property
    def is_frame_entry(self) -> bool:
        """Whether this is a frame-based entry (vs. clip-based)."""
        return self.frame_id is not None

    @property
    def duration_frames(self) -> int:
        """Duration of this entry on the timeline."""
        if self.is_frame_entry:
            return self.hold_frames
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

    def to_dict(self, base_path: Optional[Path] = None) -> dict:
        """Serialize to dictionary for JSON export.

        Args:
            base_path: If provided, store prerendered_path relative to this directory.
        """
        data = {
            "id": self.id,
            "source_clip_id": self.source_clip_id,
            "source_id": self.source_id,
            "track_index": self.track_index,
            "start_frame": self.start_frame,
            "in_point": self.in_point,
            "out_point": self.out_point,
        }
        if self.frame_id is not None:
            data["frame_id"] = self.frame_id
        if self.hold_frames != 1:
            data["hold_frames"] = self.hold_frames
        if self.hflip:
            data["hflip"] = True
        if self.vflip:
            data["vflip"] = True
        if self.reverse:
            data["reverse"] = True
        if self.prerendered_path is not None:
            if base_path:
                try:
                    rel = Path(self.prerendered_path).relative_to(base_path)
                    data["prerendered_path"] = rel.as_posix()
                except ValueError:
                    data["prerendered_path"] = self.prerendered_path
            else:
                data["prerendered_path"] = self.prerendered_path
        if self.rationale is not None:
            data["rationale"] = self.rationale
        return data

    @classmethod
    def from_dict(cls, data: dict, base_path: Optional[Path] = None) -> "SequenceClip":
        """Deserialize from dictionary.

        Args:
            data: Dictionary from JSON.
            base_path: Base directory to resolve relative prerendered_path against.
        """
        prerendered = data.get("prerendered_path")
        if prerendered and base_path:
            p = Path(prerendered)
            if not p.is_absolute():
                p = (base_path / p).resolve()
            prerendered = str(p)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_clip_id=data.get("source_clip_id", ""),
            source_id=data.get("source_id", ""),
            track_index=data.get("track_index", 0),
            start_frame=data.get("start_frame", 0),
            in_point=data.get("in_point", 0),
            out_point=data.get("out_point", 0),
            frame_id=data.get("frame_id"),
            hold_frames=data.get("hold_frames", 1),
            hflip=data.get("hflip", False),
            vflip=data.get("vflip", False),
            reverse=data.get("reverse", False),
            prerendered_path=prerendered,
            rationale=data.get("rationale"),
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

    def to_dict(self, base_path: Optional[Path] = None) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.id,
            "name": self.name,
            "clips": [clip.to_dict(base_path=base_path) for clip in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict, base_path: Optional[Path] = None) -> "Track":
        """Deserialize from dictionary."""
        clips = [
            SequenceClip.from_dict(clip_data, base_path=base_path)
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
    # Reference-guided remixing fields
    reference_source_id: Optional[str] = None  # Source used as structural guide
    dimension_weights: Optional[dict[str, float]] = None  # Dimension -> weight (0.0-1.0)
    allow_repeats: bool = False  # Allow same clip matched to multiple positions
    show_chromatic_color_bar: bool = False  # Optional Chromatic Flow bottom bar
    music_path: Optional[str] = None  # Path to music file (staccato sequences)

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

    def to_dict(self, base_path: Optional[Path] = None) -> dict:
        """Serialize to dictionary for JSON export."""
        data = {
            "id": self.id,
            "name": self.name,
            "fps": self.fps,
            "tracks": [track.to_dict(base_path=base_path) for track in self.tracks],
        }
        if self.algorithm:
            data["algorithm"] = self.algorithm
        if self.reference_source_id:
            data["reference_source_id"] = self.reference_source_id
        if self.dimension_weights:
            data["dimension_weights"] = self.dimension_weights
        if self.allow_repeats:
            data["allow_repeats"] = self.allow_repeats
        if self.show_chromatic_color_bar:
            data["show_chromatic_color_bar"] = self.show_chromatic_color_bar
        if self.music_path:
            if base_path:
                try:
                    rel = Path(self.music_path).relative_to(base_path)
                    data["music_path"] = rel.as_posix()
                except ValueError:
                    data["music_path"] = self.music_path
            else:
                data["music_path"] = self.music_path
        return data

    @classmethod
    def from_dict(cls, data: dict, base_path: Optional[Path] = None) -> "Sequence":
        """Deserialize from dictionary."""
        tracks = [
            Track.from_dict(track_data, base_path=base_path)
            for track_data in data.get("tracks", [])
        ]
        # Resolve music_path relative to base_path if needed
        music_path = data.get("music_path")
        if music_path and base_path:
            p = Path(music_path)
            if not p.is_absolute():
                resolved = (base_path / p).resolve()
                music_path = str(resolved) if resolved.exists() else None
                if not music_path:
                    logger.warning(
                        "Music file not found: %s", base_path / p
                    )

        seq = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Sequence"),
            fps=data.get("fps", 30.0),
            tracks=tracks if tracks else [],  # Empty list to skip __post_init__ default
            algorithm=data.get("algorithm"),
            reference_source_id=data.get("reference_source_id"),
            dimension_weights=data.get("dimension_weights"),
            allow_repeats=data.get("allow_repeats", False),
            show_chromatic_color_bar=data.get("show_chromatic_color_bar", False),
            music_path=music_path,
        )
        # If no tracks were loaded, ensure at least one exists
        if not seq.tracks:
            seq.tracks.append(Track(name="Video 1"))
        return seq
