"""Data models for video sources and clips."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid


@dataclass
class Source:
    """Represents an imported video file."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = field(default_factory=Path)
    duration_seconds: float = 0.0
    fps: float = 30.0
    width: int = 0
    height: int = 0

    @property
    def filename(self) -> str:
        return self.file_path.name

    @property
    def total_frames(self) -> int:
        return int(self.duration_seconds * self.fps)


@dataclass
class Clip:
    """Represents a detected scene/clip within a source video."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    start_frame: int = 0
    end_frame: int = 0
    thumbnail_path: Optional[Path] = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def start_time(self, fps: float) -> float:
        """Get start time in seconds."""
        return self.start_frame / fps

    def end_time(self, fps: float) -> float:
        """Get end time in seconds."""
        return self.end_frame / fps

    def duration_seconds(self, fps: float) -> float:
        """Get duration in seconds."""
        return self.duration_frames / fps
