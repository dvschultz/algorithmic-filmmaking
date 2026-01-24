"""Data models for video sources and clips."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from core.transcription import TranscriptSegment


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

    def to_dict(self, base_path: Optional[Path] = None) -> dict:
        """Serialize to dictionary for JSON export.

        Args:
            base_path: If provided, store file_path relative to this directory
        """
        data = {
            "id": self.id,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }
        # Store relative path if base_path provided, with absolute fallback
        if base_path:
            try:
                data["file_path"] = self.file_path.relative_to(base_path).as_posix()
            except ValueError:
                # Can't make relative (different drives on Windows, etc.)
                data["file_path"] = self.file_path.as_posix()
            data["_absolute_path"] = self.file_path.as_posix()
        else:
            data["file_path"] = self.file_path.as_posix()
        return data

    @classmethod
    def from_dict(cls, data: dict, base_path: Optional[Path] = None) -> "Source":
        """Deserialize from dictionary.

        Args:
            data: Dictionary from JSON
            base_path: Base directory to resolve relative paths against
        """
        file_path_str = data.get("file_path", "")
        file_path = Path(file_path_str)

        # Resolve relative path against base_path
        if base_path and not file_path.is_absolute():
            file_path = (base_path / file_path).resolve()

        # If resolved path doesn't exist, try absolute fallback
        if not file_path.exists() and "_absolute_path" in data:
            fallback = Path(data["_absolute_path"])
            if fallback.exists():
                file_path = fallback

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            file_path=file_path,
            duration_seconds=data.get("duration_seconds", 0.0),
            fps=data.get("fps", 30.0),
            width=data.get("width", 0),
            height=data.get("height", 0),
        )


@dataclass
class Clip:
    """Represents a detected scene/clip within a source video."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    start_frame: int = 0
    end_frame: int = 0
    thumbnail_path: Optional[Path] = None
    dominant_colors: Optional[list[tuple[int, int, int]]] = None  # RGB tuples
    shot_type: Optional[str] = None  # e.g., "wide", "medium", "close-up"
    transcript: Optional[list["TranscriptSegment"]] = None  # Speech transcription segments

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

    def get_transcript_text(self) -> str:
        """Get full transcript text from all segments."""
        if not self.transcript:
            return ""
        return " ".join(seg.text for seg in self.transcript)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data = {
            "id": self.id,
            "source_id": self.source_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        # Colors as RGB objects
        if self.dominant_colors:
            data["dominant_colors"] = [
                {"r": int(r), "g": int(g), "b": int(b)}
                for r, g, b in self.dominant_colors
            ]
        if self.shot_type:
            data["shot_type"] = self.shot_type
        # Transcript segments
        if self.transcript:
            data["transcript"] = [seg.to_dict() for seg in self.transcript]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Clip":
        """Deserialize from dictionary."""
        from core.transcription import TranscriptSegment

        # Parse colors back to tuples
        colors = None
        if "dominant_colors" in data:
            colors = [
                (c["r"], c["g"], c["b"])
                for c in data["dominant_colors"]
            ]

        # Parse transcript segments
        transcript = None
        if "transcript" in data:
            transcript = [
                TranscriptSegment.from_dict(seg)
                for seg in data["transcript"]
            ]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data.get("source_id", ""),
            start_frame=data.get("start_frame", 0),
            end_frame=data.get("end_frame", 0),
            thumbnail_path=None,  # Regenerate on load
            dominant_colors=colors,
            shot_type=data.get("shot_type"),
            transcript=transcript,
        )
