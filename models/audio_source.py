"""Data model for imported audio files."""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.transcription import TranscriptSegment


@dataclass
class AudioSource:
    """Represents an imported audio file (music, podcast, voiceover, etc.).

    Audio sources are not cut into clips and never appear in sequencer
    output — they exist to feed audio-consuming tools like Staccato and
    transcription.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = field(default_factory=Path)
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    transcript: Optional[list["TranscriptSegment"]] = None

    @property
    def filename(self) -> str:
        return self.file_path.name

    @property
    def duration_str(self) -> str:
        """Formatted duration as MM:SS or HH:MM:SS."""
        total = int(self.duration_seconds)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def to_dict(self, base_path: Optional[Path] = None) -> dict:
        """Serialize to dictionary for JSON export.

        Args:
            base_path: If provided, store file_path relative to this directory
        """
        data: dict = {
            "id": self.id,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
        if self.transcript is not None:
            data["transcript"] = [seg.to_dict() for seg in self.transcript]

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
    def from_dict(cls, data: dict, base_path: Optional[Path] = None) -> "AudioSource":
        """Deserialize from dictionary.

        Args:
            data: Dictionary from JSON
            base_path: Base directory to resolve relative paths against

        Raises:
            ValueError: If path traversal is detected (path escapes base_path)
        """
        from core.transcription import TranscriptSegment

        file_path_str = data.get("file_path", "")
        file_path = Path(file_path_str)

        # Resolve relative path against base_path
        if base_path and not file_path.is_absolute():
            resolved = (base_path / file_path).resolve()
            # Security: validate path doesn't escape base directory
            try:
                resolved.relative_to(base_path.resolve())
            except ValueError:
                raise ValueError(
                    f"Path traversal detected: {file_path_str} escapes project directory"
                )
            file_path = resolved

        # If resolved path doesn't exist, try absolute fallback
        if not file_path.exists() and "_absolute_path" in data:
            fallback = Path(data["_absolute_path"])
            if fallback.exists():
                file_path = fallback

        transcript = None
        if "transcript" in data and data["transcript"] is not None:
            transcript = [TranscriptSegment.from_dict(seg) for seg in data["transcript"]]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            file_path=file_path,
            duration_seconds=data.get("duration_seconds", 0.0),
            sample_rate=data.get("sample_rate", 0),
            channels=data.get("channels", 0),
            transcript=transcript,
        )
