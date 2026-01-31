"""Data models for video sources and clips."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from core.transcription import TranscriptSegment


@dataclass
class ExtractedText:
    """Text extracted from a video frame using OCR or VLM.

    Attributes:
        frame_number: Frame number where text was extracted
        text: The extracted text content
        confidence: Confidence score from 0.0 to 1.0
        source: Extraction method ("tesseract" or "vlm")
        bounding_boxes: Optional list of text bounding boxes with format
            [{"x": int, "y": int, "w": int, "h": int, "text": str}, ...]
    """

    frame_number: int
    text: str
    confidence: float
    source: str  # "tesseract" or "vlm"
    bounding_boxes: Optional[list[dict]] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data = {
            "frame_number": self.frame_number,
            "text": self.text,
            "confidence": self.confidence,
            "source": self.source,
        }
        if self.bounding_boxes:
            data["bounding_boxes"] = self.bounding_boxes
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedText":
        """Deserialize from dictionary."""
        return cls(
            frame_number=data.get("frame_number", 0),
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            source=data.get("source", "unknown"),
            bounding_boxes=data.get("bounding_boxes"),
        )


@dataclass
class Source:
    """Represents an imported video file."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = field(default_factory=Path)
    duration_seconds: float = 0.0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    analyzed: bool = False  # Has this source been analyzed for scenes?
    thumbnail_path: Optional[Path] = None  # Thumbnail for library grid

    @property
    def filename(self) -> str:
        return self.file_path.name

    @property
    def total_frames(self) -> int:
        return int(self.duration_seconds * self.fps)

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height).

        Returns:
            Aspect ratio as float, or 0.0 if height is 0.
        """
        if self.height == 0:
            return 0.0
        return self.width / self.height

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
            "analyzed": self.analyzed,
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

        Raises:
            ValueError: If path traversal is detected (path escapes base_path)
        """
        file_path_str = data.get("file_path", "")
        file_path = Path(file_path_str)

        # Resolve relative path against base_path
        if base_path and not file_path.is_absolute():
            resolved = (base_path / file_path).resolve()
            # Security: Validate path doesn't escape base directory
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

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            file_path=file_path,
            duration_seconds=data.get("duration_seconds", 0.0),
            fps=data.get("fps", 30.0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            analyzed=data.get("analyzed", False),
            thumbnail_path=None,  # Regenerate on load
        )


@dataclass
class Clip:
    """Represents a detected scene/clip within a source video."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    start_frame: int = 0
    end_frame: int = 0
    name: str = ""  # Custom clip name (empty = use auto-generated)
    thumbnail_path: Optional[Path] = None
    dominant_colors: Optional[list[tuple[int, int, int]]] = None  # RGB tuples
    shot_type: Optional[str] = None  # e.g., "wide", "medium", "close-up"
    transcript: Optional[list["TranscriptSegment"]] = None  # Speech transcription segments
    tags: list[str] = field(default_factory=list)  # User-defined tags for organization
    notes: str = ""  # User notes/comments about the clip
    # Content analysis fields
    object_labels: Optional[list[str]] = None  # ImageNet labels, e.g., ["dog", "car", "tree"]
    detected_objects: Optional[list[dict]] = None  # [{label, confidence, bbox}]
    person_count: Optional[int] = None  # Number of people detected
    # Video description fields
    description: Optional[str] = None  # Natural language description
    description_model: Optional[str] = None  # Model that generated it (e.g., "moondream-2b", "gpt-4o")
    description_frames: Optional[int] = None  # 1 for single frame, N for temporal
    # OCR extracted text
    extracted_texts: Optional[list[ExtractedText]] = None  # Text extracted from frames

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

    def display_name(self, source_filename: str = "", fps: float = 30.0) -> str:
        """Get display name (custom name or auto-generated fallback).

        Args:
            source_filename: Source video filename for fallback generation
            fps: Frames per second for timecode calculation

        Returns:
            Custom name if set, otherwise "filename - timecode" format
        """
        if self.name:
            return self.name
        # Auto-generate: source filename - start timecode
        start_time = self.start_time(fps)
        m = int(start_time // 60)
        s = int(start_time % 60)
        timecode = f"{m}:{s:02d}"
        if source_filename:
            return f"{source_filename} - {timecode}"
        return timecode

    def get_transcript_text(self) -> str:
        """Get full transcript text from all segments."""
        if not self.transcript:
            return ""
        return " ".join(seg.text for seg in self.transcript)

    @property
    def combined_text(self) -> Optional[str]:
        """Get deduplicated text from all OCR extractions.

        Combines text from all extracted frames, removing duplicates
        while preserving the original text casing.

        Returns:
            Combined text with unique phrases separated by " | ",
            or None if no text has been extracted.
        """
        if not self.extracted_texts:
            return None
        unique_texts = []
        seen = set()
        for et in self.extracted_texts:
            normalized = et.text.strip().lower()
            if normalized and normalized not in seen:
                unique_texts.append(et.text.strip())
                seen.add(normalized)
        return " | ".join(unique_texts) if unique_texts else None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        data = {
            "id": self.id,
            "source_id": self.source_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        # Custom name (only if set)
        if self.name:
            data["name"] = self.name
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
        # Tags and notes (only if non-empty)
        if self.tags:
            data["tags"] = self.tags
        if self.notes:
            data["notes"] = self.notes
        # Content analysis fields
        if self.object_labels:
            data["object_labels"] = self.object_labels
        if self.detected_objects:
            data["detected_objects"] = self.detected_objects
        if self.person_count is not None:
            data["person_count"] = self.person_count
        # Description fields
        if self.description:
            data["description"] = self.description
        if self.description_model:
            data["description_model"] = self.description_model
        if self.description_frames:
            data["description_frames"] = self.description_frames
        # OCR extracted text
        if self.extracted_texts:
            data["extracted_texts"] = [et.to_dict() for et in self.extracted_texts]
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

        # Parse extracted text
        extracted_texts = None
        if "extracted_texts" in data:
            extracted_texts = [
                ExtractedText.from_dict(et)
                for et in data["extracted_texts"]
            ]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data.get("source_id", ""),
            start_frame=data.get("start_frame", 0),
            end_frame=data.get("end_frame", 0),
            name=data.get("name", ""),  # Backwards compatible: defaults to empty
            thumbnail_path=None,  # Regenerate on load
            dominant_colors=colors,
            shot_type=data.get("shot_type"),
            transcript=transcript,
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            # Content analysis fields
            object_labels=data.get("object_labels"),
            detected_objects=data.get("detected_objects"),
            person_count=data.get("person_count"),
            # Description fields
            description=data.get("description"),
            description_model=data.get("description_model"),
            description_frames=data.get("description_frames"),
            # OCR extracted text
            extracted_texts=extracted_texts,
        )
