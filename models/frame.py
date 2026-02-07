"""Data model for individual frames (extracted or imported images)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from models.cinematography import CinematographyAnalysis


@dataclass
class Frame:
    """Represents an individual frame (extracted from video or imported image).

    Attributes:
        id: Unique identifier (UUID)
        file_path: Path to the image file on disk (PNG/JPG/etc.)
        source_id: Source video UUID (if extracted from a video)
        clip_id: Clip UUID (if extracted from a specific clip)
        frame_number: Original frame number in source video (if extracted)
        thumbnail_path: Resized thumbnail for browser display
        width: Image width in pixels
        height: Image height in pixels
        analyzed: Whether analysis has been run on this frame
        shot_type: Classification result (e.g., "wide", "medium", "close-up")
        dominant_colors: RGB color tuples from color analysis
        description: Natural language description from VLM
        detected_objects: Object detection results [{label, confidence, bbox}]
        extracted_texts: Text extracted via OCR
        cinematography: Rich cinematography analysis
        tags: User-defined tags
        notes: User notes/comments
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = field(default_factory=Path)
    source_id: Optional[str] = None
    clip_id: Optional[str] = None
    frame_number: Optional[int] = None
    thumbnail_path: Optional[Path] = None
    width: Optional[int] = None
    height: Optional[int] = None
    analyzed: bool = False
    # Analysis metadata (mirrors Clip fields)
    shot_type: Optional[str] = None
    dominant_colors: Optional[list[tuple[int, int, int]]] = None
    description: Optional[str] = None
    description_model: Optional[str] = None
    detected_objects: Optional[list[dict]] = None
    extracted_texts: Optional[list] = None
    cinematography: Optional["CinematographyAnalysis"] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def display_name(self) -> str:
        """Get a human-readable display name for this frame.

        Returns:
            Frame number if available, otherwise filename.
        """
        if self.frame_number is not None:
            return f"Frame {self.frame_number}"
        return self.file_path.name

    def to_dict(self, base_path: Optional[Path] = None) -> dict:
        """Serialize to dictionary for JSON export.

        Args:
            base_path: If provided, store file_path relative to this directory
        """
        data: dict = {
            "id": self.id,
            "analyzed": self.analyzed,
        }

        # Store path (relative if base_path provided)
        if base_path:
            try:
                data["file_path"] = self.file_path.relative_to(base_path).as_posix()
            except ValueError:
                data["file_path"] = self.file_path.as_posix()
            data["_absolute_path"] = self.file_path.as_posix()
        else:
            data["file_path"] = self.file_path.as_posix()

        # Optional provenance fields
        if self.source_id is not None:
            data["source_id"] = self.source_id
        if self.clip_id is not None:
            data["clip_id"] = self.clip_id
        if self.frame_number is not None:
            data["frame_number"] = self.frame_number

        # Dimensions
        if self.width is not None:
            data["width"] = self.width
        if self.height is not None:
            data["height"] = self.height

        # Analysis metadata
        if self.shot_type:
            data["shot_type"] = self.shot_type
        if self.dominant_colors:
            data["dominant_colors"] = [
                {"r": int(r), "g": int(g), "b": int(b)}
                for r, g, b in self.dominant_colors
            ]
        if self.description:
            data["description"] = self.description
        if self.description_model:
            data["description_model"] = self.description_model
        if self.detected_objects:
            data["detected_objects"] = self.detected_objects
        if self.extracted_texts:
            data["extracted_texts"] = [
                et.to_dict() if hasattr(et, "to_dict") else et
                for et in self.extracted_texts
            ]
        if self.cinematography:
            data["cinematography"] = self.cinematography.to_dict()
        if self.tags:
            data["tags"] = self.tags
        if self.notes:
            data["notes"] = self.notes

        return data

    @classmethod
    def from_dict(cls, data: dict, base_path: Optional[Path] = None) -> "Frame":
        """Deserialize from dictionary.

        Args:
            data: Dictionary from JSON
            base_path: Base directory to resolve relative paths against

        Raises:
            ValueError: If path traversal is detected
        """
        from models.cinematography import CinematographyAnalysis
        from models.clip import ExtractedText

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

        # Parse colors
        colors = None
        if "dominant_colors" in data:
            colors = [
                (c["r"], c["g"], c["b"])
                for c in data["dominant_colors"]
            ]

        # Parse extracted texts
        extracted_texts = None
        if "extracted_texts" in data:
            extracted_texts = [
                ExtractedText.from_dict(et) if isinstance(et, dict) else et
                for et in data["extracted_texts"]
            ]

        # Parse cinematography
        cinematography = None
        if "cinematography" in data:
            cinematography = CinematographyAnalysis.from_dict(data["cinematography"])

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            file_path=file_path,
            source_id=data.get("source_id"),
            clip_id=data.get("clip_id"),
            frame_number=data.get("frame_number"),
            thumbnail_path=None,  # Regenerate on load
            width=data.get("width"),
            height=data.get("height"),
            analyzed=data.get("analyzed", False),
            shot_type=data.get("shot_type"),
            dominant_colors=colors,
            description=data.get("description"),
            description_model=data.get("description_model"),
            detected_objects=data.get("detected_objects"),
            extracted_texts=extracted_texts,
            cinematography=cinematography,
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )
