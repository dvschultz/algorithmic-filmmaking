"""Project save/load functionality."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import uuid

from models.clip import Source, Clip
from models.sequence import Sequence

logger = logging.getLogger(__name__)

# Current project file schema version
SCHEMA_VERSION = "1.0"


class ProjectError(Exception):
    """Base exception for project errors."""
    pass


class ProjectLoadError(ProjectError):
    """Raised when project loading fails."""
    pass


class ProjectSaveError(ProjectError):
    """Raised when project saving fails."""
    pass


class MissingSourceError(ProjectError):
    """Raised when a source video file is missing."""
    def __init__(self, source_path: Path, source_id: str):
        self.source_path = source_path
        self.source_id = source_id
        super().__init__(f"Source video not found: {source_path}")


@dataclass
class ProjectMetadata:
    """Project-level metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Project"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMetadata":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("project_name", "Untitled Project"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            version=data.get("version", "1.0"),
        )


def save_project(
    filepath: Path,
    sources: list[Source],
    clips: list[Clip],
    sequence: Optional[Sequence],
    ui_state: Optional[dict] = None,
    metadata: Optional[ProjectMetadata] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Save project to JSON file with relative paths.

    Args:
        filepath: Path to save the project file
        sources: List of Source objects
        clips: List of Clip objects
        sequence: Optional Sequence object
        ui_state: Optional UI state dict (sensitivity, etc.)
        metadata: Optional ProjectMetadata (created if not provided)
        progress_callback: Optional callback(progress, message)

    Returns:
        True if save succeeded, False otherwise
    """
    if progress_callback:
        progress_callback(0.1, "Preparing project data...")

    # Use project file's parent as base for relative paths
    base_path = filepath.parent

    # Create or update metadata
    if metadata is None:
        metadata = ProjectMetadata(name=filepath.stem)
    else:
        metadata.modified_at = datetime.now().isoformat()

    # Build project data
    project_data = metadata.to_dict()

    # Serialize sources with relative paths
    if progress_callback:
        progress_callback(0.2, "Serializing sources...")

    project_data["sources"] = [
        source.to_dict(base_path=base_path)
        for source in sources
    ]

    # Serialize clips
    if progress_callback:
        progress_callback(0.4, "Serializing clips...")

    project_data["clips"] = [clip.to_dict() for clip in clips]

    # Serialize sequence
    if progress_callback:
        progress_callback(0.6, "Serializing sequence...")

    if sequence:
        project_data["sequence"] = sequence.to_dict()
    else:
        project_data["sequence"] = None

    # Add UI state
    if ui_state:
        project_data["ui_state"] = ui_state

    # Write to file
    if progress_callback:
        progress_callback(0.8, "Writing file...")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        if progress_callback:
            progress_callback(1.0, "Project saved")

        logger.info(f"Project saved to {filepath}")
        return True

    except (OSError, IOError) as e:
        logger.error(f"Failed to save project: {e}")
        if progress_callback:
            progress_callback(0, f"Save failed: {e}")
        return False


def load_project(
    filepath: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    missing_source_callback: Optional[Callable[[Path, str], Optional[Path]]] = None,
) -> tuple[list[Source], list[Clip], Optional[Sequence], ProjectMetadata, dict]:
    """Load project from JSON file, resolving paths and validating sources.

    Args:
        filepath: Path to the project file
        progress_callback: Optional callback(progress, message)
        missing_source_callback: Optional callback(missing_path, source_id) -> new_path or None
            Called when a source video is not found. Return new path to remap, or None to skip.

    Returns:
        Tuple of (sources, clips, sequence, metadata, ui_state)

    Raises:
        ProjectLoadError: If the project file cannot be loaded
        MissingSourceError: If a source video is missing and no callback handles it
    """
    if progress_callback:
        progress_callback(0.1, "Reading project file...")

    # Read and parse JSON
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ProjectLoadError(f"Invalid JSON in project file: {e}")
    except (OSError, IOError) as e:
        raise ProjectLoadError(f"Failed to read project file: {e}")

    # Validate version
    version = data.get("version", "1.0")
    if version > SCHEMA_VERSION:
        logger.warning(f"Project file version {version} is newer than supported {SCHEMA_VERSION}")

    # Use project file's parent as base for relative paths
    base_path = filepath.parent

    # Load metadata
    metadata = ProjectMetadata.from_dict(data)

    if progress_callback:
        progress_callback(0.2, "Loading sources...")

    # Load sources and resolve paths
    sources = []
    sources_by_id = {}
    for source_data in data.get("sources", []):
        source = Source.from_dict(source_data, base_path=base_path)

        # Check if source file exists
        if not source.file_path.exists():
            if missing_source_callback:
                new_path = missing_source_callback(source.file_path, source.id)
                if new_path:
                    source.file_path = new_path
                else:
                    logger.warning(f"Skipping missing source: {source.file_path}")
                    continue
            else:
                raise MissingSourceError(source.file_path, source.id)

        sources.append(source)
        sources_by_id[source.id] = source

    if progress_callback:
        progress_callback(0.5, "Loading clips...")

    # Load clips (only those whose source exists)
    clips = []
    for clip_data in data.get("clips", []):
        source_id = clip_data.get("source_id", "")
        if source_id in sources_by_id:
            clip = Clip.from_dict(clip_data)
            clips.append(clip)
        else:
            logger.warning(f"Skipping clip with missing source: {clip_data.get('id')}")

    if progress_callback:
        progress_callback(0.7, "Loading sequence...")

    # Load sequence
    sequence = None
    if data.get("sequence"):
        sequence = Sequence.from_dict(data["sequence"])

    # Load UI state
    ui_state = data.get("ui_state", {})

    if progress_callback:
        progress_callback(1.0, f"Loaded {len(clips)} clips")

    logger.info(f"Project loaded from {filepath}: {len(sources)} sources, {len(clips)} clips")
    return sources, clips, sequence, metadata, ui_state


def get_project_name_from_path(filepath: Path) -> str:
    """Extract project name from filepath."""
    return filepath.stem
