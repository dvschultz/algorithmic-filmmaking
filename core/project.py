"""Project save/load functionality."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Optional
import uuid

from models.clip import Source, Clip
from models.frame import Frame
from models.sequence import Sequence

logger = logging.getLogger(__name__)

# Current project file schema version
SCHEMA_VERSION = "1.2"


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
    frames: Optional[list[Frame]] = None,
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
        frames: Optional list of Frame objects

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

    # Serialize frames
    if frames:
        project_data["frames"] = [
            frame.to_dict(base_path=base_path)
            for frame in frames
        ]

    # Add UI state
    if ui_state:
        project_data["ui_state"] = ui_state

    # Write to file atomically (write temp, then rename)
    if progress_callback:
        progress_callback(0.8, "Writing file...")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory, then rename
        # This ensures atomic write - file is never in a partial state
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=".project_",
            dir=filepath.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            # Atomic rename (POSIX guarantees this is atomic)
            os.replace(temp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

        if progress_callback:
            progress_callback(1.0, "Project saved")

        logger.info(f"Project saved to {filepath}")
        return True

    except (OSError, IOError) as e:
        logger.error(f"Failed to save project: {e}")
        if progress_callback:
            progress_callback(0, f"Save failed: {e}")
        return False


def _validate_project_structure(data: dict) -> list[str]:
    """Validate basic project file structure.

    Args:
        data: Parsed JSON data

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required top-level fields
    if not isinstance(data, dict):
        errors.append("Project file must be a JSON object")
        return errors

    # Version is required
    if "version" not in data:
        errors.append("Missing required field: version")

    # Sources must be a list
    if "sources" in data and not isinstance(data["sources"], list):
        errors.append("Field 'sources' must be a list")

    # Clips must be a list
    if "clips" in data and not isinstance(data["clips"], list):
        errors.append("Field 'clips' must be a list")

    # Sequence must be a dict or null
    if "sequence" in data:
        seq = data["sequence"]
        if seq is not None and not isinstance(seq, dict):
            errors.append("Field 'sequence' must be an object or null")

    # Validate source entries have required fields
    for i, source in enumerate(data.get("sources", [])):
        if not isinstance(source, dict):
            errors.append(f"sources[{i}] must be an object")
            continue
        if "id" not in source:
            errors.append(f"sources[{i}] missing required field: id")
        if "file_path" not in source:
            errors.append(f"sources[{i}] missing required field: file_path")

    # Validate clip entries have required fields
    for i, clip in enumerate(data.get("clips", [])):
        if not isinstance(clip, dict):
            errors.append(f"clips[{i}] must be an object")
            continue
        if "id" not in clip:
            errors.append(f"clips[{i}] missing required field: id")
        if "source_id" not in clip:
            errors.append(f"clips[{i}] missing required field: source_id")

    # Frames must be a list (optional - not present in old projects)
    if "frames" in data and not isinstance(data["frames"], list):
        errors.append("Field 'frames' must be a list")

    # Validate frame entries have required fields
    for i, frame in enumerate(data.get("frames", [])):
        if not isinstance(frame, dict):
            errors.append(f"frames[{i}] must be an object")
            continue
        if "id" not in frame:
            errors.append(f"frames[{i}] missing required field: id")
        if "file_path" not in frame:
            errors.append(f"frames[{i}] missing required field: file_path")

    return errors


def load_project(
    filepath: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    missing_source_callback: Optional[Callable[[Path, str], Optional[Path]]] = None,
) -> tuple[list[Source], list[Clip], Optional[Sequence], ProjectMetadata, dict, list[Frame]]:
    """Load project from JSON file, resolving paths and validating sources.

    Args:
        filepath: Path to the project file
        progress_callback: Optional callback(progress, message)
        missing_source_callback: Optional callback(missing_path, source_id) -> new_path or None
            Called when a source video is not found. Return new path to remap, or None to skip.

    Returns:
        Tuple of (sources, clips, sequence, metadata, ui_state, frames)

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

    # Validate project structure
    validation_errors = _validate_project_structure(data)
    if validation_errors:
        raise ProjectLoadError(
            f"Invalid project file structure:\n  - " + "\n  - ".join(validation_errors)
        )

    # Validate version using semantic comparison
    version = data.get("version", "1.0")
    try:
        version_parts = tuple(int(x) for x in version.split("."))
        schema_parts = tuple(int(x) for x in SCHEMA_VERSION.split("."))
        if version_parts > schema_parts:
            logger.warning(f"Project file version {version} is newer than supported {SCHEMA_VERSION}")
    except (ValueError, AttributeError):
        logger.warning(f"Invalid version format: {version}")

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
                    # Validate the replacement path exists
                    new_path = Path(new_path)
                    if not new_path.exists():
                        logger.warning(
                            f"Replacement path does not exist: {new_path}, skipping source"
                        )
                        continue
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

        # Validate SequenceClip references - remove clips that reference missing sources/clips
        valid_clip_ids = {clip.id for clip in clips}
        valid_source_ids = set(sources_by_id.keys())

        for track in sequence.tracks:
            invalid_clips = []
            for seq_clip in track.clips:
                # Check that referenced source exists
                if seq_clip.source_id and seq_clip.source_id not in valid_source_ids:
                    logger.warning(
                        f"Removing sequence clip {seq_clip.id}: source {seq_clip.source_id} not found"
                    )
                    invalid_clips.append(seq_clip)
                # Check that referenced clip exists
                elif seq_clip.source_clip_id and seq_clip.source_clip_id not in valid_clip_ids:
                    logger.warning(
                        f"Removing sequence clip {seq_clip.id}: clip {seq_clip.source_clip_id} not found"
                    )
                    invalid_clips.append(seq_clip)

            # Remove invalid clips from track
            for invalid_clip in invalid_clips:
                track.clips.remove(invalid_clip)

    # Load frames (optional - not present in old projects)
    frames = []
    for frame_data in data.get("frames", []):
        frame = Frame.from_dict(frame_data, base_path=base_path)
        frames.append(frame)

    # Load UI state
    ui_state = data.get("ui_state", {})

    if progress_callback:
        progress_callback(1.0, f"Loaded {len(clips)} clips, {len(frames)} frames")

    logger.info(
        f"Project loaded from {filepath}: "
        f"{len(sources)} sources, {len(clips)} clips, {len(frames)} frames"
    )
    return sources, clips, sequence, metadata, ui_state, frames


def get_project_name_from_path(filepath: Path) -> str:
    """Extract project name from filepath."""
    return filepath.stem


class Project:
    """Application state independent of UI.

    This is the single source of truth for project data.
    Both CLI and GUI use this class to manage state.

    Observer Pattern:
        Register callbacks via add_observer() to receive state change notifications.
        Events are strings like "source_added", "clips_added", etc.
        The GUI wraps these callbacks with Qt signals via ProjectSignalAdapter.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        metadata: Optional[ProjectMetadata] = None,
        sources: Optional[list[Source]] = None,
        clips: Optional[list[Clip]] = None,
        sequence: Optional[Sequence] = None,
        ui_state: Optional[dict] = None,
        frames: Optional[list[Frame]] = None,
    ):
        """Initialize a Project.

        Args:
            path: File path (None if unsaved)
            metadata: Project metadata
            sources: List of source videos
            clips: List of detected clips
            sequence: Timeline sequence
            ui_state: Optional UI state dict
            frames: List of extracted/imported frames
        """
        self.path = path
        self.metadata = metadata or ProjectMetadata()
        self._sources = sources or []
        self._clips = clips or []
        self._frames: list[Frame] = frames or []
        self.sequence = sequence
        self.ui_state = ui_state or {}

        self._dirty: bool = False
        self._observers: list[Callable[[str, Any], None]] = []

    # --- Data access (read-only lists) ---

    @property
    def sources(self) -> list[Source]:
        """All source videos in the project."""
        return self._sources

    @property
    def clips(self) -> list[Clip]:
        """All clips from all analyzed sources."""
        return self._clips

    @property
    def frames(self) -> list[Frame]:
        """All frames (extracted and imported)."""
        return self._frames

    # --- Cached property indexes ---

    @cached_property
    def sources_by_id(self) -> dict[str, Source]:
        """Source lookup by ID."""
        return {s.id: s for s in self._sources}

    @cached_property
    def clips_by_id(self) -> dict[str, Clip]:
        """Clip lookup by ID."""
        return {c.id: c for c in self._clips}

    @cached_property
    def clips_by_source(self) -> dict[str, list[Clip]]:
        """Clips organized by source ID."""
        result: dict[str, list[Clip]] = {}
        for clip in self._clips:
            result.setdefault(clip.source_id, []).append(clip)
        return result

    @cached_property
    def frames_by_id(self) -> dict[str, Frame]:
        """Frame lookup by ID."""
        return {f.id: f for f in self._frames}

    @cached_property
    def frames_by_source(self) -> dict[str, list[Frame]]:
        """Frames organized by source ID."""
        result: dict[str, list[Frame]] = {}
        for frame in self._frames:
            if frame.source_id:
                result.setdefault(frame.source_id, []).append(frame)
        return result

    @cached_property
    def frames_by_clip(self) -> dict[str, list[Frame]]:
        """Frames organized by clip ID."""
        result: dict[str, list[Frame]] = {}
        for frame in self._frames:
            if frame.clip_id:
                result.setdefault(frame.clip_id, []).append(frame)
        return result

    def _invalidate_caches(self) -> None:
        """Clear cached properties when data changes."""
        for attr in (
            "sources_by_id", "clips_by_id", "clips_by_source",
            "frames_by_id", "frames_by_source", "frames_by_clip",
        ):
            self.__dict__.pop(attr, None)

    # --- Observer pattern ---

    def add_observer(self, callback: Callable[[str, Any], None]) -> None:
        """Register an observer for state changes.

        Args:
            callback: Function called with (event_name, data) on state changes.
                Events: source_added, clips_added, clips_updated, sequence_changed,
                        project_saved, project_loaded
        """
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[str, Any], None]) -> None:
        """Unregister an observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self, event: str, data: Any = None) -> None:
        """Notify all observers of state change."""
        for observer in self._observers:
            try:
                observer(event, data)
            except Exception as e:
                logger.warning(f"Observer error: {e}")

    # --- State operations ---

    def add_source(self, source: Source) -> None:
        """Add a source video to the project.

        Args:
            source: Source to add
        """
        self._sources.append(source)
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("source_added", source)

    def remove_source(self, source_id: str) -> Optional[Source]:
        """Remove a source by ID.

        Also removes all clips associated with this source.

        Args:
            source_id: ID of the source to remove

        Returns:
            The removed source, or None if not found
        """
        source = self.sources_by_id.get(source_id)
        if source is None:
            return None

        self._sources.remove(source)
        # Remove associated clips and frames
        self._clips = [c for c in self._clips if c.source_id != source_id]
        self._frames = [f for f in self._frames if f.source_id != source_id]
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("source_removed", source)
        return source

    def add_clips(self, clips: list[Clip]) -> None:
        """Add detected clips to the project.

        Args:
            clips: Clips to add
        """
        self._clips.extend(clips)
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("clips_added", clips)

    def update_clips(self, clips: list[Clip]) -> None:
        """Update existing clips (e.g., after color analysis).

        The clips should already be in the project (same IDs).

        Args:
            clips: Updated clips
        """
        # Clips are updated in-place, just notify and mark dirty
        self._dirty = True
        self._notify_observers("clips_updated", clips)

    def remove_clips(self, clip_ids: list[str]) -> list[Clip]:
        """Remove clips by ID.

        Args:
            clip_ids: IDs of clips to remove

        Returns:
            List of removed Clip objects
        """
        ids_to_remove = set(clip_ids)
        removed = [c for c in self._clips if c.id in ids_to_remove]
        self._clips = [c for c in self._clips if c.id not in ids_to_remove]
        if removed:
            self._invalidate_caches()
            self._dirty = True
            self._notify_observers("clips_removed", removed)
        return removed

    def replace_source_clips(self, source_id: str, new_clips: list[Clip]) -> None:
        """Replace all clips for a source (e.g., after re-detection).

        Args:
            source_id: The source whose clips are being replaced
            new_clips: New clips for this source
        """
        self._clips = [c for c in self._clips if c.source_id != source_id]
        self._clips.extend(new_clips)
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("clips_added", new_clips)

    # --- Frame state operations ---

    def add_frames(self, frames: list[Frame]) -> None:
        """Add frames to the project.

        Args:
            frames: Frame objects to add
        """
        self._frames.extend(frames)
        self._invalidate_caches()
        self._dirty = True
        self._notify_observers("frames_added", frames)

    def remove_frames(self, frame_ids: list[str]) -> list[Frame]:
        """Remove frames by ID.

        Args:
            frame_ids: IDs of frames to remove

        Returns:
            List of removed Frame objects
        """
        ids_to_remove = set(frame_ids)
        removed = [f for f in self._frames if f.id in ids_to_remove]
        self._frames = [f for f in self._frames if f.id not in ids_to_remove]
        if removed:
            self._invalidate_caches()
            self._dirty = True
            self._notify_observers("frames_removed", removed)
        return removed

    def update_frame(self, frame_id: str, **kwargs) -> Optional[Frame]:
        """Update a frame's fields.

        Args:
            frame_id: ID of the frame to update
            **kwargs: Fields to update (e.g., shot_type="wide", analyzed=True)

        Returns:
            Updated Frame, or None if not found
        """
        frame = self.frames_by_id.get(frame_id)
        if frame is None:
            logger.warning(f"Frame not found: {frame_id}")
            return None
        for key, value in kwargs.items():
            if hasattr(frame, key):
                setattr(frame, key, value)
        self._dirty = True
        self._notify_observers("frames_updated", [frame])
        return frame

    def add_frames_to_sequence(
        self, frame_ids: list[str], hold_frames: int = 1
    ) -> None:
        """Add frames to the sequence.

        Args:
            frame_ids: IDs of frames to add
            hold_frames: Number of timeline frames each frame occupies
        """
        from models.sequence import SequenceClip

        if self.sequence is None:
            fps = self._sources[0].fps if self._sources else 30.0
            self.sequence = Sequence(name=self.metadata.name, fps=fps)

        current_frame = self.sequence.duration_frames
        track = self.sequence.tracks[0]

        for frame_id in frame_ids:
            frame = self.frames_by_id.get(frame_id)
            if frame is None:
                logger.warning(f"Frame not found: {frame_id}")
                continue

            seq_clip = SequenceClip(
                frame_id=frame.id,
                source_id=frame.source_id or "",
                track_index=0,
                start_frame=current_frame,
                hold_frames=hold_frames,
            )
            track.add_clip(seq_clip)
            current_frame += seq_clip.duration_frames

        self._dirty = True
        self._notify_observers("sequence_changed", frame_ids)

    # --- Sequence operations ---

    def add_to_sequence(self, clip_ids: list[str]) -> None:
        """Add clips to the sequence by ID.

        Args:
            clip_ids: IDs of clips to add to the sequence
        """
        from models.sequence import SequenceClip

        if self.sequence is None:
            # Create sequence with FPS from first source
            fps = self._sources[0].fps if self._sources else 30.0
            self.sequence = Sequence(name=self.metadata.name, fps=fps)

        # Get current end frame
        current_frame = self.sequence.duration_frames
        track = self.sequence.tracks[0]

        for clip_id in clip_ids:
            clip = self.clips_by_id.get(clip_id)
            if clip is None:
                logger.warning(f"Clip not found: {clip_id}")
                continue

            source = self.sources_by_id.get(clip.source_id)
            if source is None:
                logger.warning(f"Source not found for clip: {clip_id}")
                continue

            seq_clip = SequenceClip(
                source_clip_id=clip.id,
                source_id=clip.source_id,
                track_index=0,
                start_frame=current_frame,
                in_point=clip.start_frame,
                out_point=clip.end_frame,
            )
            track.add_clip(seq_clip)
            current_frame += seq_clip.duration_frames

        self._dirty = True
        self._notify_observers("sequence_changed", clip_ids)

    def remove_from_sequence(self, clip_ids: list[str]) -> list[str]:
        """Remove clips from the sequence by their sequence clip IDs.

        Args:
            clip_ids: IDs of sequence clips to remove (not source clip IDs)

        Returns:
            List of IDs that were successfully removed
        """
        if self.sequence is None:
            return []

        removed = []
        track = self.sequence.tracks[0]

        for clip_id in clip_ids:
            removed_clip = track.remove_clip(clip_id)
            if removed_clip:
                removed.append(clip_id)
            else:
                logger.warning(f"Sequence clip not found: {clip_id}")

        if removed:
            # Recalculate start frames after removal
            self._recalculate_sequence_positions()
            self._dirty = True
            self._notify_observers("sequence_changed", removed)

        return removed

    def clear_sequence(self) -> int:
        """Clear all clips from the sequence.

        Returns:
            Number of clips that were cleared
        """
        if self.sequence is None:
            return 0

        track = self.sequence.tracks[0]
        count = len(track.clips)
        track.clips.clear()

        if count > 0:
            self._dirty = True
            self._notify_observers("sequence_changed", [])

        return count

    def reorder_sequence(self, clip_ids: list[str]) -> bool:
        """Reorder clips in the sequence to match the provided order.

        Args:
            clip_ids: Sequence clip IDs in the desired order

        Returns:
            True if reorder succeeded, False otherwise
        """
        if self.sequence is None:
            return False

        track = self.sequence.tracks[0]

        # Build lookup of current clips
        clips_by_id = {clip.id: clip for clip in track.clips}

        # Validate all IDs exist
        for clip_id in clip_ids:
            if clip_id not in clips_by_id:
                logger.warning(f"Cannot reorder: clip not found: {clip_id}")
                return False

        # Reorder clips based on provided order
        reordered = [clips_by_id[clip_id] for clip_id in clip_ids]

        # Add any clips not in the provided list at the end (preserve them)
        existing_ids = set(clip_ids)
        for clip in track.clips:
            if clip.id not in existing_ids:
                reordered.append(clip)

        track.clips = reordered
        self._recalculate_sequence_positions()

        self._dirty = True
        self._notify_observers("sequence_changed", clip_ids)
        return True

    def _recalculate_sequence_positions(self) -> None:
        """Recalculate start_frame for all sequence clips after reorder/removal."""
        if self.sequence is None:
            return

        for track in self.sequence.tracks:
            current_frame = 0
            for clip in track.clips:
                clip.start_frame = current_frame
                current_frame += clip.duration_frames

    # --- Dirty state tracking ---

    @property
    def is_dirty(self) -> bool:
        """Check if project has unsaved changes."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark project as having unsaved changes."""
        self._dirty = True

    def mark_clean(self) -> None:
        """Mark project as saved (no unsaved changes)."""
        self._dirty = False

    # --- Persistence ---

    def save(
        self,
        path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> bool:
        """Save project to file.

        Args:
            path: Path to save to (uses self.path if not specified)
            progress_callback: Optional progress callback

        Returns:
            True if save succeeded

        Raises:
            ValueError: If no path specified and project has no path
        """
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified for save")

        success = save_project(
            filepath=save_path,
            sources=self._sources,
            clips=self._clips,
            sequence=self.sequence,
            ui_state=self.ui_state,
            metadata=self.metadata,
            progress_callback=progress_callback,
            frames=self._frames,
        )

        if success:
            self.path = save_path
            self.mark_clean()
            self._notify_observers("project_saved", save_path)

        return success

    @classmethod
    def load(
        cls,
        path: Path,
        missing_source_callback: Optional[Callable[[Path, str], Optional[Path]]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> "Project":
        """Load project from file.

        Args:
            path: Path to the project file
            missing_source_callback: Callback when source video is missing
            progress_callback: Optional progress callback

        Returns:
            Loaded Project instance

        Raises:
            ProjectLoadError: If the project file cannot be loaded
            MissingSourceError: If a source video is missing
        """
        sources, clips, sequence, metadata, ui_state, frames = load_project(
            filepath=path,
            missing_source_callback=missing_source_callback,
            progress_callback=progress_callback,
        )

        project = cls(
            path=path,
            metadata=metadata,
            sources=sources,
            clips=clips,
            sequence=sequence,
            ui_state=ui_state,
            frames=frames,
        )
        project._dirty = False
        return project

    @classmethod
    def new(cls, name: str = "Untitled Project") -> "Project":
        """Create a new empty project.

        Args:
            name: Project name

        Returns:
            New empty Project instance
        """
        return cls(metadata=ProjectMetadata(name=name))

    def clear(self) -> None:
        """Clear all project data (for 'New Project')."""
        self._sources = []
        self._clips = []
        self._frames = []
        self.sequence = None
        self.ui_state = {}
        self.path = None
        self.metadata = ProjectMetadata()
        self._dirty = False
        self._invalidate_caches()
        self._notify_observers("project_cleared", None)

    def __repr__(self) -> str:
        return (
            f"Project(name={self.metadata.name!r}, "
            f"sources={len(self._sources)}, clips={len(self._clips)}, "
            f"frames={len(self._frames)}, dirty={self._dirty})"
        )
