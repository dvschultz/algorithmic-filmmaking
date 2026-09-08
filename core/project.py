"""Project save/load functionality."""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Optional
import uuid

from models.audio_source import AudioSource
from models.clip import Source, Clip
from models.frame import Frame
from models.sequence import Sequence, SequenceClip
from core.project_lock import ProjectWriter
from core.project_migrations import (
    SCHEMA_VERSION, is_future_schema, migrate_project_data, prepare_project_write,
)

logger = logging.getLogger(__name__)


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
    job_results: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "version": self.version,
            "job_results": dict(self.job_results),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMetadata":
        receipts = data.get("job_results", {})
        if is_future_schema(data.get("version", "1.0")):
            receipts = {}
        if not isinstance(receipts, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) != 64
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in key + value)
            for key, value in receipts.items()
        ):
            raise ValueError("Invalid project job result receipts")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("project_name", "Untitled Project"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            version=data.get("version", "1.0"),
            job_results=dict(receipts),
        )


def _prepare_prerendered_clips(
    sequence: Sequence,
    base_path: Path,
    additional_sequences: Optional[list[Sequence]] = None,
) -> dict[str, str]:
    """Copy/link pre-rendered clip files into the project's transformed_clips/ folder.

    Returns a mapping of original absolute paths to project-local absolute paths.
    Does **not** mutate any SequenceClip objects — the caller applies the mapping
    only during serialization so in-memory state is never changed by save.

    Hard links are preferred (near-instant, no extra disk space).  Falls back to
    an exclusive byte copy when a hard link fails (e.g. cross-device).

    If a destination filename already exists with different content, a numeric
    suffix is appended to avoid silent overwrites (e.g. ``name_2.mp4``).

    Args:
        sequence: The primary sequence (typically the active one).
        base_path: Project directory for relative path resolution.
        additional_sequences: Extra sequences to include in the prerender map.
    """
    dest_dir = base_path / "transformed_clips"
    mapping: dict[str, str] = {}  # original_path -> project_local_path

    # Collect clips from all sequences
    all_clips = list(sequence.get_all_clips())
    if additional_sequences:
        for seq in additional_sequences:
            all_clips.extend(seq.get_all_clips())

    for clip in all_clips:
        if not clip.prerendered_path:
            continue
        if clip.prerendered_path in mapping:
            continue  # Already processed (shared across sequences)
        src = Path(clip.prerendered_path)
        if not src.exists():
            continue
        dest = dest_dir / src.name

        # Already inside the project folder — nothing to copy
        if dest == src.resolve():
            mapping[clip.prerendered_path] = str(dest)
            continue

        # Handle filename collisions: if dest exists with different content,
        # pick a unique name by appending a numeric suffix.
        dest_dir.mkdir(parents=True, exist_ok=True)

        while True:
            dest = _unique_dest(src, dest)
            if dest.exists():
                break  # Existing content was verified by _unique_dest.
            try:
                os.link(src, dest)
            except FileExistsError:
                continue  # A competing publication won; choose again.
            except OSError:
                try:
                    output = dest.open("xb")
                except FileExistsError:
                    continue
                try:
                    with output, src.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output)
                except BaseException:
                    dest.unlink(missing_ok=True)
                    raise
            break

        mapping[clip.prerendered_path] = str(dest)

    return mapping


def _strip_absolute_paths(data: Any) -> None:
    """Remove machine-local fallbacks before publishing a portable document."""
    if isinstance(data, dict):
        data.pop("_absolute_path", None)
        data.pop("_thumbnail_absolute_path", None)
        for value in data.values():
            _strip_absolute_paths(value)
    elif isinstance(data, list):
        for value in data:
            _strip_absolute_paths(value)


def _same_content(src: Path, dest: Path) -> bool:
    """Compare media bytes without relying on file size as an identity."""
    if src.samefile(dest):
        return True
    if src.stat().st_size != dest.stat().st_size:
        return False
    with src.open("rb") as left, dest.open("rb") as right:
        while chunk := left.read(1024 * 1024):
            if chunk != right.read(len(chunk)):
                return False
        return not right.read(1)


def _unique_dest(src: Path, dest: Path) -> Path:
    """Return *dest* if it doesn't exist or has the same content as *src*.

    Otherwise append ``_2``, ``_3``, ... before the extension until a
    non-colliding name is found.
    """
    if not dest.exists():
        return dest

    if _same_content(src, dest):
        return dest

    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists() or _same_content(src, candidate):
            return candidate
        counter += 1


def save_project(
    filepath: Path,
    sources: list[Source],
    clips: list[Clip],
    sequence: Optional[Sequence],
    ui_state: Optional[dict] = None,
    metadata: Optional[ProjectMetadata] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    frames: Optional[list[Frame]] = None,
    extra_data: Optional[dict] = None,
    audio_sources: Optional[list[AudioSource]] = None,
) -> bool:
    """Save under cross-process ownership, preserving the public bool contract."""
    from core.project_lock import project_writer

    try:
        with project_writer(filepath) as writer:
            return _save_project_owned(
                writer.path, sources, clips, sequence, ui_state, metadata,
                progress_callback, frames, extra_data, audio_sources,
            )
    except OSError as exc:
        logger.error("Cannot acquire project writer: %s", exc)
        return False


def _save_project_owned(
    filepath: Path,
    sources: list[Source],
    clips: list[Clip],
    sequence: Optional[Sequence],
    ui_state: Optional[dict] = None,
    metadata: Optional[ProjectMetadata] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    frames: Optional[list[Frame]] = None,
    extra_data: Optional[dict] = None,
    audio_sources: Optional[list[AudioSource]] = None,
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
        extra_data: Optional dict merged into project_data (used by Project.save()
            to write multi-sequence data). Existing callers don't need to pass this.

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        prepare_project_write(filepath, metadata.version if metadata else SCHEMA_VERSION)
    except (OSError, ValueError) as exc:
        logger.error("Cannot save project: %s", exc)
        return False

    if progress_callback:
        progress_callback(0.1, "Preparing project data...")

    # Use project file's parent as base for relative paths
    base_path = filepath.parent

    # Create or update metadata
    if metadata is None:
        metadata = ProjectMetadata(name=filepath.stem)
    else:
        metadata.modified_at = datetime.now().isoformat()
        metadata.version = SCHEMA_VERSION

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

    project_data["clips"] = [clip.to_dict(base_path=base_path) for clip in clips]

    # Copy/link pre-rendered clips into project folder
    if progress_callback:
        progress_callback(0.5, "Copying pre-rendered clips...")

    all_sequences = extra_data.get("_all_sequences") if extra_data else None
    if all_sequences is None:
        all_sequences = [sequence] if sequence is not None else []
        active_idx = 0
    else:
        active_idx = (extra_data or {}).get("active_sequence_index", 0)
    if all_sequences and not 0 <= active_idx < len(all_sequences):
        logger.error("Cannot save project: invalid active sequence index")
        return False

    present_sequences = [item for item in all_sequences if item is not None]
    prerender_map = (
        _prepare_prerendered_clips(
            present_sequences[0], base_path, additional_sequences=present_sequences[1:],
        ) if present_sequences else {}
    )

    def serialize_sequence(item: Optional[Sequence]) -> Optional[dict]:
        if item is None:
            return None
        originals = []
        for clip in item.get_all_clips():
            if clip.prerendered_path in prerender_map:
                originals.append((clip, clip.prerendered_path))
                clip.prerendered_path = prerender_map[clip.prerendered_path]
        try:
            return item.to_dict(base_path=base_path)
        finally:
            for clip, original in originals:
                clip.prerendered_path = original

    if progress_callback:
        progress_callback(0.6, "Serializing sequences...")
    project_data["sequences"] = [serialize_sequence(item) for item in all_sequences]
    project_data["active_sequence_index"] = active_idx
    # A derived compatibility projection, never merged with the destination.
    project_data["sequence"] = (
        project_data["sequences"][active_idx] if all_sequences else None
    )

    # Serialize frames
    if frames:
        project_data["frames"] = [
            frame.to_dict(base_path=base_path)
            for frame in frames
        ]

    # Serialize audio sources
    if audio_sources:
        project_data["audio_sources"] = [
            audio.to_dict(base_path=base_path)
            for audio in audio_sources
        ]

    # Add UI state
    if ui_state:
        project_data["ui_state"] = ui_state

    if extra_data:
        project_data.update({
            key: value for key, value in extra_data.items()
            if not key.startswith("_") and key != "active_sequence_index"
        })

    # Write to file atomically (write temp, then rename)
    if extra_data and extra_data.get("_portable"):
        _strip_absolute_paths(project_data)

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
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (POSIX guarantees this is atomic)
            from core.project_lock import replace_project_file

            replace_project_file(temp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

        if project_data.get("job_results"):
            try:
                from core.jobs.gui_checkpoints import checkpoint_saved_gui_results

                checkpoint_saved_gui_results(filepath, project_data)
            except Exception:
                # The project file is already durable. A retry must acknowledge
                # its receipts, not report a failed save or repeat inference.
                logger.warning("Project saved; GUI result checkpoints remain pending", exc_info=True)

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

    # Sequences (v1.4+) must be a list of dicts if present
    if "sequences" in data:
        seqs = data["sequences"]
        if not isinstance(seqs, list):
            errors.append("Field 'sequences' must be a list")
        else:
            for i, seq in enumerate(seqs):
                if not isinstance(seq, (dict, type(None))):
                    errors.append(f"sequences[{i}] must be an object or null")

    def entries(field_name: str) -> list:
        value = data.get(field_name, [])
        return value if isinstance(value, list) else []

    # Validate source entries have required fields
    for i, source in enumerate(entries("sources")):
        if not isinstance(source, dict):
            errors.append(f"sources[{i}] must be an object")
            continue
        if "id" not in source:
            errors.append(f"sources[{i}] missing required field: id")
        if "file_path" not in source:
            errors.append(f"sources[{i}] missing required field: file_path")

    # Validate clip entries have required fields
    for i, clip in enumerate(entries("clips")):
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

    # Audio sources must be a list (optional - not present in old projects)
    if "audio_sources" in data and not isinstance(data["audio_sources"], list):
        errors.append("Field 'audio_sources' must be a list")

    # Validate frame entries have required fields
    for i, frame in enumerate(entries("frames")):
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
) -> tuple[list[Source], list[Clip], Optional[Sequence], ProjectMetadata, dict, list[Frame], list[AudioSource]]:
    """Load project from JSON file, resolving paths and validating sources.

    Args:
        filepath: Path to the project file
        progress_callback: Optional callback(progress, message)
        missing_source_callback: Optional callback(missing_path, source_id) -> new_path or None
            Return a valid replacement path to relink, or None to keep it offline.

    Returns:
        Tuple of (sources, clips, sequence, metadata, ui_state, frames, audio_sources)

    Raises:
        ProjectLoadError: If the project file cannot be loaded
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
            "Invalid project file structure:\n  - " + "\n  - ".join(validation_errors)
        )

    try:
        data = migrate_project_data(data)
    except ValueError as exc:
        raise ProjectLoadError(str(exc)) from exc

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

        # Offline media remains model data; declining a relink must not delete edits.
        if not source.file_path.exists():
            if missing_source_callback:
                new_path = missing_source_callback(source.file_path, source.id)
                if new_path:
                    # Validate the replacement path exists
                    new_path = Path(new_path)
                    if not new_path.exists():
                        logger.warning(
                            f"Replacement path does not exist: {new_path}; keeping original reference"
                        )
                    else:
                        source.file_path = new_path
                else:
                    logger.warning(f"Keeping offline source: {source.file_path}")
            else:
                logger.warning(f"Keeping offline source: {source.file_path}")

        sources.append(source)
        sources_by_id[source.id] = source

    if progress_callback:
        progress_callback(0.5, "Loading clips...")

    # Load clips whose source is declared, including offline media.
    clips = []
    for clip_data in data.get("clips", []):
        source_id = clip_data.get("source_id", "")
        if source_id in sources_by_id:
            clip = Clip.from_dict(clip_data, base_path=base_path)
            clips.append(clip)
        else:
            logger.warning(f"Skipping clip with missing source: {clip_data.get('id')}")

    if progress_callback:
        progress_callback(0.7, "Loading sequence...")

    # Load sequence (use "sequences" key if present for v1.4+, else "sequence")
    sequence = None
    if isinstance(data.get("sequences"), list) and data["sequences"]:
        active_idx = data.get("active_sequence_index", 0)
        active_idx = min(active_idx, len(data["sequences"]) - 1)
        seq_data = data["sequences"][active_idx]
        if seq_data:
            sequence = Sequence.from_dict(seq_data, base_path=base_path)
    elif data.get("sequence"):
        sequence = Sequence.from_dict(data["sequence"], base_path=base_path)

    if sequence:
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

    # Load audio sources (optional — not present in old projects)
    audio_sources: list[AudioSource] = []
    for audio_data in data.get("audio_sources", []):
        if not isinstance(audio_data, dict):
            logger.warning(f"Skipping non-dict audio_source entry: {audio_data!r}")
            continue
        try:
            audio_sources.append(AudioSource.from_dict(audio_data, base_path=base_path))
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Skipping malformed audio source: {e}")

    # Load UI state
    ui_state = data.get("ui_state", {})

    if progress_callback:
        progress_callback(1.0, f"Loaded {len(clips)} clips, {len(frames)} frames")

    logger.info(
        f"Project loaded from {filepath}: "
        f"{len(sources)} sources, {len(clips)} clips, "
        f"{len(frames)} frames, {len(audio_sources)} audio sources"
    )
    return sources, clips, sequence, metadata, ui_state, frames, audio_sources


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
        sequences: Optional[list[Sequence]] = None,
        active_sequence_index: int = 0,
        audio_sources: Optional[list[AudioSource]] = None,
    ):
        """Initialize a Project.

        Args:
            path: File path (None if unsaved)
            metadata: Project metadata
            sources: List of source videos
            clips: List of detected clips
            sequence: Legacy single-sequence parameter (used when sequences is None)
            ui_state: Optional UI state dict
            frames: List of extracted/imported frames
            sequences: List of all sequences (takes precedence over sequence)
            active_sequence_index: Index of the active sequence in the list
            audio_sources: List of imported audio files (not cut into clips)
        """
        self._retain_writer = False
        self._writer: Optional[ProjectWriter] = None
        self._pending_writer: Optional[ProjectWriter] = None
        self.path = path
        self.metadata = metadata or ProjectMetadata()
        self._sources = sources or []
        self._clips = clips or []
        self._frames: list[Frame] = frames or []
        self._audio_sources: list[AudioSource] = audio_sources or []
        self.ui_state = ui_state or {}

        # Multi-sequence storage: sequences list + active index
        if sequences is not None:
            self.sequences = sequences
        elif sequence is not None:
            self.sequences = [sequence]
        else:
            self.sequences = [Sequence()]
        self.active_sequence_index = min(active_sequence_index, max(0, len(self.sequences) - 1))

        self._dirty: bool = False
        self._mutation_generation: int = 0
        self._observers: list[Callable[[str, Any], None]] = []
        from core.project_session import ProjectSession

        self.session = ProjectSession(self)

    # --- Sequence compatibility property ---

    @property
    def sequence(self) -> Optional[Sequence]:
        """Active sequence (compatibility property).

        Returns the active sequence from the sequences list. Setter replaces
        the active entry. Assigning None substitutes a fresh empty Sequence
        to preserve the at-least-one invariant (R2).
        """
        if not self.sequences:
            return None
        return self.sequences[self.active_sequence_index]

    @sequence.setter
    def sequence(self, value: Optional[Sequence]) -> None:
        if value is None:
            value = Sequence()
        if not self.sequences:
            self.sequences = [value]
            self.active_sequence_index = 0
        else:
            self.sequences[self.active_sequence_index] = value

    # --- Multi-sequence management ---

    def add_sequence(self, sequence: Sequence, *, activate: bool = False) -> None:
        """Append a sequence and optionally activate it as one reversible edit."""
        from core.commands.sequences import EditSequences

        self.session.assert_owner()
        self.session.execute(EditSequences.add(self, sequence, activate))

    def remove_sequence(self, index: int) -> None:
        """Remove a sequence; deleting the last creates an undoable empty fallback."""
        from core.commands.sequences import EditSequences

        self.session.assert_owner()
        if not 0 <= index < len(self.sequences):
            return
        self.session.execute(EditSequences.remove(self, index))

    def rename_sequence(self, index: int, name: str) -> None:
        """Rename a sequence through shared history."""
        if not 0 <= index < len(self.sequences):
            raise IndexError("Sequence index out of range")
        self.update_sequence_metadata(self.sequences[index], name=name)

    def update_sequence_metadata(self, sequence: Sequence, **changes: Any) -> None:
        """Apply a validated group of sequence settings atomically."""
        from core.commands.sequences import EditSequenceMetadata

        self.session.assert_owner()
        self.session.execute(EditSequenceMetadata.capture(self, sequence, changes))

    def set_active_sequence(self, index: int) -> None:
        """Switch the active sequence.

        Args:
            index: Index of the sequence to activate

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.sequences):
            raise IndexError(f"Sequence index {index} out of range (0-{len(self.sequences) - 1})")
        self.active_sequence_index = index
        self._notify_observers("active_sequence_changed", self.active_sequence_index)

    def source_in_sequences(self, source_id: str) -> list[str]:
        """Return names of sequences that contain clips from the given source.

        Used as a guard before deleting a source — deletion should be blocked
        when the source's clips are in use.

        Args:
            source_id: The source ID to check

        Returns:
            List of sequence names containing clips from this source (empty if none)
        """
        names = []
        for seq in self.sequences + self.session.retained_sequences:
            if seq.name in names:
                continue
            if any(sc.source_id == source_id for sc in seq.get_all_clips()):
                names.append(seq.name)
        for name in self.session.retained_sources.get(source_id, []):
            if name not in names:
                names.append(name)
        return names

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

    @property
    def audio_sources(self) -> list[AudioSource]:
        """All imported audio files in the project."""
        return self._audio_sources

    # --- Cached property indexes ---

    @cached_property
    def sources_by_id(self) -> dict[str, Source]:
        """Source lookup by ID."""
        return {s.id: s for s in self._sources}

    @cached_property
    def audio_sources_by_id(self) -> dict[str, AudioSource]:
        """Audio source lookup by ID."""
        return {a.id: a for a in self._audio_sources}

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
            "audio_sources_by_id",
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
        self._assert_writable()
        self._sources.append(source)
        self._invalidate_caches()
        self.mark_dirty()
        self._notify_observers("source_added", source)

    def remove_source(self, source_id: str) -> Optional[Source]:
        """Remove a source and its references as one reversible edit."""
        removed = self.remove_sources([source_id])
        return removed[0] if removed else None

    def remove_sources(self, source_ids: list[str]) -> list[Source]:
        """Remove sources, clips, frames and sequence entries without deleting files."""
        from core.commands.sources import RemoveSources

        self.session.assert_owner()
        return self.session.execute(RemoveSources.capture(self, source_ids))

    # Fields that can be updated via update_source()
    _UPDATABLE_SOURCE_FIELDS = {"color_profile", "fps", "analyzed", "name", "has_analysis"}

    def update_source(self, source_id: str, **kwargs) -> Optional[Source]:
        """Update a source's fields.

        Only fields in _UPDATABLE_SOURCE_FIELDS can be modified.

        Args:
            source_id: ID of the source to update
            **kwargs: Fields to update (e.g., color_profile="grayscale", fps=29.97)

        Returns:
            Updated Source, or None if not found

        Raises:
            ValueError: If attempting to set a disallowed field
        """
        self._assert_writable()
        invalid = set(kwargs) - self._UPDATABLE_SOURCE_FIELDS
        if invalid:
            raise ValueError(f"Cannot update protected fields: {invalid}")

        source = self.sources_by_id.get(source_id)
        if source is None:
            logger.warning(f"Source not found: {source_id}")
            return None
        for key, value in kwargs.items():
            setattr(source, key, value)
        self.mark_dirty()
        self._notify_observers("source_updated", source)
        return source

    def add_audio_source(self, audio_source: AudioSource) -> None:
        """Add an imported audio file to the project.

        Args:
            audio_source: AudioSource to add
        """
        self._assert_writable()
        self._audio_sources.append(audio_source)
        self._invalidate_caches()
        self.mark_dirty()
        self._notify_observers("audio_sources_changed", self._audio_sources)

    def remove_audio_source(self, audio_source_id: str) -> Optional[AudioSource]:
        """Remove an audio source by ID.

        Args:
            audio_source_id: ID of the audio source to remove

        Returns:
            The removed AudioSource, or None if not found
        """
        self._assert_writable()
        audio = self.audio_sources_by_id.get(audio_source_id)
        if audio is None:
            return None
        self._audio_sources.remove(audio)
        self._invalidate_caches()
        self.mark_dirty()
        self._notify_observers("audio_sources_changed", self._audio_sources)
        return audio

    def get_audio_source(self, audio_source_id: str) -> Optional[AudioSource]:
        """Look up an audio source by ID."""
        return self.audio_sources_by_id.get(audio_source_id)

    def set_audio_transcript(self, audio_source_id: str, segments: list) -> None:
        """Publish an audio transcript, including a successful silent result."""
        from copy import deepcopy

        self._assert_writable()
        audio = self.get_audio_source(audio_source_id)
        if audio is None:
            raise ValueError(f"Audio source not found: {audio_source_id}")
        audio.transcript = deepcopy(segments)
        self.mark_dirty()
        self._notify_observers("audio_sources_changed", self._audio_sources)

    def add_clips(self, clips: list[Clip]) -> None:
        """Add detected clips to the project.

        Args:
            clips: Clips to add
        """
        self._assert_writable()
        self._clips.extend(clips)
        self._invalidate_caches()
        self.mark_dirty()
        self._notify_observers("clips_added", clips)

    def update_clips(self, clips: list[Clip]) -> None:
        """Update existing clips (e.g., after color analysis).

        The clips should already be in the project (same IDs).

        Args:
            clips: Updated clips
        """
        self._assert_writable()
        # Clips are updated in-place, just notify and mark dirty
        self.mark_dirty()
        self._notify_observers("clips_updated", clips)

    def edit_metadata(self, kind: str, updates: dict[str, dict[str, Any]]) -> list[Any]:
        """Apply explicit editorial fields as one edit; analysis uses update methods."""
        from core.commands.metadata import EditMetadata

        self.session.assert_owner()
        return self.session.execute(EditMetadata.capture(self, kind, updates))

    def update_clip_metadata(self, clip_id: str, **changes: Any) -> None:
        self.edit_metadata("clip", {clip_id: changes})

    def update_frame_metadata(self, frame_id: str, **changes: Any) -> None:
        self.edit_metadata("frame", {frame_id: changes})

    def update_source_metadata(self, source_id: str, **changes: Any) -> None:
        self.edit_metadata("source", {source_id: changes})

    def rename(self, name: str) -> None:
        self.edit_metadata("project", {self.metadata.id: {"name": name}})

    def remove_clips(self, clip_ids: list[str]) -> list[Clip]:
        """Remove clips by ID.

        Args:
            clip_ids: IDs of clips to remove

        Returns:
            List of removed Clip objects
        """
        self._assert_writable()
        ids_to_remove = set(clip_ids)
        removed = [c for c in self._clips if c.id in ids_to_remove]
        self._clips = [c for c in self._clips if c.id not in ids_to_remove]
        if removed:
            self._invalidate_caches()
            self.mark_dirty()
            self._notify_observers("clips_removed", removed)
        return removed

    def toggle_clips_disabled(self, clip_ids: list[str]) -> list[Clip]:
        """Toggle the disabled state of clips by ID.

        Args:
            clip_ids: IDs of clips to toggle

        Returns:
            List of toggled Clip objects
        """
        return self.set_clips_disabled(clip_ids, None)

    def set_clips_disabled(self, clip_ids: list[str], disabled: bool | None) -> list[Clip]:
        """Set or toggle clip states as one reversible session edit."""
        from core.commands.clip_disabled import SetClipsDisabled

        self.session.assert_owner()
        return self.session.execute(SetClipsDisabled.capture(self, clip_ids, disabled))

    @property
    def enabled_clips(self) -> list[Clip]:
        """All clips that are not disabled."""
        return [c for c in self._clips if not c.disabled]

    def replace_source_clips(self, source_id: str, new_clips: list[Clip]) -> None:
        """Replace all clips for a source (e.g., after re-detection).

        Args:
            source_id: The source whose clips are being replaced
            new_clips: New clips for this source
        """
        self._assert_writable()
        self._clips = [c for c in self._clips if c.source_id != source_id]
        self._clips.extend(new_clips)
        self._invalidate_caches()
        self.mark_dirty()
        self._notify_observers("clips_added", new_clips)

    # --- Frame state operations ---

    def add_frames(self, frames: list[Frame]) -> None:
        """Add frames to the project.

        Args:
            frames: Frame objects to add
        """
        self._assert_writable()
        self._frames.extend(frames)
        self._invalidate_caches()
        self.mark_dirty()
        self._notify_observers("frames_added", frames)

    def remove_frames(self, frame_ids: list[str]) -> list[Frame]:
        """Remove frames by ID.

        Args:
            frame_ids: IDs of frames to remove

        Returns:
            List of removed Frame objects
        """
        self._assert_writable()
        ids_to_remove = set(frame_ids)
        removed = [f for f in self._frames if f.id in ids_to_remove]
        self._frames = [f for f in self._frames if f.id not in ids_to_remove]
        if removed:
            self._invalidate_caches()
            self.mark_dirty()
            self._notify_observers("frames_removed", removed)
        return removed

    # Fields that can be updated via update_frame()
    _UPDATABLE_FRAME_FIELDS = {
        "shot_type", "dominant_colors", "description", "description_model", "transcript",
        "detected_objects", "person_count", "analyzed", "cinematography", "object_labels",
        "extracted_texts",
    }

    def update_frame(self, frame_id: str, **kwargs) -> Optional[Frame]:
        """Update a frame's fields.

        Only fields in _UPDATABLE_FRAME_FIELDS can be modified.

        Args:
            frame_id: ID of the frame to update
            **kwargs: Fields to update (e.g., shot_type="wide", analyzed=True)

        Returns:
            Updated Frame, or None if not found

        Raises:
            ValueError: If attempting to set a disallowed field
        """
        self._assert_writable()
        invalid = set(kwargs) - self._UPDATABLE_FRAME_FIELDS
        if invalid:
            raise ValueError(f"Cannot update protected fields: {invalid}")

        frame = self.frames_by_id.get(frame_id)
        if frame is None:
            logger.warning(f"Frame not found: {frame_id}")
            return None
        for key, value in kwargs.items():
            setattr(frame, key, value)
        self.mark_dirty()
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
        from core.sequence_time import still_entry

        if self.sequence is None:
            fps = self._sources[0].fps if self._sources else 30.0
            self.sequence = Sequence(name=self.metadata.name, fps=fps)

        position = self.sequence.duration_time
        entries = []

        for frame_id in frame_ids:
            frame = self.frames_by_id.get(frame_id)
            if frame is None:
                logger.warning(f"Frame not found: {frame_id}")
                continue

            seq_clip = still_entry(
                frame, timeline_fps=self.sequence.fps, start=position,
                hold_frames=hold_frames,
            )
            entries.append(seq_clip)
            position = seq_clip.timeline_range.end

        self.insert_sequence_clips(entries)

    # --- Sequence operations ---

    def add_to_sequence(self, clip_ids: list[str]) -> None:
        """Add clips to the sequence by ID.

        Args:
            clip_ids: IDs of clips to add to the sequence
        """
        from core.sequence_time import video_entry

        if self.sequence is None:
            # Create sequence with FPS from first source
            fps = self._sources[0].fps if self._sources else 30.0
            self.sequence = Sequence(name=self.metadata.name, fps=fps)

        # Get current end frame
        position = self.sequence.duration_time
        entries = []

        for clip_id in clip_ids:
            clip = self.clips_by_id.get(clip_id)
            if clip is None:
                logger.warning(f"Clip not found: {clip_id}")
                continue

            if clip.disabled:
                logger.info(f"Skipping disabled clip: {clip_id}")
                continue

            source = self.sources_by_id.get(clip.source_id)
            if source is None:
                logger.warning(f"Source not found for clip: {clip_id}")
                continue

            seq_clip = video_entry(
                clip, source, timeline_fps=self.sequence.fps, start=position,
            )
            entries.append(seq_clip)
            position = seq_clip.timeline_range.end

        self.insert_sequence_clips(entries)

    def resolve_sequence_timing(
        self, sequence_id: str, entry_id: str, convention: str,
    ) -> SequenceClip:
        """Resolve an ambiguous legacy trim as one reversible editorial choice."""
        from core.commands.sequence_time import ResolveSequenceTiming
        from core.legacy_sequence_time import CoordinateConvention, convert_legacy_entry
        from core.sequence_time import sequence_source_input
        from typing import cast

        self.session.assert_owner()
        if convention not in ("source", "clip-relative"):
            raise ValueError("Choose source or clip-relative coordinates")
        sequence = next((s for s in self.sequences if s.id == sequence_id), None)
        if sequence is None:
            raise ValueError("Sequence not found")
        entry = next((e for e in sequence.get_all_clips() if e.id == entry_id), None)
        if entry is None or not entry.legacy_timing:
            raise ValueError("Legacy sequence entry not found")
        clip = self.clips_by_id.get(entry.source_clip_id)
        source = self.sources_by_id.get(entry.source_id)
        before = entry.to_dict()
        after = convert_legacy_entry(
            before, clip.to_dict() if clip else None, source.to_dict() if source else None,
            sequence.fps, cast(CoordinateConvention, convention),
        )
        self.session.execute(ResolveSequenceTiming(sequence, entry, before, after, sequence_source_input(self, entry)))
        return entry

    def insert_sequence_clips(
        self, clips: list[SequenceClip], *, sequence: Sequence | None = None,
        create_tracks: bool = False,
    ) -> list[SequenceClip]:
        """Insert prepared timeline entries as one reversible edit."""
        from core.commands.sequence_clips import EditSequenceClips

        self.session.assert_owner()
        target = sequence if sequence is not None else self.sequence
        if target is None:
            return []
        return self.session.execute(EditSequenceClips.insert(target, clips, create_tracks=create_tracks))

    def remove_from_sequence(
        self, clip_ids: list[str], *, sequence: Sequence | None = None,
        ripple: bool = True,
    ) -> list[str]:
        """Remove timeline IDs in one edit; optionally close gaps on affected tracks."""
        from core.commands.sequence_clips import EditSequenceClips

        self.session.assert_owner()
        target = sequence if sequence is not None else self.sequence
        if target is None:
            return []
        removed = self.session.execute(EditSequenceClips.remove(target, clip_ids, ripple=ripple))
        return [clip.id for clip in removed]

    def clear_sequence(self, *, sequence: Sequence | None = None) -> int:
        """Clear every track in one reversible edit; return the removed count."""
        from core.commands.sequence_clips import EditSequenceClips

        self.session.assert_owner()
        target = sequence if sequence is not None else self.sequence
        if target is None:
            return 0
        return len(self.session.execute(EditSequenceClips.clear(target)))

    def reorder_sequence(
        self, clip_ids: list[str], *, sequence: Sequence | None = None, track_index: int = 0,
    ) -> bool:
        """Reorder a sequence track as one reversible edit."""
        from core.commands.sequence_clips import EditSequenceClips

        self.session.assert_owner()
        target = sequence if sequence is not None else self.sequence
        if target is None:
            return False
        try:
            command = EditSequenceClips.reorder(target, clip_ids, track_index=track_index)
        except ValueError:
            return False
        self.session.execute(command)
        return True

    def update_sequence_clip(
        self, clip_id: str, *, sequence: Sequence | None = None, **changes,
    ) -> list[SequenceClip]:
        """Validate and apply timing, track or transform changes atomically."""
        from core.commands.sequence_clips import EditSequenceClips

        self.session.assert_owner()
        target = sequence if sequence is not None else self.sequence
        if target is None:
            raise ValueError("No sequence exists")
        return self.session.execute(EditSequenceClips.update(target, clip_id, changes, project=self))

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

    @property
    def is_read_only(self) -> bool:
        """Future schemas may be inspected but cannot be edited or saved."""
        return is_future_schema(self.metadata.version)

    def _assert_writable(self) -> None:
        if self.is_read_only:
            raise RuntimeError("Project uses a newer schema and is read-only")

    @property
    def mutation_generation(self) -> int:
        """Monotonic counter incremented for every project mutation."""
        return getattr(self, "_mutation_generation", 0)

    def mark_dirty(self) -> None:
        """Mark project as having unsaved changes."""
        self._mutation_generation = self.mutation_generation + 1
        self._dirty = True
        self.session.record_external_change()

    def mark_clean(self) -> None:
        """Mark project as saved (no unsaved changes)."""
        self._dirty = False
        self.session.record_saved()

    def record_job_result(self, result_id: str, digest: str) -> None:
        """Include a validated job receipt in the next atomic project save."""
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in (result_id, digest)
        ):
            raise ValueError("Invalid job receipt identity")

        def record() -> None:
            existing = self.metadata.job_results.get(result_id)
            if existing is not None and existing != digest:
                raise ValueError("Job receipt conflicts with an existing result")
            self.metadata.job_results[result_id] = digest
            self.mark_dirty()

        self.session.apply_external(record)

    # --- Persistence ---

    @property
    def save_in_progress(self) -> bool:
        return self._pending_writer is not None

    def prepare_save(self, path: Path) -> Optional[ProjectWriter]:
        """Acquire a destination while retaining the current session lease."""
        from core.project_lock import ProjectWriter

        if self._retain_writer:
            self.session.assert_owner()
        if self._pending_writer is not None:
            raise RuntimeError("Project save already in progress")
        if not self._retain_writer:
            return None
        canonical = Path(path).expanduser().resolve()
        writer = self._writer
        if writer is None or writer.path != canonical:
            writer = ProjectWriter(canonical).acquire()
        self._pending_writer = writer
        return writer

    def finish_save(self, writer: Optional[ProjectWriter], success: bool) -> None:
        """Commit or discard the destination lease after a save operation."""
        if writer is None:
            return
        self.session.assert_owner()
        if writer is not self._pending_writer:
            raise RuntimeError("Save does not belong to this project")
        if success:
            if self._writer is not None and self._writer is not writer:
                self._writer.close()
            self._writer = writer
        elif writer is not self._writer:
            writer.close()
        self._pending_writer = None

    def close_writer(self) -> None:
        """Release session ownership only after pending saves have settled."""
        if self._retain_writer:
            self.session.assert_owner()
        if self._pending_writer is not None:
            raise RuntimeError("Project save already in progress")
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def snapshot_for_save(self) -> dict:
        """Return a deep-copied snapshot of all save-relevant project state.

        Used by `SaveProjectWorker` so the background thread serializes a
        stable view of the project even if the main thread mutates the live
        project mid-save. The snapshot returns the same kwargs `save_project()`
        already accepts plus the `extra_data` payload `Project.save()` builds.
        """
        import copy

        extra_data = {
            "_all_sequences": list(self.sequences),
            "active_sequence_index": self.active_sequence_index,
        }
        return {
            "sources": copy.deepcopy(self._sources),
            "clips": copy.deepcopy(self._clips),
            "sequence": copy.deepcopy(self.sequence),
            "ui_state": copy.deepcopy(self.ui_state),
            "metadata": copy.deepcopy(self.metadata),
            "frames": copy.deepcopy(self._frames),
            "extra_data": copy.deepcopy(extra_data),
            "audio_sources": copy.deepcopy(self._audio_sources),
        }

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
            ProjectBusyError: If a retained session cannot acquire its destination
        """
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No path specified for save")

        generation = self.mutation_generation
        snapshot = self.snapshot_for_save()

        from contextlib import nullcontext

        writer = self.prepare_save(save_path)
        success = False
        try:
            with writer.activate() if writer else nullcontext():
                success = save_project(
                    filepath=save_path, progress_callback=progress_callback, **snapshot,
                )
        finally:
            self.finish_save(writer, success)

        if success:
            self.path = writer.path if writer is not None else save_path
            self.metadata.version = snapshot["metadata"].version
            self.metadata.modified_at = snapshot["metadata"].modified_at
            if self.mutation_generation == generation:
                self.mark_clean()
            self._notify_observers("project_saved", self.path)

        return success

    @classmethod
    def load(
        cls,
        path: Path,
        missing_source_callback: Optional[Callable[[Path, str], Optional[Path]]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        *,
        retain_writer: bool = False,
    ) -> "Project":
        """Load project from file.

        Args:
            path: Path to the project file
            missing_source_callback: Callback when source video is missing
            progress_callback: Optional progress callback
            retain_writer: Acquire before loading and retain ownership for editing

        Returns:
            Loaded Project instance

        Raises:
            ProjectLoadError: If the project file cannot be loaded
        """
        if retain_writer:
            from core.project_lock import ProjectWriter

            writer = ProjectWriter(path).acquire()
            try:
                project = cls.load(writer.path, missing_source_callback, progress_callback)
                project._retain_writer = True
                project._writer = writer
                return project
            except BaseException:
                writer.close()
                raise

        # First, read the raw JSON to extract multi-sequence data (if present)
        # before load_project() processes it into the standard 6-tuple.
        sequences_list = None
        active_idx = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            validation_errors = _validate_project_structure(raw_data)
            if validation_errors:
                raise ProjectLoadError("Invalid project file structure: " + "; ".join(validation_errors))
            raw_data = migrate_project_data(raw_data)
            if isinstance(raw_data.get("sequences"), list) and raw_data["sequences"]:
                base_path = path.parent
                active_idx = raw_data.get("active_sequence_index", 0)
                active_idx = min(active_idx, len(raw_data["sequences"]) - 1)
                sequences_list = []
                for seq_data in raw_data["sequences"]:
                    if seq_data:
                        sequences_list.append(
                            Sequence.from_dict(seq_data, base_path=base_path)
                        )
                    else:
                        sequences_list.append(Sequence())
        except (json.JSONDecodeError, OSError, IOError):
            pass  # Will be handled by load_project() below
        except ValueError as exc:
            raise ProjectLoadError(str(exc)) from exc

        sources, clips, sequence, metadata, ui_state, frames, audio_sources = load_project(
            filepath=path,
            missing_source_callback=missing_source_callback,
            progress_callback=progress_callback,
        )

        # Validate non-active sequences: remove clips referencing missing sources/clips
        if sequences_list:
            valid_clip_ids = {clip.id for clip in clips}
            valid_source_ids = {s.id for s in sources}
            for i, seq in enumerate(sequences_list):
                if i == active_idx:
                    continue  # Active sequence already validated by load_project()
                for track in seq.tracks:
                    invalid = [
                        sc for sc in track.clips
                        if (sc.source_id and sc.source_id not in valid_source_ids)
                        or (sc.source_clip_id and sc.source_clip_id not in valid_clip_ids)
                    ]
                    for sc in invalid:
                        logger.warning(
                            f"Removing sequence clip {sc.id} from non-active sequence "
                            f"'{seq.name}': dangling reference"
                        )
                        track.clips.remove(sc)

        if sequences_list:
            project = cls(
                path=path,
                metadata=metadata,
                sources=sources,
                clips=clips,
                sequences=sequences_list,
                active_sequence_index=active_idx,
                ui_state=ui_state,
                frames=frames,
                audio_sources=audio_sources,
            )
        else:
            project = cls(
                path=path,
                metadata=metadata,
                sources=sources,
                clips=clips,
                sequence=sequence,
                ui_state=ui_state,
                frames=frames,
                audio_sources=audio_sources,
            )
        project._dirty = False
        return project

    @classmethod
    def new(cls, name: str = "Untitled Project", *, retain_writer: bool = False) -> "Project":
        """Create a new empty project.

        Args:
            name: Project name
            retain_writer: Keep ownership after the first successful save

        Returns:
            New empty Project instance
        """
        project = cls(metadata=ProjectMetadata(name=name), sequences=[Sequence()])
        project._retain_writer = retain_writer
        return project

    def clear(self) -> None:
        """Clear all project data (for 'New Project')."""
        self.session.assert_owner()
        self.close_writer()
        self._sources = []
        self._clips = []
        self._frames = []
        self._audio_sources = []
        self.sequences = [Sequence()]
        self.active_sequence_index = 0
        self.ui_state = {}
        self.path = None
        self.metadata = ProjectMetadata()
        self._dirty = False
        self._mutation_generation += 1
        self._invalidate_caches()
        self.session.reset()
        self._notify_observers("project_cleared", None)

    def __repr__(self) -> str:
        return (
            f"Project(name={self.metadata.name!r}, "
            f"sources={len(self._sources)}, clips={len(self._clips)}, "
            f"frames={len(self._frames)}, dirty={self._dirty})"
        )
