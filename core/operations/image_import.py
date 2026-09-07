"""Detached still-image import with isolated artifacts and owner publication."""

from dataclasses import dataclass
from pathlib import Path
import shutil
from threading import Event
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from core.operations.contracts import OutcomeStatus
from core.operations.transcription import _media_stamp

if TYPE_CHECKING:
    from core.project import Project
    from models.frame import Frame

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)


@dataclass(frozen=True)
class ImageImportInput:
    id: str
    path: Path
    media_stamp: tuple[int, ...] | None
    error: str | None = None


@dataclass(frozen=True)
class ImageImportTask:
    request_id: str
    items: tuple[ImageImportInput, ...]
    artifact_dir: Path
    copy_files: bool

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        output_dir: Path,
        *,
        copy_files: bool = False,
        validate_paths: bool = False,
    ) -> "ImageImportTask":
        from core.spine.security import validate_path

        items = []
        for path in paths:
            path = Path(path)
            error = None
            if validate_paths:
                valid, error, resolved = validate_path(
                    str(path), must_exist=True, must_be_file=True
                )
                if valid:
                    path = resolved
                    error = None
            if error is None:
                path = path.expanduser().resolve()
            items.append(
                ImageImportInput(
                    uuid4().hex,
                    path,
                    _media_stamp(path) if error is None else None,
                    error,
                )
            )
        request_id = uuid4().hex
        return cls(
            request_id,
            tuple(items),
            output_dir.expanduser().resolve() / request_id,
            copy_files,
        )


@dataclass(frozen=True)
class ImportedImage:
    id: str
    path: Path
    thumbnail_path: Path
    width: int
    height: int
    media_stamp: tuple[int, ...] | None
    thumbnail_stamp: tuple[int, ...] | None

    def to_model(self) -> "Frame":
        from models.frame import Frame

        return Frame(
            id=self.id,
            file_path=self.path,
            thumbnail_path=self.thumbnail_path,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class ImageImportOutcome:
    request_id: str
    status: OutcomeStatus
    frames: tuple[ImportedImage, ...] = ()
    errors: tuple[str, ...] = ()


def run_image_import(
    task: ImageImportTask,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ImageImportOutcome:
    """Import valid items in input order; cancellation publishes no partial batch."""
    cancel = cancel_event or Event()
    created = keep = False
    errors = []
    frames = []
    try:
        if cancel.is_set():
            return ImageImportOutcome(task.request_id, "unprocessed")
        from PIL import Image
        from core.thumbnail import generate_image_thumbnail

        if not task.items:
            return ImageImportOutcome(
                task.request_id, "failed", errors=("No image paths provided",)
            )
        task.artifact_dir.mkdir(parents=True, exist_ok=False)
        created = True
        for index, item in enumerate(task.items):
            if cancel.is_set():
                return ImageImportOutcome(task.request_id, "unprocessed")
            folder = task.artifact_dir / item.id
            try:
                if item.error:
                    raise ValueError(item.error)
                if item.path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError(f"Unsupported image format: {item.path.suffix}")
                if not item.path.is_file() or item.media_stamp is None:
                    raise ValueError("Image file is missing")
                if _media_stamp(item.path) != item.media_stamp:
                    raise ValueError("Image file changed while queued")
                folder.mkdir()
                path = folder / item.path.name if task.copy_files else item.path
                if task.copy_files:
                    shutil.copy2(item.path, path)
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                thumbnail = folder / "thumbnails" / "thumbnail.jpg"
                generate_image_thumbnail(path, thumbnail)
                if _media_stamp(item.path) != item.media_stamp:
                    raise ValueError("Image file changed during import")
                frames.append(
                    ImportedImage(
                        item.id,
                        path,
                        thumbnail,
                        width,
                        height,
                        _media_stamp(path),
                        _media_stamp(thumbnail),
                    )
                )
            except Exception as exc:
                errors.append(f"{item.path}: {exc}")
                if folder.exists():
                    shutil.rmtree(folder)
            if progress:
                progress(index + 1, len(task.items))
        if cancel.is_set():
            return ImageImportOutcome(task.request_id, "unprocessed")
        outcome = ImageImportOutcome(
            task.request_id,
            "succeeded" if frames else "failed",
            tuple(frames),
            tuple(errors),
        )
        if frames:
            validate_image_artifacts(task, outcome)
            keep = True
        return outcome
    except Exception as exc:
        return ImageImportOutcome(
            task.request_id,
            "unprocessed" if cancel.is_set() else "failed",
            errors=tuple([*errors, str(exc)]),
        )
    finally:
        if created and not keep:
            shutil.rmtree(task.artifact_dir, ignore_errors=True)


def validate_image_artifacts(
    task: ImageImportTask, outcome: ImageImportOutcome
) -> None:
    if (
        outcome.request_id != task.request_id
        or outcome.status != "succeeded"
        or not outcome.frames
    ):
        raise ValueError("Image import outcome does not match request")
    ids = [frame.id for frame in outcome.frames]
    if len(ids) != len(set(ids)) or ids != [
        item.id for item in task.items if item.id in ids
    ]:
        raise ValueError("Image import identities are duplicated or reordered")
    items = {item.id: item for item in task.items}
    for frame in outcome.frames:
        item = items[frame.id]
        folder = task.artifact_dir / item.id
        expected = folder / item.path.name if task.copy_files else item.path
        if (
            item.error
            or item.media_stamp is None
            or _media_stamp(item.path) != item.media_stamp
            or frame.path.resolve() != expected
            or frame.thumbnail_path.resolve() != folder / "thumbnails" / "thumbnail.jpg"
            or frame.media_stamp is None
            or _media_stamp(frame.path) != frame.media_stamp
            or frame.thumbnail_stamp is None
            or _media_stamp(frame.thumbnail_path) != frame.thumbnail_stamp
            or any(
                type(value) is not int or value <= 0
                for value in (frame.width, frame.height)
            )
        ):
            raise ValueError("Imported image or thumbnail changed before publication")


class ImageImportApplication:
    def __init__(self, project: "Project", task: ImageImportTask) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path.resolve() if project.path else None
        self.task = task
        self.consumed = False

    def is_current(self, project: "Project") -> bool:
        return (
            project is self.project
            and project.session.session_id == self.session_id
            and (project.path.resolve() if project.path else None) == self.path
        )

    def apply(self, project: "Project", outcome: ImageImportOutcome) -> bool:
        if (
            not self.is_current(project)
            or self.consumed
            or outcome.status != "succeeded"
        ):
            return False

        def publish() -> bool:
            self.consumed = True
            validate_image_artifacts(self.task, outcome)
            if any(frame.id in project.frames_by_id for frame in outcome.frames):
                raise ValueError("Imported frame ID already exists")
            project.add_frames([frame.to_model() for frame in outcome.frames])
            return True

        return project.session.apply_external(publish)
