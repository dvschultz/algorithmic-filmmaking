"""Detached still-image import with isolated artifacts and owner publication."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from threading import Event
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from core.operations.contracts import OutcomeStatus
from core.operations.transcription import _media_stamp
from core.operations.frame_extraction import _decode_stamp

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

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "artifact_dir": str(self.artifact_dir),
            "copy_files": self.copy_files,
            "items": [{**asdict(item), "path": str(item.path)} for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageImportTask":
        request_id = _import_id(data.get("request_id"))
        directory = data.get("artifact_dir")
        if (
            not isinstance(directory, str)
            or not Path(directory).is_absolute()
            or Path(directory).name != request_id
            or type(data.get("copy_files")) is not bool
            or not isinstance(data.get("items"), list)
        ):
            raise ValueError("Invalid image import task")
        items = []
        for value in data["items"]:
            path, error = value.get("path"), value.get("error")
            if (
                not isinstance(path, str)
                or not path
                or (error is not None and not isinstance(error, str))
                or (error is None and not Path(path).is_absolute())
            ):
                raise ValueError("Invalid image import input")
            items.append(
                ImageImportInput(
                    _import_id(value.get("id")),
                    Path(path),
                    _decode_stamp(value.get("media_stamp")),
                    error,
                )
            )
        if len({item.id for item in items}) != len(items):
            raise ValueError("Duplicate image import input IDs")
        return cls(request_id, tuple(items), Path(directory), data["copy_files"])

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
                    _media_stamp(path) if error is None and path.is_file() else None,
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

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "errors": list(self.errors),
            "frames": [
                {
                    **asdict(frame),
                    "path": str(frame.path),
                    "thumbnail_path": str(frame.thumbnail_path),
                }
                for frame in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageImportOutcome":
        request_id = _import_id(data.get("request_id"))
        if (
            data.get("status") not in ("succeeded", "failed", "unprocessed")
            or not isinstance(data.get("frames"), list)
            or not isinstance(data.get("errors"), list)
            or any(not isinstance(error, str) for error in data["errors"])
        ):
            raise ValueError("Invalid image import outcome")
        frames = []
        for value in data["frames"]:
            if any(
                not isinstance(value.get(key), str)
                or not Path(value[key]).is_absolute()
                for key in ("path", "thumbnail_path")
            ):
                raise ValueError("Invalid imported image artifact path")
            if any(
                type(value.get(key)) is not int or value[key] <= 0
                for key in ("width", "height")
            ):
                raise ValueError("Invalid imported image dimensions")
            frames.append(
                ImportedImage(
                    _import_id(value.get("id")),
                    Path(value["path"]),
                    Path(value["thumbnail_path"]),
                    value["width"],
                    value["height"],
                    _decode_stamp(value.get("media_stamp")),
                    _decode_stamp(value.get("thumbnail_stamp")),
                )
            )
        if (data["status"] == "succeeded") != bool(frames):
            raise ValueError("Image import status does not match its frames")
        return cls(request_id, data["status"], tuple(frames), tuple(data["errors"]))


def _import_id(value) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Invalid image import ID")
    return value


def image_import_task_inputs(task: ImageImportTask) -> dict:
    """Stable input identity excludes generated IDs and request workspace name."""
    return {
        "copy_files": task.copy_files,
        "output_root": str(task.artifact_dir.parent),
        "items": [
            {
                "path": str(item.path),
                "media_stamp": list(item.media_stamp)
                if item.media_stamp is not None
                else None,
                "error": item.error,
            }
            for item in task.items
        ],
    }


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

    def apply(
        self,
        project: "Project",
        outcome: ImageImportOutcome,
        *,
        recovered_task: ImageImportTask | None = None,
    ) -> bool:
        task = recovered_task or self.task
        if image_import_task_inputs(task) != image_import_task_inputs(self.task):
            return False
        if (
            not self.is_current(project)
            or self.consumed
            or outcome.status != "succeeded"
        ):
            return False

        def publish() -> bool:
            self.consumed = True
            validate_image_artifacts(task, outcome)
            if any(frame.id in project.frames_by_id for frame in outcome.frames):
                raise ValueError("Imported frame ID already exists")
            project.add_frames([frame.to_model() for frame in outcome.frames])
            return True

        return project.session.apply_external(publish)
