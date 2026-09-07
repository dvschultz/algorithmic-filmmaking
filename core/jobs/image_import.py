"""Validated image-import receipts shared by durable execution adapters."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.project import Project
from core.operations.image_import import (
    ImageImportTask,
    ImageImportOutcome,
    validate_image_artifacts,
    image_import_task_inputs,
    ImageImportApplication,
    run_image_import,
)


def image_import_runtime() -> dict:
    from importlib.metadata import version

    return {"algorithm": "image-import/v1", "pillow": version("Pillow")}


def image_import_target(task: ImageImportTask) -> str:
    return sha256(
        canonical_json(
            {
                "paths": [str(item.path) for item in task.items],
                "copy_files": task.copy_files,
                "output_root": str(task.artifact_dir.parent),
            }
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ImageImportRecord:
    batch_id: str
    status: str
    task: dict
    outcome: dict

    @classmethod
    def build(
        cls, task: ImageImportTask, outcome: ImageImportOutcome
    ) -> "ImageImportRecord":
        return cls.from_dict(
            {
                "batch_id": image_import_target(task),
                "status": outcome.status,
                "task": task.to_dict(),
                "outcome": outcome.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ImageImportRecord":
        data = json.loads(canonical_json(data))
        task = ImageImportTask.from_dict(data["task"])
        outcome = ImageImportOutcome.from_dict(data["outcome"])
        if (
            data["batch_id"] != image_import_target(task)
            or data["status"] != "succeeded"
        ):
            raise ValueError("Image import receipt does not match its batch")
        validate_image_artifacts(task, outcome)
        return cls(data["batch_id"], "succeeded", task.to_dict(), outcome.to_dict())


def image_import_media(task: ImageImportTask, fingerprints: MediaFingerprints) -> dict:
    values: list[dict | None] = []
    for item in task.items:
        if item.error is not None or item.path.is_dir():
            values.append(None)
            continue
        if media_stamp(item.path) != item.media_stamp:
            raise StaleJobResult("Image import source changed while queued")
        try:
            values.append(fingerprints.get(item.path) if item.path.is_file() else None)
        except OSError:
            values.append(None)
    return {"items": values}


def _task_for(
    project: Project, file_paths: list[str], copy_files: bool, validate_paths: bool
) -> ImageImportTask:
    if not file_paths or any(not isinstance(path, str) for path in file_paths):
        raise ValueError("Provide one or more image file paths")
    if type(copy_files) is not bool or type(validate_paths) is not bool:
        raise ValueError("Image import policies must be booleans")
    if project.path is None:
        raise ValueError("Image import requires a saved project")
    paths = []
    for value in file_paths:
        path = Path(value).expanduser()
        paths.append(path if path.is_absolute() else project.path.parent / path)
    return ImageImportTask.from_paths(
        paths,
        project.path.parent / "frames",
        copy_files=copy_files,
        validate_paths=validate_paths,
    )


def image_import_job_spec(
    project: Project,
    file_paths: list[str],
    *,
    copy_files: bool = True,
    validate_paths: bool = False,
) -> OperationSpec:
    task = _task_for(project, file_paths, copy_files, validate_paths)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="import_images",
        version=1,
        arguments={
            "file_paths": [str(item.path) for item in task.items],
            "copy_files": copy_files,
            "validate_paths": validate_paths,
        },
        inputs={
            "task": image_import_task_inputs(task),
            "runtime": image_import_runtime(),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _ImportStopped(Exception):
    pass


def run_image_import_job(
    store: JobStore,
    path: Path,
    file_paths: list[str],
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    copy_files: bool = True,
    validate_paths: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Append one batch; interrupted saves reuse its original IDs and files."""
    fingerprints = MediaFingerprints(cancel)
    try:
        with result_batch(store, path, max_items=1) as batch:
            project = batch.project
            live = image_import_job_spec(
                project,
                file_paths,
                copy_files=copy_files,
                validate_paths=validate_paths,
            )
            if operation is not None and (
                live.inputs_json != operation.inputs_json
                or live.arguments_json != operation.arguments_json
                or live.input_revision != operation.input_revision
            ):
                raise StaleJobResult("Image import inputs changed while queued")
            if cancel.is_set():
                raise _ImportStopped("cancelled")
            task = _task_for(project, file_paths, copy_files, validate_paths)
            target = image_import_target(task)
            application = ImageImportApplication(project, task)

            def inputs(current: Project) -> dict:
                current_task = _task_for(
                    current, file_paths, copy_files, validate_paths
                )
                return {
                    "project_id": current.metadata.id,
                    "task": image_import_task_inputs(current_task),
                    "media": image_import_media(current_task, fingerprints),
                    "runtime": image_import_runtime(),
                }

            captured = inputs(project)

            def decode(payload: dict) -> tuple[ImageImportTask, ImageImportOutcome]:
                recorded = ImageImportRecord.from_dict(payload)
                recovered = ImageImportTask.from_dict(recorded.task)
                if image_import_task_inputs(recovered) != image_import_task_inputs(
                    task
                ):
                    raise StaleJobResult(
                        "Recorded image import inputs differ from request"
                    )
                return recovered, ImageImportOutcome.from_dict(recorded.outcome)

            def is_applied(current: Project, payload: dict) -> bool:
                _, outcome = decode(payload)
                for imported in outcome.frames:
                    frame = current.frames_by_id.get(imported.id)
                    if frame is None:
                        return False
                    actual, expected = frame.to_dict(), imported.to_model().to_dict()
                    if any(
                        actual.get(key) != expected.get(key)
                        for key in (
                            "id",
                            "file_path",
                            "width",
                            "height",
                            "source_id",
                            "clip_id",
                            "frame_number",
                        )
                    ):
                        return False
                return True

            known = []
            for result_id, digest in project.metadata.job_results.items():
                row = store.get_result(result_id)
                if (
                    row is None
                    or sha256(row["spec_json"].encode()).hexdigest() != result_id
                ):
                    raise StaleJobResult(
                        "Committed result identity is missing or corrupt"
                    )
                identity = json.loads(row["spec_json"])
                if (
                    identity["kind"] != "import_images"
                    or identity["target_id"] != target
                ):
                    continue
                if (
                    sha256(row["payload_json"].encode()).hexdigest() != digest
                    or row["payload_digest"] != digest
                ):
                    raise StaleJobResult("Committed image import payload is corrupt")
                known.append((row, identity))
            pending = next(
                (
                    row
                    for row, identity in known
                    if not row["committed"]
                    and identity["project_path"] == str(path.resolve())
                    and identity["inputs"]["basis"] == captured
                    and identity["arguments"] == live.arguments
                    and is_applied(project, json.loads(row["payload_json"]))
                ),
                None,
            )
            spec = (
                ResultSpec(path.resolve(), pending["spec_json"])
                if pending
                else ResultSpec.build(
                    path,
                    kind="import_images",
                    version=1,
                    target_id=target,
                    arguments=live.arguments,
                    inputs={"basis": captured, "generation": len(known)},
                )
            )

            def compute() -> dict:
                outcome = run_image_import(
                    task,
                    cancel_event=cancel,
                    progress=lambda n, total: progress(
                        n / total if total else 1.0, "Importing images"
                    ),
                )
                if outcome.status != "succeeded" or cancel.is_set():
                    raise _ImportStopped(
                        "cancelled"
                        if cancel.is_set()
                        else "; ".join(outcome.errors) or "Image import failed"
                    )
                return asdict(ImageImportRecord.build(task, outcome))

            def apply(current: Project, payload: dict) -> None:
                if cancel.is_set():
                    raise _ImportStopped("cancelled")
                recovered, outcome = decode(payload)
                if not application.apply(current, outcome, recovered_task=recovered):
                    raise StaleJobResult("Image import target changed")

            receipt = batch.commit(
                spec,
                compute=compute,
                validate_input=lambda current: not cancel.is_set()
                and inputs(current) == captured,
                apply=apply,
                is_applied=is_applied,
            )
            _, outcome = decode(receipt["payload"])
            progress(1.0, "Imported images saved")
            return {
                "success": True,
                "result": {
                    "imported_count": len(outcome.frames),
                    "frame_ids": [frame.id for frame in outcome.frames],
                    "errors": list(outcome.errors),
                    "status": "succeeded" if receipt["applied"] else "recovered",
                },
            }
    except (FingerprintCancelled, _ImportStopped) as exc:
        return {"success": False, "error": "cancelled" if cancel.is_set() else str(exc)}
