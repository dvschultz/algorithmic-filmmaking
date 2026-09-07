"""Validated image-import receipts shared by durable execution adapters."""

from dataclasses import dataclass
from hashlib import sha256
import json

from core.jobs.commits import canonical_json
from core.operations.image_import import (
    ImageImportTask,
    ImageImportOutcome,
    validate_image_artifacts,
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
