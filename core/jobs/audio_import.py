"""Serializable audio-import results shared by durable execution adapters."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from core.jobs.commits import canonical_json
from core.jobs.media import media_stamp
from core.operations.audio_import import AudioImportTask, AudioImportOutcome


def audio_import_runtime() -> dict:
    from core.binary_resolver import find_binary

    binaries = {}
    for name in ("ffmpeg", "ffprobe"):
        binary = find_binary(name)
        binaries[name] = {
            "path": str(binary) if binary else None,
            "stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
        }
    return {"algorithm": "audio-import/v1", **binaries}


@dataclass(frozen=True)
class AudioImportRecord:
    path: str
    status: str
    task: dict
    outcome: dict

    @classmethod
    def build(
        cls, task: AudioImportTask, outcome: AudioImportOutcome
    ) -> "AudioImportRecord":
        return cls.from_dict(
            {
                "path": str(task.path),
                "status": outcome.status,
                "task": task.to_dict(),
                "outcome": asdict(outcome),
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "AudioImportRecord":
        data = json.loads(canonical_json(data))
        task = AudioImportTask.from_dict(data["task"])
        outcome = AudioImportOutcome.from_dict(data["outcome"])
        if (
            data["path"] != str(task.path)
            or data["status"] != "succeeded"
            or outcome.status != "succeeded"
            or outcome.audio_source_id != task.audio_source_id
            or task.media_stamp is None
        ):
            raise ValueError("Audio import receipt does not match its task")
        return cls(str(task.path), "succeeded", task.to_dict(), asdict(outcome))
