"""Serializable audio-import results shared by durable execution adapters."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.audio_import import (
    AudioImportTask,
    AudioImportOutcome,
    AudioImportApplication,
    run_audio_import,
)
from core.project import Project


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


def _task_for(project: Project, file_path: str) -> AudioImportTask:
    path = Path(file_path).expanduser()
    if not path.is_absolute() and project.path:
        path = project.path.parent / path
    return AudioImportTask.from_path(path)


def _task_inputs(task: AudioImportTask) -> dict:
    data = task.to_dict()
    data.pop("audio_source_id")
    normalized: dict = json.loads(canonical_json(data))
    return normalized


def audio_import_job_spec(project: Project, file_path: str) -> OperationSpec:
    task = _task_for(project, file_path)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="audio_import",
        version=1,
        arguments={"file_path": str(task.path)},
        inputs={"task": _task_inputs(task), "runtime": audio_import_runtime()},
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _ImportStopped(Exception):
    pass


def run_audio_import_job(
    store: JobStore,
    path: Path,
    file_path: str,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    operation: OperationSpec | None = None,
) -> dict:
    """Import and save once; interrupted commits retain the original audio ID."""
    fingerprints = MediaFingerprints(cancel)
    try:
        with result_batch(store, path, max_items=1) as batch:
            project = batch.project
            live = audio_import_job_spec(project, file_path)
            if operation is not None and (
                live.inputs_json != operation.inputs_json
                or live.arguments_json != operation.arguments_json
                or live.input_revision != operation.input_revision
            ):
                raise StaleJobResult("Audio import inputs changed while queued")
            if cancel.is_set():
                raise _ImportStopped("cancelled")
            task = _task_for(project, file_path)
            application = AudioImportApplication(project, task)

            def inputs(current: Project) -> dict:
                current_task = _task_for(current, file_path)
                return {
                    "project_id": current.metadata.id,
                    "task": _task_inputs(current_task),
                    "media": fingerprints.get(current_task.path),
                    "runtime": audio_import_runtime(),
                }

            def decode(payload: dict) -> tuple[AudioImportTask, AudioImportOutcome]:
                recorded = AudioImportRecord.from_dict(payload)
                recovered = AudioImportTask.from_dict(recorded.task)
                if (
                    replace(task, audio_source_id=recovered.audio_source_id)
                    != recovered
                ):
                    raise StaleJobResult(
                        "Recorded audio import inputs differ from request"
                    )
                return recovered, AudioImportOutcome.from_dict(recorded.outcome)

            def is_applied(current: Project, payload: dict) -> bool:
                recovered, outcome = decode(payload)
                audio = current.get_audio_source(outcome.audio_source_id)
                if audio is None:
                    return False
                actual, expected = (
                    audio.to_dict(),
                    outcome.to_model(recovered).to_dict(),
                )
                return all(
                    actual.get(key) == expected.get(key)
                    for key in (
                        "id",
                        "file_path",
                        "duration_seconds",
                        "sample_rate",
                        "channels",
                    )
                )

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
                if identity["kind"] != "audio_import" or identity["target_id"] != str(
                    task.path
                ):
                    continue
                if (
                    sha256(row["payload_json"].encode()).hexdigest() != digest
                    or row["payload_digest"] != digest
                ):
                    raise StaleJobResult("Committed audio import payload is corrupt")
                known.append((row, identity))

            existing = next(
                (
                    audio
                    for audio in project.audio_sources
                    if audio.file_path.expanduser().resolve() == task.path
                ),
                None,
            )
            # Already imported files may be offline. Only recovery of an unfinished
            # commit needs its media fingerprint and original result identity.
            pending_rows = [
                (row, identity) for row, identity in known if not row["committed"]
            ]
            if existing is not None and not pending_rows:
                return {
                    "success": True,
                    "result": {
                        "audio_source_id": existing.id,
                        "filename": existing.filename,
                        "duration": existing.duration_seconds,
                        "status": "skipped",
                    },
                }
            captured = inputs(project)
            pending = next(
                (
                    row
                    for row, identity in pending_rows
                    if identity["project_path"] == str(path.resolve())
                    and identity["inputs"]["basis"] == captured
                    and identity["arguments"] == live.arguments
                    and is_applied(project, json.loads(row["payload_json"]))
                ),
                None,
            )
            if existing is not None and pending is None:
                return {
                    "success": True,
                    "result": {
                        "audio_source_id": existing.id,
                        "filename": existing.filename,
                        "duration": existing.duration_seconds,
                        "status": "skipped",
                    },
                }
            spec = (
                ResultSpec(path.resolve(), pending["spec_json"])
                if pending
                else ResultSpec.build(
                    path,
                    kind="audio_import",
                    version=1,
                    target_id=str(task.path),
                    arguments=live.arguments,
                    inputs={"basis": captured, "generation": len(known)},
                )
            )

            def compute() -> dict:
                outcome = run_audio_import(
                    task,
                    cancel_event=cancel,
                    progress=lambda n, total: progress(
                        n / total if total else 1.0, "Importing audio"
                    ),
                )
                if outcome.status != "succeeded" or cancel.is_set():
                    raise _ImportStopped(
                        "cancelled"
                        if cancel.is_set()
                        else outcome.message or "Audio import failed"
                    )
                return asdict(AudioImportRecord.build(task, outcome))

            def apply(current: Project, payload: dict) -> None:
                if cancel.is_set():
                    raise _ImportStopped("cancelled")
                recovered, outcome = decode(payload)
                if not application.apply(current, outcome, recovered_task=recovered):
                    raise StaleJobResult("Audio import target changed")

            receipt = batch.commit(
                spec,
                compute=compute,
                validate_input=lambda current: not cancel.is_set()
                and inputs(current) == captured,
                apply=apply,
                is_applied=is_applied,
            )
            recovered, outcome = decode(receipt["payload"])
            progress(1.0, "Imported audio saved")
            return {
                "success": True,
                "result": {
                    "audio_source_id": outcome.audio_source_id,
                    "filename": recovered.path.name,
                    "duration": outcome.duration,
                    "status": "succeeded" if receipt["applied"] else "recovered",
                },
            }
    except (FingerprintCancelled, _ImportStopped) as exc:
        return {"success": False, "error": "cancelled" if cancel.is_set() else str(exc)}
