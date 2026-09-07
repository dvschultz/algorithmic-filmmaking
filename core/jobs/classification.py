"""Durable classification results for saved-project jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.classification import (
    ClassificationApplication,
    ClassificationOptions,
    ClassificationOutcome,
    ClassificationTask,
    run_classification,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for package in ("torch", "torchvision"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "model": "mobilenet_v3_small",
        "weights": "IMAGENET1K_V1",
        "vocabulary": "weight_metadata",
        "packages": packages,
    }


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown classification clip ID")
    return ids


def _task(
    project: Project, cid: str, thumbnails: dict[str, Path] | None = None
) -> ClassificationTask:
    clip = project.clips_by_id[cid]
    return ClassificationTask(cid, (thumbnails or {}).get(cid, clip.thumbnail_path))


def _task_data(task: ClassificationTask) -> dict:
    data = asdict(task)
    data["thumbnail_path"] = str(task.thumbnail_path) if task.thumbnail_path else None
    return data


def _source_data(project: Project, cid: str) -> dict:
    source_id = project.clips_by_id[cid].source_id
    source = project.sources_by_id.get(source_id)
    return {
        "source_id": source_id,
        "start_frame": project.clips_by_id[cid].start_frame,
        "end_frame": project.clips_by_id[cid].end_frame,
        "fps": source.fps if source else None,
        "actual_source_path": str(source.file_path) if source else None,
    }


def classification_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: ClassificationOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    targets = []
    for cid in _ids(project, clip_ids):
        task = _task(project, cid)
        source = _source_data(project, cid)
        source_path = source["actual_source_path"]
        targets.append(
            {
                **_task_data(task),
                **source,
                "image_stamp": media_stamp(task.thumbnail_path)
                if task.thumbnail_path
                else None,
                "source_stamp": media_stamp(Path(source_path)) if source_path else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="classification",
        version=1,
        arguments=arguments,
        inputs={
            "targets": targets,
            "options": asdict(options),
            "runtime": _runtime(),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: ClassificationOutcome):
        self.outcome = outcome


def run_classification_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: ClassificationOptions | None = None,
    force: bool = False,
    thumbnail_paths: dict[str, Path] | None = None,
    operation: OperationSpec | None = None,
) -> dict:
    """Reuse completed computation after failed saves, without implicit provider calls."""
    options = (
        ClassificationOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or ClassificationOptions())
    )
    if operation is not None and thumbnail_paths:
        raise ValueError("Queued classification inputs cannot replace thumbnails")
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    thumbnails = dict(thumbnail_paths or {})
    runtime = _runtime()
    fingerprint = MediaFingerprints(cancel).get
    with result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = classification_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Classification inputs changed while queued")
        ids = _ids(project, clip_ids)
        if any(cid not in ids for cid in thumbnails):
            raise ValueError("Thumbnail belongs to an unselected clip")
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "classification":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed classification payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(
            current: Project, cid: str, task: ClassificationTask | None = None
        ) -> dict:
            task = task or _task(current, cid, thumbnails)
            display = current.clips_by_id[cid].thumbnail_path
            source = _source_data(current, cid)
            source_path = source["actual_source_path"]
            return {
                "project_id": current.metadata.id,
                "display_thumbnail_path": str(display) if display else None,
                **source,
                "task": _task_data(task),
                "image": fingerprint(task.thumbnail_path),
                "source": fingerprint(Path(source_path)) if source_path else None,
                "runtime": _runtime(),
            }

        def is_output(current: Project, cid: str, payload: dict) -> bool:
            return current.clips_by_id[cid].object_labels == [
                label for label, _ in payload["labels"]
            ]

        result: dict = {
            "succeeded": [],
            "failed": [],
            "skipped": [],
            "unprocessed": [],
            "total_clips": len(ids),
        }
        for index, cid in enumerate(ids):
            if cancel.is_set():
                result["unprocessed"].extend(
                    {"clip_id": rest, "code": "cancelled"} for rest in ids[index:]
                )
                break
            clip = project.clips_by_id[cid]
            existing = clip.object_labels is not None
            if existing and not force and cid not in known:
                result["skipped"].append(
                    {"clip_id": cid, "reason": "already_populated"}
                )
                continue
            try:
                # CLI retries do not regenerate images for populated targets. Recover
                # the original analysis image only when its complete receipt still
                # matches the current target, media, options, and saved output.
                if (
                    existing
                    and not force
                    and cid not in thumbnails
                    and operation is None
                ):
                    for _, identity, payload in known.get(cid, []):
                        if (
                            identity["project_path"] != str(path)
                            or identity["arguments"] != asdict(options)
                            or not is_output(project, cid, payload)
                        ):
                            continue
                        image = identity["inputs"]["basis"]["task"]["thumbnail_path"]
                        if image is None:
                            continue
                        candidate_task = ClassificationTask(cid, Path(image))
                        if (
                            inputs(project, cid, candidate_task)
                            == identity["inputs"]["basis"]
                        ):
                            thumbnails[cid] = Path(image)
                            break
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Classification runtime changed")
                task = _task(project, cid, thumbnails)
                identity_inputs: dict = {"basis": basis}
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                    identity_inputs["previous_labels"] = clip.object_labels
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="classification",
                    version=1,
                    target_id=cid,
                    arguments=arguments,
                    inputs=identity_inputs,
                )
                specs = [spec]
                if force:
                    for row, identity, payload in known.get(cid, []):
                        if (
                            not row["committed"]
                            and identity["project_path"] == str(path)
                            and identity["arguments"] == arguments
                            and identity["inputs"]["basis"] == basis
                            and is_output(project, cid, payload)
                        ):
                            specs = [ResultSpec(path, row["spec_json"])]
                            break
                if existing and not force:
                    matches = [
                        row
                        for row, identity, payload in known[cid]
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and is_output(project, cid, payload)
                    ]
                    if not matches:
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "already_populated"}
                        )
                        continue
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]

                application = ClassificationApplication(project, (task,))

                def compute(task=task):
                    outcome = run_classification((task,), options, cancel_event=cancel)[
                        0
                    ]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {"labels": [list(label) for label in outcome.labels]}

                def apply(current, payload, cid=cid, application=application):
                    outcome = ClassificationOutcome(
                        cid,
                        "succeeded",
                        tuple(
                            (label, confidence)
                            for label, confidence in payload["labels"]
                        ),
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Classification target changed during application"
                        )

                def validate(current, cid=cid, basis=basis):
                    return inputs(current, cid) == basis

                def is_applied(current, payload, cid=cid):
                    return is_output(current, cid, payload)

                for candidate in specs:
                    receipt = batch.commit(
                        candidate,
                        compute=compute,
                        validate_input=validate,
                        apply=apply,
                        is_applied=is_applied,
                    )
                if receipt["applied"]:
                    result["succeeded"].append(
                        {
                            "clip_id": cid,
                            "label_count": len(receipt["payload"]["labels"]),
                        }
                    )
                else:
                    result["skipped"].append(
                        {"clip_id": cid, "reason": "already_committed"}
                    )
            except FingerprintCancelled:
                result["unprocessed"].extend(
                    {"clip_id": rest, "code": "cancelled"} for rest in ids[index:]
                )
                break
            except _OutcomeError as exc:
                outcome = exc.outcome
                result[outcome.status].append(
                    {"clip_id": cid, "code": outcome.code, "message": outcome.message}
                )
            progress(
                0.95 * (index + 1) / len(ids),
                f"Classification ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "Classification finished")
        return {"success": True, "result": result}
