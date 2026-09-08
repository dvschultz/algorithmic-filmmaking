"""Worker-side explicit acceptance of old values without claiming inference.

Callers must expose this as a deliberate reuse decision, never an automatic
migration. Publication uses the normal operation's stale-input owner guard.
"""

from dataclasses import replace
import json
from math import isfinite
from threading import Event
from typing import TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints, AnalysisInput
from core.operations.colors import ColorRequest, color_identity
from core.operations.contracts import ColorOutcome, ColorResult
from core.operations.embeddings import EmbeddingOutcome, EmbeddingTask, embedding_identity
from models.analysis_record import AnalysisIdentity, AnalysisRecord

LEGACY_REUSE_OPERATIONS = ("colors", "embeddings", "brightness", "volume", "classify", "detect_objects")

if TYPE_CHECKING:
    from core.operations.scalars import ScalarOutcome, ScalarTask
    from core.operations.classification import ClassificationOutcome, ClassificationTask
    from core.operations.object_detection import ObjectDetectionOutcome, ObjectDetectionTask


def _accept(record_json: str | None, value: dict, identity: AnalysisIdentity, inputs: AnalysisInput) -> str:
    record = AnalysisRecord.from_dict(json.loads(record_json)) if record_json else AnalysisRecord.legacy(value)
    if record.provenance != "unknown" or record.state != "succeeded":
        raise ValueError("Only successful values with unknown provenance can be explicitly reused")
    if record.artifact is not None:
        from core.artifacts import ArtifactStore

        previous = json.loads(ArtifactStore().read_bytes(record.artifact))
    else:
        previous = record.value
    if identity.operation == "colors" and isinstance(previous, dict):
        previous = {"dominant_colors": [
            [color["r"], color["g"], color["b"]] if isinstance(color, dict) else list(color)
            for color in previous["dominant_colors"]
        ]}
    if previous != value:
        raise ValueError("Legacy record no longer matches its visible value; recompute analysis")
    if not inputs.unchanged():
        raise ValueError("Media changed while preparing the reuse decision")
    accepted = replace(
        record.accept_legacy(identity), artifact=None,
        value_json=json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        input_json=json.dumps(inputs.to_dict(), sort_keys=True, separators=(",", ":")),
    )
    return json.dumps(accepted.to_dict(), sort_keys=True)


def accept_legacy_colors(request: ColorRequest, *, cancel_event: Event | None = None) -> ColorResult:
    """Hash detached color inputs; return records for normal owner publication."""
    from core.jobs.media import FingerprintCancelled

    fingerprints = AnalysisFingerprints(cancel_event)
    outcomes = []
    for target in request.targets:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if target.missing or target.inputs is None or not target.existing_colors:
                raise ValueError("A legacy palette and readable source are required")
            value = {"dominant_colors": [list(color) for color in target.existing_colors]}
            identity = color_identity(target, request.num_colors, fingerprints)
            record_json = _accept(target.record_json, value, identity, target.inputs)
            outcomes.append(ColorOutcome(target.target_id, "succeeded", target.existing_colors, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(ColorOutcome(target.target_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(ColorOutcome(target.target_id, "failed", code="legacy_reuse_unavailable", message=str(exc)))
    return ColorResult(request.request_id, tuple(outcomes))


def accept_legacy_embeddings(tasks: tuple[EmbeddingTask, ...], *, cancel_event: Event | None = None) -> tuple[EmbeddingOutcome, ...]:
    """Accept only compatible DINO vectors, retaining unknown provenance."""
    from core.analysis_model_identity import embedding_runtime
    from core.jobs.media import FingerprintCancelled

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = embedding_runtime()
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.inputs is None or task.existing_vector is None or task.thumbnail_path is None:
                raise ValueError("A legacy embedding and readable thumbnail are required")
            outcome = EmbeddingOutcome.from_vector(task.clip_id, task.existing_vector)
            if outcome.model != task.existing_model:
                raise ValueError("Legacy embedding model is unknown or incompatible; recompute analysis")
            identity = embedding_identity(task, fingerprints, runtime)
            value = {"embedding": list(outcome.vector), "embedding_model": outcome.model}
            record_json = _accept(task.record_json, value, identity, task.inputs)
            outcomes.append(replace(outcome, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(EmbeddingOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(EmbeddingOutcome(task.clip_id, "failed", code="legacy_reuse_unavailable", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_scalars(tasks: "tuple[ScalarTask, ...]", *, cancel_event: Event | None = None) -> "tuple[ScalarOutcome, ...]":
    """Accept finite legacy measurements, including zero, without probing media."""
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.scalars import FIELDS, ScalarOutcome, scalar_parameters, scalar_runtime, scalar_sampling, scalar_value

    fingerprints = AnalysisFingerprints(cancel_event)
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
            region = json.loads(snapshot.inputs.range_json)
            start, end, fps = region["start_frame"], region["end_frame"], region["fps"]
            if (
                not any(role == "video" for role, _, _ in snapshot.inputs.files)
                or type(start) is not int or type(end) is not int
                or start < 0 or end <= start or isinstance(fps, bool)
                or not isinstance(fps, (int, float)) or not isfinite(fps) or fps <= 0
            ):
                raise ValueError("A valid source range is required for scalar reuse")
            raw = json.loads(snapshot.value_json)[FIELDS[task.operation]]
            # A legacy None volume cannot distinguish an absent measurement
            # from a verified no-audio result. Recompute to establish that fact.
            if raw is None:
                raise ValueError("No legacy measurement is available; recompute analysis")
            value = scalar_value(task.operation, raw)
            identity = fingerprints.identity(
                snapshot.inputs, operation=task.operation,
                model=scalar_runtime(task.operation), parameters=scalar_parameters(task),
                sampling=scalar_sampling(task.operation),
            )
            record_json = _accept(
                json.dumps(snapshot.record.to_dict()) if snapshot.record else None,
                value, identity, snapshot.inputs,
            )
            outcomes.append(ScalarOutcome(task.clip_id, task.operation, "succeeded", record_json))
        except FingerprintCancelled:
            outcomes.append(ScalarOutcome(task.clip_id, task.operation, "unprocessed", message="Cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(ScalarOutcome(task.clip_id, task.operation, "failed", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_visuals(
    tasks: "tuple[ClassificationTask | ObjectDetectionTask, ...]",
    *, cancel_event: Event | None = None,
) -> "tuple[ClassificationOutcome | ObjectDetectionOutcome, ...]":
    """Reuse labels/detections without inventing missing classifier confidence."""
    from core.analysis_records import AnalysisSnapshot
    from core.analysis_model_identity import classification_runtime, object_detection_runtime
    from core.jobs.media import FingerprintCancelled
    from core.operations.classification import ClassificationTask, ClassificationOutcome, ClassificationOptions, classification_identity
    from core.operations.object_detection import DetectedObject, ObjectDetectionOutcome, ObjectDetectionOptions, object_detection_identity

    fingerprints = AnalysisFingerprints(cancel_event)
    outcomes: list[ClassificationOutcome | ObjectDetectionOutcome] = []
    for task in tasks:
        classification = isinstance(task, ClassificationTask)
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.analysis_json is None or task.thumbnail_path is None or not task.thumbnail_path.is_file():
                raise ValueError("A readable thumbnail and legacy values are required")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            if classification:
                labels = value["object_labels"]
                if not isinstance(labels, list) or any(not isinstance(label, str) or not label for label in labels):
                    raise ValueError("Legacy classification labels are missing or invalid")
                identity = classification_identity(snapshot, ClassificationOptions(), fingerprints, classification_runtime())
            else:
                raw, count = value["detected_objects"], value["person_count"]
                if not isinstance(raw, list) or type(count) is not int or count < 0:
                    raise ValueError("Legacy detections and person count are required")
                if any(isinstance(item.get("confidence"), bool) for item in raw):
                    raise ValueError("Legacy detection confidence is invalid")
                detections = tuple(DetectedObject.from_dict(item) for item in raw)
                if sum(item.label == "person" for item in detections) != count:
                    raise ValueError("Legacy person count does not match the detections")
                identity = object_detection_identity(snapshot, ObjectDetectionOptions(), fingerprints, object_detection_runtime())
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            if classification:
                # The normal reuse representation reads saved label names. No
                # synthetic confidences are introduced for legacy labels.
                outcomes.append(ClassificationOutcome(task.clip_id, "skipped", code="legacy_accepted", record_json=record_json))
            else:
                outcomes.append(ObjectDetectionOutcome(task.clip_id, "succeeded", detections=detections, person_count=count, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(ClassificationOutcome(task.clip_id, "unprocessed", code="cancelled") if classification else ObjectDetectionOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
            outcomes.append(ClassificationOutcome(task.clip_id, "failed", message=str(exc)) if classification else ObjectDetectionOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)
