"""Worker-side explicit acceptance of old values without claiming inference.

Callers must expose this as a deliberate reuse decision, never an automatic
migration. Publication uses the normal operation's stale-input owner guard.
"""

from dataclasses import replace
import json
from threading import Event

from core.analysis_records import AnalysisFingerprints, AnalysisInput
from core.operations.colors import ColorRequest, color_identity
from core.operations.contracts import ColorOutcome, ColorResult
from core.operations.embeddings import EmbeddingOutcome, EmbeddingTask, embedding_identity
from models.analysis_record import AnalysisIdentity, AnalysisRecord


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
