"""Immutable semantic analysis identities, states, and derived-file references.

These values do not import project models, Qt, or media runtimes. A job's retry
identity is deliberately separate from the identity of reusable analysis.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import re
from typing import Any, Callable, Literal, cast


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Content identity must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ArtifactRef:
    digest: str
    size: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _digest(self.digest)
        if type(self.size) is not int or self.size < 0:
            raise ValueError("Artifact size must be a nonnegative integer")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("Artifact media type is required")

    def to_dict(self) -> dict:
        return {"sha256": self.digest, "size": self.size, "media_type": self.media_type}

    @classmethod
    def from_dict(cls, value: dict) -> "ArtifactRef":
        return cls(value["sha256"], value["size"], value["media_type"])


@dataclass(frozen=True)
class AnalysisIdentity:
    document: str

    def __post_init__(self) -> None:
        data = json.loads(self.document)
        required = {"operation", "operation_version", "schema_version", "sources", "source_range", "model", "parameters", "sampling", "prompt_sha256"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Analysis identity fields are incomplete or unknown")
        if not isinstance(data["operation"], str) or not data["operation"]:
            raise ValueError("Analysis operation is required")
        for field in ("operation_version", "schema_version"):
            if type(data[field]) is not int or data[field] < 1:
                raise ValueError("Analysis versions must be positive integers")
        for field in ("sources", "source_range", "model", "parameters", "sampling"):
            if not isinstance(data[field], dict):
                raise ValueError(f"Analysis {field} must be an object")
        if not data["sources"]:
            raise ValueError("Analysis requires verified content identity")
        for label, digest in data["sources"].items():
            if not label:
                raise ValueError("Analysis source labels must be nonempty")
            _digest(digest)
        if data["prompt_sha256"] is not None:
            _digest(data["prompt_sha256"])
        object.__setattr__(self, "document", _json(data))

    @classmethod
    def build(
        cls, *, operation: str, sources: dict[str, str], source_range: dict,
        model: dict, parameters: dict, sampling: dict, prompt: str | None = None,
        operation_version: int = 1, schema_version: int = 1,
    ) -> "AnalysisIdentity":
        return cls(_json({
            "operation": operation, "operation_version": operation_version,
            "schema_version": schema_version, "sources": sources,
            "source_range": source_range, "model": model,
            "parameters": parameters, "sampling": sampling,
            "prompt_sha256": sha256(prompt.encode()).hexdigest() if prompt is not None else None,
        }))

    @property
    def key(self) -> str:
        return sha256(self.document.encode()).hexdigest()

    @property
    def operation(self) -> str:
        return cast(str, self.to_dict()["operation"])

    def to_dict(self) -> dict:
        return cast(dict, json.loads(self.document))

    @classmethod
    def from_dict(cls, value: dict) -> "AnalysisIdentity":
        return cls(_json(value))


_MISSING = object()

# Field projections remain readable by legacy consumers while operations move
# to records. Completion/reuse must not infer provenance from these fields.
ANALYSIS_FIELDS: dict[str, tuple[str, ...]] = {
    "colors": ("dominant_colors",), "shots": ("shot_type",),
    "classify": ("object_labels",),
    "detect_objects": ("detected_objects", "person_count"),
    "extract_text": ("extracted_texts",), "transcribe": ("transcript",),
    "describe": ("description", "description_model", "description_frames"),
    "cinematography": ("cinematography",), "face_embeddings": ("face_embeddings",),
    "gaze": ("gaze_yaw", "gaze_pitch", "gaze_category"),
    "embeddings": ("embedding", "embedding_model"),
    "boundary_embeddings": ("first_frame_embedding", "last_frame_embedding", "embedding_model"),
    "custom_query": ("custom_queries",),
    "brightness": ("average_brightness",), "volume": ("rms_volume",),
}


@dataclass(frozen=True)
class AnalysisRecord:
    identity: AnalysisIdentity | None
    state: Literal["succeeded", "failed", "missing"]
    provenance: Literal["verified", "unknown"] = "verified"
    value_json: str | None = None
    artifact: ArtifactRef | None = None
    error: str | None = None
    legacy_reuse: bool = False
    input_json: str | None = None

    def __post_init__(self) -> None:
        if self.state not in ("succeeded", "failed", "missing"):
            raise ValueError("Unknown analysis state")
        if self.provenance not in ("verified", "unknown"):
            raise ValueError("Unknown analysis provenance")
        if self.provenance == "verified" and self.identity is None:
            raise ValueError("Verified analysis requires an identity")
        if type(self.legacy_reuse) is not bool:
            raise ValueError("Legacy reuse must be explicit")
        if self.state == "succeeded" and (self.value_json is None) == (self.artifact is None):
            raise ValueError("Successful analysis requires exactly one payload")
        if self.value_json is not None:
            object.__setattr__(self, "value_json", _json(json.loads(self.value_json)))
        if self.input_json is not None:
            object.__setattr__(self, "input_json", _json(json.loads(self.input_json)))

    @classmethod
    def success(
        cls, identity: AnalysisIdentity, value: Any = _MISSING,
        *, artifact: ArtifactRef | None = None, input_snapshot: dict | None = None,
    ) -> "AnalysisRecord":
        return cls(identity, "succeeded", value_json=None if value is _MISSING else _json(value), artifact=artifact, input_json=_json(input_snapshot) if input_snapshot is not None else None)

    @classmethod
    def failure(cls, identity: AnalysisIdentity, error: str) -> "AnalysisRecord":
        return cls(identity, "failed", error=error)

    @classmethod
    def legacy(cls, value: Any) -> "AnalysisRecord":
        return cls(None, "succeeded", provenance="unknown", value_json=_json(value))

    def accept_legacy(self, identity: AnalysisIdentity) -> "AnalysisRecord":
        """Bind an explicit reuse decision without claiming new computation."""
        if self.provenance != "unknown" or self.state != "succeeded":
            raise ValueError("Only successful provenance-unknown values can be accepted")
        return replace(self, identity=identity, legacy_reuse=True)

    @property
    def value(self) -> Any:
        return json.loads(self.value_json) if self.value_json is not None else None

    def reusable(
        self, identity: AnalysisIdentity, *,
        artifact_available: Callable[[ArtifactRef], bool] | None = None,
    ) -> bool:
        if self.state != "succeeded" or self.identity != identity:
            return False
        if self.provenance == "unknown" and not self.legacy_reuse:
            return False
        return self.artifact is None or bool(artifact_available and artifact_available(self.artifact))

    def to_dict(self) -> dict:
        return {
            "version": 1, "identity": self.identity.to_dict() if self.identity else None,
            "state": self.state, "provenance": self.provenance,
            "value_json": self.value_json,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "error": self.error, "legacy_reuse": self.legacy_reuse,
            "input_json": self.input_json,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "AnalysisRecord":
        if value.get("version") != 1:
            raise ValueError("Unsupported analysis record version")
        return cls(
            AnalysisIdentity.from_dict(value["identity"]) if value["identity"] is not None else None,
            value["state"], value["provenance"], value.get("value_json"),
            ArtifactRef.from_dict(value["artifact"]) if value.get("artifact") is not None else None,
            value.get("error"), value.get("legacy_reuse", False),
            value.get("input_json"),
        )


@dataclass(frozen=True)
class UnreadableAnalysisRecord:
    """Preserve an unknown or damaged record without treating it as reusable."""

    document: str

    def to_dict(self) -> Any:
        return json.loads(self.document)


StoredAnalysisRecord = AnalysisRecord | UnreadableAnalysisRecord


def load_analysis_records(data: dict) -> dict[str, StoredAnalysisRecord]:
    """Retain unknown records and identify old projections as provenance-unknown."""
    serialized = data.get("analysis_records", {})
    if not isinstance(serialized, dict):
        serialized = {"__unreadable__": serialized}
    records: dict[str, StoredAnalysisRecord] = {}
    for operation, value in serialized.items():
        try:
            records[operation] = AnalysisRecord.from_dict(value)
        except (ValueError, KeyError, TypeError, AttributeError):
            records[operation] = UnreadableAnalysisRecord(_json(value))
    for operation, fields in ANALYSIS_FIELDS.items():
        if operation not in records and any(data.get(field) is not None for field in fields):
            records[operation] = AnalysisRecord.legacy({field: data[field] for field in fields if data.get(field) is not None})
    return records


def dump_analysis_records(records: dict[str, StoredAnalysisRecord]) -> dict:
    return {operation: record.to_dict() for operation, record in records.items()}
