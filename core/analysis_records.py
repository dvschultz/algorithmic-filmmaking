"""Detached analysis inputs and worker-side content verification."""

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

from models.analysis_record import AnalysisIdentity, AnalysisRecord

if TYPE_CHECKING:
    from core.jobs.media import MediaFingerprints
    from core.artifacts import ArtifactStore


def restore_project_artifacts(path: Path, document: dict, targets: list[Any]) -> None:
    from core.artifacts import ArtifactStore, ArtifactUnavailable, document_references

    refs = document_references(document)
    if not refs:
        return
    store = ArtifactStore()
    store.retain_loaded_manifest(path, refs)
    with store.pin(refs) as pin:
        for ref in refs:
            if not store.available(ref):
                bundled = path.parent / "artifacts" / f"{ref.digest}.blob"
                if bundled.is_file():
                    try:
                        store.restore_from(ref, bundled, pin=pin)
                    except (ArtifactUnavailable, OSError):
                        # Hydration below exposes a per-operation missing result.
                        pass
        restore_artifact_projections(targets, store)


def restore_artifact_projections(targets: list[Any], store: "ArtifactStore") -> None:
    """Hydrate read projections without sacrificing records or editorial data.

    Called on the load/worker path: payload checksums and parsing perform I/O.
    Missing payloads invalidate only their own operation and remain referenced
    so a later restore or targeted recomputation can recover them.
    """
    from dataclasses import replace
    from models.analysis_record import ANALYSIS_FIELDS
    from core.artifacts import ArtifactUnavailable

    for target in targets:
        for operation, record in tuple(target.analysis_records.items()):
            fields = ANALYSIS_FIELDS.get(operation)
            if not isinstance(record, AnalysisRecord) or record.artifact is None or fields is None:
                continue
            for field in fields:
                if field != "embedding_model" and hasattr(target, field):
                    setattr(target, field, None)
            try:
                if record.artifact.media_type != "application/json":
                    raise ArtifactUnavailable("Unsupported analysis payload type")
                payload = json.loads(store.read_bytes(record.artifact))
                if not isinstance(payload, dict) or set(payload) != set(fields):
                    raise ArtifactUnavailable("Analysis payload fields do not match its operation")
                if "embedding_model" in fields and payload["embedding_model"] != getattr(target, "embedding_model", None):
                    other_vectors = {"embedding", "first_frame_embedding", "last_frame_embedding"} - set(fields)
                    if any(getattr(target, field, None) is not None for field in other_vectors):
                        raise ArtifactUnavailable("Artifact model conflicts with another embedding result")
                restored = type(target).from_dict({**target.to_dict(), **payload})
                for field in fields:
                    if not hasattr(restored, field) or (payload[field] is not None and getattr(restored, field) is None):
                        raise ArtifactUnavailable("Analysis payload contains invalid values")
                for field in fields:
                    setattr(target, field, getattr(restored, field))
                if record.state == "missing":
                    target.analysis_records[operation] = replace(record, state="succeeded", error=None)
            except (ArtifactUnavailable, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                if record.state != "failed":
                    target.analysis_records[operation] = replace(record, state="missing", error=str(exc))


def _stamp(path: Path) -> tuple[int, ...] | None:
    from core.jobs.media import media_stamp

    return media_stamp(path)


@dataclass(frozen=True)
class AnalysisInput:
    files: tuple[tuple[str, Path, tuple[int, ...] | None], ...]
    range_json: str
    binding_json: str = "{}"

    @classmethod
    def capture(cls, files: dict[str, Path], source_range: dict, *, binding: dict | None = None) -> "AnalysisInput":
        return cls(
            tuple((role, Path(path), _stamp(Path(path))) for role, path in sorted(files.items())),
            json.dumps(source_range, sort_keys=True, separators=(",", ":"), allow_nan=False),
            json.dumps(binding or {}, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def unchanged(self) -> bool:
        return bool(self.files) and all(stamp is not None and _stamp(path) == stamp for _, path, stamp in self.files)

    def to_dict(self) -> dict:
        return {
            "files": [{"role": role, "path": str(path), "stamp": list(stamp) if stamp is not None else None} for role, path, stamp in self.files],
            "range": json.loads(self.range_json),
            "binding": json.loads(self.binding_json),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "AnalysisInput":
        return cls(
            tuple((item["role"], Path(item["path"]), tuple(item["stamp"]) if item["stamp"] is not None else None) for item in value["files"]),
            json.dumps(value["range"], sort_keys=True, separators=(",", ":"), allow_nan=False),
            json.dumps(value.get("binding", {}), sort_keys=True, separators=(",", ":"), allow_nan=False),
        )


@dataclass(frozen=True)
class AnalysisSnapshot:
    """JSON-safe computation inputs, prior record, and detached read projection."""

    inputs: AnalysisInput
    record: AnalysisRecord | None
    value_json: str

    @classmethod
    def capture(cls, target: Any, operation: str, files: dict[str, Path], source_range: dict, value: dict) -> "AnalysisSnapshot":
        record = getattr(target, "analysis_records", {}).get(operation)
        return cls(
            AnalysisInput.capture(files, source_range, binding={"target_id": target.id, "source_id": getattr(target, "source_id", None)}),
            record if isinstance(record, AnalysisRecord) else None,
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def to_json(self) -> str:
        return json.dumps({"inputs": self.inputs.to_dict(), "record": self.record.to_dict() if self.record else None, "value_json": self.value_json}, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, document: str) -> "AnalysisSnapshot":
        value = json.loads(document)
        return cls(AnalysisInput.from_dict(value["inputs"]), AnalysisRecord.from_dict(value["record"]) if value["record"] is not None else None, value["value_json"])

    def reusable_record(self, identity: AnalysisIdentity) -> AnalysisRecord | None:
        """Worker-side verification; owner delivery receives an inline payload."""
        from core.artifacts import ArtifactStore, ArtifactUnavailable

        record = self.record
        if record is None or not record.reusable(identity, artifact_available=lambda _: True):
            return None
        try:
            value = json.loads(ArtifactStore().read_bytes(record.artifact)) if record.artifact else record.value
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ArtifactUnavailable, OSError, ValueError):
            return None
        if encoded != self.value_json:
            return None
        return replace(record, artifact=None, value_json=self.value_json, input_json=json.dumps(self.inputs.to_dict(), sort_keys=True))


def current_record(target: Any, operation: str) -> AnalysisRecord | None:
    """Check the cheap owner-thread part of validity; workers verify full hashes.

    A changed stamp never proves a changed hash, but it requires worker-side
    revalidation before the owner may skip dispatch. No media is hashed here.
    """
    record = getattr(target, "analysis_records", {}).get(operation)
    if not isinstance(record, AnalysisRecord) or record.identity is None or record.input_json is None:
        return None
    if record.identity.operation != operation:
        return None
    if record.state != "succeeded" or (record.provenance == "unknown" and not record.legacy_reuse):
        return None
    try:
        inputs = AnalysisInput.from_dict(json.loads(record.input_json))
        binding = json.loads(inputs.binding_json)
        if binding.get("target_id") != target.id or binding.get("source_id") != getattr(target, "source_id", None):
            return None
        source_range = json.loads(inputs.range_json)
        for field in ("start_frame", "end_frame", "frame_number"):
            if field in source_range and source_range[field] != getattr(target, field, None):
                return None
        for role, path, _ in inputs.files:
            if role == "image":
                current_path = getattr(target, "file_path", None) if hasattr(target, "frame_number") else getattr(target, "thumbnail_path", None)
                if current_path is None or Path(current_path).resolve() != path.resolve():
                    return None
        if not inputs.unchanged():
            return None
        return record
    except (ValueError, TypeError, KeyError, OSError):
        return None


def recorded_image_path(target: Any, source: Any, operation: str) -> Path | None:
    """Locate a prior CLI analysis image only while its source binding is current.

    This selects an input image, not a reusable result. The operation must still
    verify content, parameters, and model identity on the worker path.
    """
    record = getattr(target, "analysis_records", {}).get(operation)
    if not isinstance(record, AnalysisRecord) or record.input_json is None or source is None:
        return None
    try:
        inputs = AnalysisInput.from_dict(json.loads(record.input_json))
        binding = json.loads(inputs.binding_json)
        if binding != {"target_id": target.id, "source_id": target.source_id}:
            return None
        source_range = json.loads(inputs.range_json)
        expected_range = {"start_frame": target.start_frame, "end_frame": target.end_frame}
        if "fps" in source_range:
            expected_range["fps"] = source.fps
        if source_range != expected_range:
            return None
        files = {role: path for role, path, _ in inputs.files}
        if files.get("video") != source.file_path or not inputs.unchanged():
            return None
        return files.get("image")
    except (ValueError, TypeError, KeyError, OSError):
        return None


def model_runtime(name: str, packages: tuple[str, ...], **details: Any) -> dict:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"name": name, "packages": versions, **details}


class AnalysisFingerprints:
    """Reuse full-content hashing once per unchanged file in an operation batch."""

    def __init__(self, cancel: Event | None = None, *, media_fingerprints: "MediaFingerprints | None" = None) -> None:
        from core.jobs.media import MediaFingerprints

        self._fingerprints = media_fingerprints if media_fingerprints is not None else MediaFingerprints(cancel if cancel is not None else Event())
        self._lock = Lock()

    def identity(
        self, inputs: AnalysisInput, *, operation: str, model: dict,
        parameters: dict, sampling: dict, prompt: str | None = None,
        operation_version: int = 1, schema_version: int = 1,
    ) -> AnalysisIdentity:
        if not inputs.unchanged():
            raise ValueError("Analysis media changed while queued")
        sources = {}
        with self._lock:
            for role, path, _ in inputs.files:
                fingerprint = self._fingerprints.get(path)
                if fingerprint is None:
                    raise ValueError("Analysis media is missing")
                sources[role] = fingerprint["sha256"]
        if not inputs.unchanged():
            raise ValueError("Analysis media changed while fingerprinting")
        return AnalysisIdentity.build(
            operation=operation, sources=sources, source_range=json.loads(inputs.range_json),
            model=model, parameters=parameters, sampling=sampling, prompt=prompt,
            operation_version=operation_version, schema_version=schema_version,
        )
