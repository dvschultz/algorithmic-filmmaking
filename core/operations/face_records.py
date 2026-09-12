"""Face analysis input bindings and content identities, independent of GUI code."""

from dataclasses import replace
import json
from math import isfinite
from pathlib import Path
from typing import Any

from core.analysis_records import (
    AnalysisFingerprints,
    AnalysisInput,
    AnalysisSnapshot,
    model_runtime,
)
from models.analysis_record import AnalysisRecord
from core.jobs.media import MediaFingerprints
from core.runtime_families import isolated

FACE_SAMPLING = {
    "policy": "half-open-uniform-frames/v1",
    "short_clip": "midpoint",
    "embedding_precision": 5,
}


def face_value(target: Any) -> dict:
    return face_result_value(target.face_embeddings)


def face_result_value(faces: list[dict] | None) -> dict:
    from core.operations.faces import Face

    if faces is None:
        return {"face_embeddings": None}
    validated = [Face.from_dict(face).to_dict() for face in faces]
    for face in validated:
        face["embedding"] = [round(v, 5) for v in face["embedding"]]
        if not any(v != 0 for v in face["embedding"]):
            raise ValueError("Invalid face result: stored embedding has no direction")
    return {"face_embeddings": validated}


def face_snapshot(target: Any, source: Any) -> AnalysisSnapshot:
    return AnalysisSnapshot.capture(
        target,
        "face_embeddings",
        {"video": Path(source.file_path)} if source else {},
        {
            "start_frame": target.start_frame,
            "end_frame": target.end_frame,
            "fps": source.fps if source else 0.0,
        },
        face_value(target),
    )


def face_packages() -> dict:
    return dict(
        model_runtime(
            "buffalo_l",
            ("insightface", "onnxruntime", "onnxruntime-gpu", "opencv-python", "numpy"),
        )["packages"]
    )


@isolated("vision", "faces.environment")
def face_environment() -> dict:
    """Report the native face runtime selected in the executing process."""
    import onnxruntime

    return {
        "packages": face_packages(),
        "available_providers": list(onnxruntime.get_available_providers()),
    }


def face_target_runtime() -> dict:
    """Capture queued model-file bindings without loading or hashing weights."""
    from core.analysis.faces import _get_model_cache_dir
    from core.jobs.media import media_stamp

    directory = (
        _get_model_cache_dir() / "insightface" / "models" / "buffalo_l"
    ).resolve()
    return {
        "packages": face_packages(),
        "directory": str(directory),
        "files": [
            {"path": str(path), "stamp": list(media_stamp(path) or ())}
            for path in sorted(directory.glob("*.onnx"))
        ],
    }


def face_target_matches(expected: dict, current: dict) -> bool:
    """Only the initial download may populate an absent model pack."""
    if not expected["files"]:
        current = {**current, "files": []}
    return current == expected


def face_parameters(interval: float) -> dict:
    if isinstance(interval, bool) or not isfinite(interval) or interval <= 0:
        raise ValueError("Invalid face sampling interval")
    return {"sample_interval": float(interval)}


def validate_face_frames(
    value: dict, start: int, end: int, fps: float, interval: float
) -> None:
    """Every observation must belong to an actual requested frame sample."""
    face_parameters(interval)
    if not isfinite(fps) or fps <= 0 or start < 0 or end <= start:
        raise ValueError("Invalid face sampling range")
    short = (end - start) / fps < interval
    step = max(1, int(interval * fps))
    for face in value["face_embeddings"]:
        frame = face.get("frame_number")
        if (
            frame is None
            or not start <= frame < end
            or (short and frame != start + (end - start) // 2)
            or (not short and (frame - start) % step != 0)
        ):
            raise ValueError("Face observation does not match submitted frame sampling")


def execution_inputs(snapshot: AnalysisSnapshot, execution: dict) -> AnalysisInput:
    """Bind all staged files, including unselected components in the model pack."""
    files = execution.get("weight_files", [])
    if not files:
        raise ValueError("Face execution did not report verified weight files")
    captured = list(snapshot.inputs.files)
    parents = set()
    roles = set()
    for item in files:
        path = Path(item["path"]).resolve()
        role = f"model:{path.name}"
        if path.suffix != ".onnx" or role in roles:
            raise ValueError("Invalid face model pack")
        roles.add(role)
        parents.add(path.parent)
        captured.append((role, path, tuple(item["stamp"])))
    if len(parents) != 1 or set(next(iter(parents)).glob("*.onnx")) != {
        p for role, p, _ in captured if role.startswith("model:")
    }:
        raise ValueError("Face model pack changed")
    result = replace(snapshot.inputs, files=tuple(sorted(captured)))
    if not result.unchanged():
        raise ValueError("Face media or weights changed")
    return result


def face_runtime(execution: dict, environment: dict) -> dict:
    """Keep paths and stamps in bindings; semantic runtime uses content hashes."""
    if (
        execution.get("backend") != "insightface"
        or execution.get("model") != "buffalo_l"
        or execution.get("detection_size") != [640, 640]
    ):
        raise ValueError("Unknown face execution")
    weights = {
        str(Path(item["path"]).resolve()): item for item in execution["weight_files"]
    }
    components = []
    for component in execution["components"]:
        item = weights.get(str(Path(component["path"]).resolve()))
        value = component.get("weights")
        if (
            item is None
            or value is None
            or value != {"sha256": item["sha256"], "stamp": item["stamp"]}
        ):
            raise ValueError("Face component fingerprint does not match its model pack")
        if not component["providers"] or not set(component["providers"]).issubset(
            environment["available_providers"]
        ):
            raise ValueError("Face execution providers are unavailable")
        components.append(
            {
                "task": component["task"],
                "file": Path(component["path"]).name,
                "sha256": item["sha256"],
                "providers": component["providers"],
            }
        )
    tasks = [c["task"] for c in components]
    if len(tasks) != len(set(tasks)) or not {"detection", "recognition"}.issubset(
        tasks
    ):
        raise ValueError("Face execution lacks detection or recognition")
    return {
        **environment,
        "model": "buffalo_l",
        "detection_size": [640, 640],
        "components": sorted(components, key=lambda c: c["task"]),
    }


def verified_face_execution(
    execution: dict, fingerprints: MediaFingerprints, environment: dict
) -> dict:
    """Verify reported model content before comparing reference and clip vectors."""
    from core.analysis.face_weights import FaceWeights

    runtime = face_runtime(execution, environment)
    files = tuple(
        sorted(
            (Path(item["path"]).resolve(), tuple(item["stamp"]), item["sha256"])
            for item in execution["weight_files"]
        )
    )
    if not files or len({path.parent for path, _, _ in files}) != 1:
        raise ValueError("Invalid reference face model pack")
    weights = FaceWeights(files[0][0].parent, files)
    if not weights.unchanged():
        raise ValueError("Reference face model pack changed")
    for path, stamp, digest in files:
        value = fingerprints.get(path)
        if value is None or tuple(value["stamp"]) != stamp or value["sha256"] != digest:
            raise ValueError("Reference face model content changed")
    if not weights.unchanged():
        raise ValueError("Reference face model pack changed")
    return {
        "runtime": runtime,
        "weights": [(path.name, digest) for path, _, digest in files],
    }


def saved_execution(record, *, directory: Path | None = None) -> dict:
    """Reconstruct bindings for worker-side full-content verification."""
    data = record.identity.to_dict()
    if directory is None:
        inputs = AnalysisInput.from_dict(json.loads(record.input_json))
        paths = [
            (role, path, stamp)
            for role, path, stamp in inputs.files
            if role.startswith("model:")
        ]
    else:
        from core.jobs.media import media_stamp

        paths = []
        for role in data["sources"]:
            if not role.startswith("model:"):
                continue
            name = role.removeprefix("model:")
            if Path(name).name != name or Path(name).suffix != ".onnx":
                raise ValueError("Invalid saved face model filename")
            path = directory / name
            paths.append((role, path, media_stamp(path)))
    files = [
        {
            "path": str(path),
            "stamp": list(stamp) if stamp else [],
            "sha256": data["sources"][role],
        }
        for role, path, stamp in paths
    ]
    by_name = {Path(f["path"]).name: f for f in files}
    components = []
    for component in data["model"]["components"]:
        item = by_name[component["file"]]
        components.append(
            {
                "task": component["task"],
                "path": item["path"],
                "providers": component["providers"],
                "weights": {"sha256": component["sha256"], "stamp": item["stamp"]},
            }
        )
    return {
        "backend": "insightface",
        "model": "buffalo_l",
        "detection_size": [640, 640],
        "components": components,
        "weight_files": files,
    }


def reusable_face_record(
    snapshot: AnalysisSnapshot,
    interval: float,
    fingerprints: AnalysisFingerprints,
    environment: dict,
) -> AnalysisRecord | None:
    """Fully verify saved faces without performing inference on a cache miss."""
    if snapshot.record is None or snapshot.record.identity is None:
        return None
    try:
        from core.jobs.media import media_stamp
        from core.analysis.faces import _get_model_cache_dir

        directory = (
            _get_model_cache_dir() / "insightface" / "models" / "buffalo_l"
        ).resolve()
        execution = saved_execution(snapshot.record, directory=directory)
        for item in execution["weight_files"]:
            if Path(item["path"]).parent != directory:
                return None
            item["stamp"] = list(media_stamp(Path(item["path"])) or ())
        by_path = {item["path"]: item for item in execution["weight_files"]}
        for component in execution["components"]:
            component["weights"]["stamp"] = by_path[component["path"]]["stamp"]
        bound = execution_inputs(snapshot, execution)
        runtime = face_runtime(execution, environment)
        candidate = fingerprints.identity(
            bound,
            operation="face_embeddings",
            operation_version=2,
            model=runtime,
            parameters=face_parameters(interval),
            sampling=FACE_SAMPLING,
        )
        reused = replace(snapshot, inputs=bound).reusable_record(candidate)
        if reused is not None and bound.unchanged():
            value = face_result_value(reused.value["face_embeddings"])
            if value == reused.value:
                return reused
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        pass
    return None
