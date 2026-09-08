"""Alignment identities bind media, editorial transcript, and actual execution."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot, model_runtime
from core.jobs.media import media_stamp
from models.analysis_record import AnalysisIdentity, AnalysisRecord

if TYPE_CHECKING:
    from models.clip import Clip, Source


def alignment_model_revision() -> str | None:
    """Resolve an already cached model revision without networking or model loads."""
    from core.analysis.alignment import ALIGNMENT_MODEL

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(ALIGNMENT_MODEL, "config.json")
        if isinstance(cached, str):
            return Path(cached).parent.name
    except (ImportError, OSError, ValueError):
        pass
    return None


def alignment_runtime(*, execution: list[dict] | None = None) -> dict:
    from core.analysis.alignment import ALIGNMENT_MODEL
    from core.binary_resolver import find_binary

    binary = find_binary("ffmpeg")
    return model_runtime(
        ALIGNMENT_MODEL,
        ("ctc-forced-aligner", "torch", "transformers", "tokenizers", "numpy"),
        revision=alignment_model_revision(),
        device="cpu",
        dtype="float32",
        romanize=True,
        extraction="clip-ss-to-pcm-s16le-16000-mono/v1",
        fallback="whole-clip-then-segment-then-uniform/v1",
        distribution="midpoint-nearest-segment/v1",
        ffmpeg=str(binary) if binary else None,
        ffmpeg_stamp=list(media_stamp(Path(binary)) or ()) if binary else None,
        execution=execution or [],
    )


def alignment_parameters(transcript_json: str) -> dict:
    segments = json.loads(transcript_json)
    # Existing word positions are outputs, not alignment inputs. All editorial
    # segment fields remain inputs, including boundaries and language.
    for segment in segments:
        segment.pop("words", None)
    encoded = json.dumps(
        segments, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {"transcript_sha256": sha256(encoded.encode()).hexdigest()}


def alignment_identity(
    snapshot: AnalysisSnapshot,
    transcript_json: str,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="align_words",
        operation_version=2,
        model=runtime,
        parameters=alignment_parameters(transcript_json),
        sampling={"policy": "half-open-clip-audio/v1"},
    )


def execution_is_current(runtime: dict) -> bool:
    """Unknown model revisions cannot establish reusable word alignment."""
    from core.analysis.alignment import ALIGNMENT_MODEL

    events = runtime.get("execution", [])
    if not events:
        return False
    for event in events:
        if event.get("backend") == "ctc":
            if (
                event.get("model") != ALIGNMENT_MODEL
                or not runtime.get("revision")
                or event.get("revision") != runtime["revision"]
            ):
                return False
        elif event.get("backend") not in ("uniform", "empty"):
            return False
    return True


def alignment_execution_reusable(record: AnalysisRecord | None, runtime: dict) -> bool:
    """An explicit legacy decision carries no claim about prior execution."""
    return execution_is_current(runtime) or bool(
        record is not None and record.provenance == "unknown" and record.legacy_reuse
        and runtime.get("execution") == []
    )


def alignment_snapshot(clip: Clip, source: Source) -> AnalysisSnapshot:
    from core.operations.transcription_records import transcription_value

    return AnalysisSnapshot.capture(
        clip,
        "align_words",
        {"video": source.file_path},
        {
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "fps": source.fps,
        },
        transcription_value(clip),
    )


def prior_execution(snapshot: AnalysisSnapshot) -> list[dict]:
    record = snapshot.record
    if isinstance(record, AnalysisRecord) and record.identity is not None:
        events = record.identity.to_dict()["model"].get("execution", [])
        if isinstance(events, list) and all(
            isinstance(event, dict) for event in events
        ):
            return list(events)
    return []
