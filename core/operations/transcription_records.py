"""Detached transcript projections and inference identities."""

from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot, model_runtime
from models.analysis_record import AnalysisIdentity

if TYPE_CHECKING:
    from core.operations.transcription import TranscriptionOptions, TranscriptionTask


def transcription_value(target: Any) -> dict:
    return {
        "transcript": None
        if target.transcript is None
        else [s.to_dict() for s in target.transcript]
    }


def transcription_segments_value(segments: Any) -> dict:
    """Validate inference output while preserving negative log probabilities."""
    from core.transcription_models import TranscriptSegment

    def number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
        )

    for segment in segments:
        if (
            not isinstance(segment, TranscriptSegment)
            or not isinstance(segment.text, str)
            or not all(
                number(v)
                for v in (segment.start_time, segment.end_time, segment.confidence)
            )
            or segment.start_time < 0
            or segment.end_time < segment.start_time
            or (segment.language is not None and not isinstance(segment.language, str))
        ):
            raise ValueError("Invalid transcript segment")
        for word in segment.words or ():
            if (
                not isinstance(word.text, str)
                or not number(word.start)
                or not number(word.end)
                or word.start < 0
                or word.end < word.start
                or (
                    word.probability is not None
                    and (not number(word.probability) or not 0 <= word.probability <= 1)
                )
            ):
                raise ValueError("Invalid transcript word")
    return {"transcript": [s.to_dict() for s in segments]}


def transcription_task(
    target: Any, source: Any, *, skip_existing: bool = True
) -> "TranscriptionTask":
    """Capture verified-reuse inputs without reading media or importing models."""
    from core.operations.transcription import snapshot_tasks

    task = snapshot_tasks(
        [target], {source.id: source} if source else {}, skip_existing=False
    )[0]
    if task.error:
        return task
    snapshot = AnalysisSnapshot.capture(
        target,
        "transcribe",
        {"video": task.source_path} if task.source_path else {},
        {
            "start_frame": target.start_frame,
            "end_frame": target.end_frame,
            "fps": task.fps,
        },
        transcription_value(target),
    )
    return replace(task, skip=skip_existing, analysis_json=snapshot.to_json())


def transcription_parameters(options: "TranscriptionOptions") -> dict:
    return {
        "backend": options.backend,
        "model": options.cloud_model if options.backend == "groq" else options.model,
        "language": options.language or "auto",
        "segmentation_mode": options.segmentation_mode,
        "segment_max_seconds": options.segment_max_seconds,
    }


def transcription_runtime(
    options: "TranscriptionOptions", *, execution: dict | None = None
) -> dict:
    from core.transcription import _resolve_backend, transcription_model
    from core.binary_resolver import find_binary
    from core.jobs.media import media_stamp

    if execution is None:
        backend = _resolve_backend(options.backend)
        execution = {
            "backend": backend,
            "model": transcription_model(backend, options.model, options.cloud_model),
            "input_mode": "audio",
        }
    backend = execution["backend"]
    packages = {
        "faster-whisper": ("faster-whisper", "ctranslate2", "av"),
        "mlx-whisper": ("lightning-whisper-mlx", "mlx", "numpy"),
        "groq": ("litellm", "groq"),
        "audio-probe": (),
    }[backend]
    binaries = {}
    for name in ("ffprobe",) if backend == "audio-probe" else ("ffprobe", "ffmpeg"):
        binary = find_binary(name)
        binaries[name] = {
            "path": binary,
            "stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
        }
    return {
        "execution": execution,
        "packages": model_runtime("transcribe", packages)["packages"],
        "binaries": binaries,
        "extraction": "clip-ss-to-pcm-s16le-16000-mono/v1",
        "normalization": "transcript-segmentation/v1",
        "inference": {
            "device": "auto",
            "compute_type": "int8",
            "vad_filter": True,
            "word_timestamps": True,
        }
        if backend == "faster-whisper"
        else {"batch_size": 12}
        if backend == "mlx-whisper"
        else {"response_format": "verbose_json", "timestamp_granularities": ["segment"]}
        if backend == "groq"
        else {},
    }


def transcription_identity(
    snapshot: AnalysisSnapshot,
    options: "TranscriptionOptions",
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="transcribe",
        operation_version=2,
        model=runtime,
        parameters=transcription_parameters(options),
        sampling={"policy": "half-open-clip-audio/v1"},
    )
