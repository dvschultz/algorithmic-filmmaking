"""Utilities for determining per-operation analysis availability."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.settings import Settings
    from models.audio_source import AudioSource

# These operations verify complete input identities on the worker path.
VERIFIED_ANALYSIS_OPERATIONS = frozenset({"colors", "embeddings", "boundary_embeddings", "detect_objects", "extract_text", "classify", "shots", "gaze", "describe", "cinematography", "transcribe"})


_ANALYSIS_RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    "colors": ("dominant_colors",),
    "shots": ("shot_type",),
    "classify": ("object_labels",),
    "detect_objects": ("detected_objects", "person_count"),
    "extract_text": ("extracted_texts",),
    "transcribe": ("transcript",),
    "describe": ("description", "description_model", "description_frames"),
    "cinematography": ("cinematography",),
    "face_embeddings": ("face_embeddings",),
    "gaze": ("gaze_yaw", "gaze_pitch", "gaze_category"),
    "embeddings": (
        "embedding",
        "first_frame_embedding",
        "last_frame_embedding",
        "embedding_model",
    ),
    "boundary_embeddings": ("first_frame_embedding", "last_frame_embedding"),
}


def operation_has_result(op_key: str, clip) -> bool:
    """Include legacy field projections when filtering or displaying results."""
    if op_key == "colors":
        return clip.dominant_colors is not None
    if op_key == "shots":
        return clip.shot_type is not None
    if op_key == "classify":
        return clip.object_labels is not None
    if op_key == "detect_objects":
        return clip.detected_objects is not None
    if op_key == "extract_text":
        return clip.extracted_texts is not None
    if op_key == "transcribe":
        return clip.transcript is not None
    if op_key == "describe":
        return clip.description is not None
    if op_key == "cinematography":
        return clip.cinematography is not None
    if op_key == "face_embeddings":
        return clip.face_embeddings is not None
    if op_key == "gaze":
        return clip.gaze_category is not None
    if op_key == "embeddings":
        return clip.embedding is not None
    if op_key == "boundary_embeddings":
        return clip.first_frame_embedding is not None and clip.last_frame_embedding is not None
    if op_key == "custom_query":
        # Each custom query is unique — never auto-skip, always allow rerun
        return False
    return False


def audio_transcription_is_complete(audio: AudioSource, *, settings: Settings | None = None) -> bool:
    """Check current audio completion without media hashing, probes, or inference."""
    import json
    from core.analysis_records import AnalysisSnapshot, current_record
    from core.operations.audio_transcription import AudioTranscriptionTask, audio_transcription_runtime
    from core.operations.transcription import TranscriptionOptions, resolve_transcription_options
    from core.operations.transcription_records import transcription_parameters, transcription_value
    from core.settings import load_settings

    record = current_record(audio, "transcribe")
    if record is None or record.identity is None:
        return False
    try:
        configured = settings if settings is not None else load_settings()
        options = resolve_transcription_options(TranscriptionOptions(
            model=configured.transcription_model,
            language=configured.transcription_language,
            backend=configured.transcription_backend,
            segmentation_mode=configured.transcription_segmentation_mode,
            segment_max_seconds=configured.transcription_segment_max_seconds,
        ))
        task = AudioTranscriptionTask.from_audio(audio, verified=True)
        if task.analysis_json is None:
            return False
        snapshot = AnalysisSnapshot.from_json(task.analysis_json)
        if json.loads(record.input_json or "null") != snapshot.inputs.to_dict():
            return False
        data = record.identity.to_dict()
        probe_execution = {"backend": "audio-probe", "model": None, "input_mode": "no-audio"}
        no_audio = data["model"].get("execution") == probe_execution
        runtime = audio_transcription_runtime(task, options, execution=probe_execution if no_audio else None)
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == runtime
            and data["parameters"] == transcription_parameters(options)
            and data["source_range"] == json.loads(snapshot.inputs.range_json)
            and data["sampling"] == {"policy": "whole-audio-file/v1"}
            and data["prompt_sha256"] is None
            and record.value == transcription_value(audio)
            and (not no_audio or audio.transcript == [])
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def operation_is_complete_for_clip(op_key: str, clip, *, runtime: dict | None = None, source=None) -> bool:
    """Report reusable completion; existing fields alone do not prove provenance."""
    if op_key == "transcribe":
        import json
        from core.analysis_records import AnalysisSnapshot, current_record
        from core.operations.transcription import TranscriptionOptions, resolve_transcription_options
        from core.operations.transcription_records import (
            transcription_task, transcription_runtime, transcription_parameters, transcription_value,
        )
        from core.settings import load_settings

        record = current_record(clip, op_key)
        if record is None or record.identity is None or source is None:
            return False
        try:
            settings = load_settings()
            transcript_options = resolve_transcription_options(TranscriptionOptions(
                model=settings.transcription_model, language=settings.transcription_language,
                backend=settings.transcription_backend,
                segmentation_mode=settings.transcription_segmentation_mode,
                segment_max_seconds=settings.transcription_segment_max_seconds,
            ))
            transcript_task = transcription_task(clip, source)
            if transcript_task.analysis_json is None:
                return False
            snapshot = AnalysisSnapshot.from_json(transcript_task.analysis_json)
            if json.loads(record.input_json or "null") != snapshot.inputs.to_dict():
                return False
            data = record.identity.to_dict()
            # A confirmed no-audio result remains valid for unchanged inputs.
            # Re-running ffprobe belongs on the worker, never in UI availability.
            probe_execution = {"backend": "audio-probe", "model": None, "input_mode": "no-audio"}
            no_audio = data["model"].get("execution") == probe_execution
            expected_runtime = runtime if runtime is not None else transcription_runtime(transcript_options, execution=probe_execution if no_audio else None)
            return bool(
                data["operation_version"] == 2 and data["schema_version"] == 1
                and data["model"] == expected_runtime
                and data["parameters"] == transcription_parameters(transcript_options)
                and data["source_range"] == json.loads(snapshot.inputs.range_json)
                and data["sampling"] == {"policy": "half-open-clip-audio/v1"}
                and data["prompt_sha256"] is None
                and record.value == transcription_value(clip)
                and (not no_audio or clip.transcript == [])
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False
    if op_key == "cinematography":
        import json
        from hashlib import sha256
        from core.analysis_records import AnalysisSnapshot, current_record
        from core.operations.cinematography import (
            cinematography_task, cinematography_runtime, cinematography_parameters,
            cinematography_prompt, cinematography_value, resolve_options as resolve_cinematography,
        )

        record = current_record(clip, op_key)
        if record is None or record.identity is None:
            return False
        try:
            cinema_options = resolve_cinematography()
            cinema_task = cinematography_task(clip, source)
            if cinema_task.snapshot_json is None:
                return False
            snapshot = AnalysisSnapshot.from_json(cinema_task.snapshot_json)
            if json.loads(record.input_json or "null") != snapshot.inputs.to_dict():
                return False
            expected_runtime = runtime if runtime is not None else cinematography_runtime(cinema_task, cinema_options, allow_imports=False)
            execution = expected_runtime["execution"]
            data = record.identity.to_dict()
            return bool(
                execution["backend"] != "unavailable"
                and data["operation_version"] == 2 and data["schema_version"] == 1
                and data["model"] == expected_runtime
                and data["parameters"] == cinematography_parameters(cinema_options)
                and data["sampling"] == {"policy": execution["input_mode"] + "/v1"}
                and data["prompt_sha256"] == sha256(cinematography_prompt(execution).encode()).hexdigest()
                and record.value == cinematography_value(clip)
            )
        except (OSError, ValueError, TypeError, KeyError):
            return False
    if op_key == "describe":
        import json
        from hashlib import sha256
        from core.analysis_records import AnalysisInput, current_record
        from core.operations.description import description_task, description_runtime, resolve_options

        record = current_record(clip, op_key)
        if record is None or record.identity is None:
            return False
        try:
            inputs = AnalysisInput.from_dict(json.loads(record.input_json or "null"))
            files = {role: path for role, path, _ in inputs.files}
            source_range = json.loads(inputs.range_json)
            if "video" in files and (
                source is None or source.id != clip.source_id
                or source.file_path != files["video"] or source.fps != source_range.get("fps")
            ):
                return False
            if source is not None and "video" not in files:
                return False
            description_options = resolve_options()
            task = description_task(clip, source)
            expected_runtime = runtime if runtime is not None else description_runtime(task, description_options, allow_imports=False)
            data = record.identity.to_dict()
            return bool(
                data["operation_version"] == 2 and data["schema_version"] == 1
                and data["model"] == expected_runtime
                and data["parameters"] == {"tier": description_options.tier, "model": description_options.model, "input_mode": description_options.input_mode}
                and data["sampling"] == {"policy": "description-input/v1"}
                and data["prompt_sha256"] == sha256(description_options.prompt.encode()).hexdigest()
                and record.value == {"description": clip.description, "description_model": clip.description_model, "description_frames": getattr(clip, "description_frames", None)}
            )
        except (OSError, ValueError, TypeError, KeyError):
            return False
    if op_key == "boundary_embeddings":
        import json
        from hashlib import sha256
        from core.analysis_records import current_record
        from core.analysis_model_identity import boundary_embedding_runtime, DINOV2_TAG

        record = current_record(clip, op_key)
        if record is None or record.identity is None or clip.first_frame_embedding is None or clip.last_frame_embedding is None:
            return False
        data = record.identity.to_dict()
        if (data["operation_version"] != 2 or data["schema_version"] != 1
            or data["model"] != (runtime if runtime is not None else boundary_embedding_runtime())
            or data["parameters"] != {} or data["prompt_sha256"] is not None
            or data["sampling"] != {"policy": "boundary-start/end-minus-one-v1", "processor": "dinov2-default"}
            or clip.embedding_model != DINOV2_TAG):
            return False
        value = {"first_frame_embedding": clip.first_frame_embedding, "last_frame_embedding": clip.last_frame_embedding, "embedding_model": clip.embedding_model}
        if record.artifact is not None:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            return record.artifact.media_type == "application/json" and record.artifact.digest == sha256(encoded).hexdigest()
        return bool(record.value == value)
    if op_key == "gaze":
        from core.analysis_records import current_record
        from core.analysis_model_identity import gaze_runtime
        from core.operations.gaze import gaze_values

        record = current_record(clip, op_key)
        if record is None or record.identity is None:
            return False
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == (runtime if runtime is not None else gaze_runtime())
            and data["parameters"] == {"sample_interval": 1.0}
            and data["sampling"] == {"policy": "half-open-uniform-frames/v1", "short_clip": "midpoint", "angle_precision": 2}
            and record.value == gaze_values(clip)
        )
    if op_key == "shots":
        from dataclasses import asdict
        from hashlib import sha256
        import json
        from core.analysis_records import current_record
        from core.analysis_model_identity import SHOT_CLOUD_PROMPT, SHOT_TYPE_PROMPTS, shot_runtime
        from core.operations.shots import ShotTypeOptions

        record = current_record(clip, op_key)
        if record is None or record.identity is None or not clip.shot_type:
            return False
        options = ShotTypeOptions.from_settings()
        prompt = SHOT_CLOUD_PROMPT if options.tier == "cloud" else json.dumps(SHOT_TYPE_PROMPTS, sort_keys=True)
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == {"runtime": runtime if runtime is not None else shot_runtime(), "backend": options.tier, "cloud_model": options.cloud_model}
            and data["parameters"] == asdict(options)
            and data["sampling"] == {"policy": "single-image/v1", "local_ensemble": True}
            and data["prompt_sha256"] == sha256(prompt.encode()).hexdigest()
            and record.value == {"shot_type": clip.shot_type}
        )
    if op_key == "classify":
        from core.analysis_records import current_record
        from core.analysis_model_identity import classification_runtime

        record = current_record(clip, op_key)
        if record is None or record.identity is None or clip.object_labels is None:
            return False
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == (runtime if runtime is not None else classification_runtime())
            and data["parameters"] == {"top_k": 5, "threshold": 0.1}
            and data["sampling"] == {"policy": "single-image/v1"}
            and record.value == {"object_labels": clip.object_labels}
        )
    if op_key == "extract_text":
        from dataclasses import asdict
        from hashlib import sha256
        from core.analysis_records import current_record
        from core.analysis_model_identity import OCR_PROMPT, ocr_runtime
        from core.operations.ocr import OcrOptions, resolve_ocr_options

        record = current_record(clip, op_key)
        if record is None or record.identity is None or clip.extracted_texts is None:
            return False
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == (runtime if runtime is not None else ocr_runtime())
            and data["parameters"] == asdict(resolve_ocr_options(OcrOptions()))
            and data["sampling"] == {"policy": "half-open-keyframes/v1"}
            and data["prompt_sha256"] == sha256(OCR_PROMPT.encode()).hexdigest()
            and record.value == {"extracted_texts": [text.to_dict() for text in clip.extracted_texts]}
        )
    if op_key == "detect_objects":
        from core.analysis_records import current_record
        from core.analysis_model_identity import object_detection_runtime

        record = current_record(clip, op_key)
        if record is None or record.identity is None:
            return False
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == (runtime if runtime is not None else object_detection_runtime())
            and data["parameters"] == {"confidence": 0.5, "detect_all": True}
            and data["sampling"] == {"policy": "single-image/v1"}
            and record.value == {"detected_objects": clip.detected_objects, "person_count": clip.person_count}
        )
    if op_key == "colors":
        from core.analysis_records import current_record, model_runtime

        record = current_record(clip, op_key)
        if record is None or record.identity is None or not clip.dominant_colors:
            return False
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == (runtime if runtime is not None else model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python")))
            and data["parameters"] == {"num_colors": 5, "random_state": 42, "n_init": 1, "max_iter": 100}
            and record.value == {"dominant_colors": [list(color) for color in clip.dominant_colors]}
        )
    if op_key == "embeddings":
        from hashlib import sha256
        import json
        from core.analysis_records import current_record
        from core.analysis_model_identity import embedding_runtime

        record = current_record(clip, op_key)
        if record is None or record.identity is None or clip.embedding is None:
            return False
        data = record.identity.to_dict()
        if (
            data["operation_version"] != 2 or data["schema_version"] != 1
            or data["model"] != (runtime if runtime is not None else embedding_runtime())
            or data["parameters"] != {}
            or data["sampling"] != {"policy": "thumbnail/v1", "processor": "dinov2-default"}
        ):
            return False
        value = {"embedding": clip.embedding, "embedding_model": clip.embedding_model}
        if record.artifact is None:
            return bool(record.value == value)
        from core.artifacts import ArtifactStore

        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return sha256(encoded).hexdigest() == record.artifact.digest and ArtifactStore().available_fast(record.artifact)
    return operation_has_result(op_key, clip)


def compute_operation_need_counts(clips: Iterable, op_keys: Iterable[str], *, sources_by_id: dict | None = None) -> dict[str, int]:
    """Count how many clips still need each operation."""
    clip_list = list(clips)
    counts: dict[str, int] = {}
    for op_key in op_keys:
        runtime = None
        if op_key == "boundary_embeddings":
            from core.analysis_model_identity import boundary_embedding_runtime

            runtime = boundary_embedding_runtime()
        elif op_key == "gaze":
            from core.analysis_model_identity import gaze_runtime

            runtime = gaze_runtime()
        elif op_key == "shots":
            from core.analysis_model_identity import shot_runtime

            runtime = shot_runtime()
        elif op_key == "classify":
            from core.analysis_model_identity import classification_runtime

            runtime = classification_runtime()
        elif op_key == "extract_text":
            from core.analysis_model_identity import ocr_runtime

            runtime = ocr_runtime()
        elif op_key == "detect_objects":
            from core.analysis_model_identity import object_detection_runtime

            runtime = object_detection_runtime()
        elif op_key == "colors":
            from core.analysis_records import model_runtime

            runtime = model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python"))
        elif op_key == "embeddings":
            from core.analysis_model_identity import embedding_runtime

            runtime = embedding_runtime()
        counts[op_key] = sum(
            1 for clip in clip_list if not operation_is_complete_for_clip(op_key, clip, runtime=runtime, source=(sources_by_id or {}).get(getattr(clip, "source_id", None)))
        )
    return counts


def compute_disabled_operations(clips: Iterable, op_keys: Iterable[str], *, sources_by_id: dict | None = None) -> set[str]:
    """Return operations that are already complete for all clips in scope."""
    counts = compute_operation_need_counts(clips, op_keys, sources_by_id=sources_by_id)
    return {op_key for op_key, needing in counts.items() if needing == 0}


def clear_operation_result(clip, op_key: str) -> bool:
    """Clear stored result fields for one analysis operation on a clip.

    Returns True when at least one populated field was cleared. Unknown
    operations and custom queries are ignored because custom-query results are
    query-specific and should not be globally removed for a new query run.
    """
    changed = False
    records = getattr(clip, "analysis_records", {})
    if op_key in _ANALYSIS_RESULT_FIELDS and op_key in records:
        del records[op_key]
        changed = True
    for field in _ANALYSIS_RESULT_FIELDS.get(op_key, ()):
        if hasattr(clip, field) and getattr(clip, field) is not None:
            setattr(clip, field, None)
            changed = True
    return changed


def clear_operation_results(clips: Iterable, op_keys: Iterable[str]) -> int:
    """Clear stored result fields for selected operations across clips."""
    cleared = 0
    for clip in clips:
        for op_key in op_keys:
            if clear_operation_result(clip, op_key):
                cleared += 1
    return cleared
