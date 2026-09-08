"""Worker-side explicit acceptance of old values without claiming inference.

Callers must expose this as a deliberate reuse decision, never an automatic
migration. Publication uses the normal operation's stale-input owner guard.
"""

from dataclasses import replace
import json
from math import isfinite
from threading import Event
from typing import TYPE_CHECKING, cast

from core.analysis_records import AnalysisFingerprints, AnalysisInput
from core.operations.colors import ColorRequest, color_identity
from core.operations.contracts import ColorOutcome, ColorResult
from core.operations.embeddings import EmbeddingOutcome, EmbeddingTask, embedding_identity
from models.analysis_record import AnalysisIdentity, AnalysisRecord

LEGACY_REUSE_OPERATIONS = ("colors", "embeddings", "brightness", "volume", "classify", "detect_objects", "boundary_embeddings", "gaze", "shots", "extract_text", "describe", "cinematography", "transcribe", "align_words", "custom_query")

if TYPE_CHECKING:
    from core.operations.audio_transcription import AudioTranscriptionTask, AudioTranscriptionOutcome
    from core.operations.custom_query import CustomQueryTask, CustomQueryOutcome, CustomQueryOptions
    from core.operations.alignment import AlignmentTask, AlignmentOutcome
    from core.operations.transcription import TranscriptionOptions, TranscriptionOutcome, TranscriptionTask
    from core.settings import Settings
    from core.operations.cinematography import CinematographyOptions, CinematographyOutcome, CinematographyTask
    from core.operations.description import DescriptionOptions, DescriptionOutcome, DescriptionTask
    from core.operations.scalars import ScalarOutcome, ScalarTask
    from core.operations.classification import ClassificationOutcome, ClassificationTask
    from core.operations.object_detection import ObjectDetectionOutcome, ObjectDetectionTask
    from core.operations.boundary_embeddings import BoundaryEmbeddingOutcome, BoundaryEmbeddingTask
    from core.operations.gaze import GazeOutcome, GazeTask
    from core.operations.shots import ShotTypeOutcome, ShotTypeTask, ShotTypeOptions
    from core.operations.ocr import OcrOutcome, OcrTask, OcrOptions


def _accept(record_json: str | None, value: dict, identity: AnalysisIdentity, inputs: AnalysisInput) -> str:
    record = AnalysisRecord.from_dict(json.loads(record_json)) if record_json else AnalysisRecord.legacy(value)
    if record.provenance != "unknown" or record.state != "succeeded":
        raise ValueError("Only successful values with unknown provenance can be explicitly reused")
    if record.artifact is not None:
        from core.artifacts import ArtifactStore

        previous = json.loads(ArtifactStore().read_bytes(record.artifact))
    else:
        previous = record.value
    if identity.operation == "describe" and isinstance(previous, dict):
        previous = {"description_model": None, "description_frames": None, **previous}
    if identity.operation == "cinematography" and isinstance(previous, dict) and "shot_type" not in previous and isinstance(previous.get("cinematography"), dict):
        from models.cinematography import CinematographyAnalysis

        previous = {**previous, "shot_type": CinematographyAnalysis.from_dict(previous["cinematography"]).get_simple_shot_type()}
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


def accept_legacy_audio_transcript(task: "AudioTranscriptionTask", options: "TranscriptionOptions", *, cancel_event: Event | None = None) -> "AudioTranscriptionOutcome":
    """Bind a saved whole-file transcript without transcribing the audio again."""
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.audio_transcription import AudioTranscriptionOutcome, audio_transcription_identity, audio_transcription_runtime
    from core.operations.transcription import resolve_transcription_options
    from core.operations.transcription_records import transcription_segments_value
    from core.transcription_models import TranscriptSegment

    try:
        if cancel_event is not None and cancel_event.is_set():
            raise FingerprintCancelled()
        if task.analysis_json is None:
            raise ValueError("Audio reuse requires captured project inputs")
        snapshot = AnalysisSnapshot.from_json(task.analysis_json)
        region = json.loads(snapshot.inputs.range_json)
        duration = region["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not isfinite(duration) or duration <= 0:
            raise ValueError("Audio reuse requires a positive finite duration")
        value = json.loads(snapshot.value_json)
        if not isinstance(value["transcript"], list):
            raise ValueError("No legacy audio transcript is available")
        segments = tuple(TranscriptSegment.from_dict(item) for item in value["transcript"])
        if transcription_segments_value(segments) != value:
            raise ValueError("Legacy audio transcript is not canonical")
        if any(segment.end_time > duration or any(word.start < segment.start_time or word.end > segment.end_time for word in segment.words or ()) for segment in segments):
            raise ValueError("Legacy transcript timing lies outside its audio or segment")
        options = resolve_transcription_options(options)
        identity = audio_transcription_identity(snapshot, options, AnalysisFingerprints(cancel_event), audio_transcription_runtime(task, options))
        record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
        return AudioTranscriptionOutcome(task.audio_source_id, "skipped", segments, record_json=record_json)
    except FingerprintCancelled:
        return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed", message="Cancelled")
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        return AudioTranscriptionOutcome(task.audio_source_id, "failed", message=str(exc))


def accept_legacy_queries(tasks: "tuple[CustomQueryTask, ...]", options: "CustomQueryOptions", *, cancel_event: Event | None = None) -> "tuple[CustomQueryOutcome, ...]":
    """Accept the latest saved answer to one exact question without adding history."""
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.custom_query import CustomQueryOutcome, custom_query_identity, custom_query_runtime

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = custom_query_runtime(options)
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if not task.query.strip() or task.analysis_json is None or task.thumbnail_path is None or not task.thumbnail_path.is_file():
                raise ValueError("Query reuse requires an exact question and readable thumbnail")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            answer = value["result"]
            if not isinstance(answer, dict) or answer.get("query") != task.query or type(answer.get("match")) is not bool:
                raise ValueError("No valid legacy answer is available for this question")
            confidence = answer.get("confidence")
            model = answer.get("model")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Legacy answer confidence is missing or invalid")
            if model is not None and (not isinstance(model, str) or not model.strip()):
                raise ValueError("Invalid legacy answer model")
            outcome = CustomQueryOutcome(task.clip_id, task.query, "skipped", answer["match"], confidence, model, code="legacy_accepted")
            if outcome.value != answer:
                raise ValueError("Legacy answer is not canonical; recompute this query")
            identity = custom_query_identity(snapshot, task.query, options, fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(replace(outcome, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(CustomQueryOutcome(task.clip_id, task.query, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(CustomQueryOutcome(task.clip_id, task.query, "failed", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_alignment(tasks: "tuple[AlignmentTask, ...]", *, cancel_event: Event | None = None) -> "tuple[AlignmentOutcome, ...]":
    """Accept stored word timings without claiming CTC or fallback execution."""
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.alignment import AlignmentOutcome
    from core.operations.alignment_records import alignment_identity, alignment_runtime
    from core.operations.transcription_records import transcription_segments_value
    from core.transcription_models import TranscriptSegment

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = alignment_runtime()
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            target = task.target
            if target.error or task.analysis_json is None or target.source_path is None:
                raise ValueError(target.error or "Alignment reuse requires readable source media")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            if not isinstance(value["transcript"], list) or value["transcript"] != json.loads(task.transcript_json):
                raise ValueError("No consistent legacy transcript is available")
            segments = tuple(TranscriptSegment.from_dict(item) for item in value["transcript"])
            if transcription_segments_value(segments) != value:
                raise ValueError("Legacy transcript is not canonical")
            duration = target.end_time - target.start_time
            if target.start_time < 0 or not isfinite(duration) or duration <= 0:
                raise ValueError("Invalid alignment source range")
            if any(segment.words is None or (segment.text.strip() and not segment.words) or segment.end_time > duration or any(word.start < segment.start_time or word.end > segment.end_time for word in segment.words or ()) for segment in segments):
                raise ValueError("Legacy word timings are missing or outside their clip or segment")
            identity = alignment_identity(snapshot, task.transcript_json, fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            words = tuple(word for segment in segments for word in segment.words or ())
            outcomes.append(AlignmentOutcome(task.clip_id, "skipped", words, code="legacy_accepted", record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
            outcomes.append(AlignmentOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)


def legacy_transcription_options(settings: "Settings | None" = None) -> "TranscriptionOptions":
    """Capture requested transcript settings before queueing an acceptance."""
    from core.settings import load_settings
    from core.operations.transcription import TranscriptionOptions, resolve_transcription_options

    settings = settings if settings is not None else load_settings()
    return resolve_transcription_options(TranscriptionOptions(
        model=settings.transcription_model, language=settings.transcription_language,
        backend=settings.transcription_backend, cloud_model=settings.transcription_cloud_model,
        segmentation_mode=settings.transcription_segmentation_mode,
        segment_max_seconds=settings.transcription_segment_max_seconds,
    ))


def accept_legacy_transcription(tasks: "tuple[TranscriptionTask, ...]", options: "TranscriptionOptions", *, cancel_event: Event | None = None) -> "tuple[TranscriptionOutcome, ...]":
    """Accept saved clip-relative segments without changing text or word timing."""
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.transcription import TranscriptionOutcome, resolve_transcription_options
    from core.operations.transcription_records import transcription_identity, transcription_runtime, transcription_segments_value
    from core.transcription_models import TranscriptSegment

    fingerprints = AnalysisFingerprints(cancel_event)
    options = resolve_transcription_options(options)
    runtime = transcription_runtime(options)
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.error or task.analysis_json is None or task.source_path is None:
                raise ValueError(task.error or "Transcription reuse requires readable source media")
            if not isfinite(task.start_time) or not isfinite(task.end_time) or task.start_time < 0 or task.end_time <= task.start_time:
                raise ValueError("Transcription reuse requires a valid source range")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            if not isinstance(value["transcript"], list):
                raise ValueError("No legacy transcript is available")
            segments = tuple(TranscriptSegment.from_dict(item) for item in value["transcript"])
            if transcription_segments_value(segments) != value:
                raise ValueError("Legacy transcript is not canonical; recompute analysis")
            duration = task.end_time - task.start_time
            if any(segment.end_time > duration or any(word.start < segment.start_time or word.end > segment.end_time for word in segment.words or ()) for segment in segments):
                raise ValueError("Legacy transcript timing lies outside its clip or segment")
            identity = transcription_identity(snapshot, options, fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(TranscriptionOutcome(task.clip_id, "skipped", segments, code="legacy_accepted", record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(TranscriptionOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
            outcomes.append(TranscriptionOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_cinematography(tasks: "tuple[CinematographyTask, ...]", options: "CinematographyOptions", *, cancel_event: Event | None = None) -> "tuple[CinematographyOutcome, ...]":
    """Accept valid saved film-language observations without rewriting them."""
    from core.analysis.cinematography import CINEMATOGRAPHY_SCHEMA
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.cinematography import CinematographyOutcome, cinematography_identity, cinematography_runtime
    from models.cinematography import CinematographyAnalysis

    fingerprints = AnalysisFingerprints(cancel_event)
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.snapshot_json is None or task.thumbnail_path is None or not task.thumbnail_path.is_file():
                raise ValueError("Cinematography reuse requires a readable thumbnail")
            snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
            value = json.loads(snapshot.value_json)
            raw = value["cinematography"]
            if not isinstance(raw, dict):
                raise ValueError("No legacy cinematography result is available")
            analysis = CinematographyAnalysis.from_dict(raw)
            for field, schema in cast(dict[str, dict], CINEMATOGRAPHY_SCHEMA["properties"]).items():
                if "enum" in schema and getattr(analysis, field) not in schema["enum"]:
                    raise ValueError(f"Invalid legacy cinematography field: {field}")
            confidence = analysis.shot_size_confidence
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid legacy cinematography confidence")
            if analysis.analysis_mode not in ("frame", "video") or (
                analysis.analysis_model is not None and (not isinstance(analysis.analysis_model, str) or not analysis.analysis_model.strip())
            ):
                raise ValueError("Invalid legacy cinematography metadata")
            if raw != analysis.to_dict() or value["shot_type"] != analysis.get_simple_shot_type():
                raise ValueError("Legacy cinematography and shot projection disagree; recompute analysis")
            if task.target_type == "clip" and (
                task.source_path is None or task.fps is None or isinstance(task.fps, bool)
                or not isfinite(task.fps) or task.fps <= 0
                or type(task.start_frame) is not int or type(task.end_frame) is not int
                or task.start_frame < 0 or task.end_frame <= task.start_frame
            ):
                raise ValueError("Cinematography reuse requires a valid source range")
            identity = cinematography_identity(snapshot, options, fingerprints, cinematography_runtime(task, options))
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(CinematographyOutcome(task.clip_id, "skipped", json.dumps(raw, sort_keys=True), code="legacy_accepted", record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(CinematographyOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(CinematographyOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_descriptions(tasks: "tuple[DescriptionTask, ...]", options: "DescriptionOptions", *, cancel_event: Event | None = None) -> "tuple[DescriptionOutcome, ...]":
    """Bind saved descriptions to requested inputs without inventing old metadata."""
    from core.analysis_records import AnalysisSnapshot
    from core.jobs.media import FingerprintCancelled
    from core.operations.description import DescriptionOutcome, description_identity, description_runtime

    fingerprints = AnalysisFingerprints(cancel_event)
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.analysis_json is None or task.thumbnail_path is None or not task.thumbnail_path.is_file():
                raise ValueError("Description reuse requires a readable thumbnail")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            description = value["description"]
            model = value["description_model"]
            frames = value["description_frames"]
            if not isinstance(description, str) or not description.strip() or description.startswith("Error"):
                raise ValueError("A nonempty legacy description is required")
            if model is not None and (not isinstance(model, str) or not model.strip()):
                raise ValueError("Legacy description model is invalid")
            if frames is not None and (type(frames) is not int or frames <= 0):
                raise ValueError("Legacy description frame count is invalid")
            if task.target_type == "clip" and (
                task.source_path is None or task.fps is None or isinstance(task.fps, bool)
                or not isfinite(task.fps) or task.fps <= 0
                or type(task.start_frame) is not int or type(task.end_frame) is not int
                or task.start_frame < 0 or task.end_frame <= task.start_frame
            ):
                raise ValueError("Description reuse requires a valid source range")
            identity = description_identity(snapshot, options, fingerprints, description_runtime(task, options))
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(DescriptionOutcome(task.clip_id, "skipped", description, model, code="legacy_accepted", record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(DescriptionOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)


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


def accept_legacy_boundaries(tasks: "tuple[BoundaryEmbeddingTask, ...]", *, cancel_event: Event | None = None) -> "tuple[BoundaryEmbeddingOutcome, ...]":
    """Accept complete compatible endpoint pairs without decoding or inference."""
    from core.analysis_records import AnalysisSnapshot
    from core.analysis_model_identity import boundary_embedding_runtime
    from core.jobs.media import FingerprintCancelled
    from core.operations.boundary_embeddings import BoundaryEmbeddingOutcome, boundary_embedding_identity

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = boundary_embedding_runtime()
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if (
                task.analysis_json is None or task.source_path is None
                or isinstance(task.fps, bool) or not isfinite(task.fps) or task.fps <= 0
                or type(task.start_frame) is not int or type(task.end_frame) is not int
                or task.start_frame < 0 or task.end_frame <= task.start_frame
            ):
                raise ValueError("Boundary reuse requires a valid source range")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            outcome = BoundaryEmbeddingOutcome.from_dict({
                "clip_id": task.clip_id, "status": "succeeded",
                "first": value["first_frame_embedding"], "last": value["last_frame_embedding"],
                "model": value["embedding_model"],
            })
            identity = boundary_embedding_identity(snapshot, fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(replace(outcome, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(BoundaryEmbeddingOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(BoundaryEmbeddingOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_gaze(tasks: "tuple[GazeTask, ...]", *, cancel_event: Event | None = None) -> "tuple[GazeOutcome, ...]":
    """Accept complete finite gaze observations without estimating missing angles."""
    from core.analysis_records import AnalysisSnapshot
    from core.analysis_model_identity import gaze_runtime
    from core.jobs.media import FingerprintCancelled
    from core.operations.gaze import GazeOptions, GazeOutcome, gaze_identity

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = gaze_runtime()
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if (
                task.analysis_json is None or task.source_path is None
                or isinstance(task.fps, bool) or not isfinite(task.fps) or task.fps <= 0
                or type(task.start_frame) is not int or type(task.end_frame) is not int
                or task.start_frame < 0 or task.end_frame <= task.start_frame
            ):
                raise ValueError("Gaze reuse requires a valid source range")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            # Legacy all-None fields cannot establish that no gaze was found.
            outcome = GazeOutcome.from_result(task.clip_id, value)
            identity = gaze_identity(snapshot, GazeOptions(), fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(replace(outcome, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(GazeOutcome(task.clip_id, "unprocessed", code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(GazeOutcome(task.clip_id, "failed", message=str(exc)))
    return tuple(outcomes)


def accept_legacy_shots(tasks: "tuple[ShotTypeTask, ...]", options: "ShotTypeOptions", *, cancel_event: Event | None = None) -> "tuple[ShotTypeOutcome, ...]":
    """Bind saved shot labels to explicitly captured backend/model settings."""
    from core.analysis_records import AnalysisSnapshot
    from core.analysis_model_identity import SHOT_TYPE_PROMPTS, shot_runtime
    from core.jobs.media import FingerprintCancelled
    from core.operations.shots import ShotTypeOutcome, shot_identity

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = shot_runtime()
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.analysis_json is None or task.thumbnail_path is None or not task.thumbnail_path.is_file():
                raise ValueError("Shot reuse requires a readable thumbnail")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            label = value["shot_type"]
            if not isinstance(label, str) or label not in SHOT_TYPE_PROMPTS:
                raise ValueError("A known legacy shot label is required")
            identity = shot_identity(snapshot, options, fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(ShotTypeOutcome(task.clip_id, "skipped", shot_type=label, code="legacy_accepted", target_type=task.target_type, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(ShotTypeOutcome(task.clip_id, "unprocessed", code="cancelled", target_type=task.target_type))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(ShotTypeOutcome(task.clip_id, "failed", message=str(exc), target_type=task.target_type))
    return tuple(outcomes)


def accept_legacy_ocr(tasks: "tuple[OcrTask, ...]", options: "OcrOptions", *, cancel_event: Event | None = None) -> "tuple[OcrOutcome, ...]":
    """Accept valid saved OCR observations, including explicitly stored empties."""
    from core.analysis_records import AnalysisSnapshot
    from core.analysis_model_identity import ocr_runtime
    from core.jobs.media import FingerprintCancelled
    from core.operations.ocr import OcrOutcome, OcrText, ocr_identity

    fingerprints = AnalysisFingerprints(cancel_event)
    runtime = ocr_runtime()
    outcomes = []
    for task in tasks:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise FingerprintCancelled()
            if task.analysis_json is None or task.path is None:
                raise ValueError("OCR reuse requires readable media")
            if task.target_type == "clip" and (
                isinstance(task.fps, bool) or not isfinite(task.fps) or task.fps <= 0
                or type(task.start_frame) is not int or type(task.end_frame) is not int
                or task.start_frame < 0 or task.end_frame <= task.start_frame
            ):
                raise ValueError("OCR reuse requires a valid source range")
            snapshot = AnalysisSnapshot.from_json(task.analysis_json)
            value = json.loads(snapshot.value_json)
            raw = value["extracted_texts"]
            if not isinstance(raw, list):
                raise ValueError("No legacy OCR result is available")
            texts = tuple(OcrText(**item) for item in raw)
            if task.target_type == "clip" and any(not task.start_frame <= text.frame_number < task.end_frame for text in texts):
                raise ValueError("Legacy OCR observation is outside the clip range")
            identity = ocr_identity(snapshot, options, fingerprints, runtime)
            record_json = _accept(json.dumps(snapshot.record.to_dict()) if snapshot.record else None, value, identity, snapshot.inputs)
            outcomes.append(OcrOutcome(task.clip_id, "succeeded", target_type=task.target_type, texts=texts, record_json=record_json))
        except FingerprintCancelled:
            outcomes.append(OcrOutcome(task.clip_id, "unprocessed", target_type=task.target_type, code="cancelled"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            outcomes.append(OcrOutcome(task.clip_id, "failed", target_type=task.target_type, message=str(exc)))
    return tuple(outcomes)
