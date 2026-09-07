"""Saved-project analysis plans with durable transcription steps."""

import json
from pathlib import Path
from threading import Event

from core.jobs.commits import StaleJobResult
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.jobs.transcription import run_transcription_job, transcription_job_spec
from core.operations.analysis_plan import Progress, run_analysis_plan
from core.operations.transcription import TranscriptionOptions
from core.project import Project
from core.project_revision import ProjectRevisionConflict
from core.spine.project_io import load_with_mtime, project_writer, save_with_mtime_check


def analysis_job_spec(project: Project, *, arguments: dict) -> OperationSpec:
    inputs = {}
    if "boundary_embeddings" in (arguments.get("operations") or []):
        from core.jobs.boundary_embeddings import boundary_embedding_job_spec

        boundary = boundary_embedding_job_spec(project, arguments.get("clip_ids"), arguments={})
        inputs["boundary_embeddings"] = json.loads(boundary.inputs_json)
    if "extract_text" in (arguments.get("operations") or []):
        from core.jobs.ocr import ocr_job_spec
        from core.operations.ocr import OcrOptions

        ocr = ocr_job_spec(project, arguments.get("clip_ids"), OcrOptions(), arguments={})
        inputs["ocr"] = json.loads(ocr.inputs_json)
    if "embeddings" in (arguments.get("operations") or []):
        from core.jobs.embeddings import embedding_job_spec
        from core.operations.embeddings import EmbeddingOptions

        embedding = embedding_job_spec(project, arguments.get("clip_ids"), EmbeddingOptions(), arguments={})
        inputs["embeddings"] = json.loads(embedding.inputs_json)
    if "gaze" in (arguments.get("operations") or []):
        from core.jobs.gaze import gaze_job_spec
        from core.operations.gaze import GazeOptions

        gaze = gaze_job_spec(
            project, arguments.get("clip_ids"), GazeOptions(), arguments={},
        )
        inputs["gaze"] = json.loads(gaze.inputs_json)
    if "face_embeddings" in (arguments.get("operations") or []):
        from core.jobs.faces import face_job_spec
        from core.operations.faces import FaceOptions

        faces = face_job_spec(
            project, arguments.get("clip_ids"), FaceOptions(), arguments={},
        )
        inputs["faces"] = json.loads(faces.inputs_json)
    if "detect_objects" in (arguments.get("operations") or []):
        from core.jobs.object_detection import object_detection_job_spec
        from core.operations.object_detection import ObjectDetectionOptions

        object_detection = object_detection_job_spec(
            project, arguments.get("clip_ids"), ObjectDetectionOptions(), arguments={},
        )
        inputs["object_detection"] = json.loads(object_detection.inputs_json)
    if "classify" in (arguments.get("operations") or []):
        from core.jobs.classification import classification_job_spec
        from core.operations.classification import ClassificationOptions

        classification = classification_job_spec(
            project, arguments.get("clip_ids"), ClassificationOptions(), arguments={},
        )
        inputs["classification"] = json.loads(classification.inputs_json)
    if "cinematography" in (arguments.get("operations") or []):
        from core.jobs.cinematography import cinematography_job_spec
        from core.operations.cinematography import resolve_options as resolve_cinematography

        cinematography = cinematography_job_spec(
            project, arguments.get("clip_ids"), resolve_cinematography(), arguments={},
        )
        inputs["cinematography"] = json.loads(cinematography.inputs_json)
    if "custom_query" in (arguments.get("operations") or []):
        from core.jobs.custom_query import custom_query_job_spec
        from core.operations.custom_query import resolve_options as resolve_custom_query

        custom_query = custom_query_job_spec(
            project, arguments.get("clip_ids"), resolve_custom_query(),
            arguments={"query": arguments.get("query")},
        )
        inputs["custom_query"] = json.loads(custom_query.inputs_json)
    if "describe" in (arguments.get("operations") or []):
        from core.jobs.description import description_job_spec
        from core.operations.description import resolve_options as resolve_description

        description = description_job_spec(
            project, arguments.get("clip_ids"), resolve_description(), arguments={},
        )
        inputs["description"] = json.loads(description.inputs_json)
    if "transcribe" in (arguments.get("operations") or []):
        transcription = transcription_job_spec(
            project,
            arguments.get("clip_ids"),
            TranscriptionOptions(model="base", language=None),
            arguments={},
        )
        inputs["transcription"] = json.loads(transcription.inputs_json)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="analyze_clips",
        version=1,
        arguments=arguments,
        inputs=inputs,
        persistence="job_history",
        cancellable=True,
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


def run_analysis_job(
    store: JobStore,
    path: Path,
    operation: OperationSpec,
    progress: Progress,
    cancel: Event,
) -> dict:
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    arguments = operation.arguments
    ids = arguments.get("clip_ids")
    with project_writer(path):
        project, _ = load_with_mtime(path)
        revision = project.session.file_revision
        if operation.input_revision is not None and (
            revision is None or revision.digest != operation.input_revision
        ):
            raise ProjectRevisionConflict(path)
        # Resolve the submitted backend once; changes in runtime availability
        # must not silently retarget an already queued transcription request.
        captured = json.loads(operation.inputs_json)
        transcription = captured.get("transcription")
        description = captured.get("description")
        custom_query = captured.get("custom_query")
        cinematography = captured.get("cinematography")
        classification = captured.get("classification")
        object_detection = captured.get("object_detection")
        gaze = captured.get("gaze")
        faces = captured.get("faces")
        embeddings = captured.get("embeddings")
        ocr = captured.get("ocr")
        options = (
            TranscriptionOptions(**transcription["options"]) if transcription else None
        )
        if options is not None:
            live = transcription_job_spec(project, ids, options, arguments={})
            if json.loads(live.inputs_json) != transcription:
                raise StaleJobResult("Analysis inputs changed while the job was queued")

        def execute(op: str, report: Progress | None) -> dict:
            if op == "boundary_embeddings":
                from core.jobs.boundary_embeddings import boundary_embedding_job_spec, run_boundary_embedding_job

                boundary = captured.get("boundary_embeddings")
                if boundary is None:
                    raise StaleJobResult("Analysis job has no captured boundary embedding inputs")
                current, _ = load_with_mtime(path)
                step = boundary_embedding_job_spec(current, ids, arguments={})
                if json.loads(step.inputs_json) != boundary:
                    raise StaleJobResult("Boundary embedding inputs changed before analysis")
                return run_boundary_embedding_job(store, path, ids, report or (lambda *_: None), cancel, operation=step)
            if op == "extract_text":
                from core.jobs.ocr import ocr_job_spec, run_ocr_job
                from core.operations.ocr import OcrOptions

                if ocr is None:
                    raise StaleJobResult("Analysis job has no captured OCR options")
                current, _ = load_with_mtime(path)
                step = ocr_job_spec(current, ids, OcrOptions(**ocr["options"]), arguments={})
                if json.loads(step.inputs_json) != ocr:
                    raise StaleJobResult("OCR inputs changed before analysis")
                return run_ocr_job(store, path, ids, report or (lambda *_: None), cancel, operation=step)
            if op == "embeddings":
                from core.jobs.embeddings import embedding_job_spec, run_embedding_job
                from core.operations.embeddings import EmbeddingOptions

                if embeddings is None:
                    raise StaleJobResult("Analysis job has no captured embedding options")
                current, _ = load_with_mtime(path)
                step = embedding_job_spec(current, ids, EmbeddingOptions(**embeddings["options"]), arguments={})
                if json.loads(step.inputs_json) != embeddings:
                    raise StaleJobResult("Embedding inputs changed before analysis")
                return run_embedding_job(store, path, ids, report or (lambda *_: None), cancel, operation=step)
            if op == "gaze":
                from core.jobs.gaze import gaze_job_spec, run_gaze_job
                from core.operations.gaze import GazeOptions

                if gaze is None:
                    raise StaleJobResult("Analysis job has no captured gaze options")
                current, _ = load_with_mtime(path)
                step = gaze_job_spec(
                    current, ids, GazeOptions(**gaze["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != gaze:
                    raise StaleJobResult("Gaze inputs changed before analysis")
                return run_gaze_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "face_embeddings":
                from core.jobs.faces import face_job_spec, run_face_job
                from core.operations.faces import FaceOptions

                if faces is None:
                    raise StaleJobResult("Analysis job has no captured faces options")
                current, _ = load_with_mtime(path)
                step = face_job_spec(
                    current, ids, FaceOptions(**faces["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != faces:
                    raise StaleJobResult("Face inputs changed before analysis")
                return run_face_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "detect_objects":
                from core.jobs.object_detection import object_detection_job_spec, run_object_detection_job
                from core.operations.object_detection import ObjectDetectionOptions

                if object_detection is None:
                    raise StaleJobResult("Analysis job has no captured object_detection options")
                current, _ = load_with_mtime(path)
                step = object_detection_job_spec(
                    current, ids, ObjectDetectionOptions(**object_detection["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != object_detection:
                    raise StaleJobResult("ObjectDetection inputs changed before analysis")
                return run_object_detection_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "classify":
                from core.jobs.classification import classification_job_spec, run_classification_job
                from core.operations.classification import ClassificationOptions

                if classification is None:
                    raise StaleJobResult("Analysis job has no captured classification options")
                current, _ = load_with_mtime(path)
                step = classification_job_spec(
                    current, ids, ClassificationOptions(**classification["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != classification:
                    raise StaleJobResult("Classification inputs changed before analysis")
                return run_classification_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "cinematography":
                from core.jobs.cinematography import cinematography_job_spec, run_cinematography_job
                from core.operations.cinematography import CinematographyOptions

                if cinematography is None:
                    raise StaleJobResult("Analysis job has no captured cinematography options")
                current, _ = load_with_mtime(path)
                step = cinematography_job_spec(
                    current, ids, CinematographyOptions(**cinematography["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != cinematography:
                    raise StaleJobResult("Cinematography inputs changed before analysis")
                return run_cinematography_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "custom_query":
                from core.jobs.custom_query import custom_query_job_spec, run_custom_query_job
                from core.operations.custom_query import CustomQueryOptions

                if custom_query is None:
                    raise StaleJobResult("Analysis job has no captured custom-query options")
                current, _ = load_with_mtime(path)
                step = custom_query_job_spec(
                    current, ids, CustomQueryOptions(**custom_query["options"]),
                    arguments={"query": arguments.get("query")},
                )
                if json.loads(step.inputs_json) != custom_query:
                    raise StaleJobResult("Custom-query inputs changed before analysis")
                return run_custom_query_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "describe":
                from core.jobs.description import description_job_spec, run_description_job
                from core.operations.description import DescriptionOptions

                if description is None:
                    raise StaleJobResult("Analysis job has no captured description options")
                current, _ = load_with_mtime(path)
                step = description_job_spec(
                    current, ids, DescriptionOptions(**description["options"]), arguments={},
                )
                if json.loads(step.inputs_json) != description:
                    raise StaleJobResult("Description inputs changed before analysis")
                return run_description_job(
                    store, path, ids, report or (lambda *_: None), cancel, operation=step,
                )
            if op == "transcribe":
                assert options is not None
                current, _ = load_with_mtime(path)
                step = transcription_job_spec(current, ids, options, arguments={})
                if json.loads(step.inputs_json) != transcription:
                    raise StaleJobResult("Analysis inputs changed before transcription")
                return run_transcription_job(
                    store,
                    path,
                    ids,
                    options,
                    report or (lambda *_: None),
                    cancel,
                    operation=step,
                    skip_existing=True,
                )
            # Reload after each durable step; never save an older model over
            # its result receipts. Unmigrated steps retain their spine provider.
            current, mtime = load_with_mtime(path)
            kwargs: dict[str, object] = {"skip_existing": True}
            result = ANALYZE_CLIP_OPERATION_MAP[op](
                current,
                ids,
                progress_callback=report,
                cancel_event=cancel,
                **kwargs,
            )
            save_with_mtime_check(current, path, mtime)
            return result

        return run_analysis_plan(
            arguments.get("operations"), execute, progress=progress, cancel=cancel
        )
