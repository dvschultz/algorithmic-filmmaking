"""Explicit legacy-analysis decisions shared by headless entry points."""

from typing import TYPE_CHECKING, cast
from threading import Event

if TYPE_CHECKING:
    from core.operations.ocr import OcrOptions
    from core.operations.cinematography import CinematographyOptions
    from core.operations.description import DescriptionOptions
    from core.project import Project
    from core.operations.shots import ShotTypeOptions


def accept_legacy_analysis(project: "Project", operation: str, clip_ids: list[str] | None = None, *, cancel_event: Event | None = None, shot_options: "ShotTypeOptions | None" = None, ocr_options: "OcrOptions | None" = None, description_options: "DescriptionOptions | None" = None, cinematography_options: "CinematographyOptions | None" = None) -> dict:
    """Bind selected legacy values to current inputs, retaining unknown provenance.

    Performs media hashing; desktop callers must use the detached operation
    functions on a worker and publish with their normal owner applications.
    """
    from core.operations.colors import ColorApplication, color_request
    from core.operations.embeddings import EmbeddingApplication, embedding_task
    from core.operations.legacy_reuse import LEGACY_REUSE_OPERATIONS, accept_legacy_colors, accept_legacy_embeddings, accept_legacy_scalars, accept_legacy_visuals, accept_legacy_boundaries, accept_legacy_gaze, accept_legacy_shots, accept_legacy_ocr, accept_legacy_descriptions, accept_legacy_cinematography
    from core.operations.scalars import ScalarApplication, ScalarOperation, scalar_task
    from models.analysis_record import AnalysisRecord

    project.session.assert_owner()
    if operation not in LEGACY_REUSE_OPERATIONS:
        raise ValueError("Unsupported legacy reuse operation")
    if operation == "cinematography" and cinematography_options is None:
        from core.operations.cinematography import resolve_options as resolve_cinematography_options
        cinematography_options = resolve_cinematography_options()
    if operation == "describe" and description_options is None:
        from core.operations.description import resolve_options
        description_options = resolve_options()
    if operation == "shots" and shot_options is None:
        from core.operations.shots import ShotTypeOptions
        shot_options = ShotTypeOptions.from_settings()
    if operation == "extract_text":
        from core.operations.ocr import OcrOptions, resolve_ocr_options
        ocr_options = resolve_ocr_options(ocr_options or OcrOptions())
    ids = list(dict.fromkeys(clip_ids)) if clip_ids is not None else list(project.clips_by_id)
    result: dict = {"accepted": [], "failed": [], "unprocessed": [], "provenance": "unknown"}
    for index, cid in enumerate(ids):
        if cancel_event is not None and cancel_event.is_set():
            result["unprocessed"].extend(ids[index:])
            break
        clip = project.clips_by_id.get(cid)
        if clip is None:
            result["failed"].append({"clip_id": cid, "message": "Clip not found"})
            continue
        previous = clip.analysis_records.get(operation)
        if previous is not None and not isinstance(previous, AnalysisRecord):
            result["failed"].append({"clip_id": cid, "message": "Unknown analysis record must be preserved; recompute analysis"})
            continue
        if operation == "cinematography":
            from core.operations.cinematography import CinematographyApplication, cinematography_task

            assert cinematography_options is not None
            cinematography_input = cinematography_task(clip, project.sources_by_id.get(clip.source_id))
            cinematography_application = CinematographyApplication(project, (cinematography_input,), cinematography_options)
            cinematography_result = accept_legacy_cinematography((cinematography_input,), cinematography_options, cancel_event=cancel_event)[0]
            if cinematography_result.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif cinematography_result.has_result and cinematography_application.apply(project, cinematography_result):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": cinematography_result.message or "Target changed"})
            continue
        if operation == "describe":
            from core.operations.description import DescriptionApplication, description_task

            assert description_options is not None
            description_input = description_task(clip, project.sources_by_id.get(clip.source_id))
            description_application = DescriptionApplication(project, (description_input,), description_options)
            description_result = accept_legacy_descriptions((description_input,), description_options, cancel_event=cancel_event)[0]
            if description_result.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif description_result.has_result and description_application.apply(project, description_result):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": description_result.message or "Target changed"})
            continue
        if operation == "extract_text":
            from core.operations.ocr import OcrApplication, ocr_task

            assert ocr_options is not None
            ocr_input = ocr_task(clip, project.sources_by_id.get(clip.source_id))
            ocr_application = OcrApplication(project, (ocr_input,), ocr_options)
            ocr_result = accept_legacy_ocr((ocr_input,), ocr_options, cancel_event=cancel_event)[0]
            if ocr_result.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif ocr_result.has_result and ocr_application.apply(project, ocr_result):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": ocr_result.message or "Target changed"})
            continue
        if operation == "shots":
            from core.operations.shots import ShotTypeApplication, shot_task

            assert shot_options is not None
            shot_input = shot_task(clip, project.sources_by_id.get(clip.source_id))
            shot_application = ShotTypeApplication(project, (shot_input,), shot_options)
            shot_result = accept_legacy_shots((shot_input,), shot_options, cancel_event=cancel_event)[0]
            if shot_result.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif shot_result.has_result and shot_application.apply(project, shot_result):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": shot_result.message or "Target changed"})
            continue
        if operation == "gaze":
            from core.operations.gaze import GazeApplication, GazeOptions, gaze_task

            gaze_input = gaze_task(clip, project.sources_by_id.get(clip.source_id))
            gaze_application = GazeApplication(project, (gaze_input,), GazeOptions())
            gaze_result = accept_legacy_gaze((gaze_input,), cancel_event=cancel_event)[0]
            if gaze_result.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif gaze_result.status == "succeeded" and gaze_application.apply(project, gaze_result):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": gaze_result.message or "Target changed"})
            continue
        if operation == "boundary_embeddings":
            from core.operations.boundary_embeddings import BoundaryEmbeddingApplication, boundary_embedding_task

            boundary_task = boundary_embedding_task(clip, project.sources_by_id.get(clip.source_id))
            boundary_application = BoundaryEmbeddingApplication(project, (boundary_task,))
            boundary = accept_legacy_boundaries((boundary_task,), cancel_event=cancel_event)[0]
            if boundary.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif boundary.status == "succeeded" and boundary_application.apply(project, boundary):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": boundary.message or "Target changed"})
            continue
        if operation in ("classify", "detect_objects"):
            from core.operations.classification import ClassificationApplication, ClassificationOptions, ClassificationOutcome, classification_task
            from core.operations.object_detection import ObjectDetectionApplication, ObjectDetectionOutcome, object_detection_task

            if operation == "classify":
                classification = classification_task(clip, project.sources_by_id.get(clip.source_id))
                class_app = ClassificationApplication(project, (classification,), ClassificationOptions())
                visual = accept_legacy_visuals((classification,), cancel_event=cancel_event)[0]
                accepted = visual.status in ("succeeded", "skipped") and class_app.apply(project, cast(ClassificationOutcome, visual))
            else:
                detection = object_detection_task(clip, project.sources_by_id.get(clip.source_id))
                object_app = ObjectDetectionApplication(project, (detection,))
                visual = accept_legacy_visuals((detection,), cancel_event=cancel_event)[0]
                accepted = visual.status in ("succeeded", "skipped") and object_app.apply(project, cast(ObjectDetectionOutcome, visual))
            if visual.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif accepted:
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": visual.message or "Target changed"})
            continue
        if operation in ("brightness", "volume"):
            task = scalar_task(clip, project.sources_by_id.get(clip.source_id), cast(ScalarOperation, operation))
            scalar_application = ScalarApplication(project, task)
            scalar = accept_legacy_scalars((task,), cancel_event=cancel_event)[0]
            if scalar.status == "unprocessed":
                result["unprocessed"].append(cid)
            elif scalar.status == "succeeded" and scalar_application.apply(project, scalar):
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": scalar.message or "Target changed"})
            continue
        if operation == "colors":
            request = color_request(project, [cid], skip_existing=False)
            application = ColorApplication(project, request)
            outcome = application.apply(accept_legacy_colors(request, cancel_event=cancel_event)).outcomes[0]
            if outcome.status == "unprocessed":
                result["unprocessed"].append(cid)
                continue
            accepted = outcome.status == "succeeded"
        else:
            tasks = (embedding_task(clip, project.sources_by_id.get(clip.source_id), skip_existing=False),)
            publication = EmbeddingApplication(project, tasks)
            embedding = accept_legacy_embeddings(tasks, cancel_event=cancel_event)[0]
            if embedding.status == "unprocessed":
                result["unprocessed"].append(cid)
                continue
            accepted = embedding.status == "succeeded" and publication.apply(project, embedding)
            if accepted:
                result["accepted"].append(cid)
            else:
                result["failed"].append({"clip_id": cid, "message": embedding.message or "Legacy reuse unavailable or target changed"})
            continue
        if accepted:
            result["accepted"].append(cid)
        else:
            result["failed"].append({"clip_id": cid, "message": outcome.message or outcome.code})
    return result
