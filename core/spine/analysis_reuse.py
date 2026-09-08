"""Explicit legacy-analysis decisions shared by headless entry points."""

from typing import TYPE_CHECKING
from threading import Event

if TYPE_CHECKING:
    from core.project import Project


def accept_legacy_analysis(project: "Project", operation: str, clip_ids: list[str] | None = None, *, cancel_event: Event | None = None) -> dict:
    """Bind selected legacy values to current inputs, retaining unknown provenance.

    Performs media hashing; desktop callers must use the detached operation
    functions on a worker and publish with their normal owner applications.
    """
    from core.operations.colors import ColorApplication, color_request
    from core.operations.embeddings import EmbeddingApplication, embedding_task
    from core.operations.legacy_reuse import accept_legacy_colors, accept_legacy_embeddings
    from models.analysis_record import AnalysisRecord

    project.session.assert_owner()
    if operation not in ("colors", "embeddings"):
        raise ValueError("Explicit legacy reuse currently supports colors and embeddings")
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
