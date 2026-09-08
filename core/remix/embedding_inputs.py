"""Embedding prerequisites for private sequencing snapshots."""

import logging
import json
from dataclasses import replace
from threading import Event

from models.clip import Clip, Source
from core.operations.embeddings import (
    EmbeddingOptions,
    embedding_task,
    embedding_identity,
    reusable_embedding,
    run_embeddings,
)
from core.analysis_records import AnalysisFingerprints
from core.analysis_model_identity import embedding_runtime
from models.analysis_record import AnalysisRecord

logger = logging.getLogger(__name__)


def populate_boundary_embeddings(
    clips: list[tuple[Clip, Source]], *, cancel_event: Event | None = None
) -> None:
    """Verify boundary prerequisites on private sequencing occurrences."""
    from core.operations.boundary_embeddings import (
        boundary_embedding_task,
        boundary_embedding_identity,
        run_boundary_embeddings,
        validate_boundary_model,
    )
    from core.analysis_records import AnalysisSnapshot
    from core.analysis_model_identity import boundary_embedding_runtime, DINOV2_TAG
    from core.jobs.media import FingerprintCancelled

    cancel = cancel_event or Event()
    if cancel.is_set():
        return
    for clip, _ in clips:
        validate_boundary_model(clip, DINOV2_TAG)
    tasks = tuple(
        replace(boundary_embedding_task(clip, source), clip_id=str(i))
        for i, (clip, source) in enumerate(clips)
    )
    fingerprints, runtime = AnalysisFingerprints(cancel), boundary_embedding_runtime()
    needs_inference = False
    for task in tasks:
        assert task.analysis_json is not None
        snapshot = AnalysisSnapshot.from_json(task.analysis_json)
        if task.source_path is None or not snapshot.inputs.unchanged():
            continue
        try:
            identity = boundary_embedding_identity(snapshot, fingerprints, runtime)
        except FingerprintCancelled:
            return
        needs_inference |= snapshot.reusable_record(identity) is None
    if needs_inference:
        from core.feature_registry import check_feature

        available, missing = check_feature("embeddings")
        if not available:
            raise RuntimeError(
                "DINOv2 boundary embeddings require torch and transformers. "
                f"Missing: {', '.join(missing)}. Run embedding analysis first or install dependencies via Settings."
            )
    outcomes = run_boundary_embeddings(
        tasks, cancel_event=cancel, fingerprints=fingerprints, runtime=runtime
    )
    for (clip, _), outcome in zip(clips, outcomes):
        if cancel.is_set():
            return
        if outcome.has_result and outcome.record_json is not None:
            clip.first_frame_embedding = list(outcome.first)
            clip.last_frame_embedding = list(outcome.last)
            clip.embedding_model = outcome.model
            clip.analysis_records["boundary_embeddings"] = AnalysisRecord.from_dict(
                json.loads(outcome.record_json)
            )
        else:
            clip.first_frame_embedding = None
            clip.last_frame_embedding = None
            if outcome.record_json is not None:
                clip.analysis_records["boundary_embeddings"] = AnalysisRecord.from_dict(
                    json.loads(outcome.record_json)
                )
            logger.warning(
                "Boundary prerequisite failed for %s: %s",
                clip.id,
                outcome.message or outcome.code,
            )


def populate_embeddings(
    clips: list[tuple[Clip, Source]],
    *,
    cancel_event: Event | None = None,
    require_all: bool = False,
) -> None:
    """Populate detached clips only; callers must snapshot owner models first."""
    cancel = cancel_event or Event()
    if cancel.is_set():
        return
    # Index delivery IDs support repeated occurrences; semantic identities retain
    # the actual clip binding and never include this temporary occurrence ID.
    tasks = tuple(
        replace(embedding_task(clip, source), clip_id=str(i))
        for i, (clip, source) in enumerate(clips)
    )
    fingerprints, runtime = AnalysisFingerprints(cancel), embedding_runtime()
    needs_inference = False
    for task in tasks:
        if (
            task.thumbnail_path is None
            or not task.thumbnail_path.is_file()
            or task.inputs is None
            or not task.inputs.unchanged()
        ):
            continue
        from core.jobs.media import FingerprintCancelled

        try:
            identity = embedding_identity(task, fingerprints, runtime)
        except FingerprintCancelled:
            return
        needs_inference |= reusable_embedding(task, identity) is None
    if needs_inference:
        from core.feature_registry import check_feature

        available, missing = check_feature("embeddings")
        if not available:
            raise RuntimeError(
                "DINOv2 embeddings require torch and transformers. "
                f"Missing: {', '.join(missing)}. "
                "Run embedding analysis first or install dependencies via Settings."
            )
    outcomes = run_embeddings(
        tasks,
        EmbeddingOptions(),
        cancel_event=cancel,
        fingerprints=fingerprints,
        runtime=runtime,
    )
    for (clip, _), outcome in zip(clips, outcomes):
        if cancel.is_set():
            return
        if (
            outcome.status in ("succeeded", "skipped")
            and outcome.record_json is not None
        ):
            clip.embedding = list(outcome.vector)
            clip.embedding_model = outcome.model
            clip.analysis_records["embeddings"] = AnalysisRecord.from_dict(
                json.loads(outcome.record_json)
            )
        else:
            # A stale projection must not enter a recipe if recomputation fails.
            clip.embedding = None
            logger.warning(
                "Embedding prerequisite failed for %s: %s",
                clip.id,
                outcome.message or outcome.code,
            )
    if require_all and not cancel.is_set():
        missing_count = sum(clip.embedding is None for clip, _ in clips)
        if missing_count:
            raise RuntimeError(
                f"Missing DINOv2 embeddings for {missing_count} clips. "
                "Run embedding analysis first or ensure thumbnails exist before generating Staccato."
            )
