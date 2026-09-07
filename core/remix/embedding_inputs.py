"""Embedding prerequisites for private sequencing snapshots."""

import logging
from threading import Event

from models.clip import Clip, Source
from core.operations.embeddings import EmbeddingOptions, EmbeddingTask, run_embeddings

logger = logging.getLogger(__name__)


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
    pending = [
        clip for clip, _ in clips if clip.embedding is None and clip.thumbnail_path
    ]
    if pending:
        from core.feature_registry import check_feature
        from core.jobs.media import media_stamp

        available, missing = check_feature("embeddings")
        if not available:
            raise RuntimeError(
                "DINOv2 embeddings require torch and transformers. "
                f"Missing: {', '.join(missing)}. "
                "Run embedding analysis first or install dependencies via Settings."
            )
        # Index identity also supports repeated occurrences of a clip in a recipe.
        tasks = tuple(
            EmbeddingTask(str(i), clip.thumbnail_path) for i, clip in enumerate(pending)
        )
        stamps = [
            media_stamp(task.thumbnail_path) if task.thumbnail_path else None
            for task in tasks
        ]
        outcomes = run_embeddings(tasks, EmbeddingOptions(), cancel_event=cancel)
        for clip, task, stamp, outcome in zip(pending, tasks, stamps, outcomes):
            if cancel.is_set():
                return
            if (
                outcome.status == "succeeded"
                and stamp is not None
                and task.thumbnail_path is not None
                and media_stamp(task.thumbnail_path) == stamp
            ):
                clip.embedding = list(outcome.vector)
                clip.embedding_model = outcome.model
            elif outcome.status != "succeeded":
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
