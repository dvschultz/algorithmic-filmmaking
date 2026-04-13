"""Background worker for DINOv2 visual embedding extraction.

Extracts 768-dimensional embeddings from each clip's thumbnail and writes
them to ``clip.embedding`` and ``clip.embedding_model`` in place. Processes
in chunks so progress updates and cancellation checks happen between chunks
rather than only at the start and end of the whole run.

Mirrors the shape of ``ui/workers/gaze_worker.py`` — including the always-
load-then-unload model lifecycle in ``finally``. No ``_model_lock`` race
avoidance here; sequential phase ordering guarantees no concurrent embeddings
consumer is active during this worker's run.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models.clip import Clip, Source

from PySide6.QtCore import Signal, Slot

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 16


class EmbeddingAnalysisWorker(CancellableWorker):
    """Background worker for DINOv2 visual embedding extraction.

    Filters clips to those that need embeddings, processes them in chunks
    (so progress and cancellation are responsive), mutates each clip's
    ``embedding`` and ``embedding_model`` fields in place.

    Signals:
        progress: Emitted with (current, total) after each chunk completes.
        embedding_ready: Emitted with clip_id for each clip that gets
            an embedding attached.
        analysis_completed: Emitted exactly once when the worker finishes
            (success, cancellation, or error — the pipeline can always
            advance on this signal).
        error: Emitted with an error message string on chunk-level failure
            (inherited from CancellableWorker).
    """

    progress = Signal(int, int)  # current, total
    embedding_ready = Signal(str)  # clip_id
    analysis_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        sources_by_id: Optional[dict[str, "Source"]] = None,
        skip_existing: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        parent=None,
    ):
        """
        Args:
            clips: Clips to process.
            sources_by_id: Accepted for launch-site API parity with other
                workers (e.g., GazeAnalysisWorker). Never read inside this
                worker — embeddings only need each clip's thumbnail_path.
            skip_existing: If True (default), clips that already have an
                embedding are skipped.
            chunk_size: Clips per chunk. Smaller = more responsive progress
                and cancellation; larger = slightly less overhead.
            parent: Qt parent.
        """
        super().__init__(parent)
        self._clips = clips
        # sources_by_id intentionally stored but not used — see docstring
        self._sources_by_id = sources_by_id
        self._skip_existing = skip_existing
        self._chunk_size = max(1, chunk_size)

    @Slot()
    def run(self):
        """Execute embedding extraction on the filtered clip set."""
        self._log_start()

        # Filter clips: skip those already analyzed, skip those without a thumbnail.
        clips_to_process: list[tuple["Clip", Path]] = []
        for clip in self._clips:
            if self._skip_existing and clip.embedding is not None:
                continue
            thumb_path = getattr(clip, "thumbnail_path", None)
            if not thumb_path:
                logger.warning(
                    "Skipping clip %s: no thumbnail_path", clip.id
                )
                continue
            clips_to_process.append((clip, Path(thumb_path)))

        total = len(clips_to_process)
        if total == 0:
            logger.info("No clips to process for embedding extraction")
            self.analysis_completed.emit()
            self._log_complete()
            return

        logger.info("Starting embedding extraction: %d clips", total)

        # Import lazily so tests can patch without torch present
        from core.analysis.embeddings import (
            _EMBEDDING_MODEL_TAG,
            extract_clip_embeddings_batch,
            unload_model,
        )

        # Always-unload pattern (matches gaze_worker.py). Sequential phase
        # ordering guarantees no concurrent embeddings consumer is active,
        # so we don't need a TOCTOU-safe check — just unload in `finally`.
        processed = 0
        try:
            for chunk_start in range(0, total, self._chunk_size):
                if self.is_cancelled():
                    self._log_cancelled()
                    break

                chunk = clips_to_process[chunk_start : chunk_start + self._chunk_size]
                thumbnail_paths = [thumb for _clip, thumb in chunk]

                try:
                    vectors = extract_clip_embeddings_batch(thumbnail_paths)
                except Exception as exc:  # noqa: BLE001 — treat as fatal for this run
                    logger.exception("Embedding extraction failed on chunk")
                    self.error.emit(f"Embedding extraction failed: {exc}")
                    break

                # Map vectors back to clips 1:1 by index within this chunk.
                for (clip, _thumb), vec in zip(chunk, vectors):
                    clip.embedding = vec
                    clip.embedding_model = _EMBEDDING_MODEL_TAG
                    self.embedding_ready.emit(clip.id)

                processed += len(chunk)
                self.progress.emit(processed, total)
        finally:
            # Always unload when we're done — matches gaze_worker.py.
            try:
                unload_model()
            except Exception:
                logger.debug("unload_model() raised during cleanup; ignoring")
            self.analysis_completed.emit()
            self._log_complete()
