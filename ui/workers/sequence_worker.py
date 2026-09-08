"""Background worker for sequence generation.

Runs generate_sequence() (including auto-compute for brightness, volume,
embeddings) in a background thread so the UI stays responsive.
"""

import logging
from copy import deepcopy
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project
    from core.jobs.sequence_scalars import SequenceScalarJob
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class SequenceWorker(CancellableWorker):
    """Background worker that runs generate_sequence().

    Heavy auto-compute operations (brightness, volume, CLIP embeddings)
    happen inside generate_sequence() and would otherwise block the main
    thread. This worker moves them off the UI thread.

    Signals:
        sequence_ready: Emitted with sorted (Clip, Source) list on success
        progress_message: Emitted with status text during processing
        error: Emitted with error string on failure (inherited)
    """

    sequence_ready = Signal(list)  # List[(Clip, Source)]
    progress_message = Signal(str)

    def __init__(
        self,
        algorithm: str,
        clips: List[Tuple[Any, Any]],
        direction: Optional[str] = None,
        no_color_handling: Optional[str] = None,
        parent=None,
        *,
        project: "Project | None" = None,
    ):
        super().__init__(parent)
        self._algorithm = algorithm
        self._clips = deepcopy(clips)
        self._direction = direction
        self._no_color_handling = no_color_handling
        self.prerequisite_job: "SequenceScalarJob | SequenceEmbeddingJob | None" = None
        if algorithm in ("brightness", "volume"):
            from core.jobs.sequence_scalars import SequenceScalarJob

            self.prerequisite_job = SequenceScalarJob(
                clips, operation="brightness" if algorithm == "brightness" else "volume",
                project=project,
            )
        if algorithm in ("similarity_chain", "match_cut"):
            from core.jobs.sequence_embeddings import SequenceEmbeddingJob

            self.prerequisite_job = SequenceEmbeddingJob(
                self._clips,
                mode="boundary" if algorithm == "match_cut" else "thumbnail",
                project=project,
            )

    def run(self):
        self._log_start()
        try:
            from core.remix import generate_sequence

            self.progress_message.emit(f"Computing {self._algorithm} sequence...")

            if self.prerequisite_job is not None:
                self.prerequisite_job.populate(self._clips, self._cancel_event)
                if self.is_cancelled():
                    return
                # Prerequisites were resolved once, including per-clip failures.
                # Invoke the pure sorter so missing results are not recomputed.
                if self._algorithm in ("brightness", "volume"):
                    from core.remix.scalar_inputs import sort_scalar_inputs

                    sorted_clips = sort_scalar_inputs(
                        self._clips,
                        "brightness" if self._algorithm == "brightness" else "volume",
                        direction=self._direction,
                    )
                elif self._algorithm == "match_cut":
                    from core.remix.match_cut import match_cut_chain

                    sorted_clips = match_cut_chain(self._clips)
                else:
                    from core.remix.similarity_chain import similarity_chain

                    sorted_clips = similarity_chain(self._clips)
            else:
                sorted_clips = generate_sequence(
                    algorithm=self._algorithm,
                    clips=self._clips,
                    clip_count=len(self._clips),
                    direction=self._direction,
                    no_color_handling=self._no_color_handling,
                    cancel_event=self._cancel_event,
                )

            if not self.is_cancelled():
                self.sequence_ready.emit(sorted_clips)
        except Exception as e:
            if not self.is_cancelled():
                logger.error(f"Sequence generation failed: {e}")
                self.error.emit(str(e))
        self._log_complete()
