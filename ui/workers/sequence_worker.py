"""Background worker for sequence generation.

Runs generate_sequence() (including auto-compute for brightness, volume,
embeddings) in a background thread so the UI stays responsive.
"""

import logging
from typing import Any, List, Optional, Tuple

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
        parent=None,
    ):
        super().__init__(parent)
        self._algorithm = algorithm
        self._clips = clips
        self._direction = direction

    def run(self):
        self._log_start()
        try:
            from core.remix import generate_sequence

            self.progress_message.emit(f"Computing {self._algorithm} sequence...")

            sorted_clips = generate_sequence(
                algorithm=self._algorithm,
                clips=self._clips,
                clip_count=len(self._clips),
                direction=self._direction,
            )

            if not self.is_cancelled():
                self.sequence_ready.emit(sorted_clips)
        except Exception as e:
            if not self.is_cancelled():
                logger.error(f"Sequence generation failed: {e}")
                self.error.emit(str(e))
        self._log_complete()
