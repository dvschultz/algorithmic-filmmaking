"""Single-step LLM proposal worker for the Free Association sequencer.

Each worker instance makes exactly one LLM call to propose the next clip.
The dialog spawns a fresh worker per proposal, which keeps the lifecycle
simple: create, start, emit one signal, clean up via deleteLater. This
avoids the complexity of inter-thread signaling for accept/reject loops.

No worker.wait() on cancellation — litellm.completion is a blocking HTTP
call that can take up to 120s to return, and waiting would freeze the UI.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from core.remix.free_association import propose_next_clip
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class FreeAssociationWorker(CancellableWorker):
    """Run one LLM call to propose the next clip.

    Signals:
        proposal_ready(str, str): emitted with (clip_short_id, rationale)
            when the LLM successfully returns a valid proposal.
        error(str): emitted with a human-readable message when the LLM
            call fails, returns None/malformed content, or returns a
            hallucinated/rejected clip ID.
    """

    proposal_ready = Signal(str, str)  # (clip_short_id, rationale)

    def __init__(
        self,
        current_clip_metadata: str,
        candidate_digests: list[tuple[str, str]],
        recent_rationales: list[str],
        rejected_short_ids: list[str],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._current_clip_metadata = current_clip_metadata
        self._candidate_digests = candidate_digests
        self._recent_rationales = recent_rationales
        self._rejected_short_ids = rejected_short_ids
        self._model = model
        self._temperature = temperature

    def run(self):
        """Execute the single-step proposal."""
        self._log_start()
        try:
            clip_id, rationale = propose_next_clip(
                current_clip_metadata=self._current_clip_metadata,
                candidate_digests=self._candidate_digests,
                recent_rationales=self._recent_rationales,
                rejected_short_ids=self._rejected_short_ids,
                model=self._model,
                temperature=self._temperature,
            )
        except ValueError as exc:
            # Malformed JSON, None content, hallucinated ID, rejected ID
            logger.warning("%s: ValueError from propose_next_clip: %s", self.worker_name, exc)
            if not self.is_cancelled():
                self.error.emit(str(exc))
            self._log_complete()
            return
        except Exception as exc:  # noqa: BLE001 — network / provider errors
            # Network timeout, auth failure, provider outage, etc.
            logger.exception("%s: unexpected error during LLM call", self.worker_name)
            if not self.is_cancelled():
                self.error.emit(f"LLM call failed: {exc}")
            self._log_complete()
            return

        if self.is_cancelled():
            self._log_cancelled()
            return

        self.proposal_ready.emit(clip_id, rationale)
        self._log_complete()
