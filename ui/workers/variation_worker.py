"""Background worker that recomputes a recipe as a new variation (plan U16).

The Sequence tab validates the request on the GUI thread with
``core.spine.sequences.prepare_regeneration``, snapshots the candidate clips,
and hands the algorithm half to this worker. Publishing stays on the GUI
thread (``publish_recipe``), so a cancelled or failed run publishes nothing.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

if TYPE_CHECKING:
    from core.spine.sequences import RegenerationPlan
    from models.clip import Clip, Source

logger = logging.getLogger(__name__)


class VariationWorker(CancellableWorker):
    """Run ``run_algorithm`` for a regeneration plan off the GUI thread.

    Signals:
        variation_ready(recipe, sequence_settings, notes)
        progress_message(str)
        error(str) (inherited)
    """

    variation_ready = Signal(object, dict, list)
    progress_message = Signal(str)

    def __init__(self, plan: "RegenerationPlan", candidates: list[tuple["Clip", "Source"]], parent=None) -> None:
        super().__init__(parent)
        self.plan = plan
        self._candidates = deepcopy(candidates)  # generation never mutates library clips

    def run(self) -> None:
        self._log_start()
        try:
            from core.remix.registry import registry, run_algorithm

            definition = registry.require(self.plan.algorithm)
            self.progress_message.emit(f"Regenerating {definition.key}...")
            run = run_algorithm(
                definition, self._candidates, self.plan.parameters, seed=self.plan.seed,
                cancel_event=self._cancel_event, parent_recipe_id=self.plan.parent_recipe_id,
                progress=self.progress_message.emit,
            )
            if run is None or self.is_cancelled():
                return
            if not run.recipe.realized:
                raise ValueError("Algorithm produced an empty sequence: " + "; ".join(run.proposal.notes))
            self.variation_ready.emit(run.recipe, dict(run.proposal.sequence_settings), list(run.proposal.notes))
        except Exception as exc:
            if not self.is_cancelled():
                logger.error("Variation failed: %s", exc)
                self.error.emit(str(exc))
        self._log_complete()
