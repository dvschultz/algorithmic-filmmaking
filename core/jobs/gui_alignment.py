"""Record GUI alignment computations before owner-thread publication."""

from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.gui_results import GuiResultJournal, GuiResultReceipt as AlignmentReceipt
from core.jobs.media import FingerprintCancelled
from core.operations.alignment import AlignmentOutcome, AlignmentTask, run_alignment
from core.transcription_models import WordTimestamp

__all__ = ["GuiAlignmentCache", "AlignmentReceipt"]


class GuiAlignmentCache(GuiResultJournal):
    """Alignment adapter retaining the existing recorded outcome identities."""

    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        force: bool,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_align_words",
            arguments={"force": force},
            media_stamps=media_stamps,
        )

    def run(
        self,
        tasks: tuple[AlignmentTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[AlignmentOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[AlignmentOutcome, ...]:
        if not tasks:
            return ()
        self.start(cancel)
        outcomes = []
        prepared = False
        progress(0, len(tasks))
        for index, task in enumerate(tasks):
            if cancel.is_set():
                break
            try:
                data = asdict(task)
                data["target"]["source_path"] = (
                    str(task.target.source_path) if task.target.source_path else None
                )
                request, payload = self.prepare(
                    task.clip_id, data, task.target.source_path
                )
                if payload is None:
                    if not prepared:
                        if not prepare():
                            cancel.set()
                            break
                        prepared = True
                    self.validate_media(request)
                    outcome = run_alignment((task,), cancel_event=cancel)[0]
                    if cancel.is_set():
                        break
                    if outcome.status != "succeeded":
                        outcomes.append(outcome)
                        progress(index + 1, len(tasks))
                        continue
                    payload = self.record(request, outcome)
                if cancel.is_set():
                    break
                outcome = AlignmentOutcome(
                    **{
                        **payload,
                        "words": tuple(
                            WordTimestamp.from_dict(w) for w in payload["words"]
                        ),
                    }
                )
                outcomes.append(outcome)
                deliver(deepcopy(outcome))
                progress(index + 1, len(tasks))
            except FingerprintCancelled:
                break
        outcomes.extend(
            AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
            for task in tasks[len(outcomes) :]
        )
        return tuple(outcomes)
