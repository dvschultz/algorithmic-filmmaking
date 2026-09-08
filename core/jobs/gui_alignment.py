"""Record GUI alignment computations before owner-thread publication."""

from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import json
from threading import Event
from typing import Callable

from core.jobs.gui_results import (
    GuiResultJournal,
    GuiResultRequest,
    GuiResultReceipt as AlignmentReceipt,
)
from core.jobs.commits import StaleJobResult
from core.analysis_records import AnalysisFingerprints
from core.operations.alignment_records import alignment_runtime
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
        self.runtime = alignment_runtime()
        self.transient_outcomes: dict[str, dict] = {}

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        expected = json.loads(request.spec.identity_json)["inputs"]["task"].get(
            "runtime"
        )
        if expected is not None:
            current = alignment_runtime()
            if expected.get("revision") is None:
                current["revision"] = None  # First model load may populate the cache.
            if current != expected:
                raise StaleJobResult("Alignment runtime changed")

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
        outcomes = []
        progress(0, len(tasks))
        try:
            self.start(cancel, allow_missing_receipts=True)
            for index, task in enumerate(tasks):
                if cancel.is_set():
                    break
                data = asdict(task)
                data["target"]["source_path"] = (
                    str(task.target.source_path) if task.target.source_path else None
                )
                if task.analysis_json is not None:
                    data["runtime"] = self.runtime
                request, payload = self.prepare(
                    task.clip_id, data, task.target.source_path
                )
                if payload is None:
                    self.validate_media(request)

                    def prepare_current() -> bool:
                        nonlocal request
                        self.validate_media(request)
                        if not prepare():
                            return False
                        # Dependency installation belongs to the existing GUI
                        # prepare flow. Journal the resulting runtime explicitly.
                        if task.analysis_json is not None:
                            self.runtime = alignment_runtime()
                            data["runtime"] = self.runtime
                            request, _ = self.prepare(
                                task.clip_id, data, task.target.source_path
                            )
                        self.validate_media(request)
                        return True

                    outcome = run_alignment(
                        (task,),
                        cancel_event=cancel,
                        prepare=prepare_current,
                        fingerprints=AnalysisFingerprints(
                            cancel, media_fingerprints=self.fingerprints
                        ),
                    )[0]
                    if cancel.is_set():
                        break
                    if outcome.status == "succeeded":
                        if task.analysis_json is not None:
                            self.validate_media(request)
                            self.runtime = alignment_runtime()
                            data["runtime"] = self.runtime
                            request, _ = self.prepare(
                                task.clip_id, data, task.target.source_path
                            )
                        payload = self.record(request, outcome)
                    elif outcome.can_apply:
                        self.transient_outcomes[task.clip_id] = asdict(outcome)
                if cancel.is_set():
                    break
                if payload is not None:
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
            cancel.set()
        finally:
            if hasattr(self, "store"):
                self.store.close()
        outcomes.extend(
            AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
            for task in tasks[len(outcomes) :]
        )
        return tuple(outcomes)
