"""Record GUI gaze before publication and explicit project saves."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.gaze import _runtime, _task_data
from core.jobs.commits import StaleJobResult
from core.analysis_records import AnalysisFingerprints
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled
from core.operations.gaze import (
    GazeOptions,
    GazeOutcome,
    GazeTask,
    run_gaze,
)


class GuiGazeCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: GazeOptions,
        previous_results: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_gaze",
            arguments=asdict(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.skip_existing = skip_existing
        self.runtime = _runtime()
        self.transient_outcomes: dict[str, dict] = {}
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        if _runtime() != data["runtime"]:
            raise StaleJobResult("Gaze source media or runtime changed")

    def run(
        self,
        tasks: tuple[GazeTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[GazeOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[GazeOutcome, ...]:
        if not tasks:
            return ()
        previous = json.loads(self.previous_json)
        outcomes: dict[str, GazeOutcome] = {}
        pending = []
        requests = {}

        def publish(outcome: GazeOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks))

        try:
            self.start(cancel, allow_missing_receipts=True)
            for task in tasks:
                if cancel.is_set():
                    break
                if task.skip and task.analysis_json is None:
                    publish(
                        GazeOutcome(task.clip_id, "skipped", code="already_populated")
                    )
                    continue
                if task.source_path is None or not task.source_path.is_file():
                    publish(
                        GazeOutcome(task.clip_id, "failed", code="source_file_missing")
                    )
                    continue
                data = {
                    **_task_data(task),
                    "previous_gaze": previous[task.clip_id],
                    "runtime": self.runtime,
                }
                request, payload = self.prepare(task.clip_id, data, task.source_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(GazeOutcome.from_dict(payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: GazeOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        elif outcome.can_apply:
                            self.transient_outcomes[outcome.clip_id] = asdict(outcome)
                        publish(outcome)

                    computed = run_gaze(
                        tuple(pending),
                        self.options,
                        cancel_event=cancel,
                        on_outcome=record,
                        fingerprints=AnalysisFingerprints(
                            cancel, media_fingerprints=self.fingerprints
                        ),
                        runtime=self.runtime,
                    )
                    for outcome in computed:
                        outcomes.setdefault(outcome.clip_id, outcome)
                else:
                    cancel.set()
        except FingerprintCancelled:
            cancel.set()
        finally:
            if hasattr(self, "store"):
                self.store.close()
        return tuple(
            outcomes.get(
                task.clip_id,
                GazeOutcome(task.clip_id, "unprocessed", code="cancelled"),
            )
            for task in tasks
        )
