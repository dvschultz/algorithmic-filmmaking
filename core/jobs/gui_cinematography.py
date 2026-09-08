"""Record clip/frame GUI cinematography results before owner-thread publication."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult
from core.jobs.cinematography import _task_data
from core.analysis_records import AnalysisFingerprints
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.operations.cinematography import (
    CinematographyOptions,
    CinematographyOutcome,
    CinematographyTask,
    cinematography_parameters,
    cinematography_runtime,
    run_cinematography,
)


class GuiCinematographyCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: CinematographyOptions,
        previous_results: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_cinematography",
            arguments=cinematography_parameters(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.tasks: dict[str, CinematographyTask] = {}
        self.transient_outcomes: dict[str, dict] = {}
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        source = Path(data["source_path"]) if data["source_path"] else None
        if (
            self.fingerprints.get(source) != data["source_media"]
            or cinematography_runtime(self.tasks[request.clip_id], self.options)
            != data["runtime"]
        ):
            raise StaleJobResult("Cinematography source media or runtime changed")

    def run(
        self,
        tasks: tuple[CinematographyTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[CinematographyOutcome], None],
        progress: Callable[[int, int, str], None],
    ) -> tuple[CinematographyOutcome, ...]:
        if not tasks:
            return ()
        self.tasks = {task.clip_id: task for task in tasks}
        previous = json.loads(self.previous_json)
        outcomes: dict[str, CinematographyOutcome] = {}
        pending = []
        requests = {}

        def publish(outcome: CinematographyOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks), outcome.clip_id)

        try:
            self.start(cancel, allow_missing_receipts=True)
            for task in tasks:
                if cancel.is_set():
                    break
                if task.source_path is not None and media_stamp(
                    task.source_path
                ) != self.media_stamps.get(task.source_path):
                    raise StaleJobResult("Cinematography source changed while queued")
                data = {
                    **_task_data(task),
                    "previous_cinematography": previous[task.clip_id],
                    "source_media": self.fingerprints.get(task.source_path),
                    "runtime": cinematography_runtime(task, self.options),
                }
                request, payload = self.prepare(task.clip_id, data, task.thumbnail_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(CinematographyOutcome(**payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: CinematographyOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        elif outcome.can_apply:
                            self.transient_outcomes[outcome.clip_id] = asdict(outcome)
                        publish(outcome)

                    computed = run_cinematography(
                        tuple(pending),
                        self.options,
                        cancel_event=cancel,
                        on_outcome=record,
                        fingerprints=AnalysisFingerprints(
                            cancel, media_fingerprints=self.fingerprints
                        ),
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
                CinematographyOutcome(task.clip_id, "unprocessed", code="cancelled"),
            )
            for task in tasks
        )
