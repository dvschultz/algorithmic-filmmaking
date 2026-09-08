"""Record GUI faces before publication and explicit project saves."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.faces import _runtime, _task_data
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled
from core.operations.faces import (
    FaceOptions,
    FaceOutcome,
    FaceTask,
    run_faces,
)


class GuiFaceCache(GuiResultJournal):
    def _accept(self, request: GuiResultRequest, row: dict) -> dict:
        payload = super()._accept(request, row)
        if "record_json" not in payload:
            # The base authenticates the original receipt first. Normalize only
            # its comparison shape; keep its original result ID and digest.
            payload = {**payload, "record_json": None}
            self.results[request.clip_id] = replace(
                self.results[request.clip_id], payload_json=json.dumps(payload, sort_keys=True)
            )
        return payload

    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: FaceOptions,
        previous_results: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_faces",
            arguments=asdict(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.runtime = _runtime()
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        if _runtime() != data["runtime"]:
            raise StaleJobResult("Face source media or runtime changed")

    def run(
        self,
        tasks: tuple[FaceTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[FaceOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[FaceOutcome, ...]:
        if not tasks:
            return ()
        previous = json.loads(self.previous_json)
        outcomes: dict[str, FaceOutcome] = {}
        pending = []
        requests = {}

        def publish(outcome: FaceOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks))

        try:
            self.start(cancel)
            for task in tasks:
                if cancel.is_set():
                    break
                if task.skip:
                    publish(
                        FaceOutcome(task.clip_id, "skipped", code="already_populated")
                    )
                    continue
                if task.source_path is None or not task.source_path.is_file():
                    publish(
                        FaceOutcome(task.clip_id, "failed", code="source_file_missing")
                    )
                    continue
                data = {
                    **_task_data(task),
                    "previous_faces": previous[task.clip_id],
                    "runtime": self.runtime,
                }
                request, payload = self.prepare(task.clip_id, data, task.source_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(FaceOutcome.from_dict(payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: FaceOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        publish(outcome)

                    computed = run_faces(
                        tuple(pending),
                        self.options,
                        cancel_event=cancel,
                        on_outcome=record,
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
                FaceOutcome(task.clip_id, "unprocessed", code="cancelled"),
            )
            for task in tasks
        )
