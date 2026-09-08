"""Record GUI object_detection before publication and explicit project saves."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.object_detection import _runtime, _task_data
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.analysis_records import AnalysisFingerprints
from core.operations.object_detection import (
    ObjectDetectionOptions,
    ObjectDetectionOutcome,
    ObjectDetectionTask,
    run_object_detection,
)


class GuiObjectDetectionCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: ObjectDetectionOptions,
        previous_results: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_object_detection",
            arguments=asdict(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.runtime = _runtime()
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )
        self.transient_outcomes: dict[str, dict] = {}

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        source_path = data["previous_object_detection"]["source_path"]
        source = Path(source_path) if source_path else None
        if (
            self.fingerprints.get(source) != data["source_media"]
            or _runtime() != data["runtime"]
        ):
            raise StaleJobResult("ObjectDetection source media or runtime changed")

    def run(
        self,
        tasks: tuple[ObjectDetectionTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[ObjectDetectionOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[ObjectDetectionOutcome, ...]:
        if not tasks:
            return ()
        previous = json.loads(self.previous_json)
        outcomes: dict[str, ObjectDetectionOutcome] = {}
        pending = []
        requests = {}

        def publish(outcome: ObjectDetectionOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks))

        try:
            self.start(cancel, allow_missing_receipts=True)
            for task in tasks:
                if cancel.is_set():
                    break
                source_path = previous[task.clip_id]["source_path"]
                source = Path(source_path) if source_path else None
                if source is not None and media_stamp(source) != self.media_stamps.get(
                    source
                ):
                    raise StaleJobResult("ObjectDetection source changed while queued")
                data = {
                    **_task_data(task),
                    "previous_object_detection": previous[task.clip_id],
                    "source_media": self.fingerprints.get(source),
                    "runtime": self.runtime,
                }
                request, payload = self.prepare(task.clip_id, data, task.thumbnail_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(ObjectDetectionOutcome.from_dict(payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: ObjectDetectionOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        elif outcome.can_apply:
                            self.transient_outcomes[outcome.clip_id] = asdict(outcome)
                        publish(outcome)

                    computed = run_object_detection(
                        tuple(pending),
                        self.options,
                        cancel_event=cancel,
                        on_outcome=record,
                        fingerprints=AnalysisFingerprints(cancel, media_fingerprints=self.fingerprints),
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
                ObjectDetectionOutcome(task.clip_id, "unprocessed", code="cancelled"),
            )
            for task in tasks
        )
