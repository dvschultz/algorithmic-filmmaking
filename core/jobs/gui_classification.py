"""Record GUI classification before publication and explicit project saves."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.classification import _runtime, _task_data
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.operations.classification import (
    ClassificationOptions,
    ClassificationOutcome,
    ClassificationTask,
    run_classification,
)


class GuiClassificationCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: ClassificationOptions,
        previous_results: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_classification",
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
        source_path = data["previous_classification"]["source_path"]
        source = Path(source_path) if source_path else None
        if (
            self.fingerprints.get(source) != data["source_media"]
            or _runtime() != data["runtime"]
        ):
            raise StaleJobResult("Classification source media or runtime changed")

    def run(
        self,
        tasks: tuple[ClassificationTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[ClassificationOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[ClassificationOutcome, ...]:
        if not tasks:
            return ()
        previous = json.loads(self.previous_json)
        outcomes: dict[str, ClassificationOutcome] = {}
        pending = []
        requests = {}

        def publish(outcome: ClassificationOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks))

        try:
            self.start(cancel)
            for task in tasks:
                if cancel.is_set():
                    break
                source_path = previous[task.clip_id]["source_path"]
                source = Path(source_path) if source_path else None
                if source is not None and media_stamp(source) != self.media_stamps.get(
                    source
                ):
                    raise StaleJobResult("Classification source changed while queued")
                data = {
                    **_task_data(task),
                    "previous_classification": previous[task.clip_id],
                    "source_media": self.fingerprints.get(source),
                    "runtime": self.runtime,
                }
                request, payload = self.prepare(task.clip_id, data, task.thumbnail_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(ClassificationOutcome.from_dict(payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: ClassificationOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        publish(outcome)

                    computed = run_classification(
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
                ClassificationOutcome(task.clip_id, "unprocessed", code="cancelled"),
            )
            for task in tasks
        )
