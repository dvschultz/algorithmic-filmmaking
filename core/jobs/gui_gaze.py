"""Record GUI gaze before publication and explicit project saves."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.gaze import _runtime, _task_data
from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json
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
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        if _runtime() != data["runtime"]:
            raise StaleJobResult("Gaze source media or runtime changed")

    def _completed_empty(self, task: GazeTask, previous: dict) -> bool:
        """Recognize saved empty observations without treating missing data as done."""
        if not self.skip_existing or any(
            value is not None for value in previous.values()
        ):
            return False
        for row, identity in self.empty_results.get(task.clip_id, []):
            inputs = identity["inputs"]
            data = inputs["task"]
            if (
                identity["version"] != 1
                or identity["project_path"] != str(self.path)
                or identity["arguments"] != asdict(self.options)
                or inputs["project_id"] != self.project_id
                or inputs["source_id"] != task.source_id
                or data["runtime"] != self.runtime
                or any(data[key] != value for key, value in _task_data(task).items())
                or inputs["media"] != self.fingerprints.get(task.source_path)
            ):
                continue
            request = GuiResultRequest(
                ResultSpec(self.path, row["spec_json"]),
                task.clip_id,
                task.source_path,
                canonical_json({"media": inputs["media"]}),
            )
            GazeOutcome.from_dict(self._accept(request, row))
            return True
        return False

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
            self.start(cancel)
            self.empty_results: dict[str, list[tuple[dict, dict]]] = {}
            if self.skip_existing:
                for result_id in self.receipts:
                    row = self.store.get_result(result_id)
                    identity = json.loads(row["spec_json"])
                    if identity["kind"] != self.kind:
                        continue
                    payload = json.loads(row["payload_json"])
                    if payload.get("code") == "no_gaze_detected":
                        self.empty_results.setdefault(identity["target_id"], []).append(
                            (row, identity)
                        )
            for task in tasks:
                if cancel.is_set():
                    break
                if task.skip:
                    publish(
                        GazeOutcome(task.clip_id, "skipped", code="already_populated")
                    )
                    continue
                if task.source_path is None or not task.source_path.is_file():
                    publish(
                        GazeOutcome(task.clip_id, "failed", code="source_file_missing")
                    )
                    continue
                if self._completed_empty(task, previous[task.clip_id]):
                    publish(
                        GazeOutcome(task.clip_id, "skipped", code="already_populated")
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
                        publish(outcome)

                    computed = run_gaze(
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
                GazeOutcome(task.clip_id, "unprocessed", code="cancelled"),
            )
            for task in tasks
        )
