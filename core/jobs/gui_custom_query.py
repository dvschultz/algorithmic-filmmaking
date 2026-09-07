"""Journal GUI custom-query inference before owner-thread append publication."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult
from core.jobs.custom_query import _provenance
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.operations.custom_query import (
    CustomQueryOptions,
    CustomQueryOutcome,
    CustomQueryTask,
    run_custom_query,
)


class GuiCustomQueryCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: CustomQueryOptions,
        previous_queries: dict,
        targets: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_custom_query",
            arguments=asdict(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.previous_json = json.dumps(
            previous_queries, sort_keys=True, allow_nan=False
        )
        self.targets_json = json.dumps(targets, sort_keys=True, allow_nan=False)
        self.runtime = _provenance(options)

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        path = data["target"]["source_path"]
        if (
            self.fingerprints.get(Path(path) if path else None) != data["source_media"]
            or _provenance(self.options) != data["runtime"]
        ):
            raise StaleJobResult("Custom-query source media or runtime changed")

    def run(
        self,
        tasks: tuple[CustomQueryTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[CustomQueryOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[CustomQueryOutcome, ...]:
        if not tasks:
            return ()
        self.start(cancel)
        previous, targets = (
            json.loads(self.previous_json),
            json.loads(self.targets_json),
        )
        outcomes: dict[str, CustomQueryOutcome] = {}
        pending = []
        requests = {}
        progress(0, len(tasks))

        def publish(outcome: CustomQueryOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks))

        try:
            for task in tasks:
                if cancel.is_set():
                    break
                target = targets[task.clip_id]
                source = Path(target["source_path"]) if target["source_path"] else None
                if source is not None and media_stamp(source) != self.media_stamps.get(
                    source
                ):
                    raise StaleJobResult("Custom-query source changed while queued")
                data = {
                    **asdict(task),
                    "thumbnail_path": str(task.thumbnail_path)
                    if task.thumbnail_path
                    else None,
                    "target": target,
                    "previous_queries": previous[task.clip_id],
                    "source_media": self.fingerprints.get(source),
                    "runtime": self.runtime,
                }
                request, payload = self.prepare(task.clip_id, data, task.thumbnail_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(CustomQueryOutcome(**payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: CustomQueryOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        publish(outcome)

                    computed = run_custom_query(
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
        return tuple(
            outcomes.get(
                task.clip_id,
                CustomQueryOutcome(
                    task.clip_id, task.query, "unprocessed", code="cancelled"
                ),
            )
            for task in tasks
        )
