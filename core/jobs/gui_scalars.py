"""Recover verified scalar computations before GUI publication and explicit save."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled
from core.operations.scalars import (
    FIELDS,
    ScalarOperation,
    ScalarOutcome,
    ScalarTask,
    run_scalars,
    scalar_parameters,
    scalar_runtime,
    scalar_sampling,
    scalar_value,
)
from models.analysis_record import AnalysisRecord


class GuiScalarCache(GuiResultJournal):
    """Journal successful work; authenticate records before accepting recovery."""

    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        operation: ScalarOperation,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path, project_id, source_ids, receipts,
            kind=f"gui_{operation}", arguments={}, media_stamps=media_stamps,
        )
        self.operation = operation
        self.runtime = scalar_runtime(operation)
        self.transient_outcomes: dict[str, dict] = {}

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        snapshot = AnalysisSnapshot.from_json(data["task"]["snapshot_json"])
        if (
            not snapshot.inputs.unchanged()
            or scalar_runtime(self.operation) != data["runtime"]
        ):
            raise StaleJobResult("Scalar inputs or runtime changed")

    def _accept(self, request: GuiResultRequest, row: dict) -> dict:
        payload = super()._accept(request, row)
        task = ScalarTask(**json.loads(request.spec.identity_json)["inputs"]["task"]["task"])
        snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
        outcome = ScalarOutcome(**payload)
        record = AnalysisRecord.from_dict(json.loads(outcome.record_json or "null"))
        identity = self.analysis_fingerprints.identity(
            snapshot.inputs,
            operation=task.operation,
            model=self.runtime,
            parameters=scalar_parameters(task),
            sampling=scalar_sampling(task.operation),
        )
        if (
            outcome.operation != self.operation
            or record.state != "succeeded"
            or record.provenance != "verified"
            or record.identity != identity
            or json.loads(record.input_json or "null") != snapshot.inputs.to_dict()
            or not isinstance(record.value, dict)
            or record.value != scalar_value(task.operation, record.value.get(FIELDS[task.operation]))
        ):
            self.results.pop(request.clip_id, None)
            raise StaleJobResult("Cached scalar record does not match its inputs")
        return payload

    def run(
        self,
        tasks: tuple[ScalarTask, ...],
        cancel: Event,
        deliver: Callable[[ScalarOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[ScalarOutcome, ...]:
        if any(task.operation != self.operation for task in tasks):
            raise ValueError("Scalar journal operation does not match tasks")
        if len({task.clip_id for task in tasks}) != len(tasks):
            raise ValueError("Scalar journal requires unique delivery IDs")
        outcomes: list[ScalarOutcome] = []
        try:
            self.start(cancel, allow_missing_receipts=True)
            self.analysis_fingerprints = AnalysisFingerprints(
                cancel, media_fingerprints=self.fingerprints,
            )
            for index, task in enumerate(tasks):
                if cancel.is_set():
                    break
                snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
                files = {role: path for role, path, _ in snapshot.inputs.files}
                request, payload = self.prepare(
                    task.clip_id,
                    {"task": asdict(task), "runtime": self.runtime},
                    files.get("video"),
                )
                if payload is None:
                    self.validate_media(request)
                    outcome = run_scalars(
                        (task,), cancel_event=cancel,
                        fingerprints=self.analysis_fingerprints,
                    )[0]
                    if cancel.is_set():
                        break
                    if outcome.status == "succeeded":
                        payload = self.record(request, outcome)
                    else:
                        self.transient_outcomes[task.clip_id] = asdict(outcome)
                if cancel.is_set():
                    break
                if payload is not None:
                    outcome = ScalarOutcome(**payload)
                outcomes.append(outcome)
                deliver(outcome)
                progress(index + 1, len(tasks))
        except FingerprintCancelled:
            cancel.set()
        finally:
            if hasattr(self, "store"):
                self.store.close()
        outcomes.extend(
            ScalarOutcome(task.clip_id, task.operation, "unprocessed", message="Cancelled")
            for task in tasks[len(outcomes):]
        )
        return tuple(outcomes)
