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
from core.analysis_records import AnalysisFingerprints, AnalysisInput
from core.operations.face_records import (
    face_target_runtime,
    face_target_matches,
    face_packages,
)
from core.operations.faces import (
    FaceOptions,
    FaceOutcome,
    FaceTask,
    run_faces,
    face_model_session,
)


class GuiFaceCache(GuiResultJournal):
    def _accept(self, request: GuiResultRequest, row: dict) -> dict:
        payload = super()._accept(request, row)
        if self.verified:
            from models.analysis_record import AnalysisRecord

            record = AnalysisRecord.from_dict(
                json.loads(payload.get("record_json") or "null")
            )
            if (
                record.identity is None
                or record.state != "succeeded"
                or record.identity.to_dict()["model"]["packages"] != face_packages()
                or not AnalysisInput.from_dict(
                    json.loads(record.input_json or "null")
                ).unchanged()
            ):
                raise StaleJobResult("Cached face record no longer matches its inputs")
        if "record_json" not in payload:
            # The base authenticates the original receipt first. Normalize only
            # its comparison shape; keep its original result ID and digest.
            payload = {**payload, "record_json": None}
            self.results[request.clip_id] = replace(
                self.results[request.clip_id],
                payload_json=json.dumps(payload, sort_keys=True),
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
        verified: bool = False,
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
        self.verified = verified
        self.runtime = face_target_runtime() if verified else _runtime()
        self.transient_outcomes: dict[str, dict] = {}
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        valid = (
            face_target_matches(data["runtime"], face_target_runtime())
            if self.verified
            else _runtime() == data["runtime"]
        )
        if not valid:
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
        if self.verified:
            return self._run_verified(tasks, cancel, prepare, deliver, progress)
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

    def _run_verified(
        self, tasks, cancel, prepare, deliver, progress
    ) -> tuple[FaceOutcome, ...]:
        outcomes = []
        try:
            self.start(cancel, allow_missing_receipts=True)
            fingerprints = AnalysisFingerprints(
                cancel, media_fingerprints=self.fingerprints
            )
            with face_model_session() as session:
                for index, task in enumerate(tasks):
                    if cancel.is_set():
                        break
                    data = {
                        **_task_data(task),
                        "runtime": self.runtime,
                        "record_version": 2,
                    }
                    request, payload = self.prepare(
                        task.clip_id, data, task.source_path
                    )
                    if payload is None:
                        self.validate_media(request)

                        def prepare_current() -> bool:
                            self.validate_media(request)
                            if not prepare():
                                return False
                            self.validate_media(request)
                            return True

                        outcome = run_faces(
                            (task,),
                            self.options,
                            cancel_event=cancel,
                            fingerprints=fingerprints,
                            model_session=session,
                            prepare=prepare_current,
                        )[0]
                        if cancel.is_set():
                            break
                        if outcome.status == "succeeded":
                            self.validate_media(request)
                            # First initialization may download a model pack.
                            self.runtime = face_target_runtime()
                            data["runtime"] = self.runtime
                            request, _ = self.prepare(
                                task.clip_id, data, task.source_path
                            )
                            payload = self.record(request, outcome)
                        elif outcome.can_apply:
                            self.transient_outcomes[task.clip_id] = asdict(outcome)
                    if cancel.is_set():
                        break
                    if payload is not None:
                        outcome = FaceOutcome.from_dict(payload)
                    outcomes.append(outcome)
                    deliver(outcome)
                    progress(index + 1, len(tasks))
        except FingerprintCancelled:
            cancel.set()
        finally:
            if hasattr(self, "store"):
                self.store.close()
        outcomes.extend(
            FaceOutcome(t.clip_id, "unprocessed", code="cancelled")
            for t in tasks[len(outcomes) :]
        )
        return tuple(outcomes)
