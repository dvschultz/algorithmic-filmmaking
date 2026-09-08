"""Recover recorded GUI transcripts while retaining shared parallel inference."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.commits import StaleJobResult
from core.analysis_records import AnalysisFingerprints
from core.operations.transcription_records import (
    transcription_parameters,
    transcription_runtime,
)
from core.jobs.media import FingerprintCancelled
from core.operations.transcription import (
    TranscriptionOptions,
    TranscriptionOutcome,
    TranscriptionTask,
    run_transcription,
    resolve_transcription_options,
)
from core.transcription_models import TranscriptSegment


class GuiTranscriptionCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: TranscriptionOptions,
        previous_transcripts: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        options = resolve_transcription_options(options)
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_transcribe",
            arguments=transcription_parameters(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.transient_outcomes: dict[str, dict] = {}
        self.previous_transcripts_json = json.dumps(
            previous_transcripts, sort_keys=True, allow_nan=False
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        if data.get("runtime") != transcription_runtime(self.options):
            raise StaleJobResult("Transcription runtime changed")

    def run(
        self,
        tasks: tuple[TranscriptionTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[TranscriptionOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[TranscriptionOutcome, ...]:
        if not tasks:
            return ()
        previous = json.loads(self.previous_transcripts_json)
        outcomes: dict[str, TranscriptionOutcome] = {}
        pending = []
        requests = {}
        progress(0, len(tasks))

        def publish(outcome: TranscriptionOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            deliver(deepcopy(outcome))
            progress(len(outcomes), len(tasks))

        try:
            self.start(cancel, allow_missing_receipts=True)
            for task in tasks:
                if cancel.is_set():
                    break
                data = asdict(task)
                data["source_path"] = (
                    str(task.source_path) if task.source_path else None
                )
                data["previous_transcript"] = previous[task.clip_id]
                data["runtime"] = transcription_runtime(self.options)
                request, payload = self.prepare(task.clip_id, data, task.source_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(
                        TranscriptionOutcome(
                            **{
                                **payload,
                                "segments": tuple(
                                    TranscriptSegment.from_dict(s)
                                    for s in payload["segments"]
                                ),
                            }
                        )
                    )

            if pending and not cancel.is_set():
                for task in pending:
                    self.validate_media(requests[task.clip_id])

                def record(outcome: TranscriptionOutcome) -> None:
                    if outcome.status == "succeeded":
                        self.record(requests[outcome.clip_id], outcome)
                    elif outcome.can_apply:
                        self.transient_outcomes[outcome.clip_id] = asdict(outcome)
                    publish(outcome)

                def prepare_current() -> bool:
                    ready = prepare()
                    if ready:
                        for task in pending:
                            self.validate_media(requests[task.clip_id])
                    return ready

                computed = run_transcription(
                    tuple(pending),
                    self.options,
                    cancel_event=cancel,
                    on_outcome=record,
                    prepare=prepare_current,
                    fingerprints=AnalysisFingerprints(
                        cancel, media_fingerprints=self.fingerprints
                    ),
                )
                for outcome in computed:
                    outcomes.setdefault(outcome.clip_id, outcome)
        except FingerprintCancelled:
            cancel.set()
        finally:
            if hasattr(self, "store"):
                self.store.close()
        return tuple(
            outcomes.get(
                task.clip_id,
                TranscriptionOutcome(
                    task.clip_id,
                    "unprocessed",
                    code="cancelled" if cancel.is_set() else "batch_aborted",
                ),
            )
            for task in tasks
        )
