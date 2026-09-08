"""Recover recorded GUI transcripts while retaining shared parallel inference."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.gui_results import GuiResultJournal
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
            arguments=asdict(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.previous_transcripts_json = json.dumps(
            previous_transcripts, sort_keys=True, allow_nan=False
        )

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
        self.start(cancel)
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
            for task in tasks:
                if cancel.is_set():
                    break
                data = asdict(task)
                data["source_path"] = (
                    str(task.source_path) if task.source_path else None
                )
                data["previous_transcript"] = previous[task.clip_id]
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
                if prepare():
                    for task in pending:
                        self.validate_media(requests[task.clip_id])

                    def record(outcome: TranscriptionOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.record(requests[outcome.clip_id], outcome)
                        publish(outcome)

                    computed = run_transcription(
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
                TranscriptionOutcome(
                    task.clip_id,
                    "unprocessed",
                    code="cancelled" if cancel.is_set() else "batch_aborted",
                ),
            )
            for task in tasks
        )
