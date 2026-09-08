"""Record detached GUI embedding batches before owner-thread publication."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.embeddings import _runtime
from core.jobs.commits import StaleJobResult
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.analysis_records import AnalysisFingerprints
from core.operations.embeddings import (
    EmbeddingOptions,
    EmbeddingOutcome,
    EmbeddingTask,
    embedding_model_session,
    run_embeddings,
    embedding_identity,
    reusable_embedding,
)


class GuiEmbeddingCache(GuiResultJournal):
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        options: EmbeddingOptions,
        targets: dict[str, dict],
        previous_results: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        super().__init__(
            path,
            project_id,
            source_ids,
            receipts,
            kind="gui_embeddings",
            arguments=asdict(options),
            media_stamps=media_stamps,
        )
        self.options = options
        self.runtime = _runtime()
        self.targets_json = json.dumps(targets, sort_keys=True, allow_nan=False)
        self.previous_json = json.dumps(
            previous_results, sort_keys=True, allow_nan=False
        )
        self.reused_outcomes: dict[str, dict] = {}

    @property
    def transient_outcomes(self) -> dict[str, dict]:
        """Expose exact reuse outcomes to shared controller publication."""
        return self.reused_outcomes

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        source = Path(data["source_path"]) if data["source_path"] else None
        if (
            _runtime() != data["runtime"]
            or self.fingerprints.get(source) != data["source_media"]
        ):
            raise StaleJobResult("Embedding source media or runtime changed")

    def run(
        self,
        tasks: tuple[EmbeddingTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[EmbeddingOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[EmbeddingOutcome, ...]:
        if not tasks:
            return ()
        outcomes: dict[str, EmbeddingOutcome] = {}
        targets, previous = (
            json.loads(self.targets_json),
            json.loads(self.previous_json),
        )
        pending = []
        requests = {}
        failed = False

        def publish(outcome: EmbeddingOutcome) -> None:
            outcomes[outcome.clip_id] = outcome
            if not cancel.is_set():
                deliver(outcome)
                progress(len(outcomes), len(tasks))

        try:
            self.start(cancel, allow_missing_receipts=True)
            fingerprints = AnalysisFingerprints(cancel, media_fingerprints=self.fingerprints)
            for task in tasks:
                if cancel.is_set():
                    break
                if task.inputs is not None and task.skip and task.inputs.unchanged():
                    semantic = embedding_identity(task, fingerprints, self.runtime)
                    reused = reusable_embedding(task, semantic)
                    if reused is not None:
                        self.reused_outcomes[task.clip_id] = asdict(reused)
                        publish(reused)
                        continue
                if task.skip and task.inputs is None:
                    publish(
                        EmbeddingOutcome(
                            task.clip_id, "skipped", code="already_populated"
                        )
                    )
                    continue
                if task.thumbnail_path is None or not task.thumbnail_path.is_file():
                    publish(
                        EmbeddingOutcome(
                            task.clip_id, "failed", code="thumbnail_missing"
                        )
                    )
                    continue
                target = targets[task.clip_id]
                source = Path(target["source_path"]) if target["source_path"] else None
                if source is not None and media_stamp(source) != self.media_stamps.get(
                    source
                ):
                    raise StaleJobResult("Embedding source changed while queued")
                data = {
                    **target,
                    "previous_embedding": previous[task.clip_id],
                    "runtime": self.runtime,
                    "source_media": self.fingerprints.get(source),
                }
                request, payload = self.prepare(task.clip_id, data, task.thumbnail_path)
                requests[task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(EmbeddingOutcome.from_dict(payload))
            if pending and not cancel.is_set():
                if prepare():
                    with embedding_model_session() as session:
                        for start in range(0, len(pending), self.options.chunk_size):
                            if cancel.is_set():
                                break
                            chunk = tuple(
                                pending[start : start + self.options.chunk_size]
                            )
                            for task in chunk:
                                self.validate_media(requests[task.clip_id])
                            computed = run_embeddings(
                                chunk,
                                self.options,
                                cancel_event=cancel,
                                model_session=session,
                                fingerprints=fingerprints,
                                runtime=self.runtime,
                            )
                            # Journal every valid vector in the computed batch before
                            # delivering any of them to the project owner.
                            for outcome in computed:
                                if outcome.status == "succeeded":
                                    self.record(requests[outcome.clip_id], outcome)
                            for outcome in computed:
                                publish(outcome)
                            if session.failed:
                                failed = True
                                break
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
                EmbeddingOutcome(
                    task.clip_id,
                    "unprocessed",
                    code="embedding_failed" if failed else "cancelled",
                ),
            )
            for task in tasks
        )
