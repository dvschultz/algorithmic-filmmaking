"""Record GUI alignment computations before owner-thread publication.

The job lifecycle remains session-only. Saved projects can recover computed
outputs; their ordinary project save persists the publication receipts.
"""

from dataclasses import asdict, dataclass
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.store import JobStore
from core.operations.alignment import AlignmentOutcome, AlignmentTask, run_alignment
from core.transcription_models import WordTimestamp


@dataclass(frozen=True)
class AlignmentReceipt:
    result_id: str
    digest: str
    payload_json: str

    def matches(self, outcome: AlignmentOutcome) -> bool:
        """Ensure queued mutable signal data still represents the recorded output."""
        return bool(
            json.loads(self.payload_json) == json.loads(canonical_json(asdict(outcome)))
        )


class GuiAlignmentCache:
    """Detached saved-project identity and write-once per-target receipts."""

    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        force: bool,
        media_stamps: dict[Path, tuple[int, ...] | None],
    ) -> None:
        self.path = path.expanduser().resolve()
        self.project_id = project_id
        self.source_ids = dict(source_ids)
        self.saved_receipts = dict(receipts)
        self.force = force
        self.media_stamps = dict(media_stamps)
        self.results: dict[str, AlignmentReceipt] = {}

    def run(
        self,
        tasks: tuple[AlignmentTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[AlignmentOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[AlignmentOutcome, ...]:
        if not tasks:
            return ()
        from core.settings import load_settings

        store = JobStore(load_settings().cache_dir / "jobs.db")
        fingerprint = MediaFingerprints(cancel).get
        generations: dict[str, int] = {}
        for result_id, receipt_digest in self.saved_receipts.items():
            row = store.get_result(result_id)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "gui_align_words":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != receipt_digest
                or row["payload_digest"] != receipt_digest
            ):
                raise StaleJobResult("Committed alignment payload is corrupt")
            cid = identity["target_id"]
            generations[cid] = generations.get(cid, 0) + 1

        outcomes = []
        prepared = False
        progress(0, len(tasks))
        for index, task in enumerate(tasks):
            if cancel.is_set():
                break
            try:
                if task.target.source_path is not None and media_stamp(
                    task.target.source_path
                ) != self.media_stamps.get(task.target.source_path):
                    raise StaleJobResult("Alignment media changed while queued")
                media = fingerprint(task.target.source_path)
                data = asdict(task)
                data["target"]["source_path"] = (
                    str(task.target.source_path) if task.target.source_path else None
                )
                spec = ResultSpec.build(
                    self.path,
                    kind="gui_align_words",
                    version=1,
                    target_id=task.clip_id,
                    arguments={"force": self.force},
                    inputs={
                        "project_id": self.project_id,
                        "source_id": self.source_ids[task.clip_id],
                        "task": data,
                        "media": media,
                        "generation": generations.get(task.clip_id, 0),
                    },
                )
                row = store.get_result(spec.result_id)
                if row is None:
                    if not prepared:
                        if not prepare():
                            cancel.set()
                            break
                        prepared = True
                    outcome = run_alignment((task,), cancel_event=cancel)[0]
                    if cancel.is_set():
                        break
                    if outcome.status != "succeeded":
                        outcomes.append(outcome)
                        progress(index + 1, len(tasks))
                        continue
                    payload = canonical_json(asdict(outcome))
                    if fingerprint(task.target.source_path) != media:
                        raise StaleJobResult(
                            "Alignment media changed during computation"
                        )
                    row = store.record_result(
                        spec.result_id,
                        spec.identity_json,
                        payload,
                        sha256(payload.encode()).hexdigest(),
                    )
                digest = sha256(row["payload_json"].encode()).hexdigest()
                if (
                    row["spec_json"] != spec.identity_json
                    or digest != row["payload_digest"]
                ):
                    raise StaleJobResult("Cached alignment result is corrupt")
                if fingerprint(task.target.source_path) != media:
                    raise StaleJobResult("Alignment media changed before delivery")
                if cancel.is_set():
                    break
                payload = json.loads(row["payload_json"])
                if (
                    payload["clip_id"] != task.clip_id
                    or payload["status"] != "succeeded"
                ):
                    raise StaleJobResult(
                        "Cached alignment result does not match its target"
                    )
                outcome = AlignmentOutcome(
                    **{
                        **payload,
                        "words": tuple(
                            WordTimestamp.from_dict(w) for w in payload["words"]
                        ),
                    }
                )
                self.results[task.clip_id] = AlignmentReceipt(
                    spec.result_id, digest, row["payload_json"]
                )
                outcomes.append(outcome)
                deliver(deepcopy(outcome))
                progress(index + 1, len(tasks))
            except FingerprintCancelled:
                break
        outcomes.extend(
            AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
            for task in tasks[len(outcomes) :]
        )
        return tuple(outcomes)
