"""Shared computation journal for saved GUI projects with explicit saves."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Any

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json
from core.jobs.media import MediaFingerprints, media_stamp
from core.jobs.store import JobStore


@dataclass(frozen=True)
class GuiResultReceipt:
    result_id: str
    digest: str
    payload_json: str

    def matches(self, outcome: Any) -> bool:
        return bool(
            json.loads(self.payload_json) == json.loads(canonical_json(asdict(outcome)))
        )


@dataclass(frozen=True)
class GuiResultRequest:
    spec: ResultSpec
    clip_id: str
    media_path: Path | None
    media_json: str


class GuiResultJournal:
    def __init__(
        self,
        path: Path,
        project_id: str,
        source_ids: dict[str, str],
        receipts: dict[str, str],
        *,
        kind: str,
        arguments: dict,
        media_stamps: dict[Path, tuple[int, ...] | None],
        target_id_field: str = "clip_id",
    ) -> None:
        self.path = path.expanduser().resolve()
        self.project_id = project_id
        self.source_ids = dict(source_ids)
        self.receipts = dict(receipts)
        self.kind = kind
        self.arguments_json = canonical_json(arguments)
        self.media_stamps = dict(media_stamps)
        self.results: dict[str, GuiResultReceipt] = {}
        self.target_id_field = target_id_field

    def start(self, cancel: Event) -> None:
        from core.settings import load_settings

        self.store = JobStore(load_settings().cache_dir / "jobs.db")
        self.fingerprints = MediaFingerprints(cancel)
        self.generations: dict[str, int] = {}
        for result_id, receipt_digest in self.receipts.items():
            row = self.store.get_result(result_id)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            if sha256(row["spec_json"].encode()).hexdigest() != result_id:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != self.kind:
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != receipt_digest
                or row["payload_digest"] != receipt_digest
            ):
                raise StaleJobResult("Committed analysis payload is corrupt")
            cid = identity["target_id"]
            self.generations[cid] = self.generations.get(cid, 0) + 1

    def prepare(
        self, clip_id: str, data: dict, media_path: Path | None
    ) -> tuple[GuiResultRequest, dict | None]:
        if media_path is not None and media_stamp(media_path) != self.media_stamps.get(
            media_path
        ):
            raise StaleJobResult("Analysis media changed while queued")
        media = self.fingerprints.get(media_path)
        spec = ResultSpec.build(
            self.path,
            kind=self.kind,
            version=1,
            target_id=clip_id,
            arguments=json.loads(self.arguments_json),
            inputs={
                "project_id": self.project_id,
                "source_id": self.source_ids[clip_id],
                "task": data,
                "media": media,
                "generation": self.generations.get(clip_id, 0),
            },
        )
        request = GuiResultRequest(
            spec, clip_id, media_path, canonical_json({"media": media})
        )
        row = self.store.get_result(spec.result_id)
        return request, self._accept(request, row) if row is not None else None

    def validate_media(self, request: GuiResultRequest) -> None:
        if (
            canonical_json({"media": self.fingerprints.get(request.media_path)})
            != request.media_json
        ):
            raise StaleJobResult(
                "Analysis media changed during computation or before delivery"
            )

    def record(self, request: GuiResultRequest, outcome: Any) -> dict:
        self.validate_media(request)
        payload = asdict(outcome)
        if (
            payload[self.target_id_field] != request.clip_id
            or payload["status"] != "succeeded"
        ):
            raise StaleJobResult("Analysis result does not match a successful target")
        payload_json = canonical_json(payload)
        row = self.store.record_result(
            request.spec.result_id,
            request.spec.identity_json,
            payload_json,
            sha256(payload_json.encode()).hexdigest(),
        )
        return self._accept(request, row)

    def _accept(self, request: GuiResultRequest, row: dict) -> dict:
        digest = sha256(row["payload_json"].encode()).hexdigest()
        if (
            row["spec_json"] != request.spec.identity_json
            or digest != row["payload_digest"]
        ):
            raise StaleJobResult("Cached analysis result is corrupt")
        self.validate_media(request)
        payload: dict = json.loads(row["payload_json"])
        if (
            payload[self.target_id_field] != request.clip_id
            or payload["status"] != "succeeded"
        ):
            raise StaleJobResult("Cached analysis result does not match its target")
        self.results[request.clip_id] = GuiResultReceipt(
            request.spec.result_id, digest, row["payload_json"]
        )
        return payload
