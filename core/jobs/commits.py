"""Durable computed results and project-before-checkpoint reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterator
from contextlib import contextmanager

from core.project import Project
from core.jobs.store import JobStore
from core.spine.project_io import load_with_mtime, project_writer, save_with_mtime_check


def canonical_json(value: dict) -> str:
    if not isinstance(value, dict):
        raise ValueError("Operation identities and payloads must be JSON objects")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ResultSpec:
    """Canonical immutable operation/input identity, safe to carry across retries."""

    project_path: Path
    identity_json: str

    @classmethod
    def build(
        cls,
        path: Path,
        *,
        kind: str,
        version: int,
        target_id: str,
        arguments: dict,
        inputs: dict,
    ) -> ResultSpec:
        canonical = path.expanduser().resolve()
        identity = {
            "project_path": str(canonical),
            "kind": kind,
            "version": version,
            "target_id": target_id,
            "arguments": arguments,
            "inputs": inputs,
        }
        return cls(canonical, canonical_json(identity))

    @property
    def result_id(self) -> str:
        return sha256(self.identity_json.encode()).hexdigest()


class StaleJobResult(RuntimeError):
    pass


def commit_result(
    store: JobStore,
    spec: ResultSpec,
    *,
    compute: Callable[[], dict],
    validate_input: Callable[[Project], bool],
    apply: Callable[[Project, dict], None],
    is_applied: Callable[[Project, dict], bool],
) -> dict:
    """Compute once, publish data and receipt together, then acknowledge the job.

    Only saved projects participate. A failed save discards the detached model;
    recovery reloads disk and uses its receipt to avoid applying twice.
    """
    with result_batch(store, spec.project_path, max_items=1) as batch:
        return batch.commit(
            spec,
            compute=compute,
            validate_input=validate_input,
            apply=apply,
            is_applied=is_applied,
        )


@dataclass
class _PendingResult:
    spec: ResultSpec
    payload: dict
    digest: str
    validate_input: Callable[[Project], bool]
    is_applied: Callable[[Project, dict], bool]


@contextmanager
def result_batch(
    store: JobStore, path: Path, *, max_items: int = 16
) -> Iterator["ResultBatch"]:
    """Publish bounded groups from one private model under its writer lease.

    Normal exit flushes the final partial group. Exceptional exit discards
    unsaved model changes; recorded computations remain available for retry.
    """
    if store.persistence != "job_history":
        raise ValueError("Project result commits require a durable job store")
    if type(max_items) is not int or max_items < 1:
        raise ValueError("Batch size must be a positive integer")
    if path.resolve() != path:
        raise StaleJobResult("Project path was retargeted")
    with project_writer(path):
        project, mtime = load_with_mtime(path)
        project._assert_writable()
        batch = ResultBatch(store, path, project, mtime, max_items)
        try:
            yield batch
            batch.flush()
        finally:
            batch._active = False


class ResultBatch:
    """Private staged application; construct through result_batch()."""

    def __init__(
        self,
        store: JobStore,
        path: Path,
        project: Project,
        mtime: float,
        max_items: int,
    ) -> None:
        self.store = store
        self.path = path
        self.project = project
        self.mtime = mtime
        self.max_items = max_items
        self._pending: dict[str, _PendingResult] = {}
        self._dirty = False
        self._active = True
        self._failed = False

    def _assert_active(self) -> None:
        if not self._active or self._failed:
            raise RuntimeError(
                "Result batch is closed or failed; reload before retrying"
            )
        self.project.session.assert_owner()

    def commit(
        self,
        spec: ResultSpec,
        *,
        compute: Callable[[], dict],
        validate_input: Callable[[Project], bool],
        apply: Callable[[Project, dict], None],
        is_applied: Callable[[Project, dict], bool],
    ) -> dict:
        """Record computation and stage application; flush when the group fills."""
        self._assert_active()
        if spec.project_path != self.path:
            raise ValueError("Result belongs to another project")
        project, store = self.project, self.store
        if not validate_input(project):
            raise StaleJobResult("Job inputs changed before computation")
        row = store.get_result(spec.result_id)
        if row is None:
            if spec.result_id in project.metadata.job_results:
                raise StaleJobResult(
                    "Committed result payload is missing; refusing to recompute"
                )
            payload_json = canonical_json(compute())
            digest = sha256(payload_json.encode()).hexdigest()
            row = store.record_result(
                spec.result_id, spec.identity_json, payload_json, digest
            )
        if row["spec_json"] != spec.identity_json:
            raise ValueError("Stored result identity does not match request")
        payload_json = row["payload_json"]
        digest = sha256(payload_json.encode()).hexdigest()
        if digest != row["payload_digest"]:
            raise ValueError("Stored result payload is corrupt")
        payload = json.loads(payload_json)
        if not validate_input(project):
            raise StaleJobResult("Job inputs changed during computation")
        receipt = project.metadata.job_results.get(spec.result_id)
        if receipt is not None:
            if receipt != digest or not is_applied(project, payload):
                raise StaleJobResult(
                    "Previously committed output changed; refusing stale replay"
                )
            applied = False
        else:
            if row["committed"]:
                raise StaleJobResult("Project no longer contains the committed receipt")
            try:
                apply(project, payload)
                if not is_applied(project, payload):
                    raise ValueError(
                        "Result application did not produce the expected output"
                    )
                project.record_job_result(spec.result_id, digest)
            except BaseException:
                self._failed = True
                raise
            self._dirty = True
            applied = True
        self._pending.setdefault(
            spec.result_id,
            _PendingResult(spec, payload, digest, validate_input, is_applied),
        )
        if len(self._pending) >= self.max_items:
            self.flush()
        return {
            "result_id": spec.result_id,
            "applied": applied,
            "payload": json.loads(payload_json),
        }

    def flush(self) -> None:
        """Revalidate the entire group, save once, then acknowledge all receipts."""
        self._assert_active()
        try:
            self._flush()
        except BaseException:
            self._failed = True
            raise

    def _flush(self) -> None:
        if not self._pending:
            return
        if self.path.resolve() != self.path:
            raise StaleJobResult("Project path was retargeted")
        for pending in self._pending.values():
            if (
                sha256(canonical_json(pending.payload).encode()).hexdigest()
                != pending.digest
                or self.project.metadata.job_results.get(pending.spec.result_id)
                != pending.digest
            ):
                raise StaleJobResult("Staged result payload or receipt changed")
            if not pending.validate_input(self.project) or not pending.is_applied(
                self.project, pending.payload
            ):
                raise StaleJobResult(
                    "Batch inputs or staged output changed before publication"
                )
        self.project.session.verify_file_revision()
        if self._dirty:
            save_with_mtime_check(self.project, self.path, self.mtime)
            self.mtime = self.path.stat().st_mtime
        self.store.checkpoint_results(
            [
                (pending.spec.result_id, pending.digest)
                for pending in self._pending.values()
            ]
        )
        self._pending.clear()
        self._dirty = False
