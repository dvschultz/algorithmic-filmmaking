"""Durable computed results and project-before-checkpoint reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

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
    if spec.project_path.resolve() != spec.project_path:
        raise StaleJobResult("Project path was retargeted")
    with project_writer(spec.project_path):
        project, mtime = load_with_mtime(spec.project_path)
        project._assert_writable()
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
        project.session.verify_file_revision()
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
            apply(project, payload)
            if not is_applied(project, payload):
                raise ValueError(
                    "Result application did not produce the expected output"
                )
            project.record_job_result(spec.result_id, digest)
            save_with_mtime_check(project, spec.project_path, mtime)
            applied = True
        store.checkpoint_result(spec.result_id, digest)
        return {"result_id": spec.result_id, "applied": applied, "payload": payload}
