"""Phase ordering and per-clip outcomes for a combined analysis request."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.analysis_operations import OPERATIONS_BY_KEY, PHASE_ORDER
from core.operations.contracts import OutcomeStatus


@dataclass(frozen=True)
class ClipAnalysisOptions:
    """Explicit per-request overrides shared by standalone and batch analysis."""

    top_k: int = 5
    confidence: float = 0.5
    detect_all: bool = True
    tier: str | None = None
    prompt: str | None = None


class ClipAnalysisPlan:
    """Reserve parallel phases together and serial operations one at a time."""

    def __init__(self, clip_ids: Iterable[str], operations: Iterable[str]) -> None:
        self.clip_ids = tuple(dict.fromkeys(clip_ids))
        selected = tuple(dict.fromkeys(operations))
        if not self.clip_ids or not selected:
            raise ValueError("Analysis requires clips and operations")
        unknown = set(selected) - OPERATIONS_BY_KEY.keys()
        if unknown:
            raise ValueError(
                f"Unknown analysis operations: {', '.join(sorted(unknown))}"
            )
        self.phases = tuple(
            (phase, items)
            for phase in PHASE_ORDER
            if (
                items := tuple(
                    op for op in selected if OPERATIONS_BY_KEY[op].phase == phase
                )
            )
        )
        self.operations = tuple(op for _, items in self.phases for op in items)
        self.running: set[str] = set()
        self.results: dict[str, dict[str, OutcomeStatus]] = {}
        self.cancelled = False
        self._phase_index = 0

    @property
    def active_phase(self) -> str | None:
        if self._phase_index >= len(self.phases):
            return None
        return self.phases[self._phase_index][0]

    @property
    def finished(self) -> bool:
        return not self.running and (
            self.cancelled or len(self.results) == len(self.operations)
        )

    def begin_ready(self) -> tuple[str, ...]:
        if self.cancelled or self.running:
            return ()
        while self._phase_index < len(self.phases):
            phase, operations = self.phases[self._phase_index]
            pending = tuple(op for op in operations if op not in self.results)
            if pending:
                ready = pending[:1] if phase == "sequential" else pending
                self.running.update(ready)
                return ready
            self._phase_index += 1
        return ()

    def finish(self, operation: str, outcomes: Mapping[str, OutcomeStatus]) -> bool:
        if operation not in self.running:
            return False
        if set(outcomes) - set(self.clip_ids):
            raise ValueError("Analysis outcomes include an unrequested clip")
        if any(
            status not in ("succeeded", "skipped", "failed", "unprocessed")
            for status in outcomes.values()
        ):
            raise ValueError("Invalid analysis outcome status")
        default: OutcomeStatus = "unprocessed" if self.cancelled else "failed"
        self.results[operation] = {
            cid: outcomes.get(cid, default) for cid in self.clip_ids
        }
        self.running.remove(operation)
        return True

    def cancel(self) -> None:
        self.cancelled = True

    def successful_ids(self) -> tuple[str, ...]:
        if self.cancelled:
            return ()
        return tuple(
            cid
            for cid in self.clip_ids
            if all(
                self.results.get(op, {}).get(cid) in ("succeeded", "skipped")
                for op in self.operations
            )
        )


@dataclass(frozen=True)
class ClipAnalysisFailureReport:
    """End-of-batch summary of per-clip failures, with the work to retry."""

    succeeded: int
    failed: int
    retry_clip_ids: tuple[str, ...]
    retry_operations: tuple[str, ...]
    summary: str
    details: str
    full_log: str


def clip_analysis_failure_report(
    result: Mapping[str, Any], *, preview_count: int = 10
) -> ClipAnalysisFailureReport:
    """Summarize a finished ``ClipAnalysisController`` result.

    Counts successes and failures, keeps the user-facing detail text bounded
    no matter how many clips failed, and names exactly the clips and
    operations that did not complete so they can be retried on their own.
    """
    failed_ids = tuple(result.get("failed") or ())
    succeeded = len(result.get("succeeded") or ())
    operations: Mapping[str, Mapping[str, str]] = result.get("operations") or {}
    failed_set = set(failed_ids)
    incomplete = {
        op: [
            cid
            for cid, status in outcomes.items()
            if cid in failed_set and status not in ("succeeded", "skipped")
        ]
        for op, outcomes in operations.items()
    }
    retry_operations = tuple(op for op, cids in incomplete.items() if cids)
    retry_set = {cid for cids in incomplete.values() for cid in cids}
    retry_clip_ids = tuple(cid for cid in failed_ids if cid in retry_set)

    errors = list(dict.fromkeys(result.get("errors") or ()))
    lines = errors or [
        f"{OPERATIONS_BY_KEY[op].label if op in OPERATIONS_BY_KEY else op}: "
        f"{len(cids)} incomplete clips"
        for op, cids in incomplete.items()
        if cids
    ]
    preview = lines[:preview_count]
    details = "\n".join(preview)
    if len(lines) > preview_count:
        details += f"\n... and {len(lines) - preview_count} more"
    summary = (
        f"Analysis finished: {succeeded} clips succeeded, "
        f"{len(failed_ids)} failed."
    )
    return ClipAnalysisFailureReport(
        succeeded=succeeded,
        failed=len(failed_ids),
        retry_clip_ids=retry_clip_ids,
        retry_operations=retry_operations,
        summary=summary,
        details=details,
        full_log="\n".join(lines),
    )
