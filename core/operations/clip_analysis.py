"""Phase ordering and per-clip outcomes for a combined analysis request."""

from collections.abc import Iterable, Mapping

from core.analysis_operations import OPERATIONS_BY_KEY, PHASE_ORDER
from core.operations.contracts import OutcomeStatus


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
