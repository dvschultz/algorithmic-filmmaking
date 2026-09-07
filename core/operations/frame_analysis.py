"""Ordered, Qt-free outcome tracking for frame-analysis workflows."""

FRAME_ANALYSIS_OPERATIONS = (
    "colors",
    "shots",
    "classify",
    "detect_objects",
    "extract_text",
    "describe",
    "cinematography",
)


class FrameAnalysisPlan:
    def __init__(self, frame_ids: list[str], operations: list[str]) -> None:
        self.frame_ids = tuple(dict.fromkeys(frame_ids))
        self.operations = tuple(dict.fromkeys(operations))
        if not self.frame_ids or not self.operations:
            raise ValueError("Frames and analysis operations are required")
        if any(op not in FRAME_ANALYSIS_OPERATIONS for op in self.operations):
            raise ValueError("Unsupported frame analysis operation")
        self.results: dict[str, dict[str, str]] = {}
        self.active: str | None = None
        self.cancelled = False

    def begin_next(self) -> str | None:
        if self.active is not None:
            raise RuntimeError("An analysis step is still running")
        if self.cancelled or len(self.results) == len(self.operations):
            return None
        self.active = self.operations[len(self.results)]
        return self.active

    def finish(self, operation: str, outcomes: dict[str, str]) -> bool:
        if operation != self.active or operation in self.results:
            return False
        if set(outcomes) - set(self.frame_ids):
            raise ValueError("Analysis outcome belongs to another frame")
        if any(
            status not in ("succeeded", "skipped", "failed", "unprocessed")
            for status in outcomes.values()
        ):
            raise ValueError("Invalid frame analysis outcome")
        self.results[operation] = {
            fid: outcomes.get(fid, "failed") for fid in self.frame_ids
        }
        self.active = None
        return True

    def cancel(self) -> None:
        self.cancelled = True

    def successful_ids(self) -> tuple[str, ...]:
        if self.cancelled or len(self.results) != len(self.operations):
            return ()
        return tuple(
            fid
            for fid in self.frame_ids
            if all(
                result[fid] in ("succeeded", "skipped")
                for result in self.results.values()
            )
        )
