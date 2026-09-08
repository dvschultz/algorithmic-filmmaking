"""Verified scalar prerequisites for private sequencing occurrences."""

from copy import deepcopy
from dataclasses import replace
import json
from threading import Event

from core.analysis_records import AnalysisSnapshot
from core.operations.scalars import (
    FIELDS,
    ScalarOperation,
    run_scalars,
    scalar_runtime,
    scalar_task,
)
from models.analysis_record import AnalysisRecord
from models.clip import Clip, Source


def scalar_inputs(
    clips: list[tuple[Clip, Source]],
    operation: ScalarOperation,
    *,
    cancel_event: Event | None = None,
) -> list[tuple[Clip, Source]]:
    """Return analyzed copies; never publish sequencing work into owner models."""
    cancel = cancel_event or Event()
    if cancel.is_set():
        return []
    snapshots = deepcopy(clips)
    tasks = tuple(
        replace(scalar_task(clip, source, operation), clip_id=str(index))
        for index, (clip, source) in enumerate(snapshots)
    )
    outcomes = run_scalars(tasks, cancel_event=cancel)
    if cancel.is_set():
        return []
    runtime = scalar_runtime(operation)
    for (clip, _), task, outcome in zip(snapshots, tasks, outcomes, strict=True):
        if not outcome.has_result or outcome.record_json is None:
            raise RuntimeError(
                f"{operation.capitalize()} analysis failed for clip {clip.id}: "
                f"{outcome.message or outcome.status}"
            )
        record = AnalysisRecord.from_dict(json.loads(outcome.record_json))
        if (
            record.identity is None
            or not AnalysisSnapshot.from_json(task.snapshot_json).inputs.unchanged()
            or record.identity.to_dict()["model"] != runtime
        ):
            raise RuntimeError(f"{operation.capitalize()} analysis inputs changed")
        if record.value_json is None:
            raise RuntimeError(f"{operation.capitalize()} analysis has no result")
        value = json.loads(record.value_json)
        setattr(clip, FIELDS[operation], value[FIELDS[operation]])
        clip.analysis_records[operation] = record
    return [] if cancel.is_set() else snapshots
