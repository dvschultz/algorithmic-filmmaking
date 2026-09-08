"""Journal desktop shot inference before owner-thread publication and explicit save."""

from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from typing import Callable, TYPE_CHECKING

from core.analysis_records import AnalysisFingerprints

from core.jobs.commits import StaleJobResult, canonical_json
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest, GuiResultReceipt
from core.jobs.media import FingerprintCancelled, media_stamp
from core.jobs.shots import _runtime, _task_data
from core.operations.shots import (
    ShotTypeOptions,
    ShotTypeOutcome,
    ShotTypeTask,
    run_shot_types,
)
from core.project import Project

if TYPE_CHECKING:
    from models.clip import Clip, Source
    from models.frame import Frame


def _target_snapshot(target: "Clip | Frame", source: "Source | None") -> dict:
    return {
        "source_id": target.source_id or "",
        "source_path": str(source.file_path) if source else None,
        "fps": source.fps if source else None,
        "shot_type": target.shot_type,
        "start_frame": getattr(target, "start_frame", None),
        "end_frame": getattr(target, "end_frame", None),
        "frame_clip_id": getattr(target, "clip_id", None),
        "frame_number": getattr(target, "frame_number", None),
    }


def shot_target_snapshot(project: Project, task: ShotTypeTask) -> dict:
    """Capture editorial identity without retaining live model objects."""
    target = (
        project.frames_by_id if task.target_type == "frame" else project.clips_by_id
    )[task.clip_id]
    source = project.sources_by_id.get(target.source_id or "")
    return _target_snapshot(target, source)


def saved_shot_matches(
    identity: dict,
    payload: dict,
    targets: dict[str, dict],
    sources: dict[str, dict],
    path: Path,
) -> bool:
    """Acknowledge only the exact target and output present in this saved snapshot."""
    from models.clip import Clip, Source
    from models.frame import Frame

    outcome = ShotTypeOutcome.from_dict(payload)
    kind = outcome.target_type
    if (
        outcome.status != "succeeded"
        or identity["kind"] != f"gui_shots_{kind}"
        or outcome.clip_id != identity["target_id"]
    ):
        raise StaleJobResult("Saved shot outcome does not match its target")
    data = identity["inputs"]["task"]
    if data["target_type"] != kind or data["clip_id"] != outcome.clip_id:
        raise StaleJobResult("Saved shot task does not match its outcome")
    saved = targets.get(outcome.clip_id)
    if saved is None:
        return False
    target = (
        Frame.from_dict(saved, path.parent)
        if kind == "frame"
        else Clip.from_dict(saved, path.parent)
    )
    if (
        outcome.record_json is not None
        and (record := target.analysis_records.get("shots")) is not None
    ):
        if record.to_dict() != json.loads(outcome.record_json):
            return False
    elif outcome.record_json is not None:
        return False
    source_data = sources.get(target.source_id or "")
    source = Source.from_dict(source_data, path.parent) if source_data else None
    actual = _target_snapshot(target, source)
    image = target.file_path if isinstance(target, Frame) else target.thumbnail_path
    return (
        actual == {**data["previous"], "shot_type": outcome.shot_type}
        and actual["source_id"] == identity["inputs"]["source_id"]
        and (str(image) if image else None) == data["thumbnail_path"]
    )


class _ShotJournal(GuiResultJournal):
    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        data = json.loads(request.spec.identity_json)["inputs"]["task"]
        source_path = data["previous"]["source_path"]
        if (
            self.fingerprints.get(Path(source_path) if source_path else None)
            != data["source_media"]
            or _runtime() != data["runtime"]
        ):
            raise StaleJobResult("Shot source media or runtime changed")

    def _accept(self, request: GuiResultRequest, row: dict) -> dict:
        outcome = ShotTypeOutcome.from_dict(json.loads(row["payload_json"]))
        if self.kind != f"gui_shots_{outcome.target_type}":
            raise StaleJobResult("Recorded shot target type changed")
        return super()._accept(request, row)


class GuiShotCache:
    def __init__(
        self,
        project: Project,
        tasks: tuple[ShotTypeTask, ...],
        options: ShotTypeOptions,
    ) -> None:
        assert project.path is not None
        project.session.assert_owner()
        self.path = project.path.expanduser().resolve()
        self.options = options
        self.runtime = _runtime()
        self.transient_outcomes: dict[tuple[str, str], dict] = {}
        previous_by_kind: dict[str, dict[str, dict]] = {
            kind: {} for kind in ("clip", "frame")
        }
        paths = set()
        for task in tasks:
            previous = shot_target_snapshot(project, task)
            previous_by_kind[task.target_type][task.clip_id] = previous
            if task.thumbnail_path is not None:
                paths.add(task.thumbnail_path)
            if previous["source_path"]:
                paths.add(Path(previous["source_path"]))
        self.media_stamps = {path: media_stamp(path) for path in paths}
        self.previous_json = canonical_json(previous_by_kind)
        self.journals = {
            kind: _ShotJournal(
                self.path,
                project.metadata.id,
                {cid: data["source_id"] for cid, data in targets.items()},
                project.metadata.job_results,
                kind=f"gui_shots_{kind}",
                arguments=asdict(options),
                media_stamps=self.media_stamps,
            )
            for kind, targets in previous_by_kind.items()
            if targets
        }

    def receipt(self, outcome: ShotTypeOutcome) -> GuiResultReceipt | None:
        return self.journals[outcome.target_type].results.get(outcome.clip_id)

    def run(
        self,
        tasks: tuple[ShotTypeTask, ...],
        cancel: Event,
        prepare: Callable[[], bool],
        deliver: Callable[[ShotTypeOutcome], None],
        progress: Callable[[int, int], None],
    ) -> tuple[ShotTypeOutcome, ...]:
        previous = json.loads(self.previous_json)
        outcomes: dict[tuple[str, str], ShotTypeOutcome] = {}
        pending = []
        requests = {}

        def publish(outcome: ShotTypeOutcome) -> None:
            outcomes[outcome.target_type, outcome.clip_id] = outcome
            deliver(outcome)
            progress(len(outcomes), len(tasks))

        try:
            for journal in self.journals.values():
                journal.start(cancel, allow_missing_receipts=True)
            fingerprints = AnalysisFingerprints(cancel)
            for task in tasks:
                if cancel.is_set():
                    break
                journal = self.journals[task.target_type]
                old = previous[task.target_type][task.clip_id]
                source = Path(old["source_path"]) if old["source_path"] else None
                if (
                    source is not None
                    and media_stamp(source) != self.media_stamps[source]
                ):
                    raise StaleJobResult("Shot source changed while queued")
                data = {
                    **_task_data(task),
                    "previous": old,
                    "source_media": journal.fingerprints.get(source),
                    "runtime": self.runtime,
                }
                request, payload = journal.prepare(
                    task.clip_id, data, task.thumbnail_path
                )
                requests[task.target_type, task.clip_id] = request
                if payload is None:
                    pending.append(task)
                elif not cancel.is_set():
                    publish(ShotTypeOutcome.from_dict(payload))
            if pending and not cancel.is_set():
                if prepare():
                    for task in pending:
                        self.journals[task.target_type].validate_media(
                            requests[task.target_type, task.clip_id]
                        )

                    def record(outcome: ShotTypeOutcome) -> None:
                        if outcome.status == "succeeded":
                            self.journals[outcome.target_type].record(
                                requests[outcome.target_type, outcome.clip_id], outcome
                            )
                        elif outcome.can_apply:
                            self.transient_outcomes[
                                outcome.target_type, outcome.clip_id
                            ] = asdict(outcome)
                        publish(outcome)

                    for outcome in run_shot_types(
                        tuple(pending),
                        self.options,
                        cancel_event=cancel,
                        on_outcome=record,
                        fingerprints=fingerprints,
                        runtime=self.runtime,
                    ):
                        outcomes.setdefault(
                            (outcome.target_type, outcome.clip_id), outcome
                        )
                else:
                    cancel.set()
        except FingerprintCancelled:
            cancel.set()
        finally:
            for journal in self.journals.values():
                if hasattr(journal, "store"):
                    journal.store.close()
        return tuple(
            outcomes.get(
                (task.target_type, task.clip_id),
                ShotTypeOutcome(
                    task.clip_id,
                    "unprocessed",
                    code="cancelled",
                    target_type=task.target_type,
                ),
            )
            for task in tasks
        )
