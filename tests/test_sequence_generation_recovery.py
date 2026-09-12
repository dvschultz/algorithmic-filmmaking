from pathlib import Path
from threading import Event

import pytest

from core.jobs.sequence_generation import (
    run_sequence_generation_job,
    sequence_generation_job_spec,
    sequence_generation_retry_spec,
)
from core.jobs.store import JobStore, STATUS_FAILED
from core.project import Project
from core.spine.project_io import load_with_mtime
from models.clip import Clip, Source


def _saved_project(tmp_path: Path) -> Path:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"fixture")
    project = Project()
    project.add_source(
        Source(id="source", file_path=media, fps=25.0, duration_seconds=10)
    )
    project.add_clips(
        [Clip(id="clip", source_id="source", start_frame=0, end_frame=50)]
    )
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    return path


def test_keyed_retry_restores_operation_seed_after_failed_checkpoint(
    tmp_path, monkeypatch
):
    import core.remix.registry as registry_module

    path = _saved_project(tmp_path)
    store = JobStore(tmp_path / "jobs.db")
    project, mtime = load_with_mtime(path)
    operation = sequence_generation_retry_spec(
        store,
        path,
        project,
        "shuffle",
        idempotency_key="request",
    )
    row = store.insert(
        kind=operation.kind,
        args=operation.arguments,
        project_path=str(path),
        project_mtime_at_start=mtime,
        idempotency_key="request",
        operation=operation,
    )
    calls = 0
    actual_run = registry_module.run_algorithm

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(registry_module, "run_algorithm", counted)
    checkpoint = store.checkpoint_results
    monkeypatch.setattr(
        store,
        "checkpoint_results",
        lambda receipts: (_ for _ in ()).throw(OSError("checkpoint")),
    )
    with pytest.raises(OSError, match="checkpoint"):
        run_sequence_generation_job(store, path, operation, lambda *_: None, Event())
    store.update_status(row.id, STATUS_FAILED, error="checkpoint", terminal=True)

    monkeypatch.setattr(store, "checkpoint_results", checkpoint)
    reloaded, _ = load_with_mtime(path)
    retry = sequence_generation_retry_spec(
        store,
        path,
        reloaded,
        "shuffle",
        idempotency_key="request",
    )
    assert retry == operation
    assert retry.arguments["seed"] == operation.arguments["seed"]
    result = run_sequence_generation_job(store, path, retry, lambda *_: None, Event())
    assert result["replayed"]
    assert calls == 1
    store.close()


def test_unkeyed_generation_specs_from_new_sessions_are_distinct(tmp_path):
    path = _saved_project(tmp_path)
    first_project, _ = load_with_mtime(path)
    second_project, _ = load_with_mtime(path)
    first = sequence_generation_job_spec(first_project, "sequential")
    second = sequence_generation_job_spec(second_project, "sequential")
    assert first.operation_id != second.operation_id


def test_cancellation_during_apply_prevents_sequence_save(tmp_path, monkeypatch):
    import core.spine.sequences as sequences

    path = _saved_project(tmp_path)
    before = path.read_bytes()
    project, _ = load_with_mtime(path)
    operation = sequence_generation_job_spec(project, "sequential")
    store = JobStore(tmp_path / "cancel.db")
    cancel = Event()
    publish = sequences.publish_recipe

    def publish_then_cancel(*args, **kwargs):
        sequence = publish(*args, **kwargs)
        cancel.set()
        return sequence

    monkeypatch.setattr(sequences, "publish_recipe", publish_then_cancel)
    with pytest.raises(Exception, match="before publication"):
        run_sequence_generation_job(store, path, operation, lambda *_: None, cancel)
    assert path.read_bytes() == before
    store.close()


def test_job_runtime_records_publication_race_as_cancelled(tmp_path, monkeypatch):
    import time

    import core.spine.sequences as sequences
    from core.jobs.runtime import JobRuntime
    from core.jobs.sequence_generation import run_sequence_generation_runtime_job
    from core.jobs.store import STATUS_CANCELLED

    path = _saved_project(tmp_path)
    before = path.read_bytes()
    project, mtime = load_with_mtime(path)
    operation = sequence_generation_job_spec(project, "sequential")
    store = JobStore(tmp_path / "runtime-cancel.db")
    runtime = JobRuntime(store, max_workers=1)
    cancel = Event()
    publish = sequences.publish_recipe

    def publish_then_cancel(*args, **kwargs):
        sequence = publish(*args, **kwargs)
        cancel.set()
        return sequence

    monkeypatch.setattr(sequences, "publish_recipe", publish_then_cancel)
    submitted = runtime.submit(
        kind=operation.kind,
        args=operation.arguments,
        run=lambda progress, event: run_sequence_generation_runtime_job(
            store,
            path,
            operation,
            progress,
            event,
        ),
        project_path=path,
        project_mtime_at_start=mtime,
        cancellation_event=cancel,
        operation=operation,
    )
    try:
        for _ in range(500):
            row = store.get(submitted["task_id"])
            if row.status == STATUS_CANCELLED and not runtime.is_handle_live(row.id):
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"Job did not settle as cancelled: {row.status}")
        assert path.read_bytes() == before
    finally:
        runtime.shutdown()
        store.close()


@pytest.mark.parametrize("edit", ["placement", "trim", "rational_placement"])
def test_recipe_match_compares_complete_realized_timing(tmp_path, edit):
    from fractions import Fraction

    from core.spine.sequences import generate_sequence, get_sequence_recipe

    path = _saved_project(tmp_path)
    project = Project.load(path)
    assert generate_sequence(project, "sequential")["success"]
    entry = project.sequence.get_all_clips()[0]
    if edit == "placement":
        entry.start_frame += 1
    elif edit == "trim":
        entry.in_point += 1
        entry.out_point += 1
    else:
        entry.timeline_start = str(entry.timeline_start_time + Fraction(1, 1000))
    assert not get_sequence_recipe(project)["matches_timeline"]


def test_recipe_match_detects_source_and_rate_changes(tmp_path):
    from core.spine.sequences import generate_sequence, get_sequence_recipe

    project = Project.load(_saved_project(tmp_path))
    assert generate_sequence(project, "sequential")["success"]
    entry = project.sequence.get_all_clips()[0]
    entry.source_id = "another-source"
    assert not get_sequence_recipe(project)["matches_timeline"]

    entry.source_id = "source"
    entry.timeline_rate = "24"
    assert not get_sequence_recipe(project)["matches_timeline"]


def test_recipe_match_uses_snapshot_cut_after_library_recut(tmp_path):
    from core.spine.sequences import generate_sequence, get_sequence_recipe

    project = Project.load(_saved_project(tmp_path))
    assert generate_sequence(project, "sequential")["success"]
    project.clips_by_id["clip"].start_frame = 10
    inspected = get_sequence_recipe(project)
    assert not inspected["reconstructable"]
    assert inspected["matches_timeline"]


def test_recipe_inspection_reports_mismatch_for_missing_vfr_map(tmp_path):
    from core.spine.sequences import generate_sequence, get_sequence_recipe

    media = tmp_path / "vfr.mp4"
    media.write_bytes(b"fixture")
    source = Source(
        id="source",
        file_path=media,
        fps=25.0,
        duration_seconds=3,
        variable_frame_rate=True,
        frame_timestamps=tuple(str(i / 25) for i in range(76)),
    )
    project = Project(
        sources=[source],
        clips=[Clip(id="clip", source_id="source", start_frame=0, end_frame=50)],
    )
    assert generate_sequence(project, "sequential")["success"]
    source.frame_timestamps = None
    inspected = get_sequence_recipe(project)
    assert inspected["success"]
    assert not inspected["matches_timeline"]


def test_recipe_match_parses_vfr_map_once_per_source(tmp_path, monkeypatch):
    import models.media_time as media_time
    from core.spine.sequences import generate_sequence, get_sequence_recipe

    media = tmp_path / "vfr.mp4"
    media.write_bytes(b"fixture")
    timestamps = tuple(f"{i}/25" for i in range(101))
    source = Source(
        id="source", file_path=media, fps=25.0, duration_seconds=4,
        variable_frame_rate=True, frame_timestamps=timestamps,
    )
    project = Project(
        sources=[source],
        clips=[
            Clip(id="first", source_id="source", start_frame=0, end_frame=25),
            Clip(id="second", source_id="source", start_frame=25, end_frame=50),
        ],
    )
    assert generate_sequence(project, "sequential")["success"]
    calls = 0
    original = media_time.rational

    def counted(value):
        nonlocal calls
        if value in timestamps:
            calls += 1
        return original(value)

    monkeypatch.setattr(media_time, "rational", counted)
    assert get_sequence_recipe(project)["matches_timeline"]
    assert calls == len(timestamps)
