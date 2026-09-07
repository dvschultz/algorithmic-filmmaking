"""Recovery across detection computation, project save, and ledger checkpoint."""

from threading import Event

import pytest

from core.jobs.commits import StaleJobResult
from core.jobs.detection import run_saved_detection
from core.jobs.store import JobStore
from core.project import Project, ProjectSaveError
from models.clip import Clip, Source


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = Project.new(name="detection")
    for index in range(2):
        media = tmp_path / f"source-{index}.mp4"
        media.write_bytes(b"media")
        source = Source(file_path=media)
        project.add_source(source)
        project.add_clips([Clip(source_id=source.id, start_frame=0, end_frame=90)])
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    calls = []

    def detect(request, **kwargs):
        calls.append(request.video_path)
        source = Source(file_path=request.video_path)
        return source, [Clip(source_id=source.id, start_frame=0, end_frame=30)]

    monkeypatch.setattr("core.jobs.detection.run_detection", detect)
    monkeypatch.setattr(
        "core.spine.detect._generate_detected_clip_thumbnails",
        lambda *a, **k: {"generated": [], "failed": [], "skipped": []},
    )
    return path, store, [s.id for s in project.sources], calls


def run(setup, ids=None, cancel=None):
    path, store, source_ids, _ = setup
    return run_saved_detection(
        store,
        path,
        ids if ids is not None else source_ids,
        3.0,
        lambda *a: None,
        cancel or Event(),
    )


def test_failed_save_reuses_computed_clips(setup, monkeypatch):
    path, store, ids, calls = setup
    with monkeypatch.context() as patch:
        patch.setattr(Project, "save", lambda *a, **k: False)
        with pytest.raises(ProjectSaveError):
            run(setup, ids[:1])
    assert len(calls) == 1
    assert not Project.load(path).metadata.job_results
    result = run(setup, ids[:1])
    assert len(calls) == 1
    assert result["result"]["succeeded"][0]["clip_count"] == 1
    assert len(Project.load(path).metadata.job_results) == 1


def test_saved_result_reconciles_checkpoint_without_reapplying(setup, monkeypatch):
    path, store, ids, calls = setup
    with monkeypatch.context() as patch:
        patch.setattr(
            store,
            "checkpoint_results",
            lambda *a: (_ for _ in ()).throw(RuntimeError("checkpoint failed")),
        )
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            run(setup, ids[:1])
    project = Project.load(path)
    result_id = next(iter(project.metadata.job_results))
    clip_ids = [c.id for c in project.clips]
    assert not store.get_result(result_id)["committed"]
    with monkeypatch.context() as patch:
        patch.setattr(Project, "save", lambda *a, **k: pytest.fail("saved again"))
        run(setup, ids[:1])
    assert [c.id for c in Project.load(path).clips] == clip_ids
    assert len(calls) == 1
    assert store.get_result(result_id)["committed"]


@pytest.mark.parametrize("committed", [True, False])
def test_retry_does_not_overwrite_user_edits(setup, monkeypatch, committed):
    path, store, ids, calls = setup
    if committed:
        run(setup, ids[:1])
    else:
        with monkeypatch.context() as patch:
            patch.setattr(Project, "save", lambda *a, **k: False)
            with pytest.raises(ProjectSaveError):
                run(setup, ids[:1])
    project = Project.load(path)
    project.clips_by_source[ids[0]][0].notes = "keep this"
    assert project.save(path)
    with pytest.raises(StaleJobResult):
        run(setup, ids[:1])
    assert Project.load(path).clips_by_source[ids[0]][0].notes == "keep this"
    assert len(calls) == 1


def test_failure_keeps_successful_sources(setup, monkeypatch):
    from core.jobs import detection

    path, store, ids, calls = setup
    compute = detection.run_detection

    def detect(request, **kwargs):
        if request.video_path.name == "source-1.mp4":
            raise RuntimeError("decoder failed")
        return compute(request, **kwargs)

    monkeypatch.setattr(detection, "run_detection", detect)
    result = run(setup)["result"]
    assert result["succeeded"][0]["source_id"] == ids[0]
    assert result["failed"][0]["source_id"] == ids[1]
    assert len(Project.load(path).metadata.job_results) == 1


def test_cancellation_keeps_prior_source_and_does_not_publish_current(
    setup, monkeypatch
):
    from core.jobs import detection

    path, store, ids, calls = setup
    compute = detection.run_detection
    cancel = Event()

    def detect(request, **kwargs):
        if request.video_path.name == "source-1.mp4":
            cancel.set()
            raise RuntimeError("decoder stopped")
        return compute(request, **kwargs)

    monkeypatch.setattr(detection, "run_detection", detect)
    result = run(setup, cancel=cancel)["result"]
    assert [item["source_id"] for item in result["succeeded"]] == ids[:1]
    assert result["failed"] == []
    assert result["cancelled"] == ids[1:]
    restored = Project.load(path)
    assert len(restored.metadata.job_results) == 1
    assert restored.clips_by_source[ids[1]][0].end_frame == 90


def test_progress_reaches_completion_only_after_source_is_saved(setup, monkeypatch):
    from core.jobs import detection

    path, store, ids, calls = setup
    compute = detection.run_detection
    events = []

    def detect(request, **kwargs):
        kwargs["progress_callback"](0.5, "halfway")
        return compute(request, **kwargs)

    def progress(fraction, message):
        frames = Project.load(path).clips_by_source[ids[0]][0].end_frame
        events.append((fraction, frames))

    monkeypatch.setattr(detection, "run_detection", detect)
    run_saved_detection(store, path, ids[:1], 3.0, progress, Event())
    assert events == [(0.4, 90), (1.0, 30)]
