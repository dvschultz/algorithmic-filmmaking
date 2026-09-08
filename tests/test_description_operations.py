"""Description parity, bounded dispatch, retries, and cancelled publication."""

from threading import Event, Thread

import pytest

from core.operations.description import (
    DescriptionOptions,
    DescriptionTask,
    run_description,
)
from core.spine.analyze import describe
from tests.test_spine_analyze import _build_project
from ui.workers.description_worker import DescriptionWorker


def project_with_thumbnails(tmp_path, count=3):
    project = _build_project(tmp_path, count)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")
    for clip in project.clips:
        clip.thumbnail_path = thumbnail
    return project


@pytest.mark.parametrize("response", ["", "Error: provider failed"])
def test_gui_and_spine_reject_invalid_provider_response(
    tmp_path, monkeypatch, response
):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.description.describe_frame", lambda *a, **kw: (response, "cloud")
    )
    worker = DescriptionWorker(project.clips, tier="cloud")
    worker.run()
    result = describe(project, tier="cloud")["result"]
    assert worker.error_count == 1
    assert result["failed"][0]["code"] == "description_failed"
    assert result["succeeded"] == []
    assert project.clips[0].description is None


def test_gui_and_spine_forward_same_inputs(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    calls = []

    def provider(*args, **kwargs):
        calls.append((args, kwargs))
        return "A person walking", "model"

    monkeypatch.setattr("core.analysis.description.describe_frame", provider)
    worker = DescriptionWorker(
        project.clips, sources=project.sources_by_id, tier="cloud", prompt="Action?"
    )
    worker.run()
    result = describe(project, tier="cloud", prompt="Action?")["result"]
    assert calls[0][0] == calls[1][0]
    assert {k: v for k, v in calls[0][1].items() if k != "on_execution"} == {
        k: v for k, v in calls[1][1].items() if k != "on_execution"
    }
    assert (
        worker.result[0].description
        == project.clips[0].description
        == "A person walking"
    )
    assert result["succeeded"] == [{"clip_id": "c-0", "model": "model"}]


@pytest.mark.parametrize("tier, limit", [("cloud", 2), ("local", 1), ("gpu", 1)])
def test_cancel_bounds_admission_and_suppresses_inflight(
    tmp_path, monkeypatch, tier, limit
):
    path = tmp_path / "thumb.jpg"
    path.write_bytes(b"fake")
    tasks = tuple(DescriptionTask(str(i), path, None, 0, 1, None) for i in range(8))
    entered, release, cancel = Event(), Event(), Event()
    calls, delivered, results = [], [], []

    def provider(*args, **kwargs):
        calls.append(True)
        if len(calls) == limit:
            entered.set()
        assert release.wait(5)
        return "Late output", "model"

    monkeypatch.setattr("core.analysis.description.describe_frame", provider)
    thread = Thread(
        target=lambda: results.append(
            run_description(
                tasks,
                DescriptionOptions(tier, parallelism=2),
                cancel_event=cancel,
                on_outcome=delivered.append,
            )
        )
    )
    thread.start()
    try:
        assert entered.wait(5)
        cancel.set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(calls) == limit
    assert delivered == []
    assert all(o.status == "unprocessed" for o in results[0])


def test_cancel_interrupts_retry_and_does_not_reinvoke_provider(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    worker = DescriptionWorker(project.clips, tier="cloud")
    cancel = worker._cancel_event
    calls, delays = [], []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("503 unavailable")

    def wait(delay):
        delays.append(delay)
        cancel.set()
        return True

    monkeypatch.setattr("core.analysis.description.describe_frame", fail)
    monkeypatch.setattr(cancel, "wait", wait)
    worker.run()
    assert calls == [True]
    assert delays == [2]
    assert worker.result[0].status == "unprocessed"
    assert worker.success_count == worker.error_count == 0


def test_spine_skip_missing_and_cancelled_results(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path)
    monkeypatch.setattr(
        "core.analysis.description.describe_frame",
        lambda *a, **kw: ("Existing", "model"),
    )
    describe(project, clip_ids=[project.clips[0].id], tier="cloud")
    project.clips[1].thumbnail_path = None
    cancel = Event()

    def provider(*a, **kw):
        cancel.set()
        return "Late output", "model"

    monkeypatch.setattr("core.analysis.description.describe_frame", provider)
    result = describe(project, tier="cloud", cancel_event=cancel)["result"]
    assert result["skipped"] == [{"clip_id": "c-0", "reason": "valid_analysis"}]
    assert result["failed"] == [{"clip_id": "c-1", "code": "thumbnail_missing"}]
    assert result["succeeded"] == []
    assert project.clips[2].description is None


def test_unexpected_task_failure_preserves_other_outcomes(tmp_path, monkeypatch):
    path = tmp_path / "thumb.jpg"
    path.write_bytes(b"fake")
    tasks = tuple(DescriptionTask(str(i), path, None, 0, 1, None) for i in range(2))
    from core.operations.description import DescriptionOutcome

    def compute(task, options, cancel, *, fingerprints=None):
        if task.clip_id == "0":
            raise OSError("Cannot inspect thumbnail")
        return DescriptionOutcome(task.clip_id, "succeeded", "Valid output", "model")

    monkeypatch.setattr("core.operations.description.compute_description", compute)
    outcomes = run_description(tasks, DescriptionOptions("cloud"))
    assert outcomes[0].status == "failed"
    assert outcomes[0].message == "Cannot inspect thumbnail"
    assert outcomes[1].description == "Valid output"


def test_frame_identity_and_image_are_snapshotted(tmp_path, monkeypatch):
    from core.analysis_target import AnalysisTarget

    path = tmp_path / "frame.jpg"
    path.write_bytes(b"fake")
    target = AnalysisTarget(target_type="frame", id="frame-1", image_path=path)
    worker = DescriptionWorker([], tier="cloud", analysis_targets=[target])
    calls, delivered = [], []

    def provider(image, **kwargs):
        calls.append(image)
        return "Still image", "model"

    monkeypatch.setattr("core.analysis.description.describe_frame", provider)
    worker.description_ready.connect(lambda *args: delivered.append(args))
    worker.run()
    assert calls == [path]
    assert delivered == [("frame-1", "Still image", "model")]
