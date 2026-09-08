"""Boundary analysis is available across surfaces with recoverable atomic pairs."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.boundary_embeddings import (
    BoundaryEmbeddingApplication,
    BoundaryEmbeddingOutcome,
)
from core.project import Project
from tests.test_description_operations import project_with_thumbnails
from ui.workers.boundary_embedding_worker import BoundaryEmbeddingWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    compute = Mock(return_value=([0.123456789] * 768, [0.987654321] * 768))
    monkeypatch.setattr("core.analysis.embeddings.extract_boundary_embeddings", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    return project, compute


def test_gui_recovery_then_explicit_save(setup):
    project, compute = setup
    first = BoundaryEmbeddingWorker(project.clips, project=project)
    first.run()
    assert first.job_status == "completed"
    assert all(c.first_frame_embedding is None for c in project.clips)
    reopened = Project.load(project.path)
    second = BoundaryEmbeddingWorker(reopened.clips, project=reopened)
    second.run()
    assert second.result == first.result
    assert compute.call_count == 2
    application = BoundaryEmbeddingApplication(reopened, second.tasks)
    for outcome in second.result:
        assert application.apply(reopened, outcome)
        receipt = second.cache.results[outcome.clip_id]
        assert receipt.matches(outcome)
        reopened.record_job_result(receipt.result_id, receipt.digest)
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(r)["committed"] for r in reopened.metadata.job_results
        )
        assert reopened.save()
        assert all(
            store.get_result(r)["committed"] for r in reopened.metadata.job_results
        )
    finally:
        store.close()
    saved = Project.load(project.path).clips[0]
    assert saved.first_frame_embedding == [0.123456789] * 768
    assert saved.last_frame_embedding == [0.987654321] * 768


def test_matching_thumbnail_model_is_preserved(setup):
    project, _ = setup
    clip = project.clips[0]
    clip.embedding = [0.3] * 768
    clip.embedding_model = "dinov2-vit-b-14"
    worker = BoundaryEmbeddingWorker([clip], project=project)
    application = BoundaryEmbeddingApplication(project, worker.tasks)
    worker.run()
    assert application.apply(project, worker.result[0])
    assert clip.embedding == [0.3] * 768
    assert clip.first_frame_embedding == [0.123456789] * 768


@pytest.mark.parametrize("change", ["range", "source", "media", "edit", "session"])
def test_application_rejects_changed_target(setup, change):
    project, _ = setup
    worker = BoundaryEmbeddingWorker(project.clips, project=project)
    application = BoundaryEmbeddingApplication(project, worker.tasks)
    worker.run()
    clip = project.clips[0]
    if change == "range":
        clip.end_frame += 1
    elif change == "source":
        clip.source_id = "missing"
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "edit":
        clip.last_frame_embedding = [1.0] * 768
    else:
        project.clear()
    assert not application.apply(project, worker.result[0])
    assert clip.first_frame_embedding is None


def test_invalid_second_vector_cannot_apply_first(setup):
    project, _ = setup
    worker = BoundaryEmbeddingWorker(project.clips, project=project)
    application = BoundaryEmbeddingApplication(project, worker.tasks)
    invalid = BoundaryEmbeddingOutcome(
        project.clips[0].id, "succeeded", (1.0,) * 768, (0.0,) * 768, "dinov2-vit-b-14"
    )
    with pytest.raises(ValueError):
        application.apply(project, invalid)
    assert project.clips[0].first_frame_embedding is None


def test_missing_source_is_per_item_failure(setup):
    project, compute = setup
    project.clips[0].source_id = "missing"
    worker = BoundaryEmbeddingWorker(project.clips, project=project)
    worker.run()
    assert [o.status for o in worker.result] == ["failed", "succeeded"]
    compute.assert_called_once()


def test_spine_combined_analysis_publishes_pair(setup):
    from core.spine.analyze import analyze_clips

    project, compute = setup
    result = analyze_clips(project, operations=["boundary_embeddings"])
    assert result["success"]
    assert (
        len(
            result["result"]["operations"]["boundary_embeddings"]["result"]["succeeded"]
        )
        == 2
    )
    assert project.clips[0].first_frame_embedding == [0.123456789] * 768
    assert compute.call_count == 2


def test_availability_distinguishes_missing_empty_and_complete_pairs(setup):
    from core.analysis_availability import (
        operation_is_complete_for_clip,
        clear_operation_result,
    )

    project, _ = setup
    clip = project.clips[0]
    clip.extracted_texts = []
    assert operation_is_complete_for_clip("extract_text", clip)
    clip.first_frame_embedding = [1.0] * 768
    assert not operation_is_complete_for_clip("boundary_embeddings", clip)
    clip.last_frame_embedding = [2.0] * 768
    clip.embedding = [3.0] * 768
    clip.embedding_model = "dinov2-vit-b-14"
    assert operation_is_complete_for_clip("boundary_embeddings", clip)
    assert clear_operation_result(clip, "boundary_embeddings")
    assert clip.first_frame_embedding is None and clip.last_frame_embedding is None
    assert clip.embedding == [3.0] * 768
    assert clip.embedding_model == "dinov2-vit-b-14"


def test_cancelled_cached_results_are_not_delivered(setup):
    project, compute = setup
    first = BoundaryEmbeddingWorker(project.clips, project=project)
    first.run()
    second = BoundaryEmbeddingWorker(project.clips, project=project)
    cancel = Event()
    cancel.set()
    deliver = Mock()
    outcomes = second.cache.run(
        second.tasks, cancel, lambda: True, deliver, lambda *_: None
    )
    assert all(o.status == "unprocessed" for o in outcomes)
    deliver.assert_not_called()
    assert compute.call_count == 2


def test_gui_launcher_tracks_boundary_completion_separately(setup, monkeypatch):
    from PySide6.QtCore import QObject
    from core.settings import Settings
    from ui.workers.clip_analysis import ClipAnalysisController

    project, _ = setup
    window = QObject()
    window.project = project
    window.settings = Settings()
    monkeypatch.setattr(BoundaryEmbeddingWorker, "start", Mock())
    controller = ClipAnalysisController(window, project.clips, ["boundary_embeddings"])
    controller.start()
    worker = controller.workers["boundary_embeddings"]
    assert worker.cache.path == project.path.resolve()
    worker.analysis_completed.emit()
    worker.analysis_completed.emit()
    assert not controller.plan.results
    worker.finished.emit()
    worker.finished.emit()
    assert set(controller.plan.results) == {"boundary_embeddings"}
    controller.cancel()
    assert controller.finished
