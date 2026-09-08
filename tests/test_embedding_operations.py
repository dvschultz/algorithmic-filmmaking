"""Embedding outcomes stay detached and reject unusable provider results."""

from threading import Event, Thread
from unittest.mock import Mock

import pytest

from tests.test_description_operations import project_with_thumbnails
from ui.workers.embedding_worker import EmbeddingAnalysisWorker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    compute = Mock(side_effect=lambda paths: [[0.1] * 768 for _ in paths])
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", compute
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    return project, compute


def test_worker_does_not_mutate_project_models(setup):
    project, _ = setup
    worker = EmbeddingAnalysisWorker(project.clips)
    worker.run()
    assert all(c.embedding is None for c in project.clips)


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), [[0.0] * 768] * 2])
def test_failed_rerun_preserves_vector_and_invalidates_completion(setup, failure):
    from core.spine.analyze import embeddings

    project, compute = setup
    embeddings(project)
    compute.side_effect = [failure]
    result = embeddings(project, skip_existing=False)["result"]
    assert len(result["failed"]) == 2
    assert all(c.analysis_records["embeddings"].state == "failed" for c in project.clips)
    assert all(c.embedding == [0.1] * 768 for c in project.clips)


def test_late_embedding_failure_cannot_replace_newer_success(setup):
    from core.operations.embeddings import EmbeddingApplication, EmbeddingOptions, embedding_task, run_embeddings
    from core.spine.analyze import embeddings

    project, compute = setup
    tasks = tuple(embedding_task(c, project.sources_by_id[c.source_id], skip_existing=False) for c in project.clips)
    application = EmbeddingApplication(project, tasks)
    compute.side_effect = RuntimeError("provider failed")
    failed = run_embeddings(tasks, EmbeddingOptions())
    compute.side_effect = lambda paths: [[0.1] * 768 for _ in paths]
    embeddings(project)
    assert all(not application.apply(project, outcome) for outcome in failed)
    assert all(c.analysis_records["embeddings"].state == "succeeded" for c in project.clips)


def test_cancel_during_inference_rejects_late_results(setup):
    project, compute = setup
    worker = EmbeddingAnalysisWorker(project.clips)

    def cancelled(paths):
        worker.cancel()
        return [[0.1] * 768 for _ in paths]

    compute.side_effect = cancelled
    ready = Mock()
    worker.embedding_ready.connect(ready)
    worker.run()
    ready.assert_not_called()
    assert all(c.embedding is None for c in project.clips)


@pytest.mark.parametrize("vector", [[0.0] * 768, [float("nan")] * 768, [1.0]])
def test_unusable_vectors_are_not_published(setup, vector):
    project, compute = setup
    compute.side_effect = lambda paths: [vector for _ in paths]
    worker = EmbeddingAnalysisWorker(project.clips)
    ready, error = Mock(), Mock()
    worker.embedding_ready.connect(ready)
    worker.error.connect(error)
    worker.run()
    ready.assert_not_called()
    error.assert_called_once()
    assert all(c.embedding is None for c in project.clips)


def test_short_provider_batch_is_not_silently_truncated(setup):
    project, compute = setup
    compute.side_effect = None
    compute.return_value = [[0.1] * 768]
    worker = EmbeddingAnalysisWorker(project.clips)
    ready, error = Mock(), Mock()
    worker.embedding_ready.connect(ready)
    worker.error.connect(error)
    worker.run()
    ready.assert_not_called()
    error.assert_called_once()


def test_gui_and_spine_share_the_same_vectors(setup):
    from core.spine.analyze import embeddings

    project, compute = setup
    worker = EmbeddingAnalysisWorker(project.clips)
    worker.run()
    result = embeddings(project)["result"]
    assert len(result["succeeded"]) == 2
    assert result["failed"] == result["unprocessed"] == []
    assert all(
        tuple(c.embedding) == o.vector for c, o in zip(project.clips, worker.result)
    )
    assert compute.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        "thumbnail",
        "image_bytes",
        "source",
        "range",
        "source_bytes",
        "manual_vector",
        "model",
        "session",
    ],
)
def test_application_rejects_changed_target(setup, change, tmp_path):
    from core.operations.embeddings import EmbeddingApplication

    project, _ = setup
    worker = EmbeddingAnalysisWorker(project.clips)
    application = EmbeddingApplication(project, worker.tasks)
    worker.run()
    clip = project.clips[0]
    if change == "thumbnail":
        clip.thumbnail_path = tmp_path / "other.jpg"
    elif change == "image_bytes":
        clip.thumbnail_path.write_bytes(b"replaced")
    elif change == "source":
        clip.source_id = "other"
    elif change == "range":
        clip.start_frame += 1
    elif change == "source_bytes":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "manual_vector":
        clip.embedding = [0.2] * 768
    elif change == "model":
        clip.embedding_model = "other-model"
    else:
        project.clear()
    previous = clip.embedding
    assert not application.apply(project, worker.result[0])
    assert clip.embedding == previous


def test_application_is_once_only_and_detaches_vector(setup):
    from core.operations.embeddings import EmbeddingApplication

    project, _ = setup
    worker = EmbeddingAnalysisWorker(project.clips)
    application = EmbeddingApplication(project, worker.tasks)
    worker.run()
    assert application.apply(project, worker.result[0])
    assert not application.apply(project, worker.result[0])
    project.clips[0].embedding[0] = 99.0
    assert worker.result[0].vector[0] == 0.1


def test_cancelled_waiter_does_not_unload_active_model(setup, monkeypatch):
    from core.operations.embeddings import EmbeddingOptions, run_embeddings

    project, compute = setup
    tasks = EmbeddingAnalysisWorker(project.clips).tasks
    entered, release, cancel = Event(), Event(), Event()
    unload = Mock()
    monkeypatch.setattr("core.analysis.embeddings.unload_model", unload)

    def blocked(paths):
        entered.set()
        assert release.wait(5)
        return [[0.1] * 768 for _ in paths]

    compute.side_effect = blocked
    first = Thread(target=lambda: run_embeddings(tasks, EmbeddingOptions()))
    results = []
    second = Thread(
        target=lambda: results.extend(
            run_embeddings(tasks, EmbeddingOptions(), cancel_event=cancel)
        )
    )
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        cancel.set()
        second.join(5)
        assert not second.is_alive()
        assert all(o.status == "unprocessed" for o in results)
        unload.assert_not_called()
        compute.assert_called_once()
    finally:
        release.set()
        first.join(5)
    assert not first.is_alive()
    unload.assert_called_once()


def test_missing_image_is_failure_but_valid_neighbor_succeeds(setup):
    from core.spine.analyze import embeddings

    project, compute = setup
    project.clips[0].thumbnail_path = None
    result = embeddings(project)["result"]
    assert result["failed"] == [
        {"clip_id": project.clips[0].id, "code": "thumbnail_missing"}
    ]
    assert len(result["succeeded"]) == 1
    assert len(compute.call_args.args[0]) == 1


def test_pipeline_launcher_uses_guarded_delivery(setup, monkeypatch):
    from PySide6.QtCore import QObject
    from core.settings import Settings
    from core.operations.embeddings import EmbeddingApplication
    from ui.workers.clip_analysis import ClipAnalysisController

    project, _ = setup
    window = QObject()
    window.project = project
    window.settings = Settings()
    monkeypatch.setattr(EmbeddingAnalysisWorker, "start", Mock())
    controller = ClipAnalysisController(window, project.clips, ["embeddings"])
    controller.start()
    assert isinstance(controller.applications["embeddings"], EmbeddingApplication)
    worker = controller.workers["embeddings"]
    worker.analysis_completed.emit()
    assert not controller.plan.results
    worker.finished.emit()
    assert set(controller.plan.results) == {"embeddings"}
    controller.cancel()
    assert controller.finished


def test_pipeline_retains_embedding_errors_and_ready_does_not_mark_dirty(setup):
    from types import SimpleNamespace
    from ui.main_window import MainWindow

    project, _ = setup
    window = SimpleNamespace(
        clips_by_id=project.clips_by_id,
        _mark_dirty=Mock(),
        statusBar=Mock(),
        _color_run_error=None,
        _shot_type_run_error=None,
        _classification_run_error=None,
        _object_detection_run_error=None,
        _text_extraction_run_error=None,
        _transcription_run_error=None,
        _description_run_error=None,
        _cinematography_run_error=None,
    )
    MainWindow._on_embeddings_error(window, "model unavailable")
    details = MainWindow._get_completed_analysis_error_details(window, ["embeddings"])
    assert details == [("embeddings", "model unavailable", "embedding analysis")]
    MainWindow._reset_analysis_run_error(window, "embeddings")
    assert window._embeddings_run_error is None
    MainWindow._on_embedding_ready(window, project.clips[0].id)
    window._mark_dirty.assert_not_called()
