"""Sequencing prerequisites recover without saving GUI project edits."""

from unittest.mock import Mock

import pytest

from core.project import Project
from tests.test_description_operations import project_with_thumbnails


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from core.settings import Settings

    project = project_with_thumbnails(tmp_path, 2)
    project.sources[0].file_path.write_bytes(b"video")
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: Settings(cache_dir=tmp_path / "cache")
    )
    monkeypatch.setattr("core.feature_registry.check_feature", lambda _: (True, []))
    thumbnail = Mock(side_effect=lambda paths: [[0.1] * 768 for _ in paths])
    boundary = Mock(return_value=([0.2] * 768, [0.3] * 768))
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_clip_embeddings_batch", thumbnail
    )
    monkeypatch.setattr(
        "core.analysis.embeddings.extract_boundary_embeddings", boundary
    )
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    return project, thumbnail, boundary


@pytest.mark.parametrize("algorithm", ["similarity_chain", "match_cut"])
def test_saved_worker_reuses_prerequisites_without_project_save(setup, algorithm):
    from ui.workers.sequence_worker import SequenceWorker

    project, thumbnail, boundary = setup
    project.clips[0].notes = "unsaved note"
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    outputs = []
    for _ in range(2):
        worker = SequenceWorker(algorithm, pairs, project=project)
        worker.sequence_ready.connect(outputs.append)
        worker.run()
    assert len(outputs) == 2
    assert thumbnail.call_count == (1 if algorithm == "similarity_chain" else 0)
    assert boundary.call_count == (2 if algorithm == "match_cut" else 0)
    assert all(
        c.embedding is None and c.first_frame_embedding is None for c in project.clips
    )
    assert Project.load(project.path).clips[0].notes != "unsaved note"


def test_staccato_recovers_same_saved_prerequisites(setup):
    from core.analysis.audio import AudioAnalysis
    from ui.dialogs.staccato_dialog import StaccatoGenerateWorker

    project, thumbnail, _ = setup
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    for _ in range(2):
        worker = StaccatoGenerateWorker(
            pairs, AudioAnalysis([], [], [], 1.0), "onsets", project=project
        )
        worker._auto_compute_embeddings()
        assert all(c.embedding == [0.1] * 768 for c, _ in worker._clips)
    thumbnail.assert_called_once()
    assert all(c.embedding is None for c in project.clips)


def test_interruption_between_batches_preserves_completed_batch(setup):
    from copy import deepcopy
    from threading import Event
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob

    project, thumbnail, _ = setup
    more = []
    for i in range(18):
        clip = deepcopy(project.clips[0])
        clip.id = f"extra-{i}"
        more.append(clip)
    project.add_clips(more)
    assert project.save()
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]

    def infer(paths):
        if thumbnail.call_count == 2:
            raise SystemExit("interrupted")
        return [[0.1] * 768 for _ in paths]

    thumbnail.side_effect = infer
    job = SequenceEmbeddingJob(pairs, mode="thumbnail", project=project)
    with pytest.raises(SystemExit, match="interrupted"):
        job._compute(lambda *_: None, Event())
    copied = deepcopy(pairs)
    SequenceEmbeddingJob(pairs, mode="thumbnail", project=project).populate(
        copied, Event()
    )
    assert [len(call.args[0]) for call in thumbnail.call_args_list] == [16, 4, 4]
    assert all(c.embedding == [0.1] * 768 for c, _ in copied)


@pytest.mark.parametrize("mode", ["thumbnail", "boundary"])
def test_saved_cache_invalidates_when_source_changes(setup, mode):
    from copy import deepcopy
    from threading import Event
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob

    project, thumbnail, boundary = setup
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    SequenceEmbeddingJob(pairs, mode=mode, project=project).populate(
        deepcopy(pairs), Event()
    )
    project.sources[0].file_path.write_bytes(b"replacement source")
    SequenceEmbeddingJob(pairs, mode=mode, project=project).populate(
        deepcopy(pairs), Event()
    )
    assert (thumbnail.call_count if mode == "thumbnail" else boundary.call_count) == (
        2 if mode == "thumbnail" else 4
    )


@pytest.mark.parametrize(
    "change", ["source", "range", "embedding", "save_as", "session"]
)
def test_owner_rejects_stale_proposal_inputs(setup, change):
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob
    from core.jobs.commits import StaleJobResult

    project, _, _ = setup
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    job = SequenceEmbeddingJob(pairs, mode="thumbnail", project=project)
    if change == "source":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "range":
        project.clips[0].end_frame += 1
    elif change == "embedding":
        project.clips[0].embedding = [0.4] * 768
    elif change == "save_as":
        assert project.save(project.path.with_name("new-project.json"))
    else:
        project.clear()
    with pytest.raises(StaleJobResult):
        job.validate_project(project)


def test_unsaved_job_is_session_only_and_does_not_cache(setup):
    from copy import deepcopy
    from threading import Event
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob

    project, thumbnail, _ = setup
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    for _ in range(2):
        job = SequenceEmbeddingJob(pairs, mode="thumbnail")
        assert job.operation.persistence == "session_only"
        job.populate(deepcopy(pairs), Event())
    assert thumbnail.call_count == 2


def test_cancel_during_inference_emits_no_proposal(setup):
    from ui.workers.sequence_worker import SequenceWorker

    project, thumbnail, _ = setup
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    worker = SequenceWorker("similarity_chain", pairs, project=project)
    ready = Mock()
    worker.sequence_ready.connect(ready)

    def infer(paths):
        worker.cancel()
        return [[0.1] * 768 for _ in paths]

    thumbnail.side_effect = infer
    worker.run()
    ready.assert_not_called()
    assert worker.prerequisite_job.status == "cancelled"


def test_corrupt_cache_fails_without_recomputation(setup):
    import sqlite3
    from copy import deepcopy
    from threading import Event
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob

    project, thumbnail, _ = setup
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    SequenceEmbeddingJob(pairs, mode="thumbnail", project=project).populate(
        deepcopy(pairs), Event()
    )
    with sqlite3.connect(project.path.parent / "cache" / "jobs.db") as connection:
        connection.execute("UPDATE job_results SET payload_json = '{}' ")
    with pytest.raises(RuntimeError, match="corrupt"):
        SequenceEmbeddingJob(pairs, mode="thumbnail", project=project).populate(
            deepcopy(pairs), Event()
        )
    thumbnail.assert_called_once()


def test_real_qthread_records_history_and_delivers_result(setup):
    from PySide6.QtWidgets import QApplication
    from ui.workers.sequence_worker import SequenceWorker
    from core.jobs.store import JobStore

    app = QApplication.instance() or QApplication([])
    project, _, _ = setup
    worker = SequenceWorker(
        "match_cut",
        [(c, project.sources_by_id[c.source_id]) for c in project.clips],
        project=project,
    )
    ready = []
    worker.sequence_ready.connect(ready.append)
    worker.start()
    assert worker.wait(10000)
    app.processEvents()
    assert len(ready) == 1
    store = JobStore(project.path.parent / "cache" / "jobs.db")
    try:
        assert store.get(worker.prerequisite_job.task_id).status == "completed"
    finally:
        store.close()
    assert all(c.first_frame_embedding is None for c in project.clips)


def test_staccato_result_is_not_applied_to_replacement_project(setup, monkeypatch):
    from types import SimpleNamespace
    from ui.tabs.sequence_tab import SequenceTab

    project, _, _ = setup
    apply = Mock()
    tab = SimpleNamespace(_project=project, _apply_staccato_sequence=apply)

    class Dialog:
        def __init__(self, clips, project, parent):
            self._project = project
            self.parent = parent
            self.music_path = None
            self.sequence_ready = SimpleNamespace(
                connect=lambda callback: setattr(self, "callback", callback)
            )

        def exec(self):
            self.parent._project = Project.new()
            self.callback([])

    monkeypatch.setattr("ui.dialogs.staccato_dialog.StaccatoDialog", Dialog)
    SequenceTab._show_staccato_dialog(tab, [])
    apply.assert_not_called()


@pytest.mark.parametrize("mode", ["thumbnail", "boundary"])
def test_missing_media_does_not_request_model_dependencies(setup, monkeypatch, mode):
    from copy import deepcopy
    from threading import Event
    from core.jobs.sequence_embeddings import SequenceEmbeddingJob

    project, _, _ = setup
    if mode == "thumbnail":
        for path in {c.thumbnail_path for c in project.clips}:
            path.unlink()
    else:
        project.sources[0].file_path.unlink()
    check = Mock(return_value=(False, ["torch"]))
    monkeypatch.setattr("core.feature_registry.check_feature", check)
    pairs = [(c, project.sources_by_id[c.source_id]) for c in project.clips]
    job = SequenceEmbeddingJob(pairs, mode=mode, project=project)
    if mode == "thumbnail":
        with pytest.raises(RuntimeError, match="Missing DINOv2 embeddings"):
            job.populate(deepcopy(pairs), Event(), require_all=True)
    else:
        copied = deepcopy(pairs)
        job.populate(copied, Event())
        assert all(c.first_frame_embedding is None for c, _ in copied)
    check.assert_not_called()
