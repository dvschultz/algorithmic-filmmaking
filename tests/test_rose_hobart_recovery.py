"""Rose Hobart recovers verified face computation before project publication."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.project import Project
from core.settings import Settings
from tests.test_rose_hobart_analysis import analyzed_project as rose_setup  # noqa: F401
from tests.test_face_records import setup as face_setup  # noqa: F401
from ui.dialogs.rose_hobart_dialog import RoseHobartDialog, RoseHobartWorker


@pytest.fixture
def saved(request, tmp_path, monkeypatch):
    project, provider, directory = request.getfixturevalue("rose_setup")
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: Settings(cache_dir=tmp_path, model_cache_dir=tmp_path),
    )
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    project.save(tmp_path / "project.sceneripper")
    return project, provider, directory


def worker_for(project):
    return RoseHobartWorker(
        [project.sources[0].file_path],
        [(clip, project.sources_by_id[clip.source_id]) for clip in project.clips],
        "Balanced",
        "Original Order",
        1.0,
        project=project,
    )


def test_interrupted_publication_recovers_without_clip_inference(saved):
    project, provider, _ = saved
    first = worker_for(project)
    first.run()
    assert first.result is not None, first.failure
    assert first.cache.results
    assert project.clips[0].face_embeddings is None
    restored = Project.load(project.path)
    second = worker_for(restored)
    second.run()
    assert second.result is not None, second.failure
    assert second.outcomes == first.outcomes
    assert provider.call_count == 1


@pytest.mark.parametrize("tampered", [False, True])
def test_dialog_authenticates_and_checkpoints_receipts(saved, monkeypatch, tampered):
    project, _, _ = saved
    monkeypatch.setattr(RoseHobartWorker, "start", lambda self: None)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", Mock())
    dialog = RoseHobartDialog(project.clips, project.sources_by_id, project=project)
    dialog._ref_widgets = [
        SimpleNamespace(_image_path=project.sources[0].file_path, has_face=True)
    ]
    sequences = []
    dialog.sequence_ready.connect(sequences.append)
    dialog._on_generate()
    worker = dialog.worker
    worker.run()
    assert worker.result is not None, worker.failure
    if tampered:
        worker.outcomes = (replace(worker.outcomes[0], faces=()),)
    dialog._retire_workers()
    if tampered:
        assert not sequences
        assert project.clips[0].face_embeddings is None
        assert not project.metadata.job_results
    else:
        assert sequences
        receipt = worker.cache.results[project.clips[0].id]
        assert project.metadata.job_results[receipt.result_id] == receipt.digest
        assert project.save()
        from core.jobs.store import JobStore

        store = JobStore(project.path.parent / "jobs.db")
        try:
            assert store.get_result(receipt.result_id)["committed"]
        finally:
            store.close()
    dialog._ref_widgets = []
    dialog.reject()


def test_cancellation_after_receipt_keeps_recoverable_computation(saved, monkeypatch):
    project, provider, _ = saved
    first = worker_for(project)
    original = first.cache.record

    def record(*args):
        value = original(*args)
        first.cancel()
        return value

    monkeypatch.setattr(first.cache, "record", record)
    first.run()
    assert first.result is None
    assert project.clips[0].face_embeddings is None
    second = worker_for(Project.load(project.path))
    second.run()
    assert second.result is not None, second.failure
    assert provider.call_count == 1


def test_regular_face_worker_receipt_is_reusable_by_rose_hobart(saved):
    from threading import Event
    from ui.workers.face_detection_worker import FaceDetectionWorker

    project, provider, _ = saved
    worker = FaceDetectionWorker(project.clips, project.sources_by_id, project=project)
    outcomes = worker.cache.run(
        worker.tasks, Event(), lambda: True, lambda _: None, lambda *_: None
    )
    rose = worker_for(project)
    rose.run()
    assert rose.result is not None, rose.failure
    assert rose.outcomes == outcomes
    assert provider.call_count == 1


def test_first_reference_model_download_receipt_recovers(saved, monkeypatch):
    project, provider, directory = saved
    files = {path: path.read_bytes() for path in directory.glob("*.onnx")}
    for path in files:
        path.unlink()
    original = provider.side_effect

    def reference(path, *, on_execution):
        for weight, data in files.items():
            if not weight.exists():
                weight.write_bytes(data)
        return original(on_execution=on_execution, start_frame=0)

    monkeypatch.setattr("core.analysis.faces.extract_faces_from_image", reference)
    first = worker_for(project)
    assert first.cache.runtime["files"] == []
    first.run()
    assert first.result is not None, first.failure
    recovered = worker_for(Project.load(project.path))
    recovered.run()
    assert recovered.result is not None, recovered.failure
    assert recovered.outcomes == first.outcomes
    assert provider.call_count == 1
