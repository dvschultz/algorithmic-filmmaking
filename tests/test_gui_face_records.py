"""Verified GUI face journals recover records before model preparation."""

from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import pytest

from core.operations.faces import FaceApplication
from core.project import Project
from core.jobs.store import JobStore
from tests.test_face_records import setup as face_setup  # noqa: F401
from ui.workers.face_detection_worker import FaceDetectionWorker


@pytest.fixture
def setup(request, tmp_path, monkeypatch):
    from core.settings import Settings

    project, provider, directory = request.getfixturevalue("face_setup")
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: Settings(cache_dir=tmp_path, model_cache_dir=tmp_path),
    )
    project.save(tmp_path / "project.sceneripper")
    return project, provider, directory


def run(project, *, apply=False, prepare=lambda: True, cancel=None):
    worker = FaceDetectionWorker(project.clips, project.sources_by_id, project=project)
    application = FaceApplication(project, worker.tasks, worker.options)

    def deliver(outcome):
        if apply and outcome.can_apply:
            assert application.apply(project, outcome), outcome
            receipt = worker.cache.results.get(outcome.clip_id)
            if receipt is not None:
                assert receipt.matches(outcome)
                project.record_job_result(receipt.result_id, receipt.digest)
            else:
                from dataclasses import asdict

                assert worker.cache.transient_outcomes[outcome.clip_id] == asdict(
                    outcome
                )

    outcomes = worker.cache.run(
        worker.tasks, cancel or Event(), prepare, deliver, lambda *_: None
    )
    return worker, outcomes


def test_interrupted_gui_publication_recovers_record_without_inference(setup):
    project, provider, _ = setup
    first, outcomes = run(project)
    assert outcomes[0].record_json
    reopened = Project.load(project.path)
    _, recovered = run(
        reopened, apply=True, prepare=Mock(side_effect=AssertionError("must recover"))
    )
    assert recovered == outcomes
    assert provider.call_count == 1
    assert (
        reopened.clips[0].analysis_records["face_embeddings"].provenance == "verified"
    )
    assert reopened.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert all(
            store.get_result(rid)["committed"] for rid in reopened.metadata.job_results
        )
    finally:
        store.close()


def test_semantic_reuse_avoids_prepare_and_survives_missing_receipts(
    setup, monkeypatch
):
    project, provider, _ = setup
    run(project, apply=True)
    project.save()
    assert project.metadata.job_results
    monkeypatch.setattr(JobStore, "get_result", lambda *args: None)
    worker, outcomes = run(
        project, apply=True, prepare=Mock(side_effect=AssertionError("must reuse"))
    )
    assert outcomes[0].status == "skipped"
    assert outcomes[0].clip_id in worker.cache.transient_outcomes
    assert provider.call_count == 1


@pytest.mark.parametrize("change", ["record", "value"])
def test_changed_saved_record_or_faces_does_not_acknowledge_receipt(setup, change):
    project, _, _ = setup
    run(project, apply=True)
    clip = project.clips[0]
    if change == "record":
        clip.analysis_records["face_embeddings"] = replace(
            clip.analysis_records["face_embeddings"], state="failed"
        )
    else:
        clip.face_embeddings = []
    project.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not any(
            store.get_result(rid)["committed"] for rid in project.metadata.job_results
        )
    finally:
        store.close()


def test_owned_failure_preserves_displayed_faces(setup):
    project, provider, _ = setup
    run(project, apply=True)
    before = project.clips[0].face_embeddings
    provider.side_effect = RuntimeError("failed inference")
    project.clips[0].analysis_records.clear()
    worker, outcomes = run(project, apply=True)
    assert outcomes[0].status == "failed"
    assert project.clips[0].face_embeddings == before
    assert outcomes[0].clip_id in worker.cache.transient_outcomes


def test_queued_model_change_is_rejected_before_inference(setup):
    from core.jobs.commits import StaleJobResult

    project, provider, directory = setup
    worker = FaceDetectionWorker(project.clips, project.sources_by_id, project=project)
    (directory / "recognition.onnx").write_bytes(b"changed")
    with pytest.raises(StaleJobResult):
        worker.cache.run(
            worker.tasks, Event(), lambda: True, lambda _: None, lambda *_: None
        )
    provider.assert_not_called()


def test_worker_emits_complete_records_without_mutating_project(setup):
    project, provider, _ = setup
    worker = FaceDetectionWorker(project.clips, project.sources_by_id, project=project)
    delivered = []
    worker.outcome_ready.connect(delivered.append)
    worker.run()
    assert worker.job_status == "completed"
    assert len(delivered) == 1 and delivered[0].record_json
    assert project.clips[0].face_embeddings is None
    assert provider.call_count == 1


def test_first_model_download_receipt_recovers_after_restart(setup, monkeypatch):
    project, provider, directory = setup
    contents = {path: path.read_bytes() for path in directory.glob("*.onnx")}
    for path in contents:
        path.unlink()
    original = provider.side_effect

    def compute(**kwargs):
        for path, content in contents.items():
            path.write_bytes(content)
        return original(**kwargs)

    provider.side_effect = compute
    _, first = run(project)
    assert first[0].status == "succeeded"
    _, recovered = run(
        Project.load(project.path),
        apply=True,
        prepare=Mock(side_effect=AssertionError("must recover")),
    )
    assert recovered == first
    assert provider.call_count == 1


def test_cancelled_queued_worker_drops_complete_outcomes(setup, monkeypatch):
    project, _, _ = setup
    worker = FaceDetectionWorker(project.clips, project.sources_by_id, project=project)
    delivered = []
    worker.outcome_ready.connect(delivered.append)
    original = worker.cache.run

    def compute(*args):
        held = []
        forwarded = list(args)
        forwarded[3] = held.append
        result = original(*forwarded)
        worker.cancel()
        for outcome in held:
            args[3](outcome)
        return result

    monkeypatch.setattr(worker.cache, "run", compute)
    worker.run()
    assert not delivered
    assert project.clips[0].face_embeddings is None


def test_queued_qt_delivery_authenticates_full_record(setup, tmp_path):
    import json
    import os
    import subprocess
    import sys
    from dataclasses import asdict

    project, _, _ = setup
    worker, outcomes = run(project)
    receipt = worker.cache.results[outcomes[0].clip_id]
    document = tmp_path / "delivery.json"
    document.write_text(
        json.dumps({"outcome": asdict(outcomes[0]), "receipt": asdict(receipt)})
    )
    code = r"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from dataclasses import replace
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from core.project import Project
from core.operations.faces import FaceOptions, FaceOutcome, face_task
from core.jobs.gui_results import GuiResultReceipt
from models.analysis_record import AnalysisRecord
from ui.workers.face_delivery import FaceDelivery
app = QCoreApplication([])
data = json.loads(Path(sys.argv[2]).read_text())
class Worker(QThread):
    outcome_ready = Signal(object)
    faces_ready = Signal(str, list)
    def is_cancelled(self): return self.cancelled
    def run(self):
        self.outcome_ready.emit(self.outcome)
        self.faces_ready.emit(self.outcome.clip_id, [])
        self.outcome_ready.emit(self.outcome)
for mode in ('current', 'cancel', 'edit', 'record', 'session', 'project', 'tampered', 'save_as'):
    project = Project.load(Path(sys.argv[1]))
    clip, source = project.clips[0], project.sources[0]
    worker = Worker()
    worker.cancelled = False
    worker.tasks = (face_task(clip, source),)
    worker.options = FaceOptions()
    worker.outcome = FaceOutcome.from_dict(data['outcome'])
    worker.cache = SimpleNamespace(path=project.path.resolve(), results={clip.id: GuiResultReceipt(**data['receipt'])})
    window = QObject()
    window.project = project
    window.face_detection_worker = worker
    window._on_face_detection_error = Mock()
    delivery = FaceDelivery(window, worker)
    if mode == 'cancel': worker.cancelled = True
    if mode == 'edit': clip.face_embeddings = []
    if mode == 'record': clip.analysis_records['face_embeddings'] = AnalysisRecord.legacy({'face_embeddings': []})
    if mode == 'session': project.clear()
    if mode == 'project': window.project = Project.new()
    if mode == 'tampered': worker.outcome = replace(worker.outcome, faces=())
    if mode == 'save_as': project.save(Path(sys.argv[1]).with_name('copy.sceneripper'))
    worker.start(); assert worker.wait(5000); app.processEvents()
    assert bool(project.metadata.job_results) == (mode == 'current'), mode
    if mode == 'current':
        assert clip.analysis_records['face_embeddings'].provenance == 'verified'
        assert clip.face_embeddings
        window._on_face_detection_error.assert_not_called()
    elif mode == 'edit': assert clip.face_embeddings == []
    else: assert clip.face_embeddings is None
    project.close_writer()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(project.path), str(document)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
