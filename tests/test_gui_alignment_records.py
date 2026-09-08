"""Project-backed GUI alignment carries records through recovery and saving."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.operations.alignment import AlignmentApplication
from core.project import Project
from tests.test_alignment_records import setup as setup
from ui.workers.forced_alignment_worker import ForcedAlignmentWorker


@pytest.fixture
def saved(request, tmp_path, monkeypatch):
    project, provider = request.getfixturevalue("setup")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    project.save(tmp_path / "project.json")
    yield project, provider
    project.close_writer()


def worker_for(project):
    worker = ForcedAlignmentWorker(
        project.clips, project.sources_by_id, project=project
    )
    worker._prepare = Mock(return_value=True)
    return worker


def apply(project, worker):
    application = AlignmentApplication(project, worker.tasks)
    worker.run()
    assert application.apply(project, worker.result[0]), worker.result
    receipt = worker.cache.results.get(project.clips[0].id)
    if receipt is not None:
        project.record_job_result(receipt.result_id, receipt.digest)
    return receipt


def test_verified_gui_reuse_avoids_preparation(saved):
    project, provider = saved
    apply(project, worker_for(project))
    assert project.save()
    worker = worker_for(project)
    worker._prepare.side_effect = AssertionError("reuse must not prepare")
    assert apply(project, worker) is None
    assert worker.result[0].status == "skipped"
    assert worker.cache.transient_outcomes[project.clips[0].id]
    worker._prepare.assert_not_called()
    assert provider.call_count == 1


def test_reopen_recovers_record_before_delivery(saved):
    project, provider = saved
    first = worker_for(project)
    first.run()
    project.close_writer()
    reopened = Project.load(project.path)
    try:
        worker = worker_for(reopened)
        worker._prepare.side_effect = AssertionError("must recover")
        receipt = apply(reopened, worker)
        assert worker.result == first.result
        assert reopened.save()
        store = JobStore(project.path.parent / "jobs.db")
        try:
            assert store.get_result(receipt.result_id)["committed"]
        finally:
            store.close()
        assert provider.call_count == 1
    finally:
        reopened.close_writer()


def test_changed_alignment_record_prevents_save_acknowledgement(saved):
    project, _ = saved
    receipt = apply(project, worker_for(project))
    project.clips[0].analysis_records.pop("align_words")
    assert project.save()
    store = JobStore(project.path.parent / "jobs.db")
    try:
        assert not store.get_result(receipt.result_id)["committed"]
    finally:
        store.close()


def test_gui_failure_preserves_words_and_authenticates_failure(saved):
    project, provider = saved
    apply(project, worker_for(project))
    before = project.clips[0].transcript[0].to_dict()
    project.clips[0].analysis_records.pop("align_words")
    provider.side_effect = RuntimeError("offline")
    worker = worker_for(project)
    assert apply(project, worker) is None
    assert worker.result[0].status == "failed"
    assert project.clips[0].analysis_records["align_words"].state == "failed"
    assert project.clips[0].transcript[0].to_dict() == before
    assert worker.cache.transient_outcomes[project.clips[0].id]


def test_queued_model_change_rejects_before_preparation(saved, monkeypatch):
    project, provider = saved
    worker = worker_for(project)
    monkeypatch.setattr(
        "core.operations.alignment_records.alignment_model_revision", lambda: "r2"
    )
    worker.run()
    assert worker.job_status == "failed"
    worker._prepare.assert_not_called()
    provider.assert_not_called()


def test_first_model_load_receipt_recovers_without_inference(saved, monkeypatch):
    project, provider = saved
    state = {"revision": None}
    monkeypatch.setattr(
        "core.operations.alignment_records.alignment_model_revision",
        lambda: state["revision"],
    )
    original = provider.side_effect

    def load_model(*a, **kw):
        state["revision"] = "r1"
        return original(*a, **kw)

    provider.side_effect = load_model
    first = worker_for(project)
    first.run()
    assert first.result[0].status == "succeeded"
    again = worker_for(project)
    again._prepare.side_effect = AssertionError("must recover computed receipt")
    apply(project, again)
    assert provider.call_count == 1
    assert again.result == first.result
