"""Durable alignment verifies records independently of computation receipts."""

from threading import Event
from unittest.mock import Mock
from types import SimpleNamespace
import sqlite3

import pytest

from core.jobs.alignment import run_alignment_job
from core.jobs.store import JobStore
from core.project import Project
from tests.test_alignment_records import setup as setup


@pytest.fixture
def saved(request, tmp_path, monkeypatch):
    project, provider = request.getfixturevalue("setup")
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready", lambda _: (True, [])
    )
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    path = tmp_path / "project.json"
    project.save(path)
    project.close_writer()
    store = JobStore(tmp_path / "jobs.db")
    yield path, store, provider
    store.close()


def run(saved, **kwargs):
    path, store, _ = saved
    return run_alignment_job(store, path, None, lambda *_: None, Event(), **kwargs)[
        "result"
    ]


def test_success_saves_verified_record_and_reuses(saved):
    assert len(run(saved)["succeeded"]) == 1
    project = Project.load(saved[0])
    assert project.clips[0].analysis_records["align_words"].provenance == "verified"
    project.close_writer()
    assert len(run(saved)["skipped"]) == 1
    assert saved[2].call_count == 1


@pytest.mark.parametrize("change", ["legacy", "media", "fps", "revision"])
def test_changed_inputs_recompute(saved, change, monkeypatch):
    run(saved)
    project = Project.load(saved[0])
    if change == "legacy":
        project.clips[0].analysis_records.pop("align_words")
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "fps":
        project.sources[0].fps = 24
    else:
        from core.analysis.alignment import ALIGNMENT_MODEL

        monkeypatch.setattr(
            "core.operations.alignment_records.alignment_model_revision", lambda: "r2"
        )
        saved[2].side_effect = (
            lambda *a, **kw: kw["on_execution"](
                {"backend": "ctc", "model": ALIGNMENT_MODEL, "revision": "r2"}
            )
            or []
        )
    project.save()
    project.close_writer()
    assert len(run(saved)["succeeded"]) == 1
    assert saved[2].call_count == 2


def test_missing_receipts_do_not_invalidate_verified_words(saved):
    run(saved)
    with sqlite3.connect(saved[0].parent / "jobs.db") as connection:
        connection.execute("DELETE FROM job_results")
    assert len(run(saved)["skipped"]) == 1
    assert saved[2].call_count == 1


def test_failure_record_preserves_prior_words(saved):
    run(saved)
    saved[2].side_effect = RuntimeError("provider failed")
    assert len(run(saved, force=True)["failed"]) == 1
    project = Project.load(saved[0])
    try:
        assert project.clips[0].analysis_records["align_words"].state == "failed"
        assert project.clips[0].transcript[0].words[0].text == "hello"
    finally:
        project.close_writer()


def test_first_model_load_recovers_failed_save_without_repeating_inference(
    saved, monkeypatch
):
    state = {"revision": None}
    monkeypatch.setattr(
        "core.operations.alignment_records.alignment_model_revision",
        lambda: state["revision"],
    )
    original = saved[2].side_effect

    def load(*a, **kw):
        state["revision"] = "r1"
        return original(*a, **kw)

    saved[2].side_effect = load
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "core.jobs.commits.save_with_mtime_check",
            Mock(side_effect=OSError("disk full")),
        )
        with pytest.raises(OSError):
            run(saved)
    assert len(run(saved)["succeeded"]) == 1
    assert saved[2].call_count == 1


def test_cancel_after_recording_keeps_receipt_without_publishing(saved, monkeypatch):
    path, store, provider = saved
    before = path.read_bytes()
    cancel = Event()
    original = store.record_result

    def record(*a, **kw):
        row = original(*a, **kw)
        cancel.set()
        return row

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "record_result", record)
        result = run_alignment_job(store, path, None, lambda *_: None, cancel)["result"]
    assert result["unprocessed"] == [{"clip_id": "c-0", "code": "cancelled"}]
    assert path.read_bytes() == before
    assert len(run(saved)["succeeded"]) == 1
    assert provider.call_count == 1
