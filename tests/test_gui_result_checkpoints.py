"""Project publication precedes GUI result checkpoint acknowledgement."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs.store import JobStore
from core.transcription_models import TranscriptSegment, WordTimestamp
from tests.test_spine_analyze import _build_project
from tests.test_gui_transcription_recovery import run as transcribe
from tests.test_gui_alignment_recovery import run as align


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["transcription", "alignment"])
def setup(request, tmp_path, monkeypatch):
    project = _build_project(tmp_path, 2)
    if request.param == "alignment":
        for clip in project.clips:
            clip.transcript = [TranscriptSegment(0, 1, "hello", language="en")]
    assert project.save(tmp_path / "project.json")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    compute = Mock(return_value=[TranscriptSegment(0, 1, "hello", language="en")])
    monkeypatch.setattr("core.transcription.transcribe_clip", compute)

    def extract(*args, **kwargs):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake")
        return wav

    monkeypatch.setattr("core.analysis.alignment.extract_audio_to_wav", extract)
    if request.param == "alignment":
        compute = Mock(return_value=[WordTimestamp(0, 1, "hello", 0.9)])
        monkeypatch.setattr("core.analysis.alignment.align_words", compute)

    def run():
        if request.param == "alignment":
            return align(project, force=True, apply=True)
        return transcribe(project, apply=True)

    run()
    store = JobStore(tmp_path / "jobs.db")
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )
    return project, store, compute, run


def test_save_acknowledges_matching_gui_receipts(setup):
    project, store, _, _ = setup
    assert project.save()
    saved = json.loads(project.path.read_text())
    assert saved["job_results"] == project.metadata.job_results
    assert all(store.get_result(rid)["committed"] for rid in saved["job_results"])


def test_failed_project_write_does_not_checkpoint(setup, monkeypatch):
    project, store, _, _ = setup
    checkpoint = Mock()
    monkeypatch.setattr(JobStore, "checkpoint_results", checkpoint)
    monkeypatch.setattr(
        "core.project_lock.replace_project_file", Mock(side_effect=OSError("disk full"))
    )
    assert not project.save()
    checkpoint.assert_not_called()
    assert not json.loads(project.path.read_text())["job_results"]


def test_failed_checkpoint_preserves_save_and_retries_without_inference(
    setup, monkeypatch
):
    project, store, compute, _ = setup
    with monkeypatch.context() as patch:
        patch.setattr(
            JobStore,
            "checkpoint_results",
            Mock(side_effect=RuntimeError("checkpoint failed")),
        )
        assert project.save()
    assert len(json.loads(project.path.read_text())["job_results"]) == 2
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )
    assert project.save()
    assert compute.call_count == 2
    assert all(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


def test_edited_output_is_preserved_without_false_acknowledgement(setup):
    project, store, _, _ = setup
    project.clips[0].transcript[0].text = "manual edit"
    assert project.save()
    for rid in project.metadata.job_results:
        row = store.get_result(rid)
        cid = json.loads(row["spec_json"])["target_id"]
        assert bool(row["committed"]) == (cid == project.clips[1].id)
    assert (
        json.loads(project.path.read_text())["clips"][0]["transcript"][0]["text"]
        == "manual edit"
    )


def test_save_as_does_not_acknowledge_original_project_location(setup, tmp_path):
    project, store, _, _ = setup
    assert project.save(tmp_path / "copy.json")
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


def test_all_identical_refreshes_are_acknowledged(setup):
    project, store, _, run = setup
    run()
    run()
    assert len(project.metadata.job_results) == 6
    assert project.save()
    assert all(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


def test_background_save_acknowledges_only_its_snapshot(setup, qapp):
    from ui.main_window import SaveProjectWorker

    project, store, _, _ = setup
    snapshot = project.snapshot_for_save()
    excluded = next(
        rid
        for rid in snapshot["metadata"].job_results
        if json.loads(store.get_result(rid)["spec_json"])["target_id"]
        == project.clips[1].id
    )
    del snapshot["metadata"].job_results[excluded]
    snapshot["clips"][1].transcript = None
    worker = SaveProjectWorker(snapshot, project.path)
    finished = []
    worker.save_finished.connect(lambda success, *_: finished.append(success))
    worker.run()
    assert finished == [True]
    assert len(project.metadata.job_results) == 2
    for rid in project.metadata.job_results:
        assert bool(store.get_result(rid)["committed"]) == (rid != excluded)


def test_corrupt_gui_payload_does_not_invalidate_successful_save(setup, caplog):
    project, store, _, _ = setup
    rid = next(iter(project.metadata.job_results))
    with store._connect() as connection:
        connection.execute(
            "UPDATE job_results SET payload_json='{}' WHERE result_id=?", (rid,)
        )
    assert project.save()
    assert (
        json.loads(project.path.read_text())["job_results"]
        == project.metadata.job_results
    )
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )
    assert "checkpoints remain pending" in caplog.text


def test_headless_result_receipts_are_not_checkpointed_here(setup):
    from hashlib import sha256
    from core.jobs.commits import ResultSpec

    project, store, _, _ = setup
    spec = ResultSpec.build(
        project.path,
        kind="transcribe",
        version=1,
        target_id=project.clips[0].id,
        arguments={},
        inputs={},
    )
    digest = sha256(b"{}").hexdigest()
    store.record_result(spec.result_id, spec.identity_json, "{}", digest)
    project.record_job_result(spec.result_id, digest)
    assert project.save()
    assert not store.get_result(spec.result_id)["committed"]
    assert all(
        store.get_result(rid)["committed"]
        for rid in project.metadata.job_results
        if rid != spec.result_id
    )


def test_unrelated_project_identity_does_not_acknowledge_receipts(setup):
    project, store, _, _ = setup
    project.metadata.id = "different-project"
    assert project.save()
    assert not any(
        store.get_result(rid)["committed"] for rid in project.metadata.job_results
    )


def test_save_without_receipts_does_not_open_job_cache(tmp_path, monkeypatch):
    settings = Mock(side_effect=AssertionError("No cache needed"))
    monkeypatch.setattr("core.settings.load_settings", settings)
    project = _build_project(tmp_path, 1)
    assert project.save(tmp_path / "project.json")
    settings.assert_not_called()


def test_pending_result_lookup_batches_ids_with_one_connection(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    ids = [f"{i:064x}" for i in range(502)]
    with store._connect() as connection:
        connection.executemany(
            "INSERT INTO job_results VALUES (?, '{}', '{}', ?, ?, 0)",
            [(rid, "0" * 64, int(index == 0)) for index, rid in enumerate(ids)],
        )
    connect = Mock(wraps=store._connect)
    monkeypatch.setattr(store, "_connect", connect)
    rows = store.get_pending_results(ids[:501] + ["unknown"])
    assert {row["result_id"] for row in rows} == set(ids[1:501])
    connect.assert_called_once()
