"""Batch publication keeps per-item identities and crash recovery guarantees."""

from threading import Event
from unittest.mock import Mock

import pytest

from core.jobs.colors import run_colors
from core.jobs.store import JobStore
from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.project import Project, ProjectSaveError
from tests.test_spine_analyze import _build_project


def setup_colors(tmp_path, monkeypatch, count):
    path = tmp_path / "colors.sceneripper"
    assert _build_project(tmp_path, count).save(path)
    store = JobStore(tmp_path / "jobs.db")
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)
    return path, store, extract


def run(store, path, cancel=None):
    return run_colors(store, path, None, 5, lambda *a: None, cancel or Event())


def test_failed_color_attempt_is_saved_without_success_receipt(tmp_path, monkeypatch):
    path, store, extract = setup_colors(tmp_path, monkeypatch, 1)
    try:
        run(store, path)
        previous = Project.load(path)
        receipts = dict(previous.metadata.job_results)
        previous.close_writer()
        extract.side_effect = RuntimeError("decode failed")
        result = run_colors(store, path, None, 6, lambda *a: None, Event())
        assert len(result["result"]["failed"]) == 1
        loaded = Project.load(path)
        assert loaded.clips[0].analysis_records["colors"].state == "failed"
        assert loaded.clips[0].dominant_colors == [(1, 2, 3)]
        assert loaded.metadata.job_results == receipts
        loaded.close_writer()
    finally:
        store.close()


def test_staged_analysis_rechecks_inputs_before_saving(tmp_path, monkeypatch):
    from models.analysis_record import AnalysisRecord

    path, store, _ = setup_colors(tmp_path, monkeypatch, 1)
    original = path.read_bytes()
    record = AnalysisRecord.legacy({"dominant_colors": []})
    valid = True
    try:
        with pytest.raises(StaleJobResult, match="Analysis inputs"):
            with result_batch(store, path) as batch:
                batch.stage_analysis(
                    apply=lambda project: project.record_analysis("clip", "c-0", "colors", record),
                    validate_input=lambda project: valid,
                    is_applied=lambda project: project.clips[0].analysis_records.get("colors") == record,
                )
                valid = False
        assert path.read_bytes() == original
    finally:
        store.close()


def test_color_project_serialization_is_batched(tmp_path, monkeypatch):
    path, store, extract = setup_colors(tmp_path, monkeypatch, 33)
    original = Project.save
    saves = []

    def save(project, *args, **kwargs):
        saves.append(len(project.metadata.job_results))
        return original(project, *args, **kwargs)

    monkeypatch.setattr(Project, "save", save)
    result = run(store, path)
    assert len(result["result"]["succeeded"]) == 33
    assert saves == [16, 32, 33]
    assert extract.call_count == 33


def test_failed_batch_save_reuses_all_recorded_computations(tmp_path, monkeypatch):
    path, store, extract = setup_colors(tmp_path, monkeypatch, 3)
    original = Project.save
    monkeypatch.setattr(Project, "save", lambda *a, **k: False)
    with pytest.raises(ProjectSaveError):
        run(store, path)
    assert all(c.dominant_colors is None for c in Project.load(path).clips)
    assert not Project.load(path).metadata.job_results
    assert extract.call_count == 3
    monkeypatch.setattr(Project, "save", original)
    run(store, path)
    assert extract.call_count == 3
    assert len(Project.load(path).metadata.job_results) == 3


def test_checkpoint_failure_reconciles_entire_saved_batch(tmp_path, monkeypatch):
    path, store, extract = setup_colors(tmp_path, monkeypatch, 3)
    original = store.checkpoint_results
    monkeypatch.setattr(
        store, "checkpoint_results", Mock(side_effect=RuntimeError("checkpoint crash"))
    )
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        run(store, path)
    project = Project.load(path)
    assert len(project.metadata.job_results) == 3
    assert all(c.dominant_colors == [(1, 2, 3)] for c in project.clips)
    assert all(
        not store.get_result(key)["committed"] for key in project.metadata.job_results
    )
    monkeypatch.setattr(store, "checkpoint_results", original)
    save = Mock(side_effect=AssertionError("reconciled batch must not save again"))
    monkeypatch.setattr(Project, "save", save)
    result = run(store, path)
    assert len(result["result"]["skipped"]) == 3
    assert extract.call_count == 3
    save.assert_not_called()
    assert all(
        store.get_result(key)["committed"] for key in project.metadata.job_results
    )


def test_cancel_flushes_successes_in_final_partial_batch(tmp_path, monkeypatch):
    path, store, extract = setup_colors(tmp_path, monkeypatch, 4)
    cancel = Event()

    def compute(**kwargs):
        if extract.call_count == 2:
            cancel.set()
        return [(1, 2, 3)]

    extract.side_effect = compute
    result = run(store, path, cancel)
    assert len(result["result"]["succeeded"]) == 2
    assert len(result["result"]["unprocessed"]) == 2
    project = Project.load(path)
    assert [c.dominant_colors for c in project.clips] == [
        [(1, 2, 3)],
        [(1, 2, 3)],
        None,
        None,
    ]
    assert len(project.metadata.job_results) == 2


def test_flush_revalidates_earlier_targets(tmp_path, monkeypatch):
    from dataclasses import replace

    path, store, extract = setup_colors(tmp_path, monkeypatch, 2)
    project = Project.load(path)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"other source")
    project.add_source(replace(project.sources[0], id="src-2", file_path=other))
    project.clips[1].source_id = "src-2"
    assert project.save(path)

    def compute(**kwargs):
        if kwargs["video_path"] == other:
            (tmp_path / "video.mp4").write_bytes(b"earlier input changed")
        return [(1, 2, 3)]

    extract.side_effect = compute
    with pytest.raises(StaleJobResult, match="Batch inputs"):
        run(store, path)
    restored = Project.load(path)
    assert all(c.dominant_colors is None for c in restored.clips)
    assert not restored.metadata.job_results


def test_checkpoint_group_rolls_back_when_a_record_is_missing(tmp_path):
    from hashlib import sha256

    store = JobStore(tmp_path / "jobs.db")
    digest = sha256(b"{}").hexdigest()
    store.record_result("a" * 64, "{}", "{}", digest)
    with pytest.raises(ValueError, match="missing"):
        store.checkpoint_results([("a" * 64, digest), ("b" * 64, digest)])
    assert not store.get_result("a" * 64)["committed"]


def test_caught_apply_error_cannot_publish_partial_mutation(tmp_path, monkeypatch):
    path, store, _ = setup_colors(tmp_path, monkeypatch, 1)
    before = path.read_bytes()

    def spec(target):
        return ResultSpec.build(
            path, kind="test", version=1, target_id=target, arguments={}, inputs={}
        )

    def fail_after_mutation(project, payload):
        project.edit_metadata("project", {project.metadata.id: {"name": "broken"}})
        raise ValueError("apply failed")

    with pytest.raises(RuntimeError, match="failed"):
        with result_batch(store, path) as batch:
            batch.commit(
                spec("clip"),
                compute=lambda: {"notes": "applied"},
                validate_input=lambda p: True,
                apply=lambda p, value: p.edit_metadata("clip", {"c-0": value}),
                is_applied=lambda p, value: p.clips[0].notes == value["notes"],
            )
            with pytest.raises(ValueError, match="apply failed"):
                batch.commit(
                    spec("project"),
                    compute=lambda: {},
                    validate_input=lambda p: True,
                    apply=fail_after_mutation,
                    is_applied=lambda p, value: True,
                )
    assert path.read_bytes() == before


def test_mutated_payload_cannot_be_saved_under_original_digest(tmp_path, monkeypatch):
    path, store, _ = setup_colors(tmp_path, monkeypatch, 1)
    before = path.read_bytes()
    spec = ResultSpec.build(
        path, kind="test", version=1, target_id="project", arguments={}, inputs={}
    )

    def mutate(project, payload):
        payload["name"] = "tampered"
        project.edit_metadata("project", {project.metadata.id: payload})

    with pytest.raises(StaleJobResult, match="payload"):
        with result_batch(store, path) as batch:
            batch.commit(
                spec,
                compute=lambda: {"name": "computed"},
                validate_input=lambda p: True,
                apply=mutate,
                is_applied=lambda p, value: p.metadata.name == value["name"],
            )
    assert path.read_bytes() == before
