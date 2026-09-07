"""Failure injection at the computed-result/project/checkpoint boundaries."""

from threading import Event

import pytest

from core.project import Project, ProjectSaveError
from core.jobs.commits import ResultSpec, StaleJobResult, commit_result
from core.jobs.store import JobStore


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "project.sceneripper"
    project = Project.new(name="0")
    assert project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    spec = ResultSpec.build(
        path,
        kind="increment",
        version=1,
        target_id="project",
        arguments={},
        inputs={"project_id": project.metadata.id},
    )
    counts = {"compute": 0, "apply": 0}

    def compute():
        counts["compute"] += 1
        return {"after": "1"}

    def apply(current, payload):
        counts["apply"] += 1
        current.edit_metadata(
            "project",
            {current.metadata.id: {"name": str(int(current.metadata.name) + 1)}},
        )

    kwargs = dict(
        compute=compute,
        apply=apply,
        validate_input=lambda p: p.metadata.id == project.metadata.id,
        is_applied=lambda p, value: p.metadata.name == value["after"],
    )
    return path, store, spec, counts, kwargs


def test_crash_after_project_save_reconciles_without_double_apply(setup, monkeypatch):
    path, store, spec, counts, kwargs = setup
    original = store.checkpoint_result
    monkeypatch.setattr(
        store,
        "checkpoint_result",
        lambda *a: (_ for _ in ()).throw(RuntimeError("checkpoint crash")),
    )
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        commit_result(store, spec, **kwargs)
    assert Project.load(path).metadata.name == "1"
    assert spec.result_id in Project.load(path).metadata.job_results
    assert not store.get_result(spec.result_id)["committed"]
    monkeypatch.setattr(store, "checkpoint_result", original)
    result = commit_result(store, spec, **kwargs)
    assert not result["applied"]
    assert counts == {"compute": 1, "apply": 1}
    assert store.get_result(spec.result_id)["committed"]


def test_failed_project_save_reuses_output_but_reloads_model(setup, monkeypatch):
    path, store, spec, counts, kwargs = setup
    original = Project.save
    monkeypatch.setattr(Project, "save", lambda *a, **k: False)
    with pytest.raises(ProjectSaveError):
        commit_result(store, spec, **kwargs)
    assert Project.load(path).metadata.name == "0"
    assert not Project.load(path).metadata.job_results
    monkeypatch.setattr(Project, "save", original)
    commit_result(store, spec, **kwargs)
    assert Project.load(path).metadata.name == "1"
    assert counts == {"compute": 1, "apply": 2}


def test_changed_committed_output_is_not_overwritten(setup):
    path, store, spec, counts, kwargs = setup
    commit_result(store, spec, **kwargs)
    project = Project.load(path)
    project.metadata.name = "9"
    assert project.save(path)
    with pytest.raises(StaleJobResult, match="output changed"):
        commit_result(store, spec, **kwargs)
    assert Project.load(path).metadata.name == "9"
    assert counts == {"compute": 1, "apply": 1}


def test_result_receipt_survives_ordinary_save_as(setup, tmp_path):
    path, store, spec, counts, kwargs = setup
    commit_result(store, spec, **kwargs)
    project = Project.load(path)
    copied = tmp_path / "copy.sceneripper"
    assert project.save(copied)
    assert Project.load(copied).metadata.job_results == project.metadata.job_results


def test_input_arguments_are_detached_and_bad_input_never_computes(setup):
    path, store, spec, counts, kwargs = setup
    arguments = {"nested": [1]}
    immutable = ResultSpec.build(
        path, kind="test", version=1, target_id="x", arguments=arguments, inputs={}
    )
    identity = immutable.result_id
    arguments["nested"].append(2)
    assert immutable.result_id == identity
    kwargs["validate_input"] = lambda p: False
    with pytest.raises(StaleJobResult):
        commit_result(store, spec, **kwargs)
    assert counts == {"compute": 0, "apply": 0}


def test_color_result_recovery_does_not_repeat_extraction(tmp_path, monkeypatch):
    from core.jobs.colors import run_colors
    from tests.test_spine_analyze import _build_project
    from unittest.mock import Mock

    path = tmp_path / "color.sceneripper"
    assert _build_project(tmp_path, 1).save(path)
    store = JobStore(tmp_path / "jobs.db")
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)
    original = store.checkpoint_result
    monkeypatch.setattr(
        store,
        "checkpoint_result",
        lambda *a: (_ for _ in ()).throw(RuntimeError("checkpoint crash")),
    )
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        run_colors(store, path, ["c-0"], 5, lambda *a: None, Event())
    monkeypatch.setattr(store, "checkpoint_result", original)
    result = run_colors(store, path, ["c-0"], 5, lambda *a: None, Event())
    assert result["result"]["skipped"][0]["reason"] == "already_committed"
    assert extract.call_count == 1
    assert Project.load(path).clips[0].dominant_colors == [(1, 2, 3)]


def test_missing_record_does_not_repeat_computation_for_committed_receipt(setup):
    path, store, spec, counts, kwargs = setup
    commit_result(store, spec, **kwargs)
    with store._connect() as conn:
        conn.execute("DELETE FROM job_results WHERE result_id=?", (spec.result_id,))
    with pytest.raises(StaleJobResult, match="missing"):
        commit_result(store, spec, **kwargs)
    assert counts == {"compute": 1, "apply": 1}


@pytest.mark.parametrize("change", ["media", "palette", "missing_cache"])
def test_color_retry_validates_saved_output(tmp_path, monkeypatch, change):
    from core.jobs.colors import run_colors
    from tests.test_spine_analyze import _build_project
    from unittest.mock import Mock

    path = tmp_path / "color.sceneripper"
    assert _build_project(tmp_path, 1).save(path)
    store = JobStore(tmp_path / "jobs.db")
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)

    def run():
        return run_colors(store, path, ["c-0"], 5, lambda *a: None, Event())

    run()
    if change == "media":
        (tmp_path / "video.mp4").write_bytes(b"changed media")
        extract.return_value = [(4, 5, 6)]
        assert len(run()["result"]["succeeded"]) == 1
        assert extract.call_count == 2
        assert Project.load(path).clips[0].dominant_colors == [(4, 5, 6)]
    else:
        if change == "palette":
            project = Project.load(path)
            project.clips[0].dominant_colors = [(9, 9, 9)]
            assert project.save(path)
        else:
            with store._connect() as conn:
                conn.execute("DELETE FROM job_results")
        with pytest.raises(StaleJobResult):
            run()
        assert extract.call_count == 1


def test_corrupt_cached_output_is_rejected_before_application(setup):
    path, store, spec, counts, kwargs = setup
    commit_result(store, spec, **kwargs)
    with store._connect() as conn:
        conn.execute("UPDATE job_results SET payload_json='{}'")
    with pytest.raises(ValueError, match="corrupt"):
        commit_result(store, spec, **kwargs)
    assert counts == {"compute": 1, "apply": 1}


@pytest.mark.parametrize("receipts", [[], {"bad": "digest"}, {"a" * 64: 1}])
def test_malformed_project_receipts_are_rejected(receipts):
    from core.project import ProjectMetadata

    with pytest.raises(ValueError, match="receipts"):
        ProjectMetadata.from_dict({"version": "1.5", "job_results": receipts})


def test_future_receipt_shape_does_not_prevent_read_only_inspection():
    from core.project import ProjectMetadata

    assert ProjectMetadata.from_dict(
        {"version": "99.0", "job_results": ["unknown future representation"]}
    ).job_results == {}
