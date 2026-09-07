"""Saved custom-query requests recover computation without duplicate appends."""

from threading import Event
import json
from unittest.mock import Mock, patch

import pytest

from core.jobs.commits import StaleJobResult
from core.jobs.custom_query import custom_query_job_spec, run_custom_query_job
from core.jobs.store import JobStore
from core.operations.custom_query import CustomQueryOptions
from core.project import Project
from tests.test_description_operations import project_with_thumbnails

OPTIONS = CustomQueryOptions("cloud", "model")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 2)
    assert project.save(tmp_path / "project.json")
    provider = Mock(return_value=(True, 0.9, "model"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    return project.path, JobStore(tmp_path / "jobs.db"), provider


def run(setup, **kwargs):
    path, store, _ = setup
    return run_custom_query_job(
        store,
        path,
        None,
        lambda *_: None,
        Event(),
        query="person",
        options=OPTIONS,
        **kwargs,
    )["result"]


def test_save_failure_reuses_results_without_duplicate_append(setup):
    path, store, provider = setup
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup)
    assert Project.load(path).clips[0].custom_queries is None
    provider.side_effect = AssertionError("Must reuse inference")
    assert len(run(setup)["succeeded"]) == 2
    assert provider.call_count == 2
    saved = Project.load(path)
    assert all(len(c.custom_queries) == 1 for c in saved.clips)
    assert all(store.get_result(rid)["committed"] for rid in saved.metadata.job_results)


def test_checkpoint_retry_does_not_append_but_later_request_refreshes(setup):
    path, store, provider = setup
    with patch.object(
        store, "checkpoint_results", side_effect=RuntimeError("checkpoint")
    ):
        with pytest.raises(RuntimeError, match="checkpoint"):
            run(setup)
    assert len(run(setup)["skipped"]) == 2
    assert provider.call_count == 2
    assert all(len(c.custom_queries) == 1 for c in Project.load(path).clips)
    assert len(run(setup)["succeeded"]) == 2
    assert provider.call_count == 4
    assert all(len(c.custom_queries) == 2 for c in Project.load(path).clips)


def test_manual_edit_is_preserved_as_input_to_new_append(setup):
    path, _, provider = setup
    run(setup)
    project = Project.load(path)
    project.clips[0].custom_queries[0]["match"] = False
    assert project.save()
    run(setup)
    assert Project.load(path).clips[0].custom_queries[0]["match"] is False
    assert provider.call_count == 4


def test_missing_committed_payload_never_recomputes(setup):
    path, store, provider = setup
    run(setup)
    with store._connect() as connection:
        connection.execute("DELETE FROM job_results")
    with pytest.raises(StaleJobResult, match="missing"):
        run(setup)
    assert provider.call_count == 2


def test_corrupt_committed_identity_blocks_further_inference(setup):
    _, store, provider = setup
    run(setup)
    with store._connect() as connection:
        row = connection.execute(
            "SELECT result_id, spec_json FROM job_results LIMIT 1"
        ).fetchone()
        identity = json.loads(row[1])
        identity["arguments"]["model"] = "tampered"
        connection.execute(
            "UPDATE job_results SET spec_json=? WHERE result_id=?",
            (json.dumps(identity), row[0]),
        )
    with pytest.raises(StaleJobResult, match="identity is corrupt"):
        run(setup)
    assert provider.call_count == 2


def test_queued_media_change_rejected_before_provider(setup):
    path, _, provider = setup
    project = Project.load(path)
    operation = custom_query_job_spec(
        project, None, OPTIONS, arguments={"query": "person"}
    )
    project.clips[0].thumbnail_path.write_bytes(b"changed")
    with pytest.raises(StaleJobResult, match="queued"):
        run(setup, operation=operation)
    provider.assert_not_called()


def test_operation_pins_query_and_provider(setup):
    path, _, provider = setup
    operation = custom_query_job_spec(
        Project.load(path), None, OPTIONS, arguments={"query": "other"}
    )
    run(setup, operation=operation)
    assert provider.call_args.kwargs["query"] == "other"
    assert provider.call_args.kwargs["model_name"] == "model"


def test_large_request_has_no_partial_generation_after_save_failure(
    tmp_path, monkeypatch
):
    project = project_with_thumbnails(tmp_path, 18)
    assert project.save(tmp_path / "project.json")
    provider = Mock(return_value=(True, 0.9, "model"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    setup = project.path, JobStore(tmp_path / "jobs.db"), provider
    with patch(
        "core.jobs.commits.save_with_mtime_check",
        side_effect=RuntimeError("save failed"),
    ):
        with pytest.raises(RuntimeError, match="save failed"):
            run(setup)
    assert all(c.custom_queries is None for c in Project.load(project.path).clips)
    run(setup)
    assert provider.call_count == 18
    assert all(len(c.custom_queries) == 1 for c in Project.load(project.path).clips)


def test_cancelled_provider_output_is_not_recorded(setup):
    path, store, provider = setup
    cancel = Event()

    def compute(**kwargs):
        cancel.set()
        return True, 0.9, "model"

    provider.side_effect = compute
    result = run_custom_query_job(
        store, path, None, lambda *_: None, cancel, query="person", options=OPTIONS
    )
    assert len(result["result"]["unprocessed"]) == 2
    assert not Project.load(path).metadata.job_results


def test_large_request_publishes_in_one_save(tmp_path, monkeypatch):
    from core.jobs.commits import save_with_mtime_check

    project = project_with_thumbnails(tmp_path, 18)
    assert project.save(tmp_path / "project.json")
    provider = Mock(return_value=(True, 0.9, "model"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    setup = project.path, JobStore(tmp_path / "jobs.db"), provider
    with patch(
        "core.jobs.commits.save_with_mtime_check", wraps=save_with_mtime_check
    ) as save:
        run(setup)
    save.assert_called_once()
    assert provider.call_count == 18


def test_later_analysis_failure_retains_query_receipts(setup, monkeypatch):
    from core.jobs.analysis import analysis_job_spec, run_analysis_job
    from core.spine.analyze import ANALYZE_CLIP_OPERATION_MAP

    path, store, provider = setup
    monkeypatch.setattr("core.operations.custom_query.resolve_options", lambda: OPTIONS)
    operation = analysis_job_spec(
        Project.load(path),
        arguments={"operations": ["custom_query", "colors"], "query": "person"},
    )
    monkeypatch.setitem(
        ANALYZE_CLIP_OPERATION_MAP,
        "colors",
        Mock(side_effect=RuntimeError("later step")),
    )
    with pytest.raises(RuntimeError, match="later step"):
        run_analysis_job(store, path, operation, lambda *_: None, Event())
    assert len(Project.load(path).metadata.job_results) == 2
    assert provider.call_count == 2
