"""Each query owns provenance independently of other appended query results."""

from dataclasses import replace

import pytest

from core.operations.custom_query import (
    CustomQueryApplication,
    CustomQueryOptions,
    custom_query_record_key,
    custom_query_task,
    run_custom_query,
)
from core.project import Project
from tests.test_description_operations import project_with_thumbnails


OPTIONS = CustomQueryOptions("cloud", "model")


def evaluate(project, query="person", *, reuse=True, options=OPTIONS):
    task = custom_query_task(
        project.clips[0], project.sources[0], query, skip_existing=reuse
    )
    outcome = run_custom_query((task,), options)[0]
    assert CustomQueryApplication(project, (task,)).apply(project, outcome)
    return outcome


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return False, 0.0, "model"

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    return project, calls


def test_negative_result_reuses_without_duplicate_history(setup):
    project, calls = setup
    assert evaluate(project).status == "succeeded"
    assert evaluate(project).status == "skipped"
    assert len(calls) == 1
    assert len(project.clips[0].custom_queries) == 1
    assert project.clips[0].custom_queries[0]["match"] is False


def test_other_queries_do_not_invalidate_previous_query(setup):
    project, calls = setup
    evaluate(project, "person")
    evaluate(project, "cat")
    assert evaluate(project, "person").status == "skipped"
    assert len(calls) == 2
    clip = project.clips[0]
    assert custom_query_record_key("person") in clip.analysis_records
    assert custom_query_record_key("cat") in clip.analysis_records


def test_explicit_refresh_appends_another_result(setup):
    project, calls = setup
    evaluate(project)
    evaluate(project, reuse=False)
    assert len(calls) == len(project.clips[0].custom_queries) == 2


@pytest.mark.parametrize(
    "change", ["image", "source", "range", "fps", "model", "projection"]
)
def test_changed_inputs_require_computation(setup, change):
    project, calls = setup
    evaluate(project)
    options = OPTIONS
    if change == "image":
        project.clips[0].thumbnail_path.write_bytes(b"different")
    elif change == "source":
        project.sources[0].file_path.write_bytes(b"different")
    elif change == "range":
        project.clips[0].end_frame += 1
    elif change == "fps":
        project.sources[0].fps += 1
    elif change == "model":
        options = replace(options, model="other")
    else:
        project.clips[0].custom_queries[-1]["match"] = True
    assert evaluate(project, options=options).status == "succeeded"
    assert len(calls) == 2


def test_failure_keeps_history_and_only_invalidates_its_query(setup, monkeypatch):
    project, _ = setup
    evaluate(project, "person")
    evaluate(project, "cat")

    def fail(**kwargs):
        raise RuntimeError("Invalid answer")

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", fail)
    assert evaluate(project, "person", reuse=False).status == "failed"
    clip = project.clips[0]
    assert clip.analysis_records[custom_query_record_key("person")].state == "failed"
    assert clip.analysis_records[custom_query_record_key("cat")].state == "succeeded"
    assert len(clip.custom_queries) == 2


def test_records_survive_project_round_trip(setup, tmp_path):
    project, calls = setup
    evaluate(project)
    assert project.save(tmp_path / "project.json")
    reopened = Project.load(project.path)
    assert evaluate(reopened).status == "skipped"
    assert len(calls) == 1


def test_legacy_history_does_not_prove_reuse(setup):
    project, calls = setup
    project.clips[0].custom_queries = [
        {"query": "person", "match": True, "confidence": 0.9, "model": "model"}
    ]
    assert evaluate(project).status == "succeeded"
    assert len(calls) == 1
    assert len(project.clips[0].custom_queries) == 2


def test_failed_record_requires_retry_even_with_previous_history(setup, monkeypatch):
    project, calls = setup
    evaluate(project)

    def fail(**kwargs):
        raise RuntimeError("Invalid answer")

    with monkeypatch.context() as failing:
        failing.setattr("core.analysis.custom_query.evaluate_custom_query", fail)
        assert evaluate(project, reuse=False).status == "failed"
    assert evaluate(project).status == "succeeded"
    assert len(calls) == 2


def test_media_changed_during_inference_has_no_publishable_record(setup, monkeypatch):
    project, _ = setup

    def mutate(**kwargs):
        project.sources[0].file_path.write_bytes(b"changed during inference")
        return True, 0.9, "model"

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", mutate)
    task = custom_query_task(project.clips[0], project.sources[0], "person")
    outcome = run_custom_query((task,), OPTIONS)[0]
    assert outcome.status == "failed"
    assert outcome.record_json is None


def test_parallelism_does_not_invalidate_query_reuse(setup):
    project, calls = setup
    evaluate(project)
    assert (
        evaluate(project, options=replace(OPTIONS, parallelism=4)).status == "skipped"
    )
    assert len(calls) == 1


def test_headless_reuse_selects_its_query_record(setup, monkeypatch):
    from core.spine.analyze import custom_query

    project, calls = setup
    monkeypatch.setattr(
        "core.operations.custom_query.resolve_options", lambda *args: OPTIONS
    )
    assert len(custom_query(project, query="person")["result"]["succeeded"]) == 1
    assert len(custom_query(project, query="cat")["result"]["succeeded"]) == 1
    result = custom_query(project, query="person", skip_existing=True)["result"]
    assert result["skipped"] == [{"clip_id": "c-0", "reason": "valid_analysis"}]
    assert len(calls) == len(project.clips[0].custom_queries) == 2
