"""Custom-query scheduling, cancellation and provider snapshot contracts."""

from threading import Event, Thread, get_ident
from unittest.mock import Mock

import pytest

from core.operations.custom_query import (
    CustomQueryOptions,
    CustomQueryTask,
    run_custom_query,
)
from core.spine.analyze import custom_query
from tests.test_description_operations import project_with_thumbnails
from ui.workers.custom_query_worker import CustomQueryWorker


@pytest.mark.parametrize(
    "response",
    [
        ("false", 0.5, "model"),
        (0, 0.5, "model"),
        (True, True, "model"),
        (True, float("nan"), "model"),
        (False, float("inf"), "model"),
        (False, -0.1, "model"),
        (True, 1.1, "model"),
        (True, "0.5", "model"),
        (True, 0.5, ""),
    ],
)
def test_invalid_provider_results_are_not_published(tmp_path, monkeypatch, response):
    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr(
        "core.analysis.custom_query.evaluate_custom_query", lambda **kwargs: response
    )
    result = custom_query(project, query="person", tier="cloud")["result"]
    assert result["succeeded"] == []
    assert len(result["failed"]) == 1
    assert not project.clips[0].custom_queries


def test_unparseable_provider_answer_is_failure_not_negative(tmp_path, monkeypatch):
    from core.analysis.custom_query import _parse_yes_no_response

    project = project_with_thumbnails(tmp_path, 1)

    def provider(**kwargs):
        match, confidence = _parse_yes_no_response("I'm not sure about that")
        return match, confidence, "model"

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    result = custom_query(project, query="person", tier="cloud")["result"]
    assert len(result["failed"]) == 1
    assert not project.clips[0].custom_queries


def test_gui_spine_share_options_and_trim_query(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    provider = Mock(return_value=(True, 0.875, "model"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    worker = CustomQueryWorker(project.clips, "  person  ", {}, tier="cloud")
    worker.run()
    result = custom_query(project, query="  person  ", tier="cloud")
    assert provider.call_args_list[0] == provider.call_args_list[1]
    assert worker.result[0].query == "person"
    assert result["result"]["succeeded"][0]["match"] is True


@pytest.mark.parametrize("tier", ["local", "cpu"])
def test_direct_local_options_keep_inference_on_caller_thread(
    tmp_path, monkeypatch, tier
):
    project = project_with_thumbnails(tmp_path, 1)
    owner = get_ident()
    calls = []

    def provider(**kwargs):
        calls.append(get_ident())
        assert kwargs["tier"] == "local"
        return True, 0.9, "model"

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    outcomes = run_custom_query(
        (CustomQueryTask("c-0", project.clips[0].thumbnail_path, "person"),),
        CustomQueryOptions(tier, "model", 5),
    )
    assert calls == [owner]
    assert outcomes[0].status == "succeeded"


@pytest.mark.parametrize("tier,limit", [("cloud", 2), ("local", 1)])
def test_cancel_bounds_admission_and_suppresses_inflight(
    tmp_path, monkeypatch, tier, limit
):
    project = project_with_thumbnails(tmp_path, 5)
    entered, release, cancel = Event(), Event(), Event()
    calls = []
    delivered = []

    def provider(**kwargs):
        calls.append(get_ident())
        if len(calls) == limit:
            entered.set()
        assert release.wait(5)
        return True, 0.9, "model"

    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    tasks = tuple(
        CustomQueryTask(c.id, c.thumbnail_path, "person") for c in project.clips
    )
    results = []
    thread = Thread(
        target=lambda: results.extend(
            run_custom_query(
                tasks,
                CustomQueryOptions(tier, "model", 2),
                cancel_event=cancel,
                on_outcome=delivered.append,
            )
        )
    )
    thread.start()
    try:
        assert entered.wait(5)
        cancel.set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(calls) == limit
    assert not delivered
    assert all(o.status == "unprocessed" for o in results)


def test_cancelled_provider_error_is_not_retried(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    cancel = Event()

    def provider(**kwargs):
        cancel.set()
        raise RuntimeError("429 rate limit")

    compute = Mock(side_effect=provider)
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", compute)
    result = run_custom_query(
        (CustomQueryTask("c-0", project.clips[0].thumbnail_path, "person"),),
        CustomQueryOptions("cloud", "model"),
        cancel_event=cancel,
    )
    assert result[0].status == "unprocessed"
    compute.assert_called_once()


def test_retry_wait_uses_cancellation_event(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)

    class CancelOnWait(Event):
        def wait(self, timeout=None):
            assert timeout == 2
            self.set()
            return True

    provider = Mock(side_effect=RuntimeError("429 rate limit"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    result = run_custom_query(
        (CustomQueryTask("c-0", project.clips[0].thumbnail_path, "person"),),
        CustomQueryOptions("local", "model"),
        cancel_event=CancelOnWait(),
    )
    assert result[0].status == "unprocessed"
    provider.assert_called_once()


def test_preload_failure_completes_once(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    worker = CustomQueryWorker(project.clips, "person", {}, tier="local")
    monkeypatch.setattr("core.analysis.description.is_model_loaded", lambda *_: False)
    monkeypatch.setattr(
        "core.analysis.description._load_local_model",
        Mock(side_effect=RuntimeError("load failed")),
    )
    completed, errors = [], []
    worker.analysis_completed.connect(lambda: completed.append(True))
    worker.error.connect(errors.append)
    worker.run()
    assert completed == [True]
    assert len(errors) == 1


def test_worker_keeps_model_snapshot(tmp_path, monkeypatch):
    from core.settings import Settings

    settings = Settings(description_model_cloud="first")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = project_with_thumbnails(tmp_path, 1)
    worker = CustomQueryWorker(project.clips, "person", {}, tier="cloud")
    settings.description_model_cloud = "second"
    provider = Mock(return_value=(False, 0.1, "first"))
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    worker.run()
    assert provider.call_args.kwargs["model_name"] == "first"


def test_spine_skip_uses_trimmed_query(tmp_path, monkeypatch):
    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].custom_queries = [{"query": "person", "match": True}]
    provider = Mock()
    monkeypatch.setattr("core.analysis.custom_query.evaluate_custom_query", provider)
    result = custom_query(project, query=" person ", tier="cloud", skip_existing=True)
    assert result["result"]["skipped"] == [
        {"clip_id": "c-0", "reason": "already_populated"}
    ]
    provider.assert_not_called()
