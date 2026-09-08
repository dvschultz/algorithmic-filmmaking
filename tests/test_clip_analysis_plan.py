"""Ordering, cancellation and truthful completion for combined clip analysis."""

import pytest

from core.operations.clip_analysis import ClipAnalysisPlan


def test_phase_reservations_survive_synchronous_skips():
    plan = ClipAnalysisPlan(
        ["a"], ["describe", "colors", "shots", "gaze", "transcribe"]
    )
    assert plan.begin_ready() == ("colors", "shots")
    assert plan.finish("colors", {"a": "skipped"})
    assert plan.begin_ready() == ()
    assert plan.finish("shots", {"a": "succeeded"})
    assert plan.begin_ready() == ("gaze",)
    assert plan.finish("gaze", {"a": "succeeded"})
    assert plan.begin_ready() == ("transcribe",)
    assert plan.finish("transcribe", {"a": "succeeded"})
    assert plan.begin_ready() == ("describe",)
    assert plan.finish("describe", {"a": "succeeded"})
    assert plan.begin_ready() == ()
    assert plan.finished and plan.successful_ids() == ("a",)


def test_duplicate_and_out_of_order_completion_do_not_consume_work():
    plan = ClipAnalysisPlan(["a", "a"], ["colors", "colors", "describe"])
    assert not plan.finish("describe", {"a": "succeeded"})
    assert plan.begin_ready() == ("colors",)
    assert plan.begin_ready() == ()
    assert plan.finish("colors", {"a": "succeeded"})
    assert not plan.finish("colors", {"a": "failed"})
    assert plan.begin_ready() == ("describe",)


def test_cancel_retains_running_operations_until_they_settle():
    plan = ClipAnalysisPlan(["a"], ["colors", "shots", "describe"])
    assert plan.begin_ready() == ("colors", "shots")
    plan.cancel()
    assert not plan.finished and not plan.begin_ready()
    plan.finish("colors", {"a": "succeeded"})
    assert not plan.finished
    plan.finish("shots", {})
    assert plan.finished and not plan.begin_ready()
    assert plan.results["shots"] == {"a": "unprocessed"}
    assert plan.successful_ids() == ()


def test_missing_results_fail_only_the_affected_clips():
    plan = ClipAnalysisPlan(["a", "b"], ["colors", "describe"])
    plan.begin_ready()
    plan.finish("colors", {"a": "succeeded", "b": "skipped"})
    assert plan.begin_ready() == ("describe",)
    plan.finish("describe", {"a": "succeeded"})
    assert plan.finished and plan.successful_ids() == ("a",)
    assert plan.results["describe"]["b"] == "failed"


def test_foreign_results_do_not_complete_an_operation():
    plan = ClipAnalysisPlan(["a"], ["colors"])
    plan.begin_ready()
    with pytest.raises(ValueError, match="unrequested"):
        plan.finish("colors", {"other": "succeeded"})
    assert plan.running == {"colors"} and not plan.results


def test_unknown_operation_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        ClipAnalysisPlan(["a"], ["future-operation"])
