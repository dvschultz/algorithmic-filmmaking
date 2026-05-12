"""Tests for the heuristic ChatGPT Plus quota estimator."""

from __future__ import annotations

import pytest

from core.quota_estimator import (
    BatchLoadEstimate,
    adjust_threshold,
    estimate_batch_load,
)


class TestEstimateBatchLoad:
    def test_returns_typed_estimate(self):
        result = estimate_batch_load("describe", 50)
        assert isinstance(result, BatchLoadEstimate)
        assert result.operation == "describe"
        assert result.count == 50
        assert result.estimated_messages == 50
        assert result.estimated_tokens > 0

    def test_below_threshold_no_warn(self):
        result = estimate_batch_load("describe", 10)
        assert result.should_warn is False

    def test_above_threshold_warns(self):
        # Wide v1 default for describe is 300 — push well past it.
        result = estimate_batch_load("describe", 500)
        assert result.should_warn is True

    def test_unknown_operation_silent(self):
        """Unknown operation names produce zero estimate, no warning."""
        result = estimate_batch_load("totally-unknown-op", 10000)
        assert result.should_warn is False
        assert result.estimated_messages == 0
        assert result.estimated_tokens == 0

    def test_negative_count_silent(self):
        result = estimate_batch_load("describe", -5)
        assert result.should_warn is False
        assert result.count == 0

    def test_zero_count_silent(self):
        result = estimate_batch_load("describe", 0)
        assert result.should_warn is False
        assert result.estimated_messages == 0


class TestThresholdAdjustment:
    def test_adjust_threshold_takes_effect(self):
        # Capture current threshold, lower it, confirm estimate flips,
        # then restore it so the change doesn't leak to other tests.
        original = estimate_batch_load("describe", 1).warn_threshold_units
        try:
            adjust_threshold("describe", 5)
            result = estimate_batch_load("describe", 10)
            assert result.should_warn is True
            assert result.warn_threshold_units == 5
        finally:
            adjust_threshold("describe", original)

    def test_adjust_unknown_operation_is_noop(self):
        # Should not raise even though the operation isn't in the table.
        adjust_threshold("totally-unknown-op", 10)
