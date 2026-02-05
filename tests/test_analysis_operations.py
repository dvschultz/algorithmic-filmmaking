"""Unit tests for core.analysis_operations registry."""

import pytest

from core.analysis_operations import (
    AnalysisOperation,
    ANALYSIS_OPERATIONS,
    OPERATIONS_BY_KEY,
    LOCAL_OPS,
    SEQUENTIAL_OPS,
    CLOUD_OPS,
    DEFAULT_SELECTED,
    PHASE_ORDER,
)


class TestAnalysisOperationRegistry:
    """Tests for the operation registry."""

    def test_registry_has_8_operations(self):
        """All 8 operations are registered."""
        assert len(ANALYSIS_OPERATIONS) == 8

    def test_operations_by_key_matches(self):
        """OPERATIONS_BY_KEY has an entry for every operation."""
        assert len(OPERATIONS_BY_KEY) == len(ANALYSIS_OPERATIONS)
        for op in ANALYSIS_OPERATIONS:
            assert op.key in OPERATIONS_BY_KEY
            assert OPERATIONS_BY_KEY[op.key] is op

    def test_phase_groups_cover_all_keys(self):
        """LOCAL_OPS + SEQUENTIAL_OPS + CLOUD_OPS cover all operation keys."""
        all_keys = set(LOCAL_OPS + SEQUENTIAL_OPS + CLOUD_OPS)
        expected_keys = {op.key for op in ANALYSIS_OPERATIONS}
        assert all_keys == expected_keys

    def test_no_duplicate_keys(self):
        """No two operations share the same key."""
        keys = [op.key for op in ANALYSIS_OPERATIONS]
        assert len(keys) == len(set(keys))

    def test_phase_order_covers_all_phases(self):
        """PHASE_ORDER includes all phases used by operations."""
        phases_used = {op.phase for op in ANALYSIS_OPERATIONS}
        assert phases_used == set(PHASE_ORDER)

    def test_default_selected_are_valid_keys(self):
        """DEFAULT_SELECTED contains only valid operation keys."""
        for key in DEFAULT_SELECTED:
            assert key in OPERATIONS_BY_KEY

    def test_default_selected_matches_default_enabled(self):
        """DEFAULT_SELECTED matches operations with default_enabled=True."""
        expected = [op.key for op in ANALYSIS_OPERATIONS if op.default_enabled]
        assert set(DEFAULT_SELECTED) == set(expected)

    def test_operation_is_frozen_dataclass(self):
        """AnalysisOperation instances are immutable."""
        op = ANALYSIS_OPERATIONS[0]
        with pytest.raises(AttributeError):
            op.key = "modified"

    def test_local_ops_phase(self):
        """All LOCAL_OPS have phase 'local'."""
        for key in LOCAL_OPS:
            assert OPERATIONS_BY_KEY[key].phase == "local"

    def test_sequential_ops_phase(self):
        """All SEQUENTIAL_OPS have phase 'sequential'."""
        for key in SEQUENTIAL_OPS:
            assert OPERATIONS_BY_KEY[key].phase == "sequential"

    def test_cloud_ops_phase(self):
        """All CLOUD_OPS have phase 'cloud'."""
        for key in CLOUD_OPS:
            assert OPERATIONS_BY_KEY[key].phase == "cloud"

    def test_all_operations_have_required_fields(self):
        """Every operation has non-empty key, label, tooltip, and valid phase."""
        for op in ANALYSIS_OPERATIONS:
            assert op.key, f"Operation missing key: {op}"
            assert op.label, f"Operation missing label: {op}"
            assert op.tooltip, f"Operation missing tooltip: {op}"
            assert op.phase in PHASE_ORDER, f"Invalid phase '{op.phase}' for {op.key}"
