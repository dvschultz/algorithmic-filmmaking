"""Tests for the Staccato debug HTML report generator."""

import json
import pytest
from pathlib import Path

from core.remix.staccato import StaccatoDebugInfo, StaccatoSlotDebug
from core.remix.staccato_report import (
    generate_staccato_report,
    save_staccato_report,
    _debug_to_json,
)


def _make_debug(num_slots=3) -> StaccatoDebugInfo:
    """Create a StaccatoDebugInfo with test data."""
    slots = []
    for i in range(num_slots):
        slots.append(StaccatoSlotDebug(
            slot_index=i,
            start_time=i * 0.5,
            end_time=(i + 1) * 0.5,
            onset_strength=0.3 + i * 0.2,
            clip_id=f"clip_{i:03d}",
            clip_name=f"Clip {i}",
            source_filename="video.mp4",
            cosine_distance=0.4 + i * 0.1 if i > 0 else None,
            target_distance=0.3 + i * 0.2,
            distance_score=0.05 + i * 0.01,
            needs_loop=i == 2,
        ))
    return StaccatoDebugInfo(
        strategy="onsets",
        total_slots=num_slots,
        total_clips_available=5,
        slots=slots,
    )


class TestDebugToJson:

    def test_serializes_to_valid_json(self):
        debug = _make_debug()
        json_str = _debug_to_json(debug)
        data = json.loads(json_str)
        assert data["strategy"] == "onsets"
        assert data["totalSlots"] == 3
        assert data["totalClips"] == 5
        assert len(data["slots"]) == 3

    def test_first_slot_has_null_cosine_distance(self):
        debug = _make_debug()
        data = json.loads(_debug_to_json(debug))
        assert data["slots"][0]["cosineDistance"] is None

    def test_subsequent_slots_have_cosine_distance(self):
        debug = _make_debug()
        data = json.loads(_debug_to_json(debug))
        assert data["slots"][1]["cosineDistance"] is not None
        assert data["slots"][2]["cosineDistance"] is not None

    def test_needs_loop_flag(self):
        debug = _make_debug()
        data = json.loads(_debug_to_json(debug))
        assert data["slots"][2]["needsLoop"] is True
        assert data["slots"][0]["needsLoop"] is False


class TestGenerateReport:

    def test_produces_html(self):
        debug = _make_debug()
        html = generate_staccato_report(debug)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_strategy(self):
        debug = _make_debug()
        html = generate_staccato_report(debug)
        assert "onsets" in html

    def test_contains_slot_count(self):
        debug = _make_debug()
        html = generate_staccato_report(debug)
        assert "<strong>3</strong>" in html

    def test_contains_chart_js_when_vendor_exists(self):
        debug = _make_debug()
        html = generate_staccato_report(debug)
        # If vendored Chart.js exists, it should be embedded
        vendor_path = Path(__file__).parent.parent / "core" / "remix" / "vendor" / "chart.umd.min.js"
        if vendor_path.exists():
            assert "new Chart(" in html
            assert "onsetChart" in html
            assert "distanceChart" in html
        else:
            # Fallback table
            assert "<table>" in html

    def test_contains_data_json(self):
        debug = _make_debug()
        html = generate_staccato_report(debug)
        # The JSON data should be embedded in the HTML
        assert '"strategy":"onsets"' in html or '"strategy": "onsets"' in html

    def test_single_slot(self):
        debug = _make_debug(num_slots=1)
        html = generate_staccato_report(debug)
        assert "<!DOCTYPE html>" in html
        assert "<strong>1</strong>" in html

    def test_contains_detail_table(self):
        debug = _make_debug()
        html = generate_staccato_report(debug)
        # Should have a detail table section
        assert "detailTable" in html or "<table>" in html


class TestSaveReport:

    def test_saves_to_file(self, tmp_path):
        debug = _make_debug()
        output = tmp_path / "report.html"
        result = save_staccato_report(debug, output)
        assert result == output
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_returns_path(self, tmp_path):
        debug = _make_debug()
        output = tmp_path / "test_report.html"
        result = save_staccato_report(debug, output)
        assert isinstance(result, Path)
        assert result == output
