"""Tests for Analyze tab auto-population and Clear All removal.

Behavior under test:
  - Clip.has_any_analysis() returns True when any analysis-output field is set
    and False for clips that only carry user-authored metadata.
  - The Clear All button no longer exists on the Analyze tab.
  - _run_analysis_pipeline filters out disabled clips before running.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from models.clip import Clip, Source
from tests.conftest import make_test_clip


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _bare_clip() -> Clip:
    """Return a clip with only user-authored fields populated."""
    return Clip(
        id="c1",
        source_id="s1",
        start_frame=0,
        end_frame=24,
        name="custom name",
        tags=["tag1"],
        notes="user note",
    )


class TestHasAnyAnalysis:
    def test_bare_clip_has_no_analysis(self):
        assert _bare_clip().has_any_analysis() is False

    def test_disabled_does_not_count_as_analysis(self):
        c = _bare_clip()
        c.disabled = True
        assert c.has_any_analysis() is False

    def test_user_authored_fields_excluded(self):
        # Only user-authored fields populated -> not analyzed.
        c = Clip(id="c2", source_id="s1", start_frame=0, end_frame=24,
                 name="x", tags=["a", "b"], notes="hi")
        assert c.has_any_analysis() is False

    @pytest.mark.parametrize("field, value", [
        ("dominant_colors", [(255, 0, 0)]),
        ("shot_type", "close-up"),
        ("object_labels", ["dog"]),
        ("detected_objects", [{"label": "dog", "confidence": 0.9}]),
        ("face_embeddings", [{"bbox": [0, 0, 10, 10]}]),
        ("person_count", 0),
        ("description", "a person"),
        ("custom_queries", [{"query": "x", "match": True}]),
        ("average_brightness", 0.5),
        ("rms_volume", -20.0),
        ("embedding", [0.1] * 512),
        ("first_frame_embedding", [0.1] * 512),
        ("last_frame_embedding", [0.1] * 512),
        ("gaze_yaw", 5.0),
        ("gaze_pitch", -2.0),
        ("gaze_category", "at_camera"),
    ])
    def test_each_analysis_field_flips_flag(self, field, value):
        c = _bare_clip()
        setattr(c, field, value)
        assert c.has_any_analysis() is True, f"{field}={value!r} should count as analysis"

    def test_person_count_zero_counts(self):
        # person_count=0 means "ran detection, found none" — analysis happened.
        c = _bare_clip()
        c.person_count = 0
        assert c.has_any_analysis() is True


class TestAutoIncludeAnalyzedClips:
    """Verify _auto_include_analyzed_clips routes clips into the Analyze tab."""

    class FakeAnalyzeTab:
        def __init__(self, existing_ids=()):
            self._existing = list(existing_ids)
            self.added = []

        def get_clip_ids(self):
            return list(self._existing) + list(self.added)

        def add_clips(self, clip_ids):
            self.added.extend(clip_ids)

    def test_only_clips_with_analysis_are_added(self, qapp):
        from ui.main_window import MainWindow

        analyzed = make_test_clip("a", shot_type="close-up")
        bare = make_test_clip("b")
        harness = SimpleNamespace(analyze_tab=self.FakeAnalyzeTab())

        MainWindow._auto_include_analyzed_clips(harness, [analyzed, bare])

        assert harness.analyze_tab.added == ["a"]

    def test_disabled_clip_with_prior_analysis_still_included(self, qapp):
        # Per the spec: disabled clips that were analyzed before they were
        # disabled should still appear in Analyze (rendered as disabled).
        from ui.main_window import MainWindow

        clip = make_test_clip("a", shot_type="close-up")
        clip.disabled = True
        harness = SimpleNamespace(analyze_tab=self.FakeAnalyzeTab())

        MainWindow._auto_include_analyzed_clips(harness, [clip])

        assert harness.analyze_tab.added == ["a"]

    def test_already_present_clip_not_re_added(self, qapp):
        from ui.main_window import MainWindow

        clip = make_test_clip("a", shot_type="close-up")
        harness = SimpleNamespace(
            analyze_tab=self.FakeAnalyzeTab(existing_ids=["a"]),
        )

        MainWindow._auto_include_analyzed_clips(harness, [clip])

        assert harness.analyze_tab.added == []

    def test_empty_list_is_noop(self, qapp):
        from ui.main_window import MainWindow

        harness = SimpleNamespace(analyze_tab=self.FakeAnalyzeTab())
        MainWindow._auto_include_analyzed_clips(harness, [])
        assert harness.analyze_tab.added == []

    def test_missing_analyze_tab_attr_is_noop(self, qapp):
        from ui.main_window import MainWindow

        clip = make_test_clip("a", shot_type="close-up")
        harness = SimpleNamespace()  # no analyze_tab
        # Must not raise
        MainWindow._auto_include_analyzed_clips(harness, [clip])


class TestClearAllButtonRemoved:
    def test_analyze_tab_has_no_clear_all_button(self, qapp):
        from ui.tabs.analyze_tab import AnalyzeTab
        tab = AnalyzeTab()
        assert not hasattr(tab, "clear_btn")

    def test_clear_clips_method_still_exists(self, qapp):
        # main_window calls this on new project; agent calls it via tool.
        from ui.tabs.analyze_tab import AnalyzeTab
        tab = AnalyzeTab()
        assert callable(getattr(tab, "clear_clips", None))


class TestPipelineSkipsDisabledClips:
    """Verify _run_analysis_pipeline drops disabled clips before dispatch."""

    def _make_harness(self):
        """Build a SimpleNamespace stand-in that mimics the MainWindow
        attributes _run_analysis_pipeline reads/writes."""
        return SimpleNamespace(
            _custom_query_text="dummy",
            _analysis_clips=None,
            _analysis_selected_ops=None,
            _analysis_completed_ops=None,
            _analysis_current_phase=None,
            _analysis_phase_remaining=None,
            _analysis_sequential_queue=None,
            _analysis_pending_phases=None,
            _gui_state=SimpleNamespace(set_processing=lambda *a, **kw: None),
            analyze_tab=SimpleNamespace(set_analyzing=lambda *a, **kw: None),
            _reset_analysis_run_error=lambda op: None,
            _filter_available_analysis_operations=lambda ops: ops,
            _start_next_analysis_phase=lambda: None,
        )

    def test_disabled_clips_dropped_before_pipeline_state_stored(self, qapp):
        from ui.main_window import MainWindow

        harness = self._make_harness()
        clip_enabled = make_test_clip("a")
        clip_disabled = make_test_clip("b")
        clip_disabled.disabled = True

        MainWindow._run_analysis_pipeline(
            harness, [clip_enabled, clip_disabled], ["colors"],
        )

        # State stored at line 3335 reflects only the enabled clip.
        assert [c.id for c in harness._analysis_clips] == ["a"]

    def test_all_disabled_returns_without_dispatch(self, qapp):
        from ui.main_window import MainWindow

        harness = self._make_harness()
        clip = make_test_clip("a")
        clip.disabled = True

        MainWindow._run_analysis_pipeline(harness, [clip], ["colors"])

        # Pipeline returned early — no state stored.
        assert harness._analysis_clips is None
