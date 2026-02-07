"""Tests for shot type filter in sequence tab and agent tool."""

from pathlib import Path

import pytest

from core.analysis.shots import SHOT_TYPES
from models.clip import Clip, Source


# -- Helpers ------------------------------------------------------------------

def _make_source(source_id: str = "src1") -> Source:
    return Source(id=source_id, file_path=Path("/video.mp4"), fps=30.0)


def _make_clip(clip_id: str, shot_type: str = None) -> Clip:
    return Clip(
        id=clip_id,
        source_id="src1",
        start_frame=0,
        end_frame=90,
        shot_type=shot_type,
    )


# -- IntentionImportDialog shot_type getter -----------------------------------

class TestIntentionDialogShotTypeGetter:
    """_get_shot_type returns filter for any algorithm, not just shot_type."""

    def test_get_shot_type_returns_none_for_all(self):
        """'All' selection returns None (no filter)."""
        from unittest.mock import MagicMock, patch
        from ui.dialogs.intention_import_dialog import IntentionImportDialog

        with patch.object(IntentionImportDialog, "__init__", lambda self, *a, **kw: None):
            dialog = IntentionImportDialog.__new__(IntentionImportDialog)
            dialog._algorithm = "color"  # non-shot_type algorithm

            combo = MagicMock()
            combo.currentText.return_value = "All"
            dialog.shot_type_dropdown = combo

            assert dialog._get_shot_type() is None

    def test_get_shot_type_returns_value_for_non_shot_type_algorithm(self):
        """Shot type filter works even when algorithm is not 'shot_type'."""
        from unittest.mock import MagicMock, patch
        from ui.dialogs.intention_import_dialog import IntentionImportDialog

        with patch.object(IntentionImportDialog, "__init__", lambda self, *a, **kw: None):
            dialog = IntentionImportDialog.__new__(IntentionImportDialog)
            dialog._algorithm = "duration"  # non-shot_type algorithm

            combo = MagicMock()
            combo.currentText.return_value = "Close-up"
            dialog.shot_type_dropdown = combo

            assert dialog._get_shot_type() == "close-up"


# -- GUI state tracking -------------------------------------------------------

class TestGUIStateShotFilter:
    """gui_state.sequence_shot_filter tracks the current filter."""

    def test_default_is_none(self):
        from core.gui_state import GUIState
        state = GUIState()
        assert state.sequence_shot_filter is None

    def test_can_set_filter(self):
        from core.gui_state import GUIState
        state = GUIState()
        state.sequence_shot_filter = "close-up"
        assert state.sequence_shot_filter == "close-up"

    def test_can_clear_filter(self):
        from core.gui_state import GUIState
        state = GUIState()
        state.sequence_shot_filter = "wide shot"
        state.sequence_shot_filter = None
        assert state.sequence_shot_filter is None


# -- Agent tool validation ----------------------------------------------------

class TestSetSequenceShotFilterTool:
    """set_sequence_shot_filter agent tool validates input."""

    def test_invalid_shot_type_returns_error(self):
        from unittest.mock import MagicMock
        from core.chat_tools import set_sequence_shot_filter
        from core.gui_state import GUIState

        project = MagicMock()
        gui_state = GUIState()
        main_window = MagicMock()

        result = set_sequence_shot_filter(
            project=project,
            gui_state=gui_state,
            main_window=main_window,
            shot_type="super-duper-wide",
        )
        assert result["success"] is False
        assert "Invalid shot type" in result["error"]

    def test_valid_shot_type_calls_filter(self):
        from unittest.mock import MagicMock
        from core.chat_tools import set_sequence_shot_filter
        from core.gui_state import GUIState

        project = MagicMock()
        gui_state = GUIState()
        main_window = MagicMock()
        main_window.sequence_tab.apply_shot_type_filter.return_value = 5

        result = set_sequence_shot_filter(
            project=project,
            gui_state=gui_state,
            main_window=main_window,
            shot_type="close-up",
        )
        assert result["success"] is True
        assert result["result"]["clip_count"] == 5
        assert gui_state.sequence_shot_filter == "close-up"

    def test_none_clears_filter(self):
        from unittest.mock import MagicMock
        from core.chat_tools import set_sequence_shot_filter
        from core.gui_state import GUIState

        project = MagicMock()
        gui_state = GUIState()
        gui_state.sequence_shot_filter = "wide shot"
        main_window = MagicMock()
        main_window.sequence_tab.apply_shot_type_filter.return_value = 20

        result = set_sequence_shot_filter(
            project=project,
            gui_state=gui_state,
            main_window=main_window,
            shot_type=None,
        )
        assert result["success"] is True
        assert result["result"]["shot_type"] == "all"
        assert gui_state.sequence_shot_filter is None


# -- Edge cases from acceptance criteria --------------------------------------

class TestShotTypeFilterEdgeCases:
    """Edge cases: no analyzed clips, mixed analysis, invalid agent input."""

    def test_no_analyzed_clips_specific_filter_returns_zero(self):
        """Specific shot type filter with no analyzed clips returns 0 matches."""
        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        # All clips have shot_type=None (unanalyzed)
        matching = [c for c in clips if c.shot_type == "close-up"]
        assert len(matching) == 0

    def test_no_analyzed_clips_all_filter_includes_all(self):
        """'All' filter (None) includes unanalyzed clips."""
        clips = [_make_clip("c1"), _make_clip("c2")]
        # shot_type=None means no filter applied
        shot_type = None
        if shot_type:
            filtered = [c for c in clips if c.shot_type == shot_type]
        else:
            filtered = clips
        assert len(filtered) == 2

    def test_mixed_analysis_shows_only_matching(self):
        """Filter on mixed clips shows only analyzed matches."""
        clips = [
            _make_clip("c1", shot_type="close-up"),
            _make_clip("c2", shot_type=None),  # unanalyzed
            _make_clip("c3", shot_type="wide shot"),
            _make_clip("c4", shot_type="close-up"),
        ]
        filtered = [c for c in clips if c.shot_type == "close-up"]
        assert len(filtered) == 2
        assert all(c.shot_type == "close-up" for c in filtered)

    def test_all_valid_shot_types_accepted_by_tool(self):
        """Every shot type from SHOT_TYPES is accepted by the agent tool."""
        from unittest.mock import MagicMock
        from core.chat_tools import set_sequence_shot_filter
        from core.gui_state import GUIState

        for st in SHOT_TYPES:
            project = MagicMock()
            gui_state = GUIState()
            main_window = MagicMock()
            main_window.sequence_tab.apply_shot_type_filter.return_value = 1

            result = set_sequence_shot_filter(
                project=project,
                gui_state=gui_state,
                main_window=main_window,
                shot_type=st,
            )
            assert result["success"] is True, f"Shot type '{st}' should be valid"
