"""Unit tests for content-aware chat tools."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.analysis.audio import AudioAnalysis
from core.project import Project
from models.clip import Source, Clip

# Import shared test helpers from conftest.py
from tests.conftest import make_test_clip


def _create_chat_test_project() -> Project:
    """Create a project with two sources for chat tool testing."""
    project = Project.new(name="Test Project")
    project.add_source(Source(
        id="src-1",
        file_path=Path("/test/video1.mp4"),
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    ))
    project.add_source(Source(
        id="src-2",
        file_path=Path("/test/video2.mp4"),
        duration_seconds=60.0,
        fps=24.0,
        width=1280,
        height=720,
    ))
    return project


class TestSearchTranscripts:
    """Tests for search_transcripts tool."""

    def test_search_finds_matching_clips(self):
        """Test search finds clips with matching transcript text."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="Hello world, this is a test"),
            make_test_clip("clip-2", transcript_text="Goodbye world, farewell"),
            make_test_clip("clip-3", transcript_text="No match here"),
        ])

        result = search_transcripts(project, "world")

        assert result["success"] is True
        assert result["match_count"] == 2
        assert len(result["matches"]) == 2

        clip_ids = [m["clip_id"] for m in result["matches"]]
        assert "clip-1" in clip_ids
        assert "clip-2" in clip_ids

    def test_search_case_insensitive_by_default(self):
        """Test search is case insensitive by default."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="HELLO World"),
            make_test_clip("clip-2", transcript_text="hello world"),
        ])

        result = search_transcripts(project, "HELLO")

        assert result["success"] is True
        assert result["match_count"] == 2

    def test_search_case_sensitive(self):
        """Test case sensitive search."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="HELLO World"),
            make_test_clip("clip-2", transcript_text="hello world"),
        ])

        result = search_transcripts(project, "HELLO", case_sensitive=True)

        assert result["success"] is True
        assert result["match_count"] == 1
        assert result["matches"][0]["clip_id"] == "clip-1"

    def test_search_empty_query_returns_error(self):
        """Test empty query returns error."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()

        result = search_transcripts(project, "")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_search_no_matches(self):
        """Test search returns empty results when no matches."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="Hello world"),
        ])

        result = search_transcripts(project, "nonexistent")

        assert result["success"] is True
        assert result["match_count"] == 0
        assert result["matches"] == []

    def test_search_skips_clips_without_transcripts(self):
        """Test search skips clips without transcripts."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="Hello world"),
            make_test_clip("clip-2", transcript_text=None),  # No transcript
        ])

        result = search_transcripts(project, "hello")

        assert result["success"] is True
        assert result["match_count"] == 1

    def test_search_includes_context(self):
        """Test search results include context around match."""
        from core.chat_tools import search_transcripts

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip(
                "clip-1",
                transcript_text="This is some text before TARGET and some text after"
            ),
        ])

        result = search_transcripts(project, "TARGET", context_chars=10)

        assert result["success"] is True
        assert result["match_count"] == 1
        context = result["matches"][0]["match_context"]
        assert "TARGET" in context
        # Should have ellipsis for truncation
        assert "..." in context


class TestFindSimilarClips:
    """Tests for find_similar_clips tool."""

    def test_find_by_shot_type(self):
        """Test finding clips with same shot type."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up"),
            make_test_clip("clip-2", shot_type="close_up"),
            make_test_clip("clip-3", shot_type="wide_shot"),
        ])

        result = find_similar_clips(project, "clip-1", criteria=["shot_type"])

        assert result["success"] is True
        assert result["reference_clip_id"] == "clip-1"
        assert len(result["similar_clips"]) > 0

        # clip-2 should be first (same shot type)
        similar_ids = [c["clip_id"] for c in result["similar_clips"]]
        assert "clip-2" in similar_ids

    def test_find_by_color(self):
        """Test finding clips with similar colors."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()
        # Red-ish colors
        red_colors = [(255, 0, 0), (200, 50, 50), (180, 30, 30)]
        # Blue-ish colors
        blue_colors = [(0, 0, 255), (50, 50, 200), (30, 30, 180)]

        project.add_clips([
            make_test_clip("clip-1", dominant_colors=red_colors),
            make_test_clip("clip-2", dominant_colors=red_colors),  # Similar to clip-1
            make_test_clip("clip-3", dominant_colors=blue_colors),  # Different
        ])

        result = find_similar_clips(project, "clip-1", criteria=["color"])

        assert result["success"] is True
        # clip-2 should score higher than clip-3
        if len(result["similar_clips"]) >= 2:
            scores = {c["clip_id"]: c["similarity_score"] for c in result["similar_clips"]}
            assert scores.get("clip-2", 0) > scores.get("clip-3", 0)

    def test_find_by_duration(self):
        """Test finding clips with similar duration."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", start_frame=0, end_frame=90),  # 3 seconds at 30fps
            make_test_clip("clip-2", start_frame=0, end_frame=90),  # Same duration
            make_test_clip("clip-3", start_frame=0, end_frame=900),  # 30 seconds
        ])

        result = find_similar_clips(project, "clip-1", criteria=["duration"])

        assert result["success"] is True
        # clip-2 should score higher (same duration)
        if len(result["similar_clips"]) >= 2:
            scores = {c["clip_id"]: c["similarity_score"] for c in result["similar_clips"]}
            assert scores.get("clip-2", 0) > scores.get("clip-3", 0)

    def test_invalid_clip_id_returns_error(self):
        """Test invalid clip ID returns error."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()

        result = find_similar_clips(project, "nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_invalid_criteria_returns_error(self):
        """Test invalid criteria returns error."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = find_similar_clips(project, "clip-1", criteria=["invalid_criterion"])

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_respects_limit(self):
        """Test result limit is respected."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()
        # Add many clips
        clips = [
            make_test_clip(f"clip-{i}", shot_type="close_up")
            for i in range(20)
        ]
        project.add_clips(clips)

        result = find_similar_clips(project, "clip-0", limit=5)

        assert result["success"] is True
        assert len(result["similar_clips"]) <= 5

    def test_default_criteria(self):
        """Test default criteria are used when none specified."""
        from core.chat_tools import find_similar_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up"),
            make_test_clip("clip-2", shot_type="close_up"),
        ])

        result = find_similar_clips(project, "clip-1")

        assert result["success"] is True
        # Default criteria should be ["color", "shot_type"]
        assert result["criteria"] == ["color", "shot_type"]


class TestGroupClipsBy:
    """Tests for group_clips_by tool."""

    def test_group_by_shot_type(self):
        """Test grouping clips by shot type."""
        from core.chat_tools import group_clips_by

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up"),
            make_test_clip("clip-2", shot_type="close_up"),
            make_test_clip("clip-3", shot_type="wide_shot"),
        ])

        result = group_clips_by(project, "shot_type")

        assert result["success"] is True
        assert result["criterion"] == "shot_type"
        assert result["group_count"] == 2
        assert "close_up" in result["groups"]
        assert "wide_shot" in result["groups"]
        assert result["groups"]["close_up"]["count"] == 2
        assert result["groups"]["wide_shot"]["count"] == 1

    def test_group_by_duration(self):
        """Test grouping clips by duration category."""
        from core.chat_tools import group_clips_by

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", start_frame=0, end_frame=30),   # 1 second (short)
            make_test_clip("clip-2", start_frame=0, end_frame=150),  # 5 seconds (medium)
            make_test_clip("clip-3", start_frame=0, end_frame=450),  # 15 seconds (long)
        ])

        result = group_clips_by(project, "duration")

        assert result["success"] is True
        assert result["criterion"] == "duration"
        assert "short (<2s)" in result["groups"]
        assert "medium (2-10s)" in result["groups"]
        assert "long (>10s)" in result["groups"]

    def test_group_by_source(self):
        """Test grouping clips by source file."""
        from core.chat_tools import group_clips_by

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", source_id="src-1"),
            make_test_clip("clip-2", source_id="src-1"),
            make_test_clip("clip-3", source_id="src-2"),
        ])

        result = group_clips_by(project, "source")

        assert result["success"] is True
        assert result["criterion"] == "source"
        assert result["group_count"] == 2
        assert "video1.mp4" in result["groups"]
        assert "video2.mp4" in result["groups"]

    def test_group_by_color(self):
        """Test grouping clips by color palette classification."""
        from core.chat_tools import group_clips_by

        project = _create_chat_test_project()
        # Warm colors (red/orange hues)
        warm_colors = [(255, 100, 50), (200, 80, 40)]
        # Cool colors (blue hues)
        cool_colors = [(50, 100, 255), (40, 80, 200)]

        project.add_clips([
            make_test_clip("clip-1", dominant_colors=warm_colors),
            make_test_clip("clip-2", dominant_colors=cool_colors),
            make_test_clip("clip-3", dominant_colors=None),  # Unanalyzed
        ])

        result = group_clips_by(project, "color")

        assert result["success"] is True
        assert result["criterion"] == "color"
        # Should have groups for different palette classifications
        assert "unanalyzed" in result["groups"]

    def test_invalid_criterion_returns_error(self):
        """Test invalid criterion returns error."""
        from core.chat_tools import group_clips_by

        project = _create_chat_test_project()

        result = group_clips_by(project, "invalid")

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_group_handles_unknown_shot_types(self):
        """Test clips without shot type are grouped as 'unknown'."""
        from core.chat_tools import group_clips_by

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type=None),
            make_test_clip("clip-2", shot_type="close_up"),
        ])

        result = group_clips_by(project, "shot_type")

        assert result["success"] is True
        assert "unknown" in result["groups"]


class TestFilterClipsWithSearch:
    """Tests for filter_clips with search_query parameter."""

    def test_filter_by_search_query(self):
        """Test filtering clips by transcript search query."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="Hello world"),
            make_test_clip("clip-2", transcript_text="Goodbye moon"),
            make_test_clip("clip-3", transcript_text=None),
        ])

        result = filter_clips(project, search_query="world")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_search_case_insensitive(self):
        """Test search query is case insensitive."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="HELLO WORLD"),
        ])

        result = filter_clips(project, search_query="hello")

        assert len(result) == 1

    def test_filter_combined_with_search(self):
        """Test search query combined with other filters."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip(
                "clip-1",
                shot_type="close_up",
                transcript_text="Hello world"
            ),
            make_test_clip(
                "clip-2",
                shot_type="wide_shot",
                transcript_text="Hello world"
            ),
            make_test_clip(
                "clip-3",
                shot_type="close_up",
                transcript_text="Goodbye moon"
            ),
        ])

        result = filter_clips(
            project,
            shot_type="close_up",
            search_query="Hello"
        )

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_search_no_transcript_excluded(self):
        """Test clips without transcripts are excluded from search."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text=None),
            make_test_clip("clip-2", transcript_text="Hello"),
        ])

        result = filter_clips(project, search_query="anything")

        # clip-1 should be excluded (no transcript)
        assert all(r["id"] != "clip-1" for r in result)

    def test_filter_existing_filters_still_work(self):
        """Test existing filters still work without search_query."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up"),
            make_test_clip("clip-2", shot_type="wide_shot"),
        ])

        result = filter_clips(project, shot_type="close_up")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_duration_filters(self):
        """Test duration filters still work."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", start_frame=0, end_frame=30),   # 1 second
            make_test_clip("clip-2", start_frame=0, end_frame=150),  # 5 seconds
            make_test_clip("clip-3", start_frame=0, end_frame=300),  # 10 seconds
        ])

        result = filter_clips(project, min_duration=2.0, max_duration=8.0)

        assert len(result) == 1
        assert result[0]["id"] == "clip-2"


class TestToolIntegration:
    """Integration tests for tool combinations."""

    def test_search_then_group(self):
        """Test searching transcripts then grouping results."""
        from core.chat_tools import search_transcripts, group_clips_by

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up", transcript_text="Hello world"),
            make_test_clip("clip-2", shot_type="wide_shot", transcript_text="Hello there"),
            make_test_clip("clip-3", shot_type="close_up", transcript_text="Goodbye"),
        ])

        # First search
        search_result = search_transcripts(project, "Hello")
        assert search_result["match_count"] == 2

        # Then group all clips
        group_result = group_clips_by(project, "shot_type")
        assert group_result["success"] is True

    def test_find_similar_from_filtered(self):
        """Test finding similar clips after filtering."""
        from core.chat_tools import filter_clips, find_similar_clips

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up"),
            make_test_clip("clip-2", shot_type="close_up"),
            make_test_clip("clip-3", shot_type="wide_shot"),
        ])

        # First filter
        filtered = filter_clips(project, shot_type="close_up")
        assert len(filtered) == 2

        # Then find similar to first result
        similar = find_similar_clips(project, filtered[0]["id"], criteria=["shot_type"])
        assert similar["success"] is True


class TestGUIStateSynchronization:
    """Tests for GUI state synchronization with agent tools."""

    def test_gui_state_context_string_generation(self):
        """Test GUIState generates proper context strings."""
        from core.gui_state import GUIState

        state = GUIState()
        state.active_tab = "analyze"
        state.selected_clip_ids = ["clip-1", "clip-2"]

        context = state.to_context_string()

        assert "ACTIVE TAB: analyze" in context
        assert "SELECTED CLIPS: 2" in context
        assert "clip-1" in context

    def test_gui_state_filter_tracking(self):
        """Test GUIState tracks active filters."""
        from core.gui_state import GUIState

        state = GUIState()
        state.update_active_filters({
            "shot_type": "close_up",
            "min_duration": 2.0,
            "unused": None,  # Should be excluded
        })

        assert state.active_filters["shot_type"] == "close_up"
        assert state.active_filters["min_duration"] == 2.0
        assert "unused" not in state.active_filters

        context = state.to_context_string()
        assert "ACTIVE FILTERS:" in context

    def test_gui_state_clear_filters(self):
        """Test clearing GUI state filters."""
        from core.gui_state import GUIState

        state = GUIState()
        state.update_active_filters({"shot_type": "close_up"})
        assert state.active_filters != {}

        state.clear_filters()
        assert state.active_filters == {}

    def test_gui_state_search_update(self):
        """Test updating GUI state with search results."""
        from core.gui_state import GUIState

        state = GUIState()
        results = [
            {"video_id": "abc123", "title": "Test Video"},
            {"video_id": "def456", "title": "Another Video"},
        ]

        state.update_from_search("test query", results)

        assert state.last_search_query == "test query"
        assert len(state.search_results) == 2
        assert state.selected_video_ids == []

    def test_update_clip_modifies_project(self):
        """Test update_clip modifies the actual project clips."""
        from core.chat_tools import update_clip

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", shot_type="close_up"),
        ])

        result = update_clip(
            project,
            clip_id="clip-1",
            name="My New Name",
            notes="Test notes"
        )

        assert result["success"] is True
        clip = project.clips_by_id["clip-1"]
        assert clip.name == "My New Name"
        assert clip.notes == "Test notes"

    def test_update_clip_invalid_shot_type(self):
        """Test update_clip rejects invalid shot types."""
        from core.chat_tools import update_clip

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = update_clip(
            project,
            clip_id="clip-1",
            shot_type="invalid_type"
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower() or "shot" in result["error"].lower()


class TestToolExecutor:
    """Tests for ToolExecutor functionality."""

    def test_executor_injects_project(self):
        """Test ToolExecutor injects project into tool calls."""
        from core.tool_executor import ToolExecutor

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        executor = ToolExecutor(project=project)

        # Call a tool that requires project (use get_project_state, not list_clips
        # which now requires GUI handler for main_window injection)
        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_project_state",
                "arguments": "{}"
            }
        })

        assert result["success"] is True
        assert result["result"]["clip_count"] == 1

    def test_executor_handles_missing_project(self):
        """Test ToolExecutor reports error when project required but missing."""
        from core.tool_executor import ToolExecutor

        executor = ToolExecutor(project=None)

        # Use get_project_state (list_clips now needs GUI handler)
        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_project_state",
                "arguments": "{}"
            }
        })

        assert result["success"] is False
        assert "project" in result["error"].lower()

    def test_executor_handles_unknown_tool(self):
        """Test ToolExecutor handles unknown tool names."""
        from core.tool_executor import ToolExecutor

        executor = ToolExecutor()

        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "nonexistent_tool",
                "arguments": "{}"
            }
        })

        assert result["success"] is False
        assert "unknown" in result["error"].lower()

    def test_executor_handles_invalid_json(self):
        """Test ToolExecutor handles invalid JSON arguments."""
        from core.tool_executor import ToolExecutor

        executor = ToolExecutor()

        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "list_clips",
                "arguments": "invalid json {{{}}}"
            }
        })

        assert result["success"] is False
        assert "json" in result["error"].lower()

    def test_executor_busy_check_integration(self):
        """Test ToolExecutor respects busy_check callback for tools with conflicts_with_workers."""
        from core.tool_executor import ToolExecutor
        from unittest.mock import MagicMock

        # Mock busy check that returns True (busy)
        busy_check = MagicMock(return_value=True)

        executor = ToolExecutor(
            project=_create_chat_test_project(),
            busy_check=busy_check
        )

        # detect_scenes has conflicts_with_workers=True
        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "detect_scenes",
                "arguments": '{"video_path": "/test.mp4"}'
            }
        })

        # Should fail because busy_check returns True
        assert result["success"] is False
        assert "in progress" in result["error"].lower() or "waiting" in result["error"].lower()


class TestSignalDeliveryGuards:
    """Tests for preventing duplicate signal delivery."""

    def test_gui_state_is_dataclass(self):
        """Test GUIState is a proper dataclass with defaults."""
        from core.gui_state import GUIState

        state1 = GUIState()
        state2 = GUIState()

        # Each instance should have independent lists
        state1.selected_clip_ids.append("clip-1")
        assert "clip-1" not in state2.selected_clip_ids

    def test_gui_state_clear_search(self):
        """Test clearing search state."""
        from core.gui_state import GUIState

        state = GUIState()
        state.update_from_search("query", [{"id": "1"}])
        state.selected_video_ids.append("1")

        state.clear_search_state()

        assert state.last_search_query == ""
        assert state.search_results == []
        assert state.selected_video_ids == []

    def test_gui_state_plan_management(self):
        """Test plan state management."""
        from core.gui_state import GUIState

        state = GUIState()
        assert state.current_plan is None

        # Mock plan object
        mock_plan = Mock()
        mock_plan.summary = "Test plan"
        mock_plan.status = "draft"
        mock_plan.steps = ["step1", "step2"]

        state.set_plan(mock_plan)
        assert state.current_plan is mock_plan

        state.clear_plan_state()
        assert state.current_plan is None

    def test_tool_result_format_consistency(self):
        """Test that tool results have consistent format."""
        from core.chat_tools import (
            search_transcripts,
            find_similar_clips,
            group_clips_by,
        )

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1", transcript_text="hello"),
        ])

        # All tools should return dict with "success" key
        search_result = search_transcripts(project, "hello")
        assert "success" in search_result
        assert isinstance(search_result["success"], bool)

        similar_result = find_similar_clips(project, "clip-1")
        assert "success" in similar_result
        assert isinstance(similar_result["success"], bool)

        group_result = group_clips_by(project, "shot_type")
        assert "success" in group_result
        assert isinstance(group_result["success"], bool)

    def test_filter_clips_returns_list(self):
        """Test filter_clips returns a list (not wrapped in dict)."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = filter_clips(project)

        # filter_clips returns a list directly, not a dict with success
        assert isinstance(result, list)


class TestPlanExecutionTools:
    """Tests for plan execution tools."""

    def _create_mock_main_window(self):
        """Create a mock main window with GUI state and plan controller."""
        from core.gui_state import GUIState
        from core.plan_controller import PlanController

        main_window = Mock()
        main_window._gui_state = GUIState()
        main_window.plan_controller = PlanController(main_window._gui_state)
        return main_window

    def test_start_plan_execution_no_plan(self):
        """Test start_plan_execution fails when no plan exists."""
        from core.chat_tools import start_plan_execution

        main_window = self._create_mock_main_window()

        result = start_plan_execution(main_window)

        assert result["success"] is False
        assert "No plan exists" in result["error"]

    def test_start_plan_execution_starts_plan(self):
        """Test start_plan_execution transitions plan to executing."""
        from core.chat_tools import present_plan, start_plan_execution

        main_window = self._create_mock_main_window()

        # Create a plan first
        present_plan(main_window, ["Step 1", "Step 2", "Step 3"], "Test plan")

        result = start_plan_execution(main_window)

        assert result["success"] is True
        assert result["status"] == "executing"
        assert result["current_step_number"] == 1
        assert result["total_steps"] == 3
        assert result["current_step"] == "Step 1"
        assert len(result["remaining_steps"]) == 2

    def test_start_plan_execution_already_executing(self):
        """Test start_plan_execution handles already executing plan."""
        from core.chat_tools import present_plan, start_plan_execution

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")
        start_plan_execution(main_window)

        # Call again
        result = start_plan_execution(main_window)

        assert result["success"] is True
        assert result["already_executing"] is True

    def test_complete_plan_step_no_plan(self):
        """Test complete_plan_step fails when no plan exists."""
        from core.chat_tools import complete_plan_step

        main_window = self._create_mock_main_window()

        result = complete_plan_step(main_window)

        assert result["success"] is False
        assert "No plan exists" in result["error"]

    def test_complete_plan_step_not_executing(self):
        """Test complete_plan_step fails when plan not executing."""
        from core.chat_tools import present_plan, complete_plan_step

        main_window = self._create_mock_main_window()

        # Create plan but don't start it
        present_plan(main_window, ["Step 1"], "Test plan")

        result = complete_plan_step(main_window)

        assert result["success"] is False
        assert "not executing" in result["error"]

    def test_complete_plan_step_advances(self):
        """Test complete_plan_step advances to next step."""
        from core.chat_tools import present_plan, start_plan_execution, complete_plan_step

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2", "Step 3"], "Test plan")
        start_plan_execution(main_window)

        result = complete_plan_step(main_window, "Step 1 done")

        assert result["success"] is True
        assert result["plan_completed"] is False
        assert result["completed_step"] == "Step 1"
        assert result["completed_step_number"] == 1
        assert result["current_step_number"] == 2
        assert result["current_step"] == "Step 2"
        assert "1/3" in result["progress"]

    def test_complete_plan_step_finishes_plan(self):
        """Test complete_plan_step completes plan on last step."""
        from core.chat_tools import present_plan, start_plan_execution, complete_plan_step

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")
        start_plan_execution(main_window)

        # Complete step 1
        complete_plan_step(main_window)

        # Complete step 2 (last)
        result = complete_plan_step(main_window)

        assert result["success"] is True
        assert result["plan_completed"] is True
        assert "All 2 steps finished" in result["message"]

    def test_get_plan_status_no_plan(self):
        """Test get_plan_status when no plan exists."""
        from core.chat_tools import get_plan_status

        main_window = self._create_mock_main_window()

        result = get_plan_status(main_window)

        assert result["has_plan"] is False

    def test_get_plan_status_draft(self):
        """Test get_plan_status shows draft status."""
        from core.chat_tools import present_plan, get_plan_status

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")

        result = get_plan_status(main_window)

        assert result["has_plan"] is True
        assert result["status"] == "draft"
        assert result["total_steps"] == 2
        assert "awaiting" in result["message"].lower()

    def test_get_plan_status_executing(self):
        """Test get_plan_status shows executing status with current step."""
        from core.chat_tools import present_plan, start_plan_execution, get_plan_status

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2", "Step 3"], "Test plan")
        start_plan_execution(main_window)

        result = get_plan_status(main_window)

        assert result["has_plan"] is True
        assert result["status"] == "executing"
        assert result["current_step_number"] == 1
        assert result["current_step"] == "Step 1"
        assert len(result["remaining_steps"]) == 2

    def test_get_plan_status_shows_step_details(self):
        """Test get_plan_status returns detailed step information."""
        from core.chat_tools import present_plan, start_plan_execution, complete_plan_step, get_plan_status

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")
        start_plan_execution(main_window)
        complete_plan_step(main_window, "First step done")

        result = get_plan_status(main_window)

        assert len(result["steps"]) == 2
        # First step should be completed
        assert result["steps"][0]["status"] == "completed"
        assert result["steps"][0]["result_summary"] == "First step done"
        # Second step should be running
        assert result["steps"][1]["status"] == "running"

    def test_fail_plan_step_stop(self):
        """Test fail_plan_step with stop action."""
        from core.chat_tools import present_plan, start_plan_execution, fail_plan_step

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")
        start_plan_execution(main_window)

        result = fail_plan_step(main_window, "Something went wrong", action="stop")

        assert result["success"] is True
        assert result["action"] == "stop"
        assert result["plan_status"] == "failed"
        assert result["error"] == "Something went wrong"

    def test_fail_plan_step_retry(self):
        """Test fail_plan_step with retry action."""
        from core.chat_tools import present_plan, start_plan_execution, fail_plan_step

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")
        start_plan_execution(main_window)

        result = fail_plan_step(main_window, "Temporary error", action="retry")

        assert result["success"] is True
        assert result["action"] == "retry"
        assert result["step_number"] == 1
        assert "Retrying" in result["message"]

    def test_fail_plan_step_skip(self):
        """Test fail_plan_step with skip action."""
        from core.chat_tools import present_plan, start_plan_execution, fail_plan_step

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2", "Step 3"], "Test plan")
        start_plan_execution(main_window)

        result = fail_plan_step(main_window, "Skip this step", action="skip")

        assert result["success"] is True
        assert result["action"] == "skip"
        assert result["skipped_step"] == "Step 1"
        assert result["current_step_number"] == 2
        assert result["current_step"] == "Step 2"

    def test_fail_plan_step_skip_last(self):
        """Test fail_plan_step skip on last step completes plan."""
        from core.chat_tools import present_plan, start_plan_execution, complete_plan_step, fail_plan_step

        main_window = self._create_mock_main_window()

        present_plan(main_window, ["Step 1", "Step 2"], "Test plan")
        start_plan_execution(main_window)
        complete_plan_step(main_window)  # Complete step 1

        # Skip step 2 (last step)
        result = fail_plan_step(main_window, "Skip last step", action="skip")

        assert result["success"] is True
        assert result["action"] == "skip"
        assert result["plan_completed"] is True

    def test_full_plan_workflow(self):
        """Test a complete plan execution workflow."""
        from core.chat_tools import (
            present_plan,
            start_plan_execution,
            complete_plan_step,
            get_plan_status,
        )

        main_window = self._create_mock_main_window()

        # 1. Present plan
        steps = [
            "Download videos",
            "Run scene detection",
            "Add clips to sequence",
            "Export video"
        ]
        present_result = present_plan(main_window, steps, "Video editing workflow")
        assert present_result["step_count"] == 4

        # 2. Start execution
        start_result = start_plan_execution(main_window)
        assert start_result["current_step"] == "Download videos"

        # 3. Complete each step
        complete_plan_step(main_window, "Downloaded 8 videos")
        complete_plan_step(main_window, "Detected scenes in all videos")
        complete_plan_step(main_window, "Added 24 clips to sequence")

        # 4. Check status before last step
        status = get_plan_status(main_window)
        assert status["current_step_number"] == 4
        assert status["current_step"] == "Export video"
        assert "3/4" in status["progress"]

        # 5. Complete final step
        final_result = complete_plan_step(main_window, "Exported to output.mp4")
        assert final_result["plan_completed"] is True

        # 6. Verify final status
        final_status = get_plan_status(main_window)
        assert final_status["status"] == "completed"


class TestPendingActionTracking:
    """Tests for pending action tracking in GUI state."""

    def _create_mock_main_window_with_project(self):
        """Create mock main window with GUI state, project, and plan controller."""
        from core.gui_state import GUIState
        from core.plan_controller import PlanController
        from core.project import Project

        main_window = Mock()
        main_window._gui_state = GUIState()
        main_window.plan_controller = PlanController(main_window._gui_state)
        main_window.project = Project.new(name="Untitled Project")
        return main_window

    def test_present_plan_creates_plan_for_unnamed_project(self):
        """present_plan no longer blocks on unnamed projects — the system prompt handles naming."""
        from core.chat_tools import present_plan

        main_window = self._create_mock_main_window_with_project()

        steps = ["Download videos", "Run detection", "Export"]
        result = present_plan(main_window, steps, "Test workflow")

        # Plan should be created directly (no keyword-scan gate)
        assert "plan_id" in result or "action_required" not in result

    def test_set_project_name_returns_needs_save(self):
        """set_project_name returns needs_save flag for unsaved projects."""
        from core.chat_tools import set_project_name

        main_window = self._create_mock_main_window_with_project()

        result = set_project_name(main_window, main_window.project, "My Project")

        assert result["success"] is True
        assert result["new_name"] == "My Project"
        assert "needs_save" in result

    def test_gui_state_context_shows_pending_action(self):
        """Test that GUI state context string includes pending action."""
        from core.gui_state import GUIState, NameProjectThenPlanAction

        state = GUIState()
        action = NameProjectThenPlanAction(
            pending_steps=["Step 1", "Step 2"],
            pending_summary="Test plan",
            user_response="kittyjump"
        )
        state.set_pending_action(action)

        context = state.to_context_string()

        assert "PENDING ACTION" in context
        assert "name_project_then_plan" in context
        assert "kittyjump" in context
        assert "set_project_name" in context

    def test_pending_action_cleared_after_handling(self):
        """Test that pending action can be cleared."""
        from core.gui_state import GUIState, NameProjectThenPlanAction

        state = GUIState()
        state.set_pending_action(
            NameProjectThenPlanAction(
                pending_steps=["Step 1"],
                pending_summary="Test"
            )
        )

        assert state.pending_action is not None

        state.clear_pending_action()

        assert state.pending_action is None


class TestExportBundle:
    """Tests for export_bundle tool."""

    def test_export_bundle_no_main_window(self):
        """Test export_bundle returns error when main_window is None."""
        from core.chat_tools import export_bundle

        project = _create_chat_test_project()
        result = export_bundle(None, project)

        assert result["success"] is False
        assert "Main window" in result["error"]

    def test_export_bundle_already_running(self):
        """Test export_bundle returns error when already in progress."""
        from core.chat_tools import export_bundle

        project = _create_chat_test_project()
        main_window = Mock()
        worker = Mock()
        worker.isRunning.return_value = True
        main_window.export_bundle_worker = worker

        result = export_bundle(main_window, project)

        assert result["success"] is False
        assert "already in progress" in result["error"]

    def test_export_bundle_starts_worker(self, tmp_path):
        """Test export_bundle starts the worker and returns wait marker."""
        from core.chat_tools import export_bundle

        project = _create_chat_test_project()
        main_window = Mock()
        main_window.export_bundle_worker = None
        main_window.start_agent_export_bundle.return_value = True

        dest = str(tmp_path / "test_bundle")
        result = export_bundle(main_window, project, output_path=dest)

        assert "_wait_for_worker" in result
        assert result["_wait_for_worker"] == "export_bundle"
        assert result["include_videos"] is True
        main_window.start_agent_export_bundle.assert_called_once_with(
            tmp_path / "test_bundle", True
        )

    def test_export_bundle_lightweight(self, tmp_path):
        """Test export_bundle with lightweight=True skips videos."""
        from core.chat_tools import export_bundle

        project = _create_chat_test_project()
        main_window = Mock()
        main_window.export_bundle_worker = None
        main_window.start_agent_export_bundle.return_value = True

        dest = str(tmp_path / "test_bundle")
        result = export_bundle(main_window, project, output_path=dest, lightweight=True)

        assert "_wait_for_worker" in result
        assert result["include_videos"] is False
        main_window.start_agent_export_bundle.assert_called_once_with(
            tmp_path / "test_bundle", False
        )

    def test_export_bundle_worker_fails_to_start(self, tmp_path):
        """Test export_bundle returns error when worker fails to start."""
        from core.chat_tools import export_bundle

        project = _create_chat_test_project()
        main_window = Mock()
        main_window.export_bundle_worker = None
        main_window.start_agent_export_bundle.return_value = False

        dest = str(tmp_path / "test_bundle")
        result = export_bundle(main_window, project, output_path=dest)

        assert result["success"] is False
        assert "Failed to start" in result["error"]


class TestUpdateClipCinematography:
    """Tests for update_clip_cinematography tool."""

    def test_update_creates_cinematography_if_none(self):
        """Test update creates a new CinematographyAnalysis if clip has none."""
        from core.chat_tools import update_clip_cinematography

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])
        clip = project.clips_by_id["clip-1"]
        assert clip.cinematography is None

        result = update_clip_cinematography(
            project, clip_id="clip-1", shot_size="CU"
        )

        assert result["success"] is True
        assert "shot_size" in result["updated_fields"]
        assert clip.cinematography is not None
        assert clip.cinematography.shot_size == "CU"

    def test_update_modifies_existing_cinematography(self):
        """Test update modifies existing cinematography fields."""
        from core.chat_tools import update_clip_cinematography
        from models.cinematography import CinematographyAnalysis

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])
        clip = project.clips_by_id["clip-1"]
        clip.cinematography = CinematographyAnalysis(
            shot_size="MS", camera_angle="eye_level"
        )

        result = update_clip_cinematography(
            project,
            clip_id="clip-1",
            shot_size="ECU",
            camera_angle="low_angle",
            lighting_style="dramatic",
        )

        assert result["success"] is True
        assert set(result["updated_fields"]) == {"shot_size", "camera_angle", "lighting_style"}
        assert clip.cinematography.shot_size == "ECU"
        assert clip.cinematography.camera_angle == "low_angle"
        assert clip.cinematography.lighting_style == "dramatic"

    def test_update_preserves_unmodified_fields(self):
        """Test that unspecified fields retain their values."""
        from core.chat_tools import update_clip_cinematography
        from models.cinematography import CinematographyAnalysis

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])
        clip = project.clips_by_id["clip-1"]
        clip.cinematography = CinematographyAnalysis(
            shot_size="LS", camera_angle="high_angle", lighting_style="low_key"
        )

        result = update_clip_cinematography(
            project, clip_id="clip-1", shot_size="CU"
        )

        assert result["success"] is True
        assert clip.cinematography.shot_size == "CU"
        # These should be unchanged
        assert clip.cinematography.camera_angle == "high_angle"
        assert clip.cinematography.lighting_style == "low_key"

    def test_update_invalid_shot_size(self):
        """Test update rejects invalid shot_size values."""
        from core.chat_tools import update_clip_cinematography

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = update_clip_cinematography(
            project, clip_id="clip-1", shot_size="INVALID"
        )

        assert result["success"] is False
        assert "Invalid shot_size" in result["error"]

    def test_update_invalid_camera_angle(self):
        """Test update rejects invalid camera_angle values."""
        from core.chat_tools import update_clip_cinematography

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = update_clip_cinematography(
            project, clip_id="clip-1", camera_angle="upside_down"
        )

        assert result["success"] is False
        assert "Invalid camera_angle" in result["error"]

    def test_update_clip_not_found(self):
        """Test update returns error for non-existent clip."""
        from core.chat_tools import update_clip_cinematography

        project = _create_chat_test_project()

        result = update_clip_cinematography(
            project, clip_id="nonexistent", shot_size="CU"
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_update_no_fields_provided(self):
        """Test update with no fields returns success with empty updated_fields."""
        from core.chat_tools import update_clip_cinematography

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = update_clip_cinematography(project, clip_id="clip-1")

        assert result["success"] is True
        assert result["updated_fields"] == []

    def test_update_all_field_categories(self):
        """Test that all cinematography field categories can be set."""
        from core.chat_tools import update_clip_cinematography

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = update_clip_cinematography(
            project,
            clip_id="clip-1",
            shot_size="ELS",
            camera_angle="birds_eye",
            angle_effect="omniscience",
            camera_movement="crane",
            movement_direction="up",
            dutch_tilt="moderate",
            camera_position="profile",
            subject_position="left_third",
            headroom="tight",
            lead_room="excessive",
            balance="left_heavy",
            subject_count="two_shot",
            subject_type="person",
            focus_type="shallow",
            background_type="blurred",
            estimated_lens_type="telephoto",
            lighting_style="low_key",
            lighting_direction="side",
            light_quality="hard",
            color_temperature="warm",
            emotional_intensity="high",
            suggested_pacing="fast",
        )

        assert result["success"] is True
        assert len(result["updated_fields"]) == 22
        clip = project.clips_by_id["clip-1"]
        assert clip.cinematography.shot_size == "ELS"
        assert clip.cinematography.camera_movement == "crane"
        assert clip.cinematography.dutch_tilt == "moderate"
        assert clip.cinematography.color_temperature == "warm"


class TestClearClipCinematography:
    """Tests for clear_clip_cinematography tool."""

    def test_clear_removes_cinematography(self):
        """Test clearing cinematography sets it to None."""
        from core.chat_tools import clear_clip_cinematography
        from models.cinematography import CinematographyAnalysis

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])
        clip = project.clips_by_id["clip-1"]
        clip.cinematography = CinematographyAnalysis(shot_size="CU")

        result = clear_clip_cinematography(project, clip_ids=["clip-1"])

        assert result["success"] is True
        assert result["cleared_count"] == 1
        assert "clip-1" in result["cleared_ids"]
        assert clip.cinematography is None

    def test_clear_multiple_clips(self):
        """Test clearing cinematography from multiple clips."""
        from core.chat_tools import clear_clip_cinematography
        from models.cinematography import CinematographyAnalysis

        project = _create_chat_test_project()
        project.add_clips([
            make_test_clip("clip-1"),
            make_test_clip("clip-2"),
            make_test_clip("clip-3"),
        ])
        project.clips_by_id["clip-1"].cinematography = CinematographyAnalysis()
        project.clips_by_id["clip-2"].cinematography = CinematographyAnalysis()

        result = clear_clip_cinematography(
            project, clip_ids=["clip-1", "clip-2", "clip-3"]
        )

        assert result["success"] is True
        assert result["cleared_count"] == 2
        assert "clip-3" in result["already_clear"]
        assert project.clips_by_id["clip-1"].cinematography is None
        assert project.clips_by_id["clip-2"].cinematography is None

    def test_clear_clip_not_found(self):
        """Test clearing non-existent clip reports it as not found."""
        from core.chat_tools import clear_clip_cinematography

        project = _create_chat_test_project()

        result = clear_clip_cinematography(
            project, clip_ids=["nonexistent"]
        )

        assert result["success"] is True
        assert result["cleared_count"] == 0
        assert "nonexistent" in result["not_found"]

    def test_clear_already_clear(self):
        """Test clearing a clip that already has no cinematography."""
        from core.chat_tools import clear_clip_cinematography

        project = _create_chat_test_project()
        project.add_clips([make_test_clip("clip-1")])

        result = clear_clip_cinematography(project, clip_ids=["clip-1"])

        assert result["success"] is True
        assert result["cleared_count"] == 0
        assert "clip-1" in result["already_clear"]

    def test_clear_empty_list(self):
        """Test clearing with empty list returns error."""
        from core.chat_tools import clear_clip_cinematography

        project = _create_chat_test_project()

        result = clear_clip_cinematography(project, clip_ids=[])

        assert result["success"] is False
        assert "No clip IDs" in result["error"]


class TestAnalyzeGazeTool:
    """Tests for analyze_gaze agent tool."""

    def test_analyze_gaze_in_tool_registry(self):
        """Test that analyze_gaze is registered in the tool registry."""
        from core.chat_tools import tools as registry

        tool = registry.get("analyze_gaze")
        assert tool is not None
        assert "gaze" in tool.description.lower()
        assert tool.requires_project is True

    def test_analyze_gaze_openai_format(self):
        """Test that analyze_gaze appears in OpenAI tool format."""
        from core.chat_tools import tools as registry

        openai_tools = registry.to_openai_format()
        tool_names = [t["function"]["name"] for t in openai_tools]
        assert "analyze_gaze" in tool_names


class TestGazeDataInClipListings:
    """Tests for gaze data inclusion in clip listing tools."""

    def test_list_clips_includes_gaze_when_present(self):
        """Test that list_clips includes gaze fields for analyzed clips."""
        from core.chat_tools import list_clips

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
            gaze_yaw=5.3,
            gaze_pitch=-2.1,
            gaze_category="at_camera",
        )
        project.add_clips([clip])

        result = list_clips(main_window=None, project=project)

        assert result["success"] is True
        clip_data = result["clips"][0]
        assert clip_data["gaze_yaw"] == 5.3
        assert clip_data["gaze_pitch"] == -2.1
        assert clip_data["gaze_category"] == "at_camera"

    def test_list_clips_omits_gaze_when_absent(self):
        """Test that list_clips omits gaze fields when clips have no gaze data."""
        from core.chat_tools import list_clips

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
        )
        project.add_clips([clip])

        result = list_clips(main_window=None, project=project)

        assert result["success"] is True
        clip_data = result["clips"][0]
        assert "gaze_yaw" not in clip_data
        assert "gaze_pitch" not in clip_data
        assert "gaze_category" not in clip_data

    def test_filter_clips_includes_gaze_when_present(self):
        """Test that filter_clips includes gaze fields for analyzed clips."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
            gaze_yaw=-15.7,
            gaze_pitch=3.0,
            gaze_category="looking_left",
        )
        project.add_clips([clip])

        result = filter_clips(project)

        assert len(result) == 1
        assert result[0]["gaze_yaw"] == -15.7
        assert result[0]["gaze_pitch"] == 3.0
        assert result[0]["gaze_category"] == "looking_left"

    def test_filter_clips_omits_gaze_when_absent(self):
        """Test that filter_clips omits gaze fields when clips have no gaze data."""
        from core.chat_tools import filter_clips

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
        )
        project.add_clips([clip])

        result = filter_clips(project)

        assert len(result) == 1
        assert "gaze_yaw" not in result[0]
        assert "gaze_pitch" not in result[0]
        assert "gaze_category" not in result[0]


class TestGazeAlgorithmsInAgentTools:
    """Tests for gaze algorithms in sorting/remix agent tools."""

    def test_list_sorting_algorithms_includes_gaze_sort(self):
        """Test that gaze_sort appears in list_sorting_algorithms."""
        from core.chat_tools import list_sorting_algorithms

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
            gaze_category="at_camera",
        )
        project.add_clips([clip])

        result = list_sorting_algorithms(project)

        algo_keys = [a["key"] for a in result["algorithms"]]
        assert "gaze_sort" in algo_keys
        assert "gaze_consistency" in algo_keys

    def test_list_sorting_algorithms_gaze_unavailable_without_analysis(self):
        """Test that gaze algorithms show as unavailable without gaze data."""
        from core.chat_tools import list_sorting_algorithms

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
        )
        project.add_clips([clip])

        result = list_sorting_algorithms(project)

        gaze_algos = [a for a in result["algorithms"] if a["key"] in ("gaze_sort", "gaze_consistency")]
        for algo in gaze_algos:
            assert algo["available"] is False
            assert "gaze" in algo["reason"].lower()

    def test_generate_remix_accepts_gaze_sort(self):
        """Test that generate_remix accepts gaze_sort as a valid algorithm."""
        from core.chat_tools import generate_remix

        project = _create_chat_test_project()
        clip = Clip(
            id="clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=90,
            gaze_category="at_camera",
            gaze_yaw=0.0,
            gaze_pitch=0.0,
        )
        project.add_clips([clip])

        # Mock main_window with sequence_tab
        main_window = Mock()
        main_window.sequence_tab.generate_and_apply.return_value = {
            "success": True,
            "clip_count": 1,
            "algorithm": "gaze_sort",
        }

        result = generate_remix(project, main_window, algorithm="gaze_sort", clip_count=1)

        assert result["success"] is True
        main_window.sequence_tab.generate_and_apply.assert_called_once()

    def test_generate_remix_rejects_invalid_algorithm(self):
        """Test that generate_remix rejects unknown algorithm names."""
        from core.chat_tools import generate_remix

        project = _create_chat_test_project()
        main_window = Mock()

        result = generate_remix(project, main_window, algorithm="nonexistent")

        assert result["success"] is False
        assert "Invalid algorithm" in result["error"]


class TestGazeFilterTool:
    """Tests for set_sequence_gaze_filter agent tool."""

    def test_gaze_filter_tool_registered(self):
        """Test that set_sequence_gaze_filter is in the tool registry."""
        from core.chat_tools import tools as registry

        tool = registry.get("set_sequence_gaze_filter")
        assert tool is not None
        assert "gaze" in tool.description.lower()

    def test_gaze_filter_rejects_invalid_category(self):
        """Test that invalid gaze categories are rejected."""
        from core.chat_tools import set_sequence_gaze_filter

        project = _create_chat_test_project()
        gui_state = Mock()
        main_window = Mock()

        result = set_sequence_gaze_filter(
            project, gui_state, main_window, gaze_category="invalid_direction"
        )

        assert result["success"] is False
        assert "Invalid gaze category" in result["error"]


class TestGUIStateGazeFilter:
    """Tests for gaze filter state in GUIState."""

    def test_gaze_filter_in_context_string(self):
        """Test that gaze filter appears in context string when set."""
        from core.gui_state import GUIState

        state = GUIState()
        state.sequence_gaze_filter = "at_camera"

        context = state.to_context_string()
        assert "SEQUENCE GAZE FILTER: at_camera" in context

    def test_gaze_filter_not_in_context_when_none(self):
        """Test that gaze filter does not appear in context string when not set."""
        from core.gui_state import GUIState

        state = GUIState()
        assert state.sequence_gaze_filter is None

        context = state.to_context_string()
        assert "GAZE FILTER" not in context

    def test_clear_resets_gaze_filter(self):
        """Test that clear() resets the gaze filter."""
        from core.gui_state import GUIState

        state = GUIState()
        state.sequence_gaze_filter = "looking_left"
        state.clear()

        assert state.sequence_gaze_filter is None


class TestProjectSummaryGazeCount:
    """Tests for gaze analysis count in project summary."""

    def test_project_summary_includes_gaze_count(self):
        """Test that get_project_summary includes gaze analysis count."""
        from core.chat_tools import get_project_summary

        project = _create_chat_test_project()
        project.add_clips([
            Clip(
                id="clip-1",
                source_id="src-1",
                start_frame=0,
                end_frame=90,
                gaze_category="at_camera",
            ),
            Clip(
                id="clip-2",
                source_id="src-1",
                start_frame=90,
                end_frame=180,
            ),
        ])

        result = get_project_summary(project)

        assert "Gaze analyzed**: 1/2" in result["summary"]


class TestStorytellerTool:
    """Tests for the Storyteller chat tool."""

    def test_generate_storyteller_applies_resolved_sequence_order(self):
        """Storyteller should resolve NarrativeLine clip IDs back to project clips."""
        from core.chat_tools import generate_storyteller
        from core.remix.storyteller import NarrativeLine

        project = _create_chat_test_project()
        clip_1 = make_test_clip(
            "clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=60,
            description="First clip",
        )
        clip_2 = make_test_clip(
            "clip-2",
            source_id="src-2",
            start_frame=0,
            end_frame=48,
            description="Second clip",
        )
        project.add_clips([clip_1, clip_2])

        main_window = Mock()
        main_window._gui_state = Mock(analyze_selected_ids=["clip-1", "clip-2"], cut_selected_ids=[])
        main_window.sequence_tab = Mock()
        main_window.sequence_tab.timeline = Mock()
        main_window.sequence_tab.STATE_TIMELINE = "timeline"

        def fake_generate_narrative(
            clips_with_descriptions,
            target_duration_minutes,
            narrative_structure,
            theme,
        ):
            clips_by_id = {clip.id: clip for clip, _desc in clips_with_descriptions}
            assert clips_by_id["clip-1"]._duration_seconds == pytest.approx(2.0)
            assert clips_by_id["clip-2"]._duration_seconds == pytest.approx(2.0)
            return [
                NarrativeLine("clip-2", "Second clip", "opening", 1),
                NarrativeLine("clip-1", "First clip", "closing", 2),
            ]

        with patch("core.remix.storyteller.generate_narrative", side_effect=fake_generate_narrative):
            result = generate_storyteller(
                project,
                main_window,
                theme="contrast",
                structure="auto",
                target_duration_minutes=10,
            )

        assert result["success"] is True
        assert result["clip_count"] == 2

        add_calls = main_window.sequence_tab.timeline.add_clip.call_args_list
        assert [call.args[0].id for call in add_calls] == ["clip-2", "clip-1"]
        assert [call.args[1].id for call in add_calls] == ["src-2", "src-1"]
        main_window.sequence_tab.timeline.clear_timeline.assert_called_once()
        main_window.sequence_tab.timeline._on_zoom_fit.assert_called_once()
        main_window.sequence_tab._set_state.assert_called_once_with("timeline")


class TestStaccatoTool:
    """Tests for the Staccato chat tool."""

    def test_generate_staccato_rejects_missing_embeddings(self):
        """The agent tool should fail clearly instead of silently degrading."""
        from core.chat_tools import generate_staccato

        project = _create_chat_test_project()
        clip = make_test_clip(
            "clip-1",
            source_id="src-1",
            start_frame=0,
            end_frame=60,
        )
        clip.embedding = None
        project.add_clips([clip])

        main_window = Mock()
        main_window.sequence_tab = Mock()

        with patch("core.chat_tools._validate_path", return_value=(True, "", Path("/test/audio.mp3"))), \
             patch("core.analysis.audio.analyze_music_file", return_value=AudioAnalysis(beat_times=[1.0], duration_seconds=2.0)):
            result = generate_staccato(
                project,
                main_window,
                audio_path="/test/audio.mp3",
                strategy="beats",
            )

        assert result["success"] is False
        assert "embeddings" in result["error"].lower()
