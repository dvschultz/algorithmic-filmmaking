"""Unit tests for content-aware chat tools."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from core.project import Project
from core.transcription import TranscriptSegment
from models.clip import Source, Clip


def create_test_project():
    """Create a project with test data for chat tool testing."""
    project = Project.new(name="Test Project")

    # Add sources
    source1 = Source(
        id="src-1",
        file_path=Path("/test/video1.mp4"),
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    source2 = Source(
        id="src-2",
        file_path=Path("/test/video2.mp4"),
        duration_seconds=60.0,
        fps=24.0,
        width=1280,
        height=720,
    )
    project.add_source(source1)
    project.add_source(source2)

    return project


def create_test_clip(
    clip_id: str,
    source_id: str = "src-1",
    start_frame: int = 0,
    end_frame: int = 90,
    shot_type: str = None,
    transcript_text: str = None,
    dominant_colors: list = None,
):
    """Helper to create test clips with optional attributes."""
    transcript = None
    if transcript_text:
        transcript = [
            TranscriptSegment(
                start_time=0.0,
                end_time=3.0,
                text=transcript_text,
                confidence=0.95,
            )
        ]

    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        shot_type=shot_type,
        transcript=transcript,
        dominant_colors=dominant_colors,
    )


class TestSearchTranscripts:
    """Tests for search_transcripts tool."""

    def test_search_finds_matching_clips(self):
        """Test search finds clips with matching transcript text."""
        from core.chat_tools import search_transcripts

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="Hello world, this is a test"),
            create_test_clip("clip-2", transcript_text="Goodbye world, farewell"),
            create_test_clip("clip-3", transcript_text="No match here"),
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="HELLO World"),
            create_test_clip("clip-2", transcript_text="hello world"),
        ])

        result = search_transcripts(project, "HELLO")

        assert result["success"] is True
        assert result["match_count"] == 2

    def test_search_case_sensitive(self):
        """Test case sensitive search."""
        from core.chat_tools import search_transcripts

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="HELLO World"),
            create_test_clip("clip-2", transcript_text="hello world"),
        ])

        result = search_transcripts(project, "HELLO", case_sensitive=True)

        assert result["success"] is True
        assert result["match_count"] == 1
        assert result["matches"][0]["clip_id"] == "clip-1"

    def test_search_empty_query_returns_error(self):
        """Test empty query returns error."""
        from core.chat_tools import search_transcripts

        project = create_test_project()

        result = search_transcripts(project, "")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_search_no_matches(self):
        """Test search returns empty results when no matches."""
        from core.chat_tools import search_transcripts

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="Hello world"),
        ])

        result = search_transcripts(project, "nonexistent")

        assert result["success"] is True
        assert result["match_count"] == 0
        assert result["matches"] == []

    def test_search_skips_clips_without_transcripts(self):
        """Test search skips clips without transcripts."""
        from core.chat_tools import search_transcripts

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="Hello world"),
            create_test_clip("clip-2", transcript_text=None),  # No transcript
        ])

        result = search_transcripts(project, "hello")

        assert result["success"] is True
        assert result["match_count"] == 1

    def test_search_includes_context(self):
        """Test search results include context around match."""
        from core.chat_tools import search_transcripts

        project = create_test_project()
        project.add_clips([
            create_test_clip(
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up"),
            create_test_clip("clip-2", shot_type="close_up"),
            create_test_clip("clip-3", shot_type="wide_shot"),
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

        project = create_test_project()
        # Red-ish colors
        red_colors = [(255, 0, 0), (200, 50, 50), (180, 30, 30)]
        # Blue-ish colors
        blue_colors = [(0, 0, 255), (50, 50, 200), (30, 30, 180)]

        project.add_clips([
            create_test_clip("clip-1", dominant_colors=red_colors),
            create_test_clip("clip-2", dominant_colors=red_colors),  # Similar to clip-1
            create_test_clip("clip-3", dominant_colors=blue_colors),  # Different
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", start_frame=0, end_frame=90),  # 3 seconds at 30fps
            create_test_clip("clip-2", start_frame=0, end_frame=90),  # Same duration
            create_test_clip("clip-3", start_frame=0, end_frame=900),  # 30 seconds
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

        project = create_test_project()

        result = find_similar_clips(project, "nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_invalid_criteria_returns_error(self):
        """Test invalid criteria returns error."""
        from core.chat_tools import find_similar_clips

        project = create_test_project()
        project.add_clips([create_test_clip("clip-1")])

        result = find_similar_clips(project, "clip-1", criteria=["invalid_criterion"])

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_respects_limit(self):
        """Test result limit is respected."""
        from core.chat_tools import find_similar_clips

        project = create_test_project()
        # Add many clips
        clips = [
            create_test_clip(f"clip-{i}", shot_type="close_up")
            for i in range(20)
        ]
        project.add_clips(clips)

        result = find_similar_clips(project, "clip-0", limit=5)

        assert result["success"] is True
        assert len(result["similar_clips"]) <= 5

    def test_default_criteria(self):
        """Test default criteria are used when none specified."""
        from core.chat_tools import find_similar_clips

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up"),
            create_test_clip("clip-2", shot_type="close_up"),
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up"),
            create_test_clip("clip-2", shot_type="close_up"),
            create_test_clip("clip-3", shot_type="wide_shot"),
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", start_frame=0, end_frame=30),   # 1 second (short)
            create_test_clip("clip-2", start_frame=0, end_frame=150),  # 5 seconds (medium)
            create_test_clip("clip-3", start_frame=0, end_frame=450),  # 15 seconds (long)
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", source_id="src-1"),
            create_test_clip("clip-2", source_id="src-1"),
            create_test_clip("clip-3", source_id="src-2"),
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

        project = create_test_project()
        # Warm colors (red/orange hues)
        warm_colors = [(255, 100, 50), (200, 80, 40)]
        # Cool colors (blue hues)
        cool_colors = [(50, 100, 255), (40, 80, 200)]

        project.add_clips([
            create_test_clip("clip-1", dominant_colors=warm_colors),
            create_test_clip("clip-2", dominant_colors=cool_colors),
            create_test_clip("clip-3", dominant_colors=None),  # Unanalyzed
        ])

        result = group_clips_by(project, "color")

        assert result["success"] is True
        assert result["criterion"] == "color"
        # Should have groups for different palette classifications
        assert "unanalyzed" in result["groups"]

    def test_invalid_criterion_returns_error(self):
        """Test invalid criterion returns error."""
        from core.chat_tools import group_clips_by

        project = create_test_project()

        result = group_clips_by(project, "invalid")

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_group_handles_unknown_shot_types(self):
        """Test clips without shot type are grouped as 'unknown'."""
        from core.chat_tools import group_clips_by

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type=None),
            create_test_clip("clip-2", shot_type="close_up"),
        ])

        result = group_clips_by(project, "shot_type")

        assert result["success"] is True
        assert "unknown" in result["groups"]


class TestFilterClipsWithSearch:
    """Tests for filter_clips with search_query parameter."""

    def test_filter_by_search_query(self):
        """Test filtering clips by transcript search query."""
        from core.chat_tools import filter_clips

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="Hello world"),
            create_test_clip("clip-2", transcript_text="Goodbye moon"),
            create_test_clip("clip-3", transcript_text=None),
        ])

        result = filter_clips(project, search_query="world")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_search_case_insensitive(self):
        """Test search query is case insensitive."""
        from core.chat_tools import filter_clips

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="HELLO WORLD"),
        ])

        result = filter_clips(project, search_query="hello")

        assert len(result) == 1

    def test_filter_combined_with_search(self):
        """Test search query combined with other filters."""
        from core.chat_tools import filter_clips

        project = create_test_project()
        project.add_clips([
            create_test_clip(
                "clip-1",
                shot_type="close_up",
                transcript_text="Hello world"
            ),
            create_test_clip(
                "clip-2",
                shot_type="wide_shot",
                transcript_text="Hello world"
            ),
            create_test_clip(
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text=None),
            create_test_clip("clip-2", transcript_text="Hello"),
        ])

        result = filter_clips(project, search_query="anything")

        # clip-1 should be excluded (no transcript)
        assert all(r["id"] != "clip-1" for r in result)

    def test_filter_existing_filters_still_work(self):
        """Test existing filters still work without search_query."""
        from core.chat_tools import filter_clips

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up"),
            create_test_clip("clip-2", shot_type="wide_shot"),
        ])

        result = filter_clips(project, shot_type="close_up")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_duration_filters(self):
        """Test duration filters still work."""
        from core.chat_tools import filter_clips

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", start_frame=0, end_frame=30),   # 1 second
            create_test_clip("clip-2", start_frame=0, end_frame=150),  # 5 seconds
            create_test_clip("clip-3", start_frame=0, end_frame=300),  # 10 seconds
        ])

        result = filter_clips(project, min_duration=2.0, max_duration=8.0)

        assert len(result) == 1
        assert result[0]["id"] == "clip-2"


class TestToolIntegration:
    """Integration tests for tool combinations."""

    def test_search_then_group(self):
        """Test searching transcripts then grouping results."""
        from core.chat_tools import search_transcripts, group_clips_by

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up", transcript_text="Hello world"),
            create_test_clip("clip-2", shot_type="wide_shot", transcript_text="Hello there"),
            create_test_clip("clip-3", shot_type="close_up", transcript_text="Goodbye"),
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up"),
            create_test_clip("clip-2", shot_type="close_up"),
            create_test_clip("clip-3", shot_type="wide_shot"),
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", shot_type="close_up"),
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

        project = create_test_project()
        project.add_clips([create_test_clip("clip-1")])

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

        project = create_test_project()
        project.add_clips([create_test_clip("clip-1")])

        executor = ToolExecutor(project=project)

        # Call a tool that requires project
        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "list_clips",
                "arguments": "{}"
            }
        })

        assert result["success"] is True
        assert len(result["result"]) == 1

    def test_executor_handles_missing_project(self):
        """Test ToolExecutor reports error when project required but missing."""
        from core.tool_executor import ToolExecutor

        executor = ToolExecutor(project=None)

        result = executor.execute({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "list_clips",
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
        """Test ToolExecutor respects busy_check callback."""
        from core.tool_executor import ToolExecutor, CONFLICTING_TOOLS
        from unittest.mock import MagicMock

        # Mock busy check that returns True (busy)
        busy_check = MagicMock(return_value=True)

        executor = ToolExecutor(
            project=create_test_project(),
            busy_check=busy_check
        )

        # detect_scenes is in CONFLICTING_TOOLS
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

        project = create_test_project()
        project.add_clips([
            create_test_clip("clip-1", transcript_text="hello"),
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

        project = create_test_project()
        project.add_clips([create_test_clip("clip-1")])

        result = filter_clips(project)

        # filter_clips returns a list directly, not a dict with success
        assert isinstance(result, list)


class TestPlanExecutionTools:
    """Tests for plan execution tools."""

    def _create_mock_main_window(self):
        """Create a mock main window with GUI state."""
        from core.gui_state import GUIState

        main_window = Mock()
        main_window._gui_state = GUIState()
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
        """Create mock main window with GUI state and project."""
        from core.gui_state import GUIState
        from core.project import Project

        main_window = Mock()
        main_window._gui_state = GUIState()
        main_window.project = Project.new(name="Untitled Project")
        return main_window

    def test_present_plan_stores_pending_action_for_unnamed_project(self):
        """Test that present_plan stores pending action when project is unnamed."""
        from core.chat_tools import present_plan

        main_window = self._create_mock_main_window_with_project()

        steps = ["Download videos", "Run detection", "Export"]
        result = present_plan(main_window, steps, "Test workflow")

        # Should return action_required
        assert result["success"] is True
        assert result["action_required"] == "name_project"

        # Should store pending action in GUI state
        pending = main_window._gui_state.pending_action
        assert pending is not None
        assert pending["type"] == "name_project_then_plan"
        assert pending["pending_steps"] == steps

    def test_set_project_name_includes_next_action_when_pending(self):
        """Test that set_project_name reminds agent to call present_plan."""
        from core.chat_tools import present_plan, set_project_name

        main_window = self._create_mock_main_window_with_project()

        # First, present_plan with unnamed project
        # Steps must contain project keywords (download, detect, etc.) to trigger the check
        steps = ["Download videos", "Detect scenes"]
        present_plan(main_window, steps, "Test plan")

        # Now set the project name
        result = set_project_name(main_window, main_window.project, "My Project")

        assert result["success"] is True
        assert result["new_name"] == "My Project"
        # Should include next action
        assert result.get("next_action") == "present_plan"
        assert result["next_action_args"]["steps"] == ["Download videos", "Detect scenes"]
        assert "present_plan" in result["message"]

        # Pending action should be cleared
        assert main_window._gui_state.pending_action is None

    def test_gui_state_context_shows_pending_action(self):
        """Test that GUI state context string includes pending action."""
        from core.gui_state import GUIState

        state = GUIState()
        state.set_pending_action(
            "name_project_then_plan",
            pending_steps=["Step 1", "Step 2"],
            user_response="kittyjump"
        )

        context = state.to_context_string()

        assert "PENDING ACTION" in context
        assert "name_project_then_plan" in context
        assert "kittyjump" in context
        assert "set_project_name" in context

    def test_pending_action_cleared_after_handling(self):
        """Test that pending action can be cleared."""
        from core.gui_state import GUIState

        state = GUIState()
        state.set_pending_action("test_action", data="test")

        assert state.pending_action is not None

        state.clear_pending_action()

        assert state.pending_action is None
