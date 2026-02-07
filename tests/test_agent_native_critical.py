"""Tests for agent-native critical recommendations.

Critical #1: Analysis metadata summary in context
Critical #2: Background worker status in context
Critical #3: delete_clips tool
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.gui_state import GUIState
from models.clip import Clip, Source


# -- Helpers ------------------------------------------------------------------

def _make_clip(clip_id: str, **kwargs) -> Clip:
    return Clip(
        id=clip_id,
        source_id="src1",
        start_frame=0,
        end_frame=90,
        **kwargs,
    )


# -- Critical #1: Analysis metadata in system prompt -------------------------

class TestAnalysisMetadataContext:
    """Analysis coverage counting logic produces correct summaries."""

    @staticmethod
    def _compute_analysis_lines(clips):
        """Replicate the analysis coverage logic from _build_system_prompt."""
        total_clips = len(clips)
        analysis_counts = {}
        if total_clips > 0:
            for field_name in ("shot_type", "dominant_colors", "description",
                               "transcript", "detected_objects", "person_count",
                               "extracted_texts", "cinematography"):
                count = sum(
                    1 for c in clips
                    if getattr(c, field_name, None) is not None
                )
                if count > 0:
                    analysis_counts[field_name] = count
        if analysis_counts:
            parts = [f"{k}: {v}/{total_clips}" for k, v in analysis_counts.items()]
            return f"Analysis Coverage: {', '.join(parts)}"
        return ""

    def test_analysis_counts_computed(self):
        """Analysis coverage includes non-zero fields."""
        clips = [
            _make_clip("c1", shot_type="wide", description="A wide shot"),
            _make_clip("c2", shot_type="close-up"),
            _make_clip("c3"),  # no analysis
        ]
        result = self._compute_analysis_lines(clips)

        assert "shot_type: 2/3" in result
        assert "description: 1/3" in result

    def test_no_analysis_coverage_when_no_clips(self):
        """No analysis line when no clips exist."""
        result = self._compute_analysis_lines([])
        assert result == ""

    def test_no_analysis_coverage_when_nothing_analyzed(self):
        """No analysis line when clips have no analysis."""
        clips = [_make_clip("c1"), _make_clip("c2")]
        result = self._compute_analysis_lines(clips)
        assert result == ""

    def test_all_fields_counted(self):
        """All analysis field types are counted when present."""
        clip = _make_clip(
            "c1",
            shot_type="wide",
            dominant_colors=[(255, 0, 0)],
            description="test",
            person_count=2,
        )
        result = self._compute_analysis_lines([clip])

        assert "shot_type: 1/1" in result
        assert "dominant_colors: 1/1" in result
        assert "description: 1/1" in result
        assert "person_count: 1/1" in result


# -- Critical #2: Background worker status in context ------------------------

class TestProcessingStatusContext:
    """GUIState.processing_operations appears in context string."""

    def test_processing_shows_in_context(self):
        state = GUIState()
        state.set_processing("scene_detection", "running on video.mp4")

        ctx = state.to_context_string()
        assert "PROCESSING:" in ctx
        assert "scene_detection: running on video.mp4" in ctx

    def test_no_processing_omits_line(self):
        state = GUIState()
        ctx = state.to_context_string()
        assert "PROCESSING:" not in ctx

    def test_clear_processing_removes(self):
        state = GUIState()
        state.set_processing("analysis", "colors on 5 clips")
        state.clear_processing("analysis")

        ctx = state.to_context_string()
        assert "PROCESSING:" not in ctx

    def test_multiple_processing_ops(self):
        state = GUIState()
        state.set_processing("download", "https://example.com/video.mp4")
        state.set_processing("analysis", "shot_type on 10 clips")

        ctx = state.to_context_string()
        assert "download:" in ctx
        assert "analysis:" in ctx

    def test_clear_resets_processing(self):
        state = GUIState()
        state.set_processing("download", "running")
        state.clear()

        assert state.processing_operations == {}


# -- Critical #3: delete_clips tool ------------------------------------------

class TestDeleteClipsTool:
    """delete_clips agent tool validates and removes clips."""

    def _make_project(self, clips, sequence_clip_ids=None):
        """Create a mock project with given clips and optional sequence."""
        from core.project import Project

        project = Project.__new__(Project)
        project._sources = [Source(id="src1", file_path=Path("/video.mp4"), fps=30.0)]
        project._clips = list(clips)
        project._frames = []
        project._dirty = False
        project._caches = {}
        project._observers = []

        # Set up sequence
        if sequence_clip_ids:
            seq = MagicMock()
            track = MagicMock()
            track.clips = [
                MagicMock(source_clip_id=cid) for cid in sequence_clip_ids
            ]
            seq.tracks = [track]
            project.sequence = seq
        else:
            project.sequence = None

        return project

    def test_delete_clips_removes_from_project(self):
        from core.chat_tools import delete_clips

        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        project = self._make_project(clips)

        result = delete_clips(project=project, clip_ids=["c1", "c3"])

        assert result["success"] is True
        assert result["removed_count"] == 2
        assert set(result["removed_ids"]) == {"c1", "c3"}
        assert result["remaining_clips"] == 1

    def test_delete_clips_rejects_sequence_clips(self):
        from core.chat_tools import delete_clips

        clips = [_make_clip("c1"), _make_clip("c2")]
        project = self._make_project(clips, sequence_clip_ids=["c1"])

        result = delete_clips(project=project, clip_ids=["c1"])

        assert result["success"] is False
        assert "sequence" in result["error"].lower()
        assert result["clips_in_sequence"] == ["c1"]

    def test_delete_clips_force_removes_from_sequence(self):
        from core.chat_tools import delete_clips

        clips = [_make_clip("c1"), _make_clip("c2")]
        project = self._make_project(clips, sequence_clip_ids=["c1"])

        result = delete_clips(project=project, clip_ids=["c1"], force=True)

        assert result["success"] is True
        assert result["removed_count"] == 1
        assert result["removed_from_sequence"] == 1

    def test_delete_clips_empty_ids(self):
        from core.chat_tools import delete_clips

        clips = [_make_clip("c1")]
        project = self._make_project(clips)

        result = delete_clips(project=project, clip_ids=[])

        assert result["success"] is False
        assert "No clip IDs" in result["error"]

    def test_delete_clips_not_found(self):
        from core.chat_tools import delete_clips

        clips = [_make_clip("c1")]
        project = self._make_project(clips)

        result = delete_clips(project=project, clip_ids=["nonexistent"])

        assert result["success"] is False
        assert "nonexistent" in result["not_found"]


class TestProjectRemoveClips:
    """Project.remove_clips method works correctly."""

    def test_remove_clips_returns_removed(self):
        from core.project import Project

        project = Project.__new__(Project)
        project._sources = []
        project._clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        project._frames = []
        project._dirty = False
        project._caches = {}
        project._observers = []
        project.sequence = None

        removed = project.remove_clips(["c1", "c3"])

        assert len(removed) == 2
        assert {c.id for c in removed} == {"c1", "c3"}
        assert len(project.clips) == 1
        assert project.clips[0].id == "c2"

    def test_remove_clips_nonexistent_returns_empty(self):
        from core.project import Project

        project = Project.__new__(Project)
        project._sources = []
        project._clips = [_make_clip("c1")]
        project._frames = []
        project._dirty = False
        project._caches = {}
        project._observers = []
        project.sequence = None

        removed = project.remove_clips(["nonexistent"])

        assert len(removed) == 0
        assert len(project.clips) == 1


# -- High #7: list_clips filtering -------------------------------------------

class TestListClipsFiltering:
    """list_clips supports filtering and sorting."""

    def _make_project_mock(self, clips):
        project = MagicMock()
        project.clips = clips
        project.sources = []
        project.sources_by_id = {
            "src1": Source(id="src1", file_path=Path("/video.mp4"), fps=30.0),
            "src2": Source(id="src2", file_path=Path("/other.mp4"), fps=24.0),
        }
        project.clips_by_source = {}
        return project

    def test_filter_by_source_id(self):
        from core.chat_tools import list_clips

        clips = [
            _make_clip("c1"),
            Clip(id="c2", source_id="src2", start_frame=0, end_frame=60),
            _make_clip("c3"),
        ]
        project = self._make_project_mock(clips)

        result = list_clips(main_window=None, project=project, source_id="src1")
        assert result["count"] == 2
        assert all(c["source_id"] == "src1" for c in result["clips"])

    def test_filter_by_shot_type(self):
        from core.chat_tools import list_clips

        clips = [
            _make_clip("c1", shot_type="wide shot"),
            _make_clip("c2", shot_type="close-up"),
            _make_clip("c3", shot_type="wide shot"),
        ]
        project = self._make_project_mock(clips)

        result = list_clips(main_window=None, project=project, shot_type="wide shot")
        assert result["count"] == 2

    def test_filter_has_description(self):
        from core.chat_tools import list_clips

        clips = [
            _make_clip("c1", description="A beautiful scene"),
            _make_clip("c2"),
            _make_clip("c3", description="Another clip"),
        ]
        project = self._make_project_mock(clips)

        result = list_clips(main_window=None, project=project, has_description=True)
        assert result["count"] == 2

        result = list_clips(main_window=None, project=project, has_description=False)
        assert result["count"] == 1

    def test_sort_by_duration(self):
        from core.chat_tools import list_clips

        clips = [
            Clip(id="short", source_id="src1", start_frame=0, end_frame=30),
            Clip(id="long", source_id="src1", start_frame=0, end_frame=300),
            Clip(id="medium", source_id="src1", start_frame=0, end_frame=90),
        ]
        project = self._make_project_mock(clips)

        result = list_clips(main_window=None, project=project, sort_by="duration")
        ids = [c["id"] for c in result["clips"]]
        assert ids == ["long", "medium", "short"]

    def test_limit_with_total(self):
        from core.chat_tools import list_clips

        clips = [_make_clip(f"c{i}") for i in range(10)]
        project = self._make_project_mock(clips)

        result = list_clips(main_window=None, project=project, limit=3)
        assert result["count"] == 3
        assert result["total_matching"] == 10

    def test_no_filters_returns_all(self):
        from core.chat_tools import list_clips

        clips = [_make_clip("c1"), _make_clip("c2")]
        project = self._make_project_mock(clips)

        result = list_clips(main_window=None, project=project)
        assert result["count"] == 2
