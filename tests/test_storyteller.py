"""Unit tests for the Storyteller narrative generation module."""

import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from core.remix.storyteller import (
    NarrativeLine,
    StorytellerResult,
    generate_narrative,
    sequence_by_narrative,
    calculate_sequence_duration,
    _determine_narrative_role,
    DURATION_TARGETS,
    NARRATIVE_STRUCTURES,
)


class TestNarrativeLine:
    """Tests for the NarrativeLine dataclass."""

    def test_create_narrative_line(self):
        """Test creating a NarrativeLine."""
        line = NarrativeLine(
            clip_id="clip-123",
            description="A sunset over the ocean",
            narrative_role="opening",
            line_number=1,
        )
        assert line.clip_id == "clip-123"
        assert line.description == "A sunset over the ocean"
        assert line.narrative_role == "opening"
        assert line.line_number == 1


class TestStorytellerResult:
    """Tests for the StorytellerResult dataclass."""

    def test_create_result(self):
        """Test creating a StorytellerResult."""
        lines = [
            NarrativeLine("c1", "Desc 1", "opening", 1),
            NarrativeLine("c2", "Desc 2", "development", 2),
        ]
        result = StorytellerResult(
            narrative_lines=lines,
            theme="urban isolation",
            structure="three_act",
            target_duration_minutes=30,
            excluded_clip_ids=["c3"],
        )
        assert result.theme == "urban isolation"
        assert result.structure == "three_act"
        assert result.target_duration_minutes == 30
        assert len(result.narrative_lines) == 2
        assert result.excluded_clip_ids == ["c3"]

    def test_clip_order_property(self):
        """Test the clip_order property returns IDs in order."""
        lines = [
            NarrativeLine("c3", "Desc", "opening", 1),
            NarrativeLine("c1", "Desc", "development", 2),
            NarrativeLine("c5", "Desc", "closing", 3),
        ]
        result = StorytellerResult(
            narrative_lines=lines,
            theme=None,
            structure="auto",
            target_duration_minutes=None,
            excluded_clip_ids=[],
        )
        assert result.clip_order == ["c3", "c1", "c5"]


class TestDetermineNarrativeRole:
    """Tests for the _determine_narrative_role function."""

    def test_three_act_setup(self):
        """Test three-act setup role assignment."""
        role = _determine_narrative_role(1, 8, "three_act")
        assert role == "setup"
        role = _determine_narrative_role(2, 8, "three_act")
        assert role == "setup"

    def test_three_act_confrontation(self):
        """Test three-act confrontation role assignment."""
        role = _determine_narrative_role(3, 8, "three_act")
        assert role == "confrontation"
        role = _determine_narrative_role(6, 8, "three_act")
        assert role == "confrontation"

    def test_three_act_resolution(self):
        """Test three-act resolution role assignment."""
        role = _determine_narrative_role(7, 8, "three_act")
        assert role == "resolution"
        role = _determine_narrative_role(8, 8, "three_act")
        assert role == "resolution"

    def test_chronological_beginning(self):
        """Test chronological beginning role."""
        role = _determine_narrative_role(1, 9, "chronological")
        assert role == "beginning"

    def test_chronological_middle(self):
        """Test chronological middle role."""
        role = _determine_narrative_role(5, 9, "chronological")
        assert role == "middle"

    def test_chronological_end(self):
        """Test chronological end role."""
        role = _determine_narrative_role(9, 9, "chronological")
        assert role == "end"

    def test_thematic_theme_a(self):
        """Test thematic theme_a role."""
        role = _determine_narrative_role(1, 4, "thematic")
        assert role == "theme_a"
        role = _determine_narrative_role(2, 4, "thematic")
        assert role == "theme_a"

    def test_thematic_theme_b(self):
        """Test thematic theme_b role."""
        role = _determine_narrative_role(3, 4, "thematic")
        assert role == "theme_b"
        role = _determine_narrative_role(4, 4, "thematic")
        assert role == "theme_b"

    def test_auto_opening_closing(self):
        """Test auto structure opening/closing roles."""
        role = _determine_narrative_role(1, 5, "auto")
        assert role == "opening"
        role = _determine_narrative_role(5, 5, "auto")
        assert role == "closing"

    def test_auto_development(self):
        """Test auto structure development role."""
        role = _determine_narrative_role(3, 5, "auto")
        assert role == "development"

    def test_empty_total(self):
        """Test handling of zero total clips."""
        role = _determine_narrative_role(1, 0, "three_act")
        assert role == "unknown"


class TestGenerateNarrative:
    """Tests for the generate_narrative function with mocked LLM."""

    @pytest.fixture
    def mock_clips(self):
        """Create mock clips with descriptions."""
        clips = []
        for i in range(5):
            clip = Mock()
            clip.id = f"clip-{i}"
            clip.duration_frames = 300  # 10 seconds at 30fps
            clip._duration_seconds = 10.0
            clips.append(clip)
        return clips

    def test_generate_narrative_success(self, mock_clips):
        """Test successful narrative generation."""
        # Create clips with descriptions
        clips_with_desc = [
            (mock_clips[0], "A busy city street at dawn"),
            (mock_clips[1], "A person sitting alone in a park"),
            (mock_clips[2], "Traffic rushing by"),
            (mock_clips[3], "Someone looking at the sky"),
            (mock_clips[4], "An empty bench"),
        ]

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "selected": ["c1", "c3", "c2", "c5"],
            "excluded": ["c4"],
            "structure_used": "three_act",
        })

        # Mock the settings module
        mock_settings = Mock()
        mock_settings.exquisite_corpus_model = "gemini-2.5-flash"
        mock_settings.exquisite_corpus_temperature = 0.7

        with patch("core.settings.load_settings", return_value=mock_settings), \
             patch("core.settings.get_llm_api_key", return_value="test-api-key"), \
             patch("litellm.completion", return_value=mock_response):

            result = generate_narrative(
                clips_with_descriptions=clips_with_desc,
                target_duration_minutes=30,
                narrative_structure="three_act",
                theme="urban isolation",
            )

            assert len(result) == 4
            assert result[0].clip_id == "clip-0"  # c1 maps to clip-0
            assert result[1].clip_id == "clip-2"  # c3 maps to clip-2
            assert result[0].line_number == 1
            assert result[1].line_number == 2

    def test_generate_narrative_with_markdown_response(self, mock_clips):
        """Test handling of markdown-wrapped JSON response."""
        clips_with_desc = [(mock_clips[0], "Test description")]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """```json
{"selected": ["c1"], "excluded": [], "structure_used": "auto"}
```"""

        mock_settings = Mock()
        mock_settings.exquisite_corpus_model = "gemini-2.5-flash"
        mock_settings.exquisite_corpus_temperature = 0.7

        with patch("core.settings.load_settings", return_value=mock_settings), \
             patch("core.settings.get_llm_api_key", return_value="test-api-key"), \
             patch("litellm.completion", return_value=mock_response):

            result = generate_narrative(
                clips_with_descriptions=clips_with_desc,
                target_duration_minutes=None,
                narrative_structure="auto",
            )

            assert len(result) == 1

    def test_generate_narrative_invalid_json(self, mock_clips):
        """Test handling of invalid JSON response."""
        clips_with_desc = [(mock_clips[0], "Test description")]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"

        mock_settings = Mock()
        mock_settings.exquisite_corpus_model = "gemini-2.5-flash"
        mock_settings.exquisite_corpus_temperature = 0.7

        with patch("core.settings.load_settings", return_value=mock_settings), \
             patch("core.settings.get_llm_api_key", return_value="test-api-key"), \
             patch("litellm.completion", return_value=mock_response):

            with pytest.raises(ValueError, match="LLM did not return valid JSON"):
                generate_narrative(
                    clips_with_descriptions=clips_with_desc,
                    target_duration_minutes=None,
                    narrative_structure="auto",
                )


class TestSequenceByNarrative:
    """Tests for the sequence_by_narrative function."""

    def test_sequence_by_narrative_success(self):
        """Test creating sequence from narrative lines."""
        # Create mock clips and sources
        clips = {}
        sources = {}
        for i in range(3):
            clip = Mock()
            clip.id = f"clip-{i}"
            clip.source_id = f"source-{i}"
            clips[clip.id] = clip

            source = Mock()
            source.id = f"source-{i}"
            sources[source.id] = source

        narrative_lines = [
            NarrativeLine("clip-2", "Desc", "opening", 1),
            NarrativeLine("clip-0", "Desc", "development", 2),
            NarrativeLine("clip-1", "Desc", "closing", 3),
        ]

        result = sequence_by_narrative(narrative_lines, clips, sources)

        assert len(result) == 3
        assert result[0][0].id == "clip-2"
        assert result[1][0].id == "clip-0"
        assert result[2][0].id == "clip-1"

    def test_sequence_by_narrative_missing_clip(self):
        """Test handling of missing clip in narrative."""
        clips = {"clip-1": Mock(id="clip-1", source_id="source-1")}
        sources = {"source-1": Mock(id="source-1")}

        narrative_lines = [
            NarrativeLine("clip-1", "Desc", "opening", 1),
            NarrativeLine("clip-missing", "Desc", "closing", 2),
        ]

        result = sequence_by_narrative(narrative_lines, clips, sources)

        # Should only include the valid clip
        assert len(result) == 1
        assert result[0][0].id == "clip-1"

    def test_sequence_by_narrative_missing_source(self):
        """Test handling of missing source for clip."""
        clip = Mock(id="clip-1", source_id="source-missing")
        clips = {"clip-1": clip}
        sources = {}  # No sources

        narrative_lines = [
            NarrativeLine("clip-1", "Desc", "opening", 1),
        ]

        result = sequence_by_narrative(narrative_lines, clips, sources)

        # Should skip clips without sources
        assert len(result) == 0


class TestCalculateSequenceDuration:
    """Tests for the calculate_sequence_duration function."""

    def test_calculate_duration(self):
        """Test calculating total duration."""
        clips = []
        for i in range(3):
            clip = Mock()
            clip.duration_seconds = Mock(return_value=10.0)  # 10 seconds each
            source = Mock()
            source.fps = 30.0
            clips.append((clip, source))

        duration = calculate_sequence_duration(clips)

        assert duration == 30.0

    def test_calculate_duration_empty(self):
        """Test calculating duration of empty sequence."""
        duration = calculate_sequence_duration([])
        assert duration == 0.0


class TestDurationTargets:
    """Tests for duration target constants."""

    def test_all_targets_defined(self):
        """Test that all expected duration targets are defined."""
        assert "10min" in DURATION_TARGETS
        assert "30min" in DURATION_TARGETS
        assert "1hr" in DURATION_TARGETS
        assert "90min" in DURATION_TARGETS
        assert "all" in DURATION_TARGETS

    def test_target_has_required_keys(self):
        """Test that each target has required keys."""
        for key, target in DURATION_TARGETS.items():
            assert "target" in target
            assert "min" in target
            assert "max" in target


class TestNarrativeStructures:
    """Tests for narrative structure constants."""

    def test_all_structures_defined(self):
        """Test that all expected structures are defined."""
        assert "three_act" in NARRATIVE_STRUCTURES
        assert "chronological" in NARRATIVE_STRUCTURES
        assert "thematic" in NARRATIVE_STRUCTURES
        assert "auto" in NARRATIVE_STRUCTURES
