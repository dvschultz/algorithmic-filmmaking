"""Tests for expanded filter_clips parameters and return fields (Units 4-5)."""

from pathlib import Path

import pytest

from core.project import Project
from models.clip import Source, Clip, ExtractedText
from models.cinematography import CinematographyAnalysis


def _create_test_project() -> Project:
    """Create a project with one source for filter_clips testing."""
    project = Project.new(name="Filter Test Project")
    project.add_source(Source(
        id="src-1",
        file_path=Path("/test/video1.mp4"),
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    ))
    return project


def _make_clip(clip_id: str, **kwargs) -> Clip:
    """Create a Clip with specific fields set for testing."""
    defaults = {
        "id": clip_id,
        "source_id": "src-1",
        "start_frame": 0,
        "end_frame": 90,
    }
    defaults.update(kwargs)
    return Clip(**defaults)


class TestGazeCategoryFilter:
    """Tests for gaze_category filter parameter."""

    def test_gaze_category_exact_match(self):
        """Test gaze_category filters to only matching clips."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", gaze_category="at_camera"),
            _make_clip("clip-2", gaze_category="looking_left"),
            _make_clip("clip-3", gaze_category="at_camera"),
            _make_clip("clip-4"),  # No gaze data
        ])

        result = filter_clips(project, gaze_category="at_camera")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1", "clip-3"]

    def test_gaze_category_no_match(self):
        """Test gaze_category returns empty when no clips match."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", gaze_category="looking_left"),
        ])

        result = filter_clips(project, gaze_category="at_camera")

        assert result == []

    def test_gaze_category_none_clips_excluded(self):
        """Test clips without gaze_category are excluded."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1"),  # gaze_category is None
        ])

        result = filter_clips(project, gaze_category="at_camera")

        assert result == []


class TestBrightnessFilter:
    """Tests for min_brightness / max_brightness filter parameters."""

    def test_min_brightness_filters_dark_clips(self):
        """Test min_brightness excludes clips below threshold."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-bright", average_brightness=0.8),
            _make_clip("clip-dark", average_brightness=0.2),
            _make_clip("clip-mid", average_brightness=0.5),
        ])

        result = filter_clips(project, min_brightness=0.5)

        ids = [r["id"] for r in result]
        assert "clip-bright" in ids
        assert "clip-mid" in ids
        assert "clip-dark" not in ids

    def test_max_brightness_filters_bright_clips(self):
        """Test max_brightness excludes clips above threshold."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-bright", average_brightness=0.9),
            _make_clip("clip-dark", average_brightness=0.1),
        ])

        result = filter_clips(project, max_brightness=0.5)

        ids = [r["id"] for r in result]
        assert ids == ["clip-dark"]

    def test_brightness_range_filter(self):
        """Test combined min/max brightness range."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", average_brightness=0.1),
            _make_clip("clip-2", average_brightness=0.5),
            _make_clip("clip-3", average_brightness=0.9),
        ])

        result = filter_clips(project, min_brightness=0.3, max_brightness=0.7)

        ids = [r["id"] for r in result]
        assert ids == ["clip-2"]

    def test_brightness_none_excluded(self):
        """Test clips with no brightness data are excluded when filter is active."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-analyzed", average_brightness=0.5),
            _make_clip("clip-unanalyzed"),  # average_brightness is None
        ])

        result = filter_clips(project, min_brightness=0.0)

        ids = [r["id"] for r in result]
        assert "clip-analyzed" in ids
        assert "clip-unanalyzed" not in ids


class TestVolumeFilter:
    """Tests for min_volume / max_volume filter parameters."""

    def test_min_volume_filters_quiet_clips(self):
        """Test min_volume excludes clips below threshold."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-loud", rms_volume=-5.0),
            _make_clip("clip-quiet", rms_volume=-40.0),
            _make_clip("clip-mid", rms_volume=-20.0),
        ])

        result = filter_clips(project, min_volume=-20.0)

        ids = [r["id"] for r in result]
        assert "clip-loud" in ids
        assert "clip-mid" in ids
        assert "clip-quiet" not in ids

    def test_max_volume_filters_loud_clips(self):
        """Test max_volume excludes clips above threshold."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-loud", rms_volume=-5.0),
            _make_clip("clip-quiet", rms_volume=-40.0),
        ])

        result = filter_clips(project, max_volume=-20.0)

        ids = [r["id"] for r in result]
        assert ids == ["clip-quiet"]

    def test_volume_none_excluded(self):
        """Test clips without volume data are excluded when filter is active."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-with-vol", rms_volume=-10.0),
            _make_clip("clip-no-vol"),  # rms_volume is None
        ])

        result = filter_clips(project, min_volume=-60.0)

        ids = [r["id"] for r in result]
        assert "clip-with-vol" in ids
        assert "clip-no-vol" not in ids


class TestOcrTextFilter:
    """Tests for search_ocr_text filter parameter."""

    def test_search_ocr_text_finds_match(self):
        """Test search_ocr_text finds clips with matching OCR text."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", extracted_texts=[
                ExtractedText(frame_number=10, text="EXIT DOOR", confidence=0.9, source="paddleocr"),
            ]),
            _make_clip("clip-2", extracted_texts=[
                ExtractedText(frame_number=20, text="ENTER HERE", confidence=0.9, source="paddleocr"),
            ]),
            _make_clip("clip-3"),  # No OCR data
        ])

        result = filter_clips(project, search_ocr_text="EXIT")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_search_ocr_text_case_insensitive(self):
        """Test OCR text search is case insensitive."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", extracted_texts=[
                ExtractedText(frame_number=10, text="Warning Sign", confidence=0.9, source="paddleocr"),
            ]),
        ])

        result = filter_clips(project, search_ocr_text="warning")

        assert len(result) == 1

    def test_search_ocr_text_no_data_excluded(self):
        """Test clips without OCR data are excluded."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1"),  # No extracted_texts
        ])

        result = filter_clips(project, search_ocr_text="anything")

        assert result == []


class TestTagsFilter:
    """Tests for search_tags filter parameter."""

    def test_search_tags_finds_match(self):
        """Test search_tags finds clips with matching tags."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", tags=["landscape", "sunset"]),
            _make_clip("clip-2", tags=["portrait", "indoor"]),
            _make_clip("clip-3", tags=[]),  # Empty tags
        ])

        result = filter_clips(project, search_tags="landscape")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_search_tags_case_insensitive(self):
        """Test tags search is case insensitive."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", tags=["Landscape", "Sunset"]),
        ])

        result = filter_clips(project, search_tags="landscape")

        assert len(result) == 1

    def test_search_tags_substring_match(self):
        """Test tags search uses substring matching."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", tags=["time-lapse", "nightscape"]),
        ])

        result = filter_clips(project, search_tags="scape")

        assert len(result) == 1

    def test_search_tags_no_tags_excluded(self):
        """Test clips without tags are excluded."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1"),  # tags defaults to empty list
        ])

        result = filter_clips(project, search_tags="anything")

        assert result == []


class TestNotesFilter:
    """Tests for search_notes filter parameter."""

    def test_search_notes_finds_match(self):
        """Test search_notes finds clips with matching notes."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", notes="Great establishing shot of the city"),
            _make_clip("clip-2", notes="Interior scene, needs color grading"),
            _make_clip("clip-3"),  # Empty notes
        ])

        result = filter_clips(project, search_notes="establishing")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_search_notes_case_insensitive(self):
        """Test notes search is case insensitive."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", notes="KEY SCENE for the edit"),
        ])

        result = filter_clips(project, search_notes="key scene")

        assert len(result) == 1

    def test_search_notes_empty_excluded(self):
        """Test clips with empty notes are excluded."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", notes=""),
        ])

        result = filter_clips(project, search_notes="anything")

        assert result == []


class TestCinematographyFilters:
    """Tests for cinematography_* filter parameters."""

    def _make_cine(self, **kwargs) -> CinematographyAnalysis:
        """Create a CinematographyAnalysis with specific overrides."""
        return CinematographyAnalysis(**kwargs)

    def test_cinematography_lighting_style(self):
        """Test cinematography_lighting_style filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(lighting_style="dramatic")),
            _make_clip("clip-2", cinematography=self._make_cine(lighting_style="natural")),
            _make_clip("clip-3"),  # No cinematography
        ])

        result = filter_clips(project, cinematography_lighting_style="dramatic")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_shot_size(self):
        """Test cinematography_shot_size filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(shot_size="CU")),
            _make_clip("clip-2", cinematography=self._make_cine(shot_size="MS")),
        ])

        result = filter_clips(project, cinematography_shot_size="CU")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_camera_angle(self):
        """Test cinematography_camera_angle filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(camera_angle="low_angle")),
            _make_clip("clip-2", cinematography=self._make_cine(camera_angle="eye_level")),
        ])

        result = filter_clips(project, cinematography_camera_angle="low_angle")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_camera_movement(self):
        """Test cinematography_camera_movement filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(camera_movement="pan")),
            _make_clip("clip-2", cinematography=self._make_cine(camera_movement="static")),
        ])

        result = filter_clips(project, cinematography_camera_movement="pan")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_subject_count(self):
        """Test cinematography_subject_count filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(subject_count="group")),
            _make_clip("clip-2", cinematography=self._make_cine(subject_count="single")),
        ])

        result = filter_clips(project, cinematography_subject_count="group")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_emotional_intensity(self):
        """Test cinematography_emotional_intensity filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(emotional_intensity="high")),
            _make_clip("clip-2", cinematography=self._make_cine(emotional_intensity="low")),
        ])

        result = filter_clips(project, cinematography_emotional_intensity="high")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_suggested_pacing(self):
        """Test cinematography_suggested_pacing filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(suggested_pacing="fast")),
            _make_clip("clip-2", cinematography=self._make_cine(suggested_pacing="slow")),
        ])

        result = filter_clips(project, cinematography_suggested_pacing="fast")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_cinematography_no_data_excluded(self):
        """Test clips without cinematography are excluded by cinematography filters."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1"),  # No cinematography
        ])

        result = filter_clips(project, cinematography_lighting_style="dramatic")

        assert result == []

    def test_cinematography_multiple_filters_combined(self):
        """Test multiple cinematography filters combine with AND logic."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=self._make_cine(
                lighting_style="dramatic", shot_size="CU")),
            _make_clip("clip-2", cinematography=self._make_cine(
                lighting_style="dramatic", shot_size="MS")),
            _make_clip("clip-3", cinematography=self._make_cine(
                lighting_style="natural", shot_size="CU")),
        ])

        result = filter_clips(
            project,
            cinematography_lighting_style="dramatic",
            cinematography_shot_size="CU"
        )

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]


class TestSimilarToClipId:
    """Tests for similar_to_clip_id filter parameter."""

    def _make_embedding(self, base_value: float, dim: int = 768) -> list[float]:
        """Create a simple normalized embedding for testing."""
        import numpy as np
        vec = np.full(dim, base_value, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def test_similar_to_clip_id_sorts_by_similarity(self):
        """Test results are sorted by embedding similarity."""
        from core.chat_tools import filter_clips
        import numpy as np

        # Create embeddings: anchor is similar to clip-2, less similar to clip-3
        anchor_emb = self._make_embedding(1.0)
        similar_emb = self._make_embedding(1.0)  # Same direction = similarity 1.0
        # Different direction
        different_emb = np.zeros(768, dtype=np.float64)
        different_emb[0] = 1.0  # Only one dimension set

        project = _create_test_project()
        project.add_clips([
            _make_clip("anchor", embedding=anchor_emb),
            _make_clip("clip-similar", embedding=similar_emb),
            _make_clip("clip-different", embedding=different_emb.tolist()),
        ])

        result = filter_clips(project, similar_to_clip_id="anchor")

        assert len(result) == 3  # All clips have embeddings
        # All results should have similarity_score
        for r in result:
            assert "similarity_score" in r
        # First result should be anchor itself (perfect match) or clip-similar
        # Both have score ~1.0, so just check ordering is descending
        scores = [r["similarity_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_similar_to_clip_id_invalid_id_returns_empty(self):
        """Test invalid clip ID returns empty list."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", embedding=self._make_embedding(1.0)),
        ])

        result = filter_clips(project, similar_to_clip_id="nonexistent")

        assert result == []

    def test_similar_to_clip_id_no_embedding_on_anchor(self):
        """Test anchor without embedding returns unranked results (no similarity_score)."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("anchor"),  # No embedding
            _make_clip("clip-2", embedding=self._make_embedding(1.0)),
        ])

        # Anchor has no embedding, so anchor_embedding will be None.
        # The function should return normal (unranked) results.
        result = filter_clips(project, similar_to_clip_id="anchor")

        # With no valid anchor embedding, the similarity branch won't execute,
        # so all clips pass through without similarity_score
        for r in result:
            assert "similarity_score" not in r

    def test_similar_to_clip_id_excludes_zero_embeddings(self):
        """Test clips with zero-vector embeddings are excluded from similarity results."""
        from core.chat_tools import filter_clips

        zero_embedding = [0.0] * 768

        project = _create_test_project()
        project.add_clips([
            _make_clip("anchor", embedding=self._make_embedding(1.0)),
            _make_clip("clip-zero", embedding=zero_embedding),
            _make_clip("clip-valid", embedding=self._make_embedding(0.5)),
        ])

        result = filter_clips(project, similar_to_clip_id="anchor")

        ids = [r["id"] for r in result]
        assert "clip-zero" not in ids
        assert "anchor" in ids
        assert "clip-valid" in ids

    def test_similar_to_clip_id_combined_with_gaze_filter(self):
        """Test similarity ranking combined with gaze filter (filter first, rank second)."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("anchor", embedding=self._make_embedding(1.0), gaze_category="at_camera"),
            _make_clip("clip-match", embedding=self._make_embedding(1.0), gaze_category="at_camera"),
            _make_clip("clip-nomatch", embedding=self._make_embedding(1.0), gaze_category="looking_left"),
        ])

        result = filter_clips(
            project,
            gaze_category="at_camera",
            similar_to_clip_id="anchor"
        )

        ids = [r["id"] for r in result]
        assert "clip-match" in ids
        assert "anchor" in ids
        assert "clip-nomatch" not in ids

    def test_similar_to_clip_id_clips_without_embedding_excluded(self):
        """Test clips without any embedding are excluded from similarity results."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("anchor", embedding=self._make_embedding(1.0)),
            _make_clip("clip-no-emb"),  # No embedding at all
            _make_clip("clip-has-emb", embedding=self._make_embedding(0.5)),
        ])

        result = filter_clips(project, similar_to_clip_id="anchor")

        ids = [r["id"] for r in result]
        assert "clip-no-emb" not in ids
        assert "clip-has-emb" in ids


class TestHasObjectSubstring:
    """Tests for upgraded has_object substring matching."""

    def test_has_object_substring_match(self):
        """Test has_object uses substring matching (not exact)."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", object_labels=["racecar", "track"]),
            _make_clip("clip-2", object_labels=["bicycle", "road"]),
        ])

        result = filter_clips(project, has_object="car")

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_has_object_exact_still_works(self):
        """Test exact label match still works with substring."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", object_labels=["car", "tree"]),
        ])

        result = filter_clips(project, has_object="car")

        assert len(result) == 1

    def test_has_object_substring_case_insensitive(self):
        """Test has_object is case insensitive."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", object_labels=["SportsCar"]),
        ])

        result = filter_clips(project, has_object="sportscar")

        assert len(result) == 1

    def test_has_object_checks_detected_objects(self):
        """Test has_object also searches detected_objects labels."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", detected_objects=[
                {"label": "motorcycle", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
            ]),
        ])

        result = filter_clips(project, has_object="motor")

        assert len(result) == 1


class TestExpandedReturnFields:
    """Tests for expanded return fields (Unit 5)."""

    def test_return_includes_average_brightness(self):
        """Test result includes average_brightness field."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", average_brightness=0.75),
        ])

        result = filter_clips(project)

        assert result[0]["average_brightness"] == 0.75

    def test_return_brightness_none_when_missing(self):
        """Test average_brightness is None when not analyzed."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([_make_clip("clip-1")])

        result = filter_clips(project)

        assert result[0]["average_brightness"] is None

    def test_return_includes_rms_volume(self):
        """Test result includes rms_volume field."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", rms_volume=-15.5),
        ])

        result = filter_clips(project)

        assert result[0]["rms_volume"] == -15.5

    def test_return_includes_tags(self):
        """Test result includes tags as list."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", tags=["landscape", "sunset"]),
        ])

        result = filter_clips(project)

        assert result[0]["tags"] == ["landscape", "sunset"]

    def test_return_tags_empty_list_default(self):
        """Test tags defaults to empty list when no tags set."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([_make_clip("clip-1")])

        result = filter_clips(project)

        assert result[0]["tags"] == []

    def test_return_includes_notes(self):
        """Test result includes notes field."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", notes="Important scene"),
        ])

        result = filter_clips(project)

        assert result[0]["notes"] == "Important scene"

    def test_return_notes_empty_string_default(self):
        """Test notes defaults to empty string."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([_make_clip("clip-1")])

        result = filter_clips(project)

        assert result[0]["notes"] == ""

    def test_return_includes_extracted_text(self):
        """Test result includes extracted_text from combined_text property."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", extracted_texts=[
                ExtractedText(frame_number=10, text="EXIT", confidence=0.9, source="paddleocr"),
                ExtractedText(frame_number=20, text="DOOR", confidence=0.8, source="paddleocr"),
            ]),
        ])

        result = filter_clips(project)

        assert result[0]["extracted_text"] == "EXIT | DOOR"

    def test_return_extracted_text_none_when_missing(self):
        """Test extracted_text is None when no OCR data."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([_make_clip("clip-1")])

        result = filter_clips(project)

        assert result[0]["extracted_text"] is None

    def test_return_includes_cinematography_dict(self):
        """Test result includes cinematography as dict with 7 curated fields."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", cinematography=CinematographyAnalysis(
                shot_size="CU",
                camera_angle="low_angle",
                camera_movement="pan",
                lighting_style="dramatic",
                subject_count="single",
                emotional_intensity="high",
                suggested_pacing="fast",
            )),
        ])

        result = filter_clips(project)
        cine = result[0]["cinematography"]

        assert cine is not None
        assert cine["shot_size"] == "CU"
        assert cine["camera_angle"] == "low_angle"
        assert cine["camera_movement"] == "pan"
        assert cine["lighting_style"] == "dramatic"
        assert cine["subject_count"] == "single"
        assert cine["emotional_intensity"] == "high"
        assert cine["suggested_pacing"] == "fast"
        # Should have exactly 7 keys
        assert len(cine) == 7

    def test_return_cinematography_none_when_missing(self):
        """Test cinematography is None when not analyzed."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([_make_clip("clip-1")])

        result = filter_clips(project)

        assert result[0]["cinematography"] is None

    def test_return_similarity_score_only_when_similar_to_clip_id(self):
        """Test similarity_score is only present when similar_to_clip_id is used."""
        from core.chat_tools import filter_clips
        import numpy as np

        # Create a simple normalized embedding
        vec = np.ones(768, dtype=np.float64)
        vec = vec / np.linalg.norm(vec)
        emb = vec.tolist()

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", embedding=emb),
            _make_clip("clip-2", embedding=emb),
        ])

        # Without similar_to_clip_id
        result_no_sim = filter_clips(project)
        for r in result_no_sim:
            assert "similarity_score" not in r

        # With similar_to_clip_id
        result_sim = filter_clips(project, similar_to_clip_id="clip-1")
        for r in result_sim:
            assert "similarity_score" in r


class TestFilterCombinations:
    """Tests for combining new filters with existing ones."""

    def test_gaze_combined_with_brightness(self):
        """Test gaze filter combined with brightness filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", gaze_category="at_camera", average_brightness=0.8),
            _make_clip("clip-2", gaze_category="at_camera", average_brightness=0.2),
            _make_clip("clip-3", gaze_category="looking_left", average_brightness=0.8),
        ])

        result = filter_clips(
            project,
            gaze_category="at_camera",
            min_brightness=0.5
        )

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_new_filters_with_existing_shot_type(self):
        """Test new filters combine with existing shot_type filter."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", shot_type="close_up", average_brightness=0.7),
            _make_clip("clip-2", shot_type="wide_shot", average_brightness=0.7),
            _make_clip("clip-3", shot_type="close_up", average_brightness=0.2),
        ])

        result = filter_clips(
            project,
            shot_type="close_up",
            min_brightness=0.5
        )

        ids = [r["id"] for r in result]
        assert ids == ["clip-1"]

    def test_backward_compatible_no_new_params(self):
        """Test existing behavior is unchanged when no new params are used."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1", shot_type="close_up"),
            _make_clip("clip-2", shot_type="wide_shot"),
        ])

        result = filter_clips(project, shot_type="close_up")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_no_filters_returns_all_clips(self):
        """Test calling with no filters returns all clips."""
        from core.chat_tools import filter_clips

        project = _create_test_project()
        project.add_clips([
            _make_clip("clip-1"),
            _make_clip("clip-2"),
            _make_clip("clip-3"),
        ])

        result = filter_clips(project)

        assert len(result) == 3
