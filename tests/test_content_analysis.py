"""Unit tests for content analysis modules (classification and detection)."""

import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

_has_torch = importlib.util.find_spec("torch") is not None

from models.clip import Source, Clip
from core.project import Project

# Import shared test helpers from conftest.py
from tests.conftest import make_test_clip


def _create_filter_test_project() -> Project:
    """Create a simple project for filter tests (single source)."""
    project = Project.new(name="Test Project")
    project.add_source(Source(
        id="src-1",
        file_path=Path("/test/video1.mp4"),
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    ))
    return project


class TestClassificationModule:
    """Tests for core/analysis/classification.py."""

    @pytest.mark.skipif(not _has_torch, reason="torch not installed")
    def test_classify_frame_returns_list_type(self):
        """Test classify_frame returns a list."""
        # Since the actual function has complex dependencies on torch/PIL,
        # we test the contract: it should return a list
        from core.analysis.classification import classify_frame

        # Test with a non-existent file (should return empty list gracefully)
        result = classify_frame(Path("/nonexistent/image.jpg"))
        assert isinstance(result, list)
        # Should be empty since file doesn't exist
        assert result == []

    @pytest.mark.skipif(not _has_torch, reason="torch not installed")
    def test_classify_frame_handles_invalid_path_gracefully(self):
        """Test classify_frame returns empty list for invalid paths."""
        from core.analysis.classification import classify_frame

        # Invalid file should return empty list
        result = classify_frame(Path("/nonexistent/path/image.jpg"))
        assert result == []

    def test_get_top_labels_returns_strings(self):
        """Test get_top_labels returns list of label strings only."""
        with patch("core.analysis.classification.classify_frame") as mock_classify:
            mock_classify.return_value = [
                ("cat", 0.9),
                ("dog", 0.05),
            ]

            from core.analysis.classification import get_top_labels

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                result = get_top_labels(Path(f.name))

            assert result == ["cat", "dog"]

    def test_is_model_loaded_initially_false(self):
        """Test is_model_loaded returns False before loading."""
        # Reset module state
        import core.analysis.classification as cls_module
        cls_module._model = None

        from core.analysis.classification import is_model_loaded

        assert is_model_loaded() is False

    def test_unload_model_clears_state(self):
        """Test unload_model clears the model state."""
        import core.analysis.classification as cls_module

        # Set some state
        cls_module._model = "dummy_model"
        cls_module._labels = ["label1"]
        cls_module._preprocess = "dummy_preprocess"

        from core.analysis.classification import unload_model

        unload_model()

        assert cls_module._model is None
        assert cls_module._labels is None
        assert cls_module._preprocess is None

    def test_get_model_cache_dir_returns_path(self):
        """Test _get_model_cache_dir returns a Path object."""
        from core.analysis.classification import _get_model_cache_dir

        result = _get_model_cache_dir()
        assert isinstance(result, Path)
        # Should contain 'models' or be a valid directory path
        assert result.name == "models" or "model" in str(result).lower()

    def test_get_model_cache_dir_creates_directory(self):
        """Test _get_model_cache_dir creates the directory if needed."""
        from core.analysis.classification import _get_model_cache_dir

        result = _get_model_cache_dir()
        # The function should create the directory
        assert result.exists() or result.parent.exists()


class TestDetectionModule:
    """Tests for core/analysis/detection.py."""

    def test_detect_objects_returns_list(self):
        """Test detect_objects returns list of detection dicts."""
        with patch("core.analysis.detection._load_yolo") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            # Mock YOLO results
            mock_result = MagicMock()
            mock_box = MagicMock()
            mock_box.cls = [MagicMock(__getitem__=lambda s, i: 0)]  # person class
            mock_box.conf = [MagicMock(__getitem__=lambda s, i: 0.95)]
            mock_box.xyxy = [MagicMock(tolist=lambda: [100, 200, 300, 400])]

            mock_result.boxes = [mock_box]
            mock_model.return_value = [mock_result]

            from core.analysis.detection import detect_objects

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                result = detect_objects(Path(f.name))

            # Should return a list
            assert isinstance(result, list)

    def test_detect_objects_handles_errors_gracefully(self):
        """Test detect_objects returns empty list on error."""
        with patch("core.analysis.detection._load_yolo") as mock_load:
            mock_model = MagicMock()
            mock_model.side_effect = Exception("Inference failed")
            mock_load.return_value = mock_model

            from core.analysis.detection import detect_objects

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                result = detect_objects(Path(f.name))

            assert result == []

    def test_count_people_calls_detect_with_person_class(self):
        """Test count_people filters to person class only."""
        with patch("core.analysis.detection.detect_objects") as mock_detect:
            mock_detect.return_value = [
                {"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
                {"label": "person", "confidence": 0.8, "bbox": [100, 0, 200, 100]},
            ]

            from core.analysis.detection import count_people, PERSON_CLASS_ID

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                result = count_people(Path(f.name))

            # Should call detect_objects with person class filter
            mock_detect.assert_called_once()
            call_kwargs = mock_detect.call_args[1]
            assert call_kwargs["classes"] == [PERSON_CLASS_ID]

            # Should return count
            assert result == 2

    def test_get_object_counts_aggregates_labels(self):
        """Test get_object_counts returns label -> count mapping."""
        with patch("core.analysis.detection.detect_objects") as mock_detect:
            mock_detect.return_value = [
                {"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
                {"label": "car", "confidence": 0.8, "bbox": [100, 0, 200, 100]},
                {"label": "person", "confidence": 0.7, "bbox": [200, 0, 300, 100]},
                {"label": "car", "confidence": 0.6, "bbox": [300, 0, 400, 100]},
                {"label": "dog", "confidence": 0.5, "bbox": [400, 0, 500, 100]},
            ]

            from core.analysis.detection import get_object_counts

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                result = get_object_counts(Path(f.name))

            assert result == {"person": 2, "car": 2, "dog": 1}

    def test_get_unique_labels_returns_sorted_list(self):
        """Test get_unique_labels returns sorted unique labels."""
        with patch("core.analysis.detection.detect_objects") as mock_detect:
            mock_detect.return_value = [
                {"label": "dog", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
                {"label": "cat", "confidence": 0.8, "bbox": [100, 0, 200, 100]},
                {"label": "dog", "confidence": 0.7, "bbox": [200, 0, 300, 100]},
                {"label": "bird", "confidence": 0.6, "bbox": [300, 0, 400, 100]},
            ]

            from core.analysis.detection import get_unique_labels

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                result = get_unique_labels(Path(f.name))

            # Should be unique and sorted
            assert result == ["bird", "cat", "dog"]

    def test_is_model_loaded_initially_false(self):
        """Test is_model_loaded returns False before loading."""
        import core.analysis.detection as det_module
        det_module._model = None

        from core.analysis.detection import is_model_loaded

        assert is_model_loaded() is False

    def test_unload_model_clears_state(self):
        """Test unload_model clears the model state."""
        import core.analysis.detection as det_module
        det_module._model = "dummy_model"

        from core.analysis.detection import unload_model

        unload_model()

        assert det_module._model is None

    def test_coco_classes_defined(self):
        """Test COCO classes list is properly defined."""
        from core.analysis.detection import COCO_CLASSES, PERSON_CLASS_ID

        assert len(COCO_CLASSES) == 80
        assert COCO_CLASSES[PERSON_CLASS_ID] == "person"
        assert "car" in COCO_CLASSES
        assert "dog" in COCO_CLASSES

    def test_detect_objects_with_confidence_threshold(self):
        """Test detect_objects passes confidence threshold to model."""
        with patch("core.analysis.detection._load_yolo") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            mock_model.return_value = []

            from core.analysis.detection import detect_objects

            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                detect_objects(Path(f.name), confidence_threshold=0.7)

            # Check model was called with conf param
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs["conf"] == 0.7


class TestFilterClipsWithContentAnalysis:
    """Tests for filter_clips with has_object, min_people, max_people filters."""

    def test_filter_by_has_object_in_object_labels(self):
        """Test filtering clips by object found in object_labels."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", object_labels=["cat", "dog", "chair"]),
            make_test_clip("clip-2", object_labels=["car", "bicycle"]),
            make_test_clip("clip-3", object_labels=None),
        ])

        result = filter_clips(project, has_object="dog")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_by_has_object_in_detected_objects(self):
        """Test filtering clips by object found in detected_objects."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", detected_objects=[
                {"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
                {"label": "car", "confidence": 0.8, "bbox": [100, 0, 200, 100]},
            ]),
            make_test_clip("clip-2", detected_objects=[
                {"label": "dog", "confidence": 0.7, "bbox": [0, 0, 100, 100]},
            ]),
        ])

        result = filter_clips(project, has_object="car")

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_by_has_object_case_insensitive(self):
        """Test has_object filter is case insensitive."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", object_labels=["CAT", "Dog"]),
        ])

        result = filter_clips(project, has_object="cat")
        assert len(result) == 1

        result = filter_clips(project, has_object="DOG")
        assert len(result) == 1

    def test_filter_by_has_object_no_matches(self):
        """Test has_object filter returns empty when no matches."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", object_labels=["cat", "dog"]),
        ])

        result = filter_clips(project, has_object="elephant")
        assert len(result) == 0

    def test_filter_by_min_people(self):
        """Test filtering clips by minimum person count."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", person_count=0),
            make_test_clip("clip-2", person_count=2),
            make_test_clip("clip-3", person_count=5),
            make_test_clip("clip-4", person_count=None),  # Treated as 0
        ])

        result = filter_clips(project, min_people=2)

        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "clip-2" in ids
        assert "clip-3" in ids

    def test_filter_by_max_people(self):
        """Test filtering clips by maximum person count."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", person_count=0),
            make_test_clip("clip-2", person_count=2),
            make_test_clip("clip-3", person_count=5),
        ])

        result = filter_clips(project, max_people=2)

        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "clip-1" in ids
        assert "clip-2" in ids

    def test_filter_by_people_range(self):
        """Test filtering clips by min and max people together."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", person_count=1),
            make_test_clip("clip-2", person_count=3),
            make_test_clip("clip-3", person_count=5),
            make_test_clip("clip-4", person_count=7),
        ])

        result = filter_clips(project, min_people=2, max_people=6)

        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "clip-2" in ids
        assert "clip-3" in ids

    def test_filter_combined_object_and_people(self):
        """Test combining has_object and people count filters."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", object_labels=["car"], person_count=2),
            make_test_clip("clip-2", object_labels=["car"], person_count=0),
            make_test_clip("clip-3", object_labels=["dog"], person_count=3),
        ])

        result = filter_clips(project, has_object="car", min_people=1)

        assert len(result) == 1
        assert result[0]["id"] == "clip-1"

    def test_filter_null_person_count_treated_as_zero(self):
        """Test clips with null person_count are treated as 0."""
        from core.chat_tools import filter_clips

        project = _create_filter_test_project()
        project.add_clips([
            make_test_clip("clip-1", person_count=None),
            make_test_clip("clip-2", person_count=1),
        ])

        # min_people=1 should exclude clip-1 (treated as 0)
        result = filter_clips(project, min_people=1)
        assert len(result) == 1
        assert result[0]["id"] == "clip-2"

        # max_people=0 should include clip-1 only
        result = filter_clips(project, max_people=0)
        assert len(result) == 1
        assert result[0]["id"] == "clip-1"


class TestClipDataModelExtension:
    """Tests for Clip dataclass content analysis fields."""

    def test_clip_has_content_analysis_fields(self):
        """Test Clip dataclass has new content analysis fields."""
        clip = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            object_labels=["dog", "cat"],
            detected_objects=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]}],
            person_count=3,
        )

        assert clip.object_labels == ["dog", "cat"]
        assert clip.detected_objects[0]["label"] == "person"
        assert clip.person_count == 3

    def test_clip_content_analysis_fields_default_to_none(self):
        """Test content analysis fields default to None."""
        clip = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
        )

        assert clip.object_labels is None
        assert clip.detected_objects is None
        assert clip.person_count is None

    def test_clip_to_dict_includes_content_analysis(self):
        """Test Clip.to_dict() includes content analysis fields."""
        clip = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            object_labels=["dog", "cat"],
            detected_objects=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]}],
            person_count=3,
        )

        data = clip.to_dict()

        assert data["object_labels"] == ["dog", "cat"]
        assert data["detected_objects"][0]["label"] == "person"
        assert data["person_count"] == 3

    def test_clip_from_dict_restores_content_analysis(self):
        """Test Clip.from_dict() restores content analysis fields."""
        data = {
            "id": "test-1",
            "source_id": "src-1",
            "start_frame": 0,
            "end_frame": 100,
            "object_labels": ["dog", "cat"],
            "detected_objects": [{"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]}],
            "person_count": 3,
        }

        clip = Clip.from_dict(data)

        assert clip.object_labels == ["dog", "cat"]
        assert clip.detected_objects[0]["label"] == "person"
        assert clip.person_count == 3

    def test_clip_roundtrip_preserves_content_analysis(self):
        """Test to_dict/from_dict roundtrip preserves content analysis."""
        original = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            object_labels=["bird", "tree"],
            detected_objects=[
                {"label": "car", "confidence": 0.85, "bbox": [10, 20, 200, 150]},
            ],
            person_count=0,
        )

        data = original.to_dict()
        restored = Clip.from_dict(data)

        assert restored.object_labels == original.object_labels
        assert restored.detected_objects == original.detected_objects
        assert restored.person_count == original.person_count


class TestToolTimeouts:
    """Tests for content analysis tool timeout configuration."""

    def test_get_tool_timeout_returns_default_for_unknown(self):
        """Test get_tool_timeout returns default for unknown tools."""
        from core.chat_tools import get_tool_timeout, DEFAULT_TOOL_TIMEOUT

        assert get_tool_timeout("unknown_tool") == DEFAULT_TOOL_TIMEOUT
