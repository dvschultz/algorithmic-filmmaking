"""Tests for video description analysis."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from core.analysis.description import describe_frame
from core.settings import Settings

class TestDescriptionAnalysis(unittest.TestCase):
    """Test description generation logic."""

    def setUp(self):
        # Create a dummy image file
        self.test_dir = tempfile.mkdtemp()
        self.image_path = Path(self.test_dir) / "test_frame.jpg"
        with open(self.image_path, "wb") as f:
            f.write(b"fake image data")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("core.analysis.description.load_settings")
    @patch("core.analysis.description.describe_frame_local")
    def test_describe_frame_local(self, mock_local, mock_settings):
        """Test routing to local tier."""
        mock_settings.return_value = Settings(
            description_model_tier="local",
            description_model_local="qwen3-vl-test"
        )
        mock_local.return_value = "A cat sitting on a mat."

        desc, model = describe_frame(self.image_path)

        self.assertEqual(desc, "A cat sitting on a mat.")
        self.assertEqual(model, "qwen3-vl-test")
        mock_local.assert_called_once()

    @patch("core.analysis.description.load_settings")
    @patch("core.analysis.description.describe_frame_cloud")
    def test_describe_frame_cloud(self, mock_cloud, mock_settings):
        """Test routing to Cloud tier."""
        mock_settings.return_value = Settings(
            description_model_tier="cloud",
            description_model_cloud="gpt-4o-test"
        )
        mock_cloud.return_value = "A sunny beach scene."

        desc, model = describe_frame(self.image_path)

        self.assertEqual(desc, "A sunny beach scene.")
        self.assertEqual(model, "gpt-4o-test")
        mock_cloud.assert_called_once()

    @patch("core.analysis.description.load_settings")
    @patch("core.analysis.description.describe_frame_local")
    def test_describe_frame_explicit_tier(self, mock_local, mock_settings):
        """Test explicit tier override."""
        mock_settings.return_value = Settings(
            description_model_tier="cloud",  # Default is cloud
            description_model_local="qwen3-vl-test"
        )
        mock_local.return_value = "A car on the road."

        # Request local explicitly
        desc, model = describe_frame(self.image_path, tier="local")

        self.assertEqual(desc, "A car on the road.")
        self.assertEqual(model, "qwen3-vl-test")
        mock_local.assert_called_once()

    @patch("core.analysis.description.is_mlx_vlm_available", return_value=False)
    @patch("core.analysis.description._load_local_model")
    @patch("PIL.Image.open")
    @patch("core.analysis.description.load_settings")
    def test_local_inference_logic(self, mock_settings, mock_image_open, mock_load_model, _mock_mlx):
        """Test local inference logic (mocking Moondream fallback)."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        mock_model.encode_image.return_value = "encoded_image"
        mock_model.answer_question.return_value = "Generated description"

        mock_settings.return_value = Settings(description_model_local="moondream-test")

        # Call internal function directly
        from core.analysis.description import describe_frame_local
        desc = describe_frame_local(self.image_path, "Test prompt")

        self.assertEqual(desc, "Generated description")
        mock_model.encode_image.assert_called_with(mock_image)
        mock_model.answer_question.assert_called_with("encoded_image", "Test prompt", mock_tokenizer)

    @patch("litellm.completion")
    @patch("core.analysis.description.encode_image_base64")
    @patch("core.analysis.description.load_settings")
    @patch("core.settings.get_openai_api_key")
    def test_cloud_inference_logic(self, mock_get_key, mock_settings, mock_encode, mock_completion):
        """Test Cloud inference logic (mocking litellm)."""
        # Setup mocks
        mock_encode.return_value = "base64_data"
        mock_get_key.return_value = "sk-test-key"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Cloud description"))]
        mock_completion.return_value = mock_response

        mock_settings.return_value = Settings(description_model_cloud="gpt-4o")

        # Call internal function
        from core.analysis.description import describe_frame_cloud
        desc = describe_frame_cloud(self.image_path, "Test prompt")

        self.assertEqual(desc, "Cloud description")
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o")
        self.assertEqual(call_kwargs["api_key"], "sk-test-key")
        self.assertEqual(len(call_kwargs["messages"]), 1)
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")

class TestFilterClipsSearchDescription(unittest.TestCase):
    """Test filter_clips with search_description parameter."""

    def setUp(self):
        from core.project import Project
        from models.clip import Source, Clip

        self.project = Project.new(name="Test Project")

        source = Source(
            id="src-1",
            file_path=Path("/test/video1.mp4"),
            duration_seconds=120.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        self.project.add_source(source)

    def test_filter_by_search_description(self):
        """Test filtering clips by description text."""
        from core.chat_tools import filter_clips
        from models.clip import Clip

        self.project.add_clips([
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90,
                 description="A person walking on a beach at sunset"),
            Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180,
                 description="A dog playing in the park"),
            Clip(id="clip-3", source_id="src-1", start_frame=180, end_frame=270,
                 description=None),
        ])

        result = filter_clips(self.project, search_description="beach")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "clip-1")

    def test_filter_search_description_case_insensitive(self):
        """Test search_description is case insensitive."""
        from core.chat_tools import filter_clips
        from models.clip import Clip

        self.project.add_clips([
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90,
                 description="A SUNSET over the ocean"),
        ])

        result = filter_clips(self.project, search_description="sunset")
        self.assertEqual(len(result), 1)

        result = filter_clips(self.project, search_description="OCEAN")
        self.assertEqual(len(result), 1)

    def test_filter_search_description_excludes_none(self):
        """Test clips without descriptions are excluded from search."""
        from core.chat_tools import filter_clips
        from models.clip import Clip

        self.project.add_clips([
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90,
                 description=None),
            Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180,
                 description="A mountain landscape"),
        ])

        result = filter_clips(self.project, search_description="mountain")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "clip-2")

    def test_filter_search_description_no_matches(self):
        """Test search returns empty when no descriptions match."""
        from core.chat_tools import filter_clips
        from models.clip import Clip

        self.project.add_clips([
            Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90,
                 description="A sunset beach scene"),
        ])

        result = filter_clips(self.project, search_description="mountain")
        self.assertEqual(len(result), 0)


class TestClipDescriptionSerialization(unittest.TestCase):
    """Test Clip serialization with description fields."""

    def test_clip_to_dict_includes_description(self):
        """Test to_dict includes description fields."""
        from models.clip import Clip

        clip = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            description="A person walking on a beach",
            description_model="gpt-4o",
            description_frames=1,
        )

        data = clip.to_dict()

        self.assertEqual(data["description"], "A person walking on a beach")
        self.assertEqual(data["description_model"], "gpt-4o")
        self.assertEqual(data["description_frames"], 1)

    def test_clip_from_dict_restores_description(self):
        """Test from_dict restores description fields."""
        from models.clip import Clip

        data = {
            "id": "test-1",
            "source_id": "src-1",
            "start_frame": 0,
            "end_frame": 100,
            "description": "A dog playing in snow",
            "description_model": "moondream-2b",
            "description_frames": 4,
        }

        clip = Clip.from_dict(data)

        self.assertEqual(clip.description, "A dog playing in snow")
        self.assertEqual(clip.description_model, "moondream-2b")
        self.assertEqual(clip.description_frames, 4)

    def test_clip_roundtrip_preserves_description(self):
        """Test to_dict/from_dict roundtrip preserves description."""
        from models.clip import Clip

        original = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            description="A dramatic sunset over mountains",
            description_model="claude-3-opus",
            description_frames=8,
        )

        data = original.to_dict()
        restored = Clip.from_dict(data)

        self.assertEqual(restored.description, original.description)
        self.assertEqual(restored.description_model, original.description_model)
        self.assertEqual(restored.description_frames, original.description_frames)

    def test_clip_to_dict_excludes_none_description(self):
        """Test to_dict excludes None description fields."""
        from models.clip import Clip

        clip = Clip(
            id="test-1",
            source_id="src-1",
            start_frame=0,
            end_frame=100,
            description=None,
        )

        data = clip.to_dict()

        # Should not include description keys when None
        self.assertNotIn("description", data)
        self.assertNotIn("description_model", data)


class TestModelManagement(unittest.TestCase):
    """Test model loading/unloading helpers."""

    def test_is_model_loaded_initially_false(self):
        """Test is_model_loaded returns False when no model loaded."""
        import core.analysis.description as desc_module
        desc_module._LOCAL_MODEL = None
        desc_module._CPU_MODEL = None

        from core.analysis.description import is_model_loaded
        self.assertFalse(is_model_loaded())

    def test_unload_model_clears_state(self):
        """Test unload_model clears model state."""
        import core.analysis.description as desc_module
        desc_module._LOCAL_MODEL = "dummy_model"
        desc_module._LOCAL_PROCESSOR = "dummy_processor"
        desc_module._CPU_MODEL = "dummy_model"
        desc_module._CPU_TOKENIZER = "dummy_tokenizer"

        from core.analysis.description import unload_model
        unload_model()

        self.assertIsNone(desc_module._LOCAL_MODEL)
        self.assertIsNone(desc_module._LOCAL_PROCESSOR)
        self.assertIsNone(desc_module._CPU_MODEL)
        self.assertIsNone(desc_module._CPU_TOKENIZER)


if __name__ == "__main__":
    unittest.main()
