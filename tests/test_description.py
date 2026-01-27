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
    @patch("core.analysis.description.describe_frame_cpu")
    def test_describe_frame_cpu(self, mock_cpu, mock_settings):
        """Test routing to CPU tier."""
        mock_settings.return_value = Settings(
            description_model_tier="cpu",
            description_model_cpu="moondream-test"
        )
        mock_cpu.return_value = "A cat sitting on a mat."

        desc, model = describe_frame(self.image_path)

        self.assertEqual(desc, "A cat sitting on a mat.")
        self.assertEqual(model, "moondream-test")
        mock_cpu.assert_called_once()

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
    @patch("core.analysis.description.describe_frame_cpu")
    def test_describe_frame_explicit_tier(self, mock_cpu, mock_settings):
        """Test explicit tier override."""
        mock_settings.return_value = Settings(
            description_model_tier="cloud",  # Default is cloud
            description_model_cpu="moondream-test"
        )
        mock_cpu.return_value = "A car on the road."

        # Request CPU explicitly
        desc, model = describe_frame(self.image_path, tier="cpu")

        self.assertEqual(desc, "A car on the road.")
        self.assertEqual(model, "moondream-test")
        mock_cpu.assert_called_once()

    @patch("core.analysis.description._load_cpu_model")
    @patch("PIL.Image.open")
    @patch("core.analysis.description.load_settings")
    def test_cpu_inference_logic(self, mock_settings, mock_image_open, mock_load_model):
        """Test CPU inference logic (mocking transformers)."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        
        mock_model.encode_image.return_value = "encoded_image"
        mock_model.answer_question.return_value = "Generated description"
        
        mock_settings.return_value = Settings(description_model_cpu="moondream-test")

        # Call internal function directly or via public API
        from core.analysis.description import describe_frame_cpu
        desc = describe_frame_cpu(self.image_path, "Test prompt")

        self.assertEqual(desc, "Generated description")
        mock_model.encode_image.assert_called_with(mock_image)
        mock_model.answer_question.assert_called_with("encoded_image", "Test prompt", mock_tokenizer)

    @patch("core.analysis.description.litellm")
    @patch("core.analysis.description.encode_image_base64")
    @patch("core.analysis.description.load_settings")
    @patch("core.settings.get_openai_api_key")
    def test_cloud_inference_logic(self, mock_get_key, mock_settings, mock_encode, mock_litellm):
        """Test Cloud inference logic (mocking litellm)."""
        # Setup mocks
        mock_encode.return_value = "base64_data"
        mock_get_key.return_value = "sk-test-key"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Cloud description"))]
        mock_litellm.completion.return_value = mock_response
        
        mock_settings.return_value = Settings(description_model_cloud="gpt-4o")

        # Call internal function
        from core.analysis.description import describe_frame_cloud
        desc = describe_frame_cloud(self.image_path, "Test prompt")

        self.assertEqual(desc, "Cloud description")
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o")
        self.assertEqual(call_kwargs["api_key"], "sk-test-key")
        self.assertEqual(len(call_kwargs["messages"]), 1)
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")

if __name__ == "__main__":
    unittest.main()
