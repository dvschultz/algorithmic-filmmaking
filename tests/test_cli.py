"""Unit tests for CLI commands and utilities."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from cli.main import cli, register_commands
from cli.utils.errors import ExitCode, exit_with, handle_error
from cli.utils.config import CLIConfig, get_config_dir, get_cache_dir
from cli.utils.output import output_result, output_table, _serialize_value


class TestExitCodes:
    """Tests for exit code utilities."""

    def test_exit_code_values(self):
        """Test that exit codes have expected values."""
        assert ExitCode.SUCCESS == 0
        assert ExitCode.GENERAL_ERROR == 1
        assert ExitCode.USAGE_ERROR == 2
        assert ExitCode.FILE_NOT_FOUND == 3
        assert ExitCode.DEPENDENCY_MISSING == 4
        assert ExitCode.NETWORK_ERROR == 5
        assert ExitCode.PERMISSION_ERROR == 6
        assert ExitCode.VALIDATION_ERROR == 7

    def test_exit_with_message(self):
        """Test exit_with outputs message to stderr."""
        with pytest.raises(SystemExit) as exc_info:
            exit_with(ExitCode.GENERAL_ERROR, "Test error message")
        assert exc_info.value.code == ExitCode.GENERAL_ERROR

    def test_handle_error_file_not_found(self):
        """Test handle_error maps FileNotFoundError correctly."""
        with pytest.raises(SystemExit) as exc_info:
            handle_error(FileNotFoundError("File not found"))
        assert exc_info.value.code == ExitCode.FILE_NOT_FOUND

    def test_handle_error_permission_error(self):
        """Test handle_error maps PermissionError correctly."""
        with pytest.raises(SystemExit) as exc_info:
            handle_error(PermissionError("Permission denied"))
        assert exc_info.value.code == ExitCode.PERMISSION_ERROR

    def test_handle_error_value_error(self):
        """Test handle_error maps ValueError correctly."""
        with pytest.raises(SystemExit) as exc_info:
            handle_error(ValueError("Invalid value"))
        assert exc_info.value.code == ExitCode.VALIDATION_ERROR


class TestCLIConfig:
    """Tests for CLI configuration."""

    def test_config_defaults(self):
        """Test that config has sensible defaults."""
        # Isolate from user's actual config by mocking config path and keyring
        with patch("core.settings._get_config_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/config.json")
            with patch("core.settings._get_api_key_from_keyring", return_value=""):
                config = CLIConfig()
                assert config.default_sensitivity == 3.0
                assert config.min_scene_length_seconds == 0.5
                assert config.transcription_model == "small.en"
                assert config.transcription_language == "en"
                assert config.youtube_results_count == 25

    def test_config_load_with_env_vars(self):
        """Test that environment variables override config."""
        with patch.dict(os.environ, {
            "YOUTUBE_API_KEY": "test_api_key",
            "SCENE_RIPPER_SENSITIVITY": "5.0",
        }):
            config = CLIConfig.load()
            assert config.youtube_api_key == "test_api_key"
            assert config.default_sensitivity == 5.0

    def test_config_load_from_file(self):
        """Test loading config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_path = config_dir / "config.json"
            # Use the new nested JSON format
            config_path.write_text(json.dumps({
                "detection": {
                    "default_sensitivity": 7.0,
                },
                "transcription": {
                    "model": "medium.en",
                },
            }))

            with patch("core.settings._get_config_path", return_value=config_path):
                config = CLIConfig.load()
                assert config.default_sensitivity == 7.0
                assert config.transcription_model == "medium.en"

    def test_config_save(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                # Mock keyring to prevent writing test values to real keyring
                with patch("core.settings._set_api_key_in_keyring") as mock_keyring:
                    config = CLIConfig()
                    config.default_sensitivity = 8.0
                    config.youtube_api_key = "test_api_key"

                    assert config.save() is True
                    assert config_path.exists()

                    # Verify keyring was called with the API key
                    mock_keyring.assert_called_once_with("test_api_key")

                    # Verify saved content (uses nested JSON format)
                    saved_data = json.loads(config_path.read_text())
                    assert saved_data["detection"]["default_sensitivity"] == 8.0
                    # youtube_api_key is stored in keyring, not JSON

    def test_get_config_dir_xdg(self):
        """Test XDG config directory on Linux/macOS."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}, clear=False):
            with patch("os.name", "posix"):
                config_dir = get_config_dir()
                assert config_dir == Path("/custom/config/scene-ripper")

    def test_get_cache_dir_xdg(self):
        """Test XDG cache directory."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}, clear=False):
            with patch("os.name", "posix"):
                cache_dir = get_cache_dir()
                assert cache_dir == Path("/custom/cache/scene-ripper")


class TestOutputFormatting:
    """Tests for output formatting utilities."""

    def test_serialize_value_path(self):
        """Test Path serialization."""
        value = Path("/home/user/video.mp4")
        result = _serialize_value(value)
        assert result == "/home/user/video.mp4"

    def test_serialize_value_dict(self):
        """Test dict serialization with nested values."""
        value = {"path": Path("/test"), "count": 5}
        result = _serialize_value(value)
        assert result == {"path": "/test", "count": 5}

    def test_serialize_value_list(self):
        """Test list serialization."""
        value = [Path("/a"), Path("/b")]
        result = _serialize_value(value)
        assert result == ["/a", "/b"]


class TestCLIMain:
    """Tests for main CLI entry point."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_version(self, runner):
        """Test --version flag."""
        register_commands()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "scene_ripper" in result.output
        assert "0.1.0" in result.output

    def test_cli_help(self, runner):
        """Test --help flag."""
        register_commands()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Scene Ripper" in result.output
        assert "detect" in result.output
        assert "analyze" in result.output
        assert "export" in result.output


class TestDetectCommand:
    """Tests for the detect command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_detect_missing_video(self, runner):
        """Test detect with non-existent video file."""
        result = runner.invoke(cli, ["detect", "/nonexistent/video.mp4"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "no such file" in result.output.lower()

    def test_detect_invalid_sensitivity(self, runner):
        """Test detect with invalid sensitivity value."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            try:
                result = runner.invoke(cli, [
                    "detect", f.name, "--sensitivity", "15.0"
                ])
                assert result.exit_code == ExitCode.VALIDATION_ERROR
                assert "1.0 and 10.0" in result.output
            finally:
                os.unlink(f.name)

    def test_detect_output_exists_no_force(self, runner):
        """Test detect refuses to overwrite without --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()
            # Default output is <video>.sceneripper, not .json
            output_path = Path(tmpdir) / "test.sceneripper"
            output_path.touch()

            result = runner.invoke(cli, ["detect", str(video_path)])
            assert result.exit_code == ExitCode.VALIDATION_ERROR
            assert "exists" in result.output.lower()


class TestProjectCommands:
    """Tests for project management commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    @pytest.fixture
    def sample_project(self):
        """Create a sample project file with a valid source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy video file that the project references
            video_path = Path(tmpdir) / "test_video.mp4"
            video_path.touch()

            project_path = Path(tmpdir) / "project.json"
            project_data = {
                "version": "1.0",
                "id": "test-project-id",
                "project_name": "Test Project",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "sources": [
                    {
                        "id": "source-1",
                        "file_path": "test_video.mp4",
                        "duration_seconds": 60.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "analyzed": False,
                    }
                ],
                "clips": [],
                "sequence": None,
            }
            project_path.write_text(json.dumps(project_data))
            yield str(project_path)

    def test_project_info_missing_file(self, runner):
        """Test project info with missing file."""
        result = runner.invoke(cli, ["project", "info", "/nonexistent/project.json"])
        assert result.exit_code != 0

    def test_project_info_success(self, runner, sample_project):
        """Test project info with valid project."""
        result = runner.invoke(cli, ["project", "info", sample_project])
        assert result.exit_code == 0
        assert "Test Project" in result.output or "test" in result.output.lower()

    def test_project_info_json_output(self, runner, sample_project):
        """Test project info with JSON output."""
        result = runner.invoke(cli, ["--json", "project", "info", sample_project])
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "project_name" in data or "Project Name" in str(data)

    def test_project_list_clips_empty(self, runner, sample_project):
        """Test listing clips from empty project."""
        result = runner.invoke(cli, ["project", "list-clips", sample_project])
        assert result.exit_code == 0
        assert "0 clips" in result.output.lower()

    def test_project_add_to_sequence_no_selection(self, runner, sample_project):
        """Test add-to-sequence requires selection."""
        result = runner.invoke(cli, ["project", "add-to-sequence", sample_project])
        assert result.exit_code == ExitCode.USAGE_ERROR
        assert "specify" in result.output.lower() or "--all" in result.output


class TestAnalyzeCommands:
    """Tests for analyze commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_analyze_colors_help(self, runner):
        """Test analyze colors help."""
        result = runner.invoke(cli, ["analyze", "colors", "--help"])
        assert result.exit_code == 0
        assert "colors" in result.output.lower()
        assert "--clip" in result.output or "-c" in result.output

    def test_analyze_shots_help(self, runner):
        """Test analyze shots help."""
        result = runner.invoke(cli, ["analyze", "shots", "--help"])
        assert result.exit_code == 0
        assert "shot" in result.output.lower()
        assert "CLIP" in result.output.upper() or "--force" in result.output


class TestExportCommands:
    """Tests for export commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_export_clips_help(self, runner):
        """Test export clips help."""
        result = runner.invoke(cli, ["export", "clips", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output or "-o" in result.output
        assert "--format" in result.output

    def test_export_dataset_help(self, runner):
        """Test export dataset help."""
        result = runner.invoke(cli, ["export", "dataset", "--help"])
        assert result.exit_code == 0
        assert "JSON" in result.output or "json" in result.output

    def test_export_edl_help(self, runner):
        """Test export edl help."""
        result = runner.invoke(cli, ["export", "edl", "--help"])
        assert result.exit_code == 0
        assert "EDL" in result.output or "edl" in result.output.lower()

    def test_export_video_help(self, runner):
        """Test export video help."""
        result = runner.invoke(cli, ["export", "video", "--help"])
        assert result.exit_code == 0
        assert "--quality" in result.output
        assert "--resolution" in result.output


class TestTranscribeCommand:
    """Tests for transcribe command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_transcribe_list_models(self, runner):
        """Test transcribe --list-models."""
        result = runner.invoke(cli, ["transcribe", "--list-models"])
        assert result.exit_code == 0
        assert "tiny.en" in result.output
        assert "small.en" in result.output
        assert "medium.en" in result.output

    def test_transcribe_help(self, runner):
        """Test transcribe help."""
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output or "-m" in result.output
        assert "--language" in result.output


class TestYouTubeCommands:
    """Tests for YouTube search and download commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_search_no_api_key(self, runner):
        """Test search fails gracefully without API key."""
        # Clear any API key from environment, config file, AND keyring
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.settings._get_config_path") as mock_path:
                mock_path.return_value = Path("/nonexistent/config.json")
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    result = runner.invoke(cli, ["search", "test query"])
                    assert result.exit_code == ExitCode.VALIDATION_ERROR
                    assert "API key" in result.output

    def test_search_help(self, runner):
        """Test search help."""
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        assert "--max-results" in result.output or "-n" in result.output
        assert "--order" in result.output

    def test_download_help(self, runner):
        """Test download help."""
        result = runner.invoke(cli, ["download", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output or "-o" in result.output
        assert "--detect" in result.output


class TestIntegration:
    """Integration tests for CLI pipeline."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_json_output_flag(self, runner):
        """Test that --json flag works for all commands."""
        # Test with project info on a valid project
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            project_data = {
                "version": "1.0",
                "id": "test",
                "project_name": "Test",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "sources": [],
                "clips": [],
                "sequence": None,
            }
            json.dump(project_data, f)
            f.flush()

            try:
                result = runner.invoke(cli, ["--json", "project", "info", f.name])
                assert result.exit_code == 0
                # Verify it's valid JSON
                parsed = json.loads(result.output)
                assert isinstance(parsed, dict)
            finally:
                os.unlink(f.name)

    def test_command_registration(self, runner):
        """Test all expected commands are registered."""
        result = runner.invoke(cli, ["--help"])
        expected_commands = [
            "detect",
            "analyze",
            "transcribe",
            "export",
            "project",
            "search",
            "download",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in help output"
