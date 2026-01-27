"""Integration tests for CLI and GUI interoperability.

These tests verify that projects, settings, and data can be shared
between CLI and GUI modes of the application.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cli.main import cli, register_commands
from cli.utils.errors import ExitCode
from core.project import Project
from core.settings import Settings, load_settings, save_settings
from models.clip import Source, Clip


class TestCLIProjectGUIInterop:
    """Tests for CLI-created projects being loadable by GUI."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_cli_creates_valid_project_structure(self, runner):
        """CLI-created project files have valid structure for GUI loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "cli_project.json"

            # Create project data directly (simulating CLI output)
            project_data = {
                "version": "1.0",
                "id": "cli-test-project",
                "project_name": "CLI Test Project",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "sources": [
                    {
                        "id": "source-1",
                        "file_path": "video.mp4",
                        "duration_seconds": 60.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "analyzed": True,
                    }
                ],
                "clips": [
                    {
                        "id": "clip-1",
                        "source_id": "source-1",
                        "start_frame": 0,
                        "end_frame": 150,
                        "shot_type": "medium_shot",
                    },
                    {
                        "id": "clip-2",
                        "source_id": "source-1",
                        "start_frame": 150,
                        "end_frame": 300,
                        "shot_type": "close_up",
                    },
                ],
                "sequence": None,
            }

            # Create dummy video file
            (Path(tmpdir) / "video.mp4").touch()

            # Write project file
            project_path.write_text(json.dumps(project_data))

            # Load via core.project (same as GUI uses)
            project = Project.load(project_path)

            assert project is not None
            assert project.metadata.name == "CLI Test Project"
            assert len(project.sources) == 1
            assert len(project.clips) == 2
            assert project.clips[0].shot_type == "medium_shot"

    def test_gui_project_loadable_by_project_class(self):
        """GUI-style project creation is loadable by Project.load()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "gui_project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            # Create via core.project (simulating GUI creation)
            project = Project.new(name="GUI Test Project")
            source = Source(
                id="src-gui-1",
                file_path=video_path,
                duration_seconds=120.0,
                fps=24.0,
                width=3840,
                height=2160,
            )
            project.add_source(source)

            clip = Clip(
                id="clip-gui-1",
                source_id="src-gui-1",
                start_frame=0,
                end_frame=240,
                shot_type="wide_shot",
            )
            project.add_clips([clip])
            project.save(project_path)

            # Verify can be loaded again (simulating CLI loading)
            loaded = Project.load(project_path)

            assert loaded.metadata.name == "GUI Test Project"
            assert len(loaded.sources) == 1
            assert loaded.sources[0].fps == 24.0
            assert len(loaded.clips) == 1
            assert loaded.clips[0].shot_type == "wide_shot"

    def test_project_info_command_reads_gui_project(self, runner):
        """CLI project info command can read GUI-created projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            # Create project via Project class (GUI-style)
            project = Project.new(name="Info Test Project")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)
            project.add_clips([
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90),
                Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180),
            ])
            project.save(project_path)

            # Read via CLI
            result = runner.invoke(cli, ["project", "info", str(project_path)])

            assert result.exit_code == 0
            assert "Info Test Project" in result.output or "info" in result.output.lower()

    def test_project_list_clips_on_gui_project(self, runner):
        """CLI list-clips command works on GUI-created projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            # Create project with clips
            project = Project.new(name="Clips Project")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)
            project.add_clips([
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90),
                Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180),
            ])
            project.save(project_path)

            # List clips via CLI
            result = runner.invoke(cli, ["project", "list-clips", str(project_path)])

            assert result.exit_code == 0
            assert "2 clips" in result.output.lower()


class TestGUIProjectCLIModification:
    """Tests for CLI modifying GUI-created projects."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_cli_add_to_sequence(self, runner):
        """CLI can add clips to sequence in GUI-created project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            # Create project with clips but no sequence
            project = Project.new(name="Sequence Project")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)
            project.add_clips([
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90),
                Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180),
            ])
            project.save(project_path)

            # Add all clips to sequence via CLI
            result = runner.invoke(cli, [
                "project", "add-to-sequence", str(project_path), "--all"
            ])

            assert result.exit_code == 0

            # Verify sequence was created (reload project)
            loaded = Project.load(project_path)
            assert loaded.sequence is not None
            # Sequence has tracks, each track has clips
            assert len(loaded.sequence.get_all_clips()) >= 1

    def test_round_trip_preserves_data(self, runner):
        """Data survives CLI modification and GUI re-loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            # Create via GUI (Project class)
            original = Project.new(name="Round Trip Test")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            original.add_source(source)
            original.add_clips([
                Clip(
                    id="clip-1",
                    source_id="src-1",
                    start_frame=0,
                    end_frame=90,
                    shot_type="close_up",
                    tags=["important", "hero"],
                    notes="Key moment in the video",
                ),
            ])
            original.save(project_path)

            # Modify via CLI (add to sequence)
            result = runner.invoke(cli, [
                "project", "add-to-sequence", str(project_path), "--all"
            ])
            assert result.exit_code == 0

            # Reload and verify original data preserved
            loaded = Project.load(project_path)
            clip = loaded.clips_by_id.get("clip-1")

            assert clip is not None
            assert clip.shot_type == "close_up"
            assert "important" in clip.tags
            assert "hero" in clip.tags
            assert clip.notes == "Key moment in the video"


class TestSettingsSync:
    """Tests for settings synchronization between CLI and GUI."""

    def test_settings_saved_by_cli_config_loadable(self):
        """Settings saved via CLIConfig are loadable by load_settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                # Save settings (simulating CLI)
                settings = Settings()
                settings.default_sensitivity = 5.5
                settings.export_quality = "high"
                settings.transcription_model = "medium.en"

                # Mock keyring to avoid writing to real keyring
                with patch("core.settings._set_api_key_in_keyring"):
                    with patch("core.settings._set_provider_api_key_in_keyring"):
                        save_settings(settings)

                # Load settings (simulating GUI)
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch("core.settings._get_provider_api_key_from_keyring", return_value=""):
                        loaded = load_settings()

                assert loaded.default_sensitivity == 5.5
                assert loaded.export_quality == "high"
                assert loaded.transcription_model == "medium.en"

    def test_gui_settings_readable_by_cli(self):
        """Settings saved by GUI (Settings class) are readable by CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("core.settings._get_config_path", return_value=config_path):
                # Save via Settings (simulating GUI)
                settings = Settings()
                settings.youtube_results_count = 50
                settings.min_scene_length_seconds = 1.0

                with patch("core.settings._set_api_key_in_keyring"):
                    with patch("core.settings._set_provider_api_key_in_keyring"):
                        save_settings(settings)

                # Load via load_settings (simulating CLI)
                with patch("core.settings._get_api_key_from_keyring", return_value=""):
                    with patch("core.settings._get_provider_api_key_from_keyring", return_value=""):
                        loaded = load_settings()

                assert loaded.youtube_results_count == 50
                assert loaded.min_scene_length_seconds == 1.0


class TestEnvironmentVariables:
    """Tests for environment variable handling in both CLI and GUI."""

    def test_env_sensitivity_in_cli_config(self):
        """Environment variable overrides sensitivity in CLI config."""
        from cli.utils.config import CLIConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch.dict(os.environ, {"SCENE_RIPPER_SENSITIVITY": "7.5"}):
                with patch("core.settings._get_config_path", return_value=config_path):
                    config = CLIConfig.load()
                    assert config.default_sensitivity == 7.5

    def test_env_sensitivity_in_gui_settings(self):
        """Environment variable overrides sensitivity in GUI settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch.dict(os.environ, {"SCENE_RIPPER_SENSITIVITY": "7.5"}):
                with patch("core.settings._get_config_path", return_value=config_path):
                    with patch("core.settings._get_api_key_from_keyring", return_value=""):
                        with patch("core.settings._get_provider_api_key_from_keyring", return_value=""):
                            settings = load_settings()
                            assert settings.default_sensitivity == 7.5

    def test_env_vars_override_json_config(self):
        """Environment variables take priority over JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Save config with one value
            with patch("core.settings._get_config_path", return_value=config_path):
                settings = Settings()
                settings.default_sensitivity = 3.0

                with patch("core.settings._set_api_key_in_keyring"):
                    with patch("core.settings._set_provider_api_key_in_keyring"):
                        save_settings(settings)

            # Load with env var override
            with patch.dict(os.environ, {"SCENE_RIPPER_SENSITIVITY": "9.0"}):
                with patch("core.settings._get_config_path", return_value=config_path):
                    with patch("core.settings._get_api_key_from_keyring", return_value=""):
                        with patch("core.settings._get_provider_api_key_from_keyring", return_value=""):
                            loaded = load_settings()
                            # Env var should win
                            assert loaded.default_sensitivity == 9.0

    def test_youtube_api_key_from_env(self):
        """YouTube API key from environment is available in both modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test_key_123"}):
                with patch("core.settings._get_config_path", return_value=config_path):
                    with patch("core.settings._get_api_key_from_keyring", return_value=""):
                        with patch("core.settings._get_provider_api_key_from_keyring", return_value=""):
                            settings = load_settings()
                            assert settings.youtube_api_key == "test_key_123"


class TestProjectFileFormat:
    """Tests for project file format consistency."""

    def test_project_json_has_required_fields(self):
        """Project JSON files contain all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            project = Project.new(name="Format Test")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)
            project.save(project_path)

            # Read raw JSON
            data = json.loads(project_path.read_text())

            # Check required fields
            assert "version" in data
            assert "id" in data
            assert "project_name" in data
            assert "sources" in data
            assert "clips" in data
            assert isinstance(data["sources"], list)
            assert isinstance(data["clips"], list)

    def test_source_serialization_roundtrip(self):
        """Source data survives serialization/deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            project = Project.new()
            source = Source(
                id="src-123",
                file_path=video_path,
                duration_seconds=123.456,
                fps=29.97,
                width=4096,
                height=2160,
                analyzed=True,
            )
            project.add_source(source)
            project.save(project_path)

            loaded = Project.load(project_path)
            loaded_source = loaded.sources[0]

            assert loaded_source.id == "src-123"
            assert abs(loaded_source.duration_seconds - 123.456) < 0.001
            assert abs(loaded_source.fps - 29.97) < 0.001
            assert loaded_source.width == 4096
            assert loaded_source.height == 2160
            assert loaded_source.analyzed is True

    def test_clip_serialization_roundtrip(self):
        """Clip data survives serialization/deserialization."""
        from core.transcription import TranscriptSegment

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project.json"
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            project = Project.new()
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)

            clip = Clip(
                id="clip-special",
                source_id="src-1",
                start_frame=100,
                end_frame=500,
                name="My Custom Name",
                shot_type="extreme_close_up",
                dominant_colors=[(255, 100, 50), (0, 50, 200)],
                transcript=[
                    TranscriptSegment(
                        start_time=0.0,
                        end_time=2.0,
                        text="Hello world",
                        confidence=0.95,
                    )
                ],
                tags=["favorite", "intro"],
                notes="This is an important clip",
            )
            project.add_clips([clip])
            project.save(project_path)

            loaded = Project.load(project_path)
            loaded_clip = loaded.clips_by_id["clip-special"]

            assert loaded_clip.name == "My Custom Name"
            assert loaded_clip.shot_type == "extreme_close_up"
            assert loaded_clip.dominant_colors == [(255, 100, 50), (0, 50, 200)]
            assert len(loaded_clip.transcript) == 1
            assert loaded_clip.transcript[0].text == "Hello world"
            assert "favorite" in loaded_clip.tags
            assert loaded_clip.notes == "This is an important clip"
