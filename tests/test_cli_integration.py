"""Integration tests for CLI pipeline operations.

These tests verify the full CLI workflow:
detect → analyze → export
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cli.main import cli, register_commands
from cli.utils.errors import ExitCode
from core.project import Project
from models.clip import Source, Clip


class TestSignalHandling:
    """Tests for signal handling infrastructure."""

    def test_progress_checkpoint_save_load(self):
        """Test checkpoint save and load cycle."""
        from cli.utils.signals import ProgressCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Save checkpoint
            checkpoint = ProgressCheckpoint(checkpoint_path)
            checkpoint.set_total(100)
            checkpoint.update_progress(50)
            checkpoint.add_result({"item": "data1"})
            checkpoint.add_result({"item": "data2"})
            checkpoint.set_metadata("video_path", "/test/video.mp4")

            assert checkpoint.save() is True
            assert checkpoint_path.exists()

            # Load checkpoint
            loaded = ProgressCheckpoint(checkpoint_path)
            assert loaded.load() is True
            assert loaded._current_item == 50
            assert loaded._total_items == 100
            assert len(loaded._partial_results) == 2
            assert loaded._metadata["video_path"] == "/test/video.mp4"

    def test_progress_checkpoint_clear(self):
        """Test checkpoint file cleanup."""
        from cli.utils.signals import ProgressCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            checkpoint = ProgressCheckpoint(checkpoint_path)
            checkpoint.add_result({"test": True})
            checkpoint.save()

            assert checkpoint_path.exists()

            checkpoint.clear()
            assert not checkpoint_path.exists()

    def test_graceful_exit_exception(self):
        """Test GracefulExit exception can be raised and caught."""
        from cli.utils.signals import GracefulExit

        with pytest.raises(GracefulExit):
            raise GracefulExit()

    def test_interruptible_loop_completes(self):
        """Test interruptible_loop completes normally without interrupt."""
        from cli.utils.signals import interruptible_loop

        items = [1, 2, 3, 4, 5]
        results, completed = interruptible_loop(
            items,
            lambda x: x * 2,
        )

        assert completed is True
        assert results == [2, 4, 6, 8, 10]


class TestCLIPipelineSetup:
    """Tests for CLI pipeline component setup."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_detect_command_validates_sensitivity(self, runner):
        """Test detect validates sensitivity range."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")  # Minimal file to pass path check
            video_path = f.name

        try:
            result = runner.invoke(cli, [
                "detect", video_path, "--sensitivity", "15.0"
            ])
            assert result.exit_code == ExitCode.VALIDATION_ERROR
            assert "1.0 and 10.0" in result.output
        finally:
            os.unlink(video_path)

    def test_detect_command_refuses_overwrite(self, runner):
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


class TestProjectPipeline:
    """Tests for project-based CLI workflows."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    @pytest.fixture
    def sample_project(self):
        """Create a sample project with clips for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_video.mp4"
            video_path.touch()

            project = Project.new(name="Pipeline Test")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
                analyzed=True,
            )
            project.add_source(source)

            clips = [
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90),
                Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180),
                Clip(id="clip-3", source_id="src-1", start_frame=180, end_frame=270),
            ]
            project.add_clips(clips)

            project_path = Path(tmpdir) / "project.json"
            project.save(project_path)

            yield str(project_path)

    def test_project_info_shows_clip_count(self, runner, sample_project):
        """Test project info shows clip count."""
        result = runner.invoke(cli, ["project", "info", sample_project])
        assert result.exit_code == 0
        # Should mention the clips
        assert "3" in result.output or "clips" in result.output.lower()

    def test_project_list_clips_shows_all(self, runner, sample_project):
        """Test list-clips shows all clips."""
        result = runner.invoke(cli, ["project", "list-clips", sample_project])
        assert result.exit_code == 0
        assert "3 clips" in result.output.lower()

    def test_project_add_to_sequence_all(self, runner, sample_project):
        """Test adding all clips to sequence."""
        result = runner.invoke(cli, [
            "project", "add-to-sequence", sample_project, "--all"
        ])
        assert result.exit_code == 0

        # Verify sequence was created
        project = Project.load(Path(sample_project))
        assert project.sequence is not None
        assert len(project.sequence.get_all_clips()) >= 1

    def test_project_json_output(self, runner, sample_project):
        """Test JSON output format."""
        result = runner.invoke(cli, ["--json", "project", "info", sample_project])
        assert result.exit_code == 0

        # Should be valid JSON
        data = json.loads(result.output)
        assert isinstance(data, dict)


class TestAnalyzePipeline:
    """Tests for analysis commands in pipeline context."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_analyze_colors_help(self, runner):
        """Test analyze colors help is available."""
        result = runner.invoke(cli, ["analyze", "colors", "--help"])
        assert result.exit_code == 0
        assert "--clip" in result.output or "CLIP" in result.output

    def test_analyze_shots_help(self, runner):
        """Test analyze shots help is available."""
        result = runner.invoke(cli, ["analyze", "shots", "--help"])
        assert result.exit_code == 0


class TestExportPipeline:
    """Tests for export commands in pipeline context."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    @pytest.fixture
    def project_with_sequence(self):
        """Create a project with clips in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_video.mp4"
            video_path.touch()

            project = Project.new(name="Export Test")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=60.0,
                fps=30.0,
                width=1920,
                height=1080,
                analyzed=True,
            )
            project.add_source(source)

            clips = [
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=90),
                Clip(id="clip-2", source_id="src-1", start_frame=90, end_frame=180),
            ]
            project.add_clips(clips)
            project.add_to_sequence(["clip-1", "clip-2"])

            project_path = Path(tmpdir) / "project.json"
            project.save(project_path)

            yield str(project_path), tmpdir

    def test_export_clips_help(self, runner):
        """Test export clips help."""
        result = runner.invoke(cli, ["export", "clips", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output or "-o" in result.output

    def test_export_edl_help(self, runner):
        """Test export edl help."""
        result = runner.invoke(cli, ["export", "edl", "--help"])
        assert result.exit_code == 0
        assert "EDL" in result.output or "edl" in result.output.lower()

    def test_export_dataset_help(self, runner):
        """Test export dataset help."""
        result = runner.invoke(cli, ["export", "dataset", "--help"])
        assert result.exit_code == 0


class TestFullPipelineSimulated:
    """Simulated full pipeline tests (without real video processing)."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        register_commands()
        return CliRunner()

    def test_project_roundtrip(self, runner):
        """Test creating and loading project through CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test.json"
            video_path = Path(tmpdir) / "video.mp4"
            video_path.touch()

            # Create project via Project class (simulates detect output)
            project = Project.new(name="Roundtrip Test")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=30.0,
                fps=24.0,
                width=1280,
                height=720,
            )
            project.add_source(source)
            project.add_clips([
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=48),
                Clip(id="clip-2", source_id="src-1", start_frame=48, end_frame=96),
            ])
            project.save(project_path)

            # Read via CLI
            result = runner.invoke(cli, ["project", "info", str(project_path)])
            assert result.exit_code == 0

            # Add to sequence via CLI
            result = runner.invoke(cli, [
                "project", "add-to-sequence", str(project_path), "--all"
            ])
            assert result.exit_code == 0

            # Verify via Project class
            loaded = Project.load(project_path)
            assert loaded.sequence is not None

    def test_cli_workflow_json_output(self, runner):
        """Test CLI workflow with JSON output mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test.json"
            video_path = Path(tmpdir) / "video.mp4"
            video_path.touch()

            # Create project
            project = Project.new(name="JSON Test")
            source = Source(
                id="src-1",
                file_path=video_path,
                duration_seconds=10.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
            project.add_source(source)
            project.add_clips([
                Clip(id="clip-1", source_id="src-1", start_frame=0, end_frame=60),
            ])
            project.save(project_path)

            # Get info as JSON
            result = runner.invoke(cli, [
                "--json", "project", "info", str(project_path)
            ])
            assert result.exit_code == 0

            # Verify valid JSON
            data = json.loads(result.output)
            assert isinstance(data, dict)
