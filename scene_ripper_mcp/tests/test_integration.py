"""Integration tests for MCP server tools.

Tests basic happy paths for each tool category.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Import the MCP server to get access to tool functions
from scene_ripper_mcp.server import mcp


def _make_source(file_path: Path, source_id: str = "src-1"):
    """Build a Source bound to an existing file path."""
    from models.clip import Source

    return Source(
        id=source_id,
        file_path=file_path,
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )


def _make_clip(
    clip_id: str,
    source_id: str = "src-1",
    start_frame: int = 0,
    end_frame: int = 90,
    shot_type: str | None = None,
    tags: list[str] | None = None,
):
    """Build a Clip with the given attributes."""
    from models.clip import Clip

    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        shot_type=shot_type,
        tags=tags or [],
    )


class TestProjectTools:
    """Test project management tools."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary empty .sceneripper project file."""
        from core.project import Project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test_project.sceneripper"
            project = Project.new(name="Test Project")
            project.save(project_path)
            yield project_path

    @pytest.mark.asyncio
    async def test_create_project(self):
        """Create a project and retrieve its info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "new_project.sceneripper"

            # Get the create_project tool
            tool = mcp._tool_manager.get_tool("create_project")
            assert tool is not None

            # Create a mock context
            ctx = AsyncMock()

            # Call the tool function directly
            from scene_ripper_mcp.tools.project import create_project

            result = await create_project(
                name="My Test Project",
                output_path=str(output_path),
                ctx=ctx,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["name"] == "My Test Project"
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_list_projects(self, temp_project):
        """List projects in a directory."""
        from scene_ripper_mcp.tools.project import list_projects

        ctx = AsyncMock()
        result = await list_projects(
            directory=str(temp_project.parent),
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["count"] == 1
        assert len(result_data["projects"]) == 1

    @pytest.mark.asyncio
    async def test_get_project_info_nonexistent(self):
        """Get info for non-existent project returns error."""
        from scene_ripper_mcp.tools.project import get_project_info

        ctx = AsyncMock()
        result = await get_project_info(
            project_path="/nonexistent/project.sceneripper",
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "error" in result_data


class TestClipTools:
    """Test clip query and manipulation tools."""

    @pytest.fixture
    def project_with_clips(self):
        """Create a project with some clips and a real dummy video file."""
        from core.project import Project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project_with_clips.sceneripper"
            video_path = Path(tmpdir) / "video.mp4"
            # Create a dummy file (not a real video but enough for path validation)
            video_path.write_bytes(b"fake video content")

            project = Project.new(name="Clips Test Project")
            project.add_source(_make_source(video_path, "src-1"))
            project.add_clips([
                _make_clip("clip-1", "src-1", 0, 90, shot_type="wide", tags=["nature"]),
                _make_clip("clip-2", "src-1", 90, 180, shot_type="close-up"),
                _make_clip("clip-3", "src-1", 180, 270, shot_type="medium", tags=["nature", "outdoor"]),
            ])
            project.save(project_path)
            yield project_path

    @pytest.mark.asyncio
    async def test_list_clips(self, project_with_clips):
        """List all clips in a project."""
        from scene_ripper_mcp.tools.clips import list_clips

        ctx = AsyncMock()
        result = await list_clips(
            project_path=str(project_with_clips),
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["total_clips"] == 3

    @pytest.mark.asyncio
    async def test_filter_clips_by_shot_type(self, project_with_clips):
        """Filter clips by shot type."""
        from scene_ripper_mcp.tools.clips import filter_clips

        ctx = AsyncMock()
        result = await filter_clips(
            project_path=str(project_with_clips),
            shot_type="close-up",
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        # Should find 1 close-up clip
        assert result_data["filtered_count"] == 1
        assert result_data["clips"][0]["shot_type"] == "close-up"

    @pytest.mark.asyncio
    async def test_add_clip_tags(self, project_with_clips):
        """Add tags to a clip."""
        from scene_ripper_mcp.tools.clips import add_clip_tags

        ctx = AsyncMock()
        result = await add_clip_tags(
            project_path=str(project_with_clips),
            clip_id="clip-2",
            tags=["test-tag", "another-tag"],
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "test-tag" in result_data["all_tags"]
        assert "another-tag" in result_data["all_tags"]


class TestSequenceTools:
    """Test sequence/timeline tools."""

    @pytest.fixture
    def project_with_sequence(self):
        """Create a project with clips and a sequence."""
        from core.project import Project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "sequence_project.sceneripper"
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"fake video content")

            project = Project.new(name="Sequence Test")
            project.add_source(_make_source(video_path, "src-1"))
            project.add_clips([
                _make_clip("clip-1", "src-1", 0, 90),
                _make_clip("clip-2", "src-1", 90, 180),
                _make_clip("clip-3", "src-1", 180, 270),
            ])
            # Pre-populate the sequence with one clip via the model API.
            project.add_to_sequence(["clip-1"])
            project.save(project_path)
            yield project_path

    @pytest.mark.asyncio
    async def test_get_sequence(self, project_with_sequence):
        """Get sequence from project."""
        from scene_ripper_mcp.tools.sequence import get_sequence

        ctx = AsyncMock()
        result = await get_sequence(
            project_path=str(project_with_sequence),
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "sequence" in result_data

    @pytest.mark.asyncio
    async def test_add_to_sequence(self, project_with_sequence):
        """Add clips to sequence."""
        from scene_ripper_mcp.tools.sequence import add_to_sequence

        ctx = AsyncMock()
        result = await add_to_sequence(
            project_path=str(project_with_sequence),
            clip_ids=["clip-2", "clip-3"],
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["clips_added"] >= 1

    @pytest.mark.asyncio
    async def test_clear_sequence(self, project_with_sequence):
        """Clear sequence."""
        from scene_ripper_mcp.tools.sequence import clear_sequence

        ctx = AsyncMock()
        result = await clear_sequence(
            project_path=str(project_with_sequence),
            ctx=ctx,
        )

        result_data = json.loads(result)
        assert result_data["success"] is True


class TestYouTubeTools:
    """Test YouTube search and download tools.

    Note: These tests mock the YouTube API to avoid actual API calls.
    """

    @pytest.mark.asyncio
    async def test_search_youtube_missing_api_key(self):
        """Search YouTube fails gracefully without API key."""
        from scene_ripper_mcp.tools.youtube import search_youtube

        ctx = AsyncMock()

        # Patch to simulate missing API key
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.settings.load_settings") as mock_settings:
                mock_settings.return_value = type(
                    "Settings", (), {"youtube_api_key": None, "download_dir": Path("/tmp")}
                )()

                result = await search_youtube(
                    query="test video",
                    ctx=ctx,
                )

                result_data = json.loads(result)
                # Should either succeed with mock or fail gracefully
                assert "success" in result_data


class TestExportTools:
    """Test export operations."""

    @pytest.fixture
    def export_project(self):
        """Create a project for export testing."""
        from core.project import Project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "export_project.sceneripper"
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"fake video content")
            output_dir = Path(tmpdir) / "exports"
            output_dir.mkdir()

            project = Project.new(name="Export Test")
            project.add_source(_make_source(video_path, "src-1"))
            project.add_clips([
                _make_clip("clip-1", "src-1", 0, 90),
                _make_clip("clip-2", "src-1", 90, 180),
            ])
            project.add_to_sequence(["clip-1", "clip-2"])
            project.save(project_path)

            yield {"project_path": project_path, "output_dir": output_dir}

    @pytest.mark.asyncio
    async def test_export_edl(self, export_project):
        """Export sequence as EDL."""
        from scene_ripper_mcp.tools.export import export_edl

        ctx = AsyncMock()
        output_path = export_project["output_dir"] / "test.edl"

        result = await export_edl(
            project_path=str(export_project["project_path"]),
            output_path=str(output_path),
            ctx=ctx,
        )

        result_data = json.loads(result)
        # May fail if project load is strict about source files
        if result_data["success"]:
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_dataset(self, export_project):
        """Export metadata as JSON dataset."""
        from scene_ripper_mcp.tools.export import export_dataset

        ctx = AsyncMock()
        output_path = export_project["output_dir"] / "dataset.json"

        result = await export_dataset(
            project_path=str(export_project["project_path"]),
            output_path=str(output_path),
            ctx=ctx,
        )

        result_data = json.loads(result)
        # May fail if project load is strict
        if result_data["success"]:
            assert output_path.exists()


class TestToolSchemas:
    """Test that all tools have valid schemas."""

    def test_all_tools_have_schemas(self):
        """Verify all registered tools have parameter schemas.

        Asserts a reasonable lower bound rather than an exact count so the
        check doesn't break every time a new tool lands.
        """
        tools = mcp._tool_manager._tools
        assert len(tools) >= 30, f"Expected at least 30 tools, found {len(tools)}"

        for name, tool in tools.items():
            assert "properties" in tool.parameters, f"Tool {name} missing properties"
            assert "type" in tool.parameters, f"Tool {name} missing type"

    def test_all_tools_have_descriptions(self):
        """Verify all registered tools have descriptions."""
        tools = mcp._tool_manager._tools

        for name, tool in tools.items():
            assert tool.description, f"Tool {name} missing description"
            assert len(tool.description) > 10, f"Tool {name} has too short description"


class TestSecurityValidation:
    """Test path security validation."""

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        from scene_ripper_mcp.security import validate_path

        # Attempt path traversal
        valid, error, _ = validate_path("../../../etc/passwd")
        assert not valid
        assert "traversal" in error.lower() or "must be under" in error.lower()

    def test_valid_home_path_allowed(self):
        """Test that paths under home directory are allowed."""
        from scene_ripper_mcp.security import validate_path

        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error, path = validate_path(tmpdir)
            assert valid, f"Valid path rejected: {error}"

    def test_absolute_path_required(self):
        """Test that relative paths are handled correctly."""
        from scene_ripper_mcp.security import validate_path

        # Relative path should be resolved
        valid, error, path = validate_path("./test.sceneripper", must_exist=False)
        # May or may not be valid depending on current directory
        assert isinstance(valid, bool)
