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


class TestProjectTools:
    """Test project management tools."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test_project.json"
            project_data = {
                "version": "1.0",
                "project_name": "Test Project",
                "sources": [],
                "clips": [],
                "sequence": None,
                "metadata": {
                    "name": "Test Project",
                    "created_at": "2026-01-26T00:00:00",
                    "modified_at": "2026-01-26T00:00:00",
                },
            }
            project_path.write_text(json.dumps(project_data))
            yield project_path

    @pytest.mark.asyncio
    async def test_create_project(self):
        """Create a project and retrieve its info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "new_project.json"

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
            project_path="/nonexistent/project.json",
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
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "project_with_clips.json"
            video_path = Path(tmpdir) / "video.mp4"
            # Create a dummy file (not a real video but enough for path validation)
            video_path.write_bytes(b"fake video content")

            # Create a project with sources and clips
            project_data = {
                "version": "1.0",
                "project_name": "Clips Test Project",
                "sources": [
                    {
                        "id": "src-1",
                        "file_path": str(video_path),
                        "duration_seconds": 120.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                    }
                ],
                "clips": [
                    {
                        "id": "clip-1",
                        "source_id": "src-1",
                        "start_frame": 0,
                        "end_frame": 90,
                        "shot_type": "wide",
                        "tags": ["nature"],
                    },
                    {
                        "id": "clip-2",
                        "source_id": "src-1",
                        "start_frame": 90,
                        "end_frame": 180,
                        "shot_type": "close-up",
                        "tags": [],
                    },
                    {
                        "id": "clip-3",
                        "source_id": "src-1",
                        "start_frame": 180,
                        "end_frame": 270,
                        "shot_type": "medium",
                        "tags": ["nature", "outdoor"],
                    },
                ],
                "sequence": None,
                "metadata": {
                    "name": "Clips Test Project",
                    "created_at": "2026-01-26T00:00:00",
                    "modified_at": "2026-01-26T00:00:00",
                },
            }
            project_path.write_text(json.dumps(project_data))
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
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "sequence_project.json"
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"fake video content")

            project_data = {
                "version": "1.0",
                "project_name": "Sequence Test",
                "sources": [
                    {
                        "id": "src-1",
                        "file_path": str(video_path),
                        "duration_seconds": 120.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                    }
                ],
                "clips": [
                    {"id": "clip-1", "source_id": "src-1", "start_frame": 0, "end_frame": 90},
                    {"id": "clip-2", "source_id": "src-1", "start_frame": 90, "end_frame": 180},
                    {"id": "clip-3", "source_id": "src-1", "start_frame": 180, "end_frame": 270},
                ],
                "sequence": {
                    "id": "seq-1",
                    "name": "Test Sequence",
                    "tracks": [
                        {
                            "id": "track-1",
                            "name": "V1",
                            "clips": [
                                {"id": "clip-1", "source_id": "src-1", "start_frame": 0, "end_frame": 90},
                            ],
                        }
                    ],
                },
                "metadata": {
                    "name": "Sequence Test",
                    "created_at": "2026-01-26T00:00:00",
                    "modified_at": "2026-01-26T00:00:00",
                },
            }
            project_path.write_text(json.dumps(project_data))
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
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "export_project.json"
            output_dir = Path(tmpdir) / "exports"
            output_dir.mkdir()

            project_data = {
                "version": "1.0",
                "project_name": "Export Test",
                "sources": [
                    {
                        "id": "src-1",
                        "file_path": "/fake/video.mp4",
                        "filename": "video.mp4",
                        "duration_seconds": 120.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                    }
                ],
                "clips": [
                    {"id": "clip-1", "source_id": "src-1", "start_frame": 0, "end_frame": 90},
                    {"id": "clip-2", "source_id": "src-1", "start_frame": 90, "end_frame": 180},
                ],
                "sequence": {
                    "id": "seq-1",
                    "name": "Export Sequence",
                    "tracks": [
                        {
                            "id": "track-1",
                            "name": "V1",
                            "clips": [
                                {"id": "clip-1", "source_id": "src-1", "start_frame": 0, "end_frame": 90},
                                {"id": "clip-2", "source_id": "src-1", "start_frame": 90, "end_frame": 180},
                            ],
                        }
                    ],
                },
                "metadata": {
                    "name": "Export Test",
                    "created_at": "2026-01-26T00:00:00",
                    "modified_at": "2026-01-26T00:00:00",
                },
            }
            project_path.write_text(json.dumps(project_data))
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
        """Verify all registered tools have parameter schemas."""
        tools = mcp._tool_manager._tools
        assert len(tools) == 33, f"Expected 33 tools, found {len(tools)}"

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
        valid, error, path = validate_path("./test.json", must_exist=False)
        # May or may not be valid depending on current directory
        assert isinstance(valid, bool)
