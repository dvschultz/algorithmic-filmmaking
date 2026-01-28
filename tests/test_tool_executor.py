"""Unit tests for tool executor module."""

import pytest
from unittest.mock import MagicMock, patch

from core.tool_executor import ToolExecutor
from core.chat_tools import ToolRegistry


class MockProjectMetadata:
    """Mock project metadata."""
    def __init__(self, name="Untitled Project"):
        self.name = name


class MockProject:
    """Mock project for testing."""
    def __init__(self, name="Untitled Project"):
        self.metadata = MockProjectMetadata(name)
        self.clips = []
        self.sources = []


class TestProjectNamingEnforcement:
    """Tests for project naming enforcement in tool executor."""

    def test_unnamed_project_blocks_state_modifying_tools(self):
        """Test that state-modifying tools are blocked on unnamed projects."""
        # Create a mock registry with a state-modifying tool
        registry = ToolRegistry()

        @registry.register(
            description="Test tool that modifies state",
            requires_project=True,
            modifies_project_state=True
        )
        def test_state_tool(project):
            return {"success": True}

        # Create executor with unnamed project
        project = MockProject(name="Untitled Project")
        executor = ToolExecutor(registry=registry, project=project)

        # Try to execute the state-modifying tool
        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "test_state_tool",
                "arguments": "{}"
            }
        })

        # Should fail with naming requirement error
        assert result["success"] is False
        assert "must be named" in result["error"]
        assert "set_project_name" in result["error"]
        assert "save" not in result["error"].lower()  # No saving prompt - auto-save handles it

    def test_named_project_allows_state_modifying_tools(self):
        """Test that state-modifying tools work on named projects."""
        registry = ToolRegistry()

        @registry.register(
            description="Test tool that modifies state",
            requires_project=True,
            modifies_project_state=True
        )
        def test_state_tool(project):
            return {"success": True, "message": "Tool executed"}

        # Create executor with named project
        project = MockProject(name="My Awesome Project")
        executor = ToolExecutor(registry=registry, project=project)

        # Execute the state-modifying tool
        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "test_state_tool",
                "arguments": "{}"
            }
        })

        # Should succeed
        assert result["success"] is True
        assert result["result"]["message"] == "Tool executed"

    def test_set_project_name_allowed_on_unnamed_project(self):
        """Test that set_project_name is allowed even on unnamed projects."""
        registry = ToolRegistry()

        @registry.register(
            description="Set project name",
            requires_project=True,
            modifies_project_state=True
        )
        def set_project_name(project, name: str):
            project.metadata.name = name
            return {"success": True, "new_name": name}

        # Create executor with unnamed project
        project = MockProject(name="Untitled Project")
        executor = ToolExecutor(registry=registry, project=project)

        # set_project_name should be allowed
        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "set_project_name",
                "arguments": '{"name": "New Name"}'
            }
        })

        # Should succeed (exception to the naming rule)
        assert result["success"] is True
        assert project.metadata.name == "New Name"

    def test_non_state_modifying_tools_allowed_on_unnamed_project(self):
        """Test that read-only tools work on unnamed projects."""
        registry = ToolRegistry()

        @registry.register(
            description="Read-only tool",
            requires_project=True,
            modifies_project_state=False  # Does NOT modify state
        )
        def get_project_info(project):
            return {"name": project.metadata.name}

        # Create executor with unnamed project
        project = MockProject(name="Untitled Project")
        executor = ToolExecutor(registry=registry, project=project)

        # Read-only tools should be allowed
        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "get_project_info",
                "arguments": "{}"
            }
        })

        # Should succeed
        assert result["success"] is True
        assert result["result"]["name"] == "Untitled Project"
