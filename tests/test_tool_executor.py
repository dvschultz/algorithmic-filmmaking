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


class TestToolExecution:
    """Tests for tool execution in the executor."""

    def test_state_modifying_tools_execute_on_any_project(self):
        """Test that state-modifying tools execute regardless of project name."""
        registry = ToolRegistry()

        @registry.register(
            description="Test tool that modifies state",
            requires_project=True,
            modifies_project_state=True
        )
        def test_state_tool(project):
            return {"success": True, "message": "Tool executed"}

        # Works on unnamed project (no code gate -- system prompt handles naming)
        project = MockProject(name="Untitled Project")
        executor = ToolExecutor(registry=registry, project=project)

        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "test_state_tool",
                "arguments": "{}"
            }
        })

        assert result["success"] is True
        assert result["result"]["message"] == "Tool executed"

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

        project = MockProject(name="My Awesome Project")
        executor = ToolExecutor(registry=registry, project=project)

        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "test_state_tool",
                "arguments": "{}"
            }
        })

        assert result["success"] is True
        assert result["result"]["message"] == "Tool executed"

    def test_read_only_tools_execute_on_unnamed_project(self):
        """Test that read-only tools work on unnamed projects."""
        registry = ToolRegistry()

        @registry.register(
            description="Read-only tool",
            requires_project=True,
            modifies_project_state=False
        )
        def get_project_info(project):
            return {"name": project.metadata.name}

        project = MockProject(name="Untitled Project")
        executor = ToolExecutor(registry=registry, project=project)

        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "get_project_info",
                "arguments": "{}"
            }
        })

        assert result["success"] is True
        assert result["result"]["name"] == "Untitled Project"

    def test_conflicts_with_workers_blocks_when_busy(self):
        """Test that tools with conflicts_with_workers=True are blocked when busy."""
        from unittest.mock import MagicMock

        registry = ToolRegistry()

        @registry.register(
            description="Conflicting tool",
            requires_project=True,
            conflicts_with_workers=True
        )
        def conflicting_tool(project):
            return {"success": True}

        busy_check = MagicMock(return_value=True)
        project = MockProject(name="Test Project")
        executor = ToolExecutor(registry=registry, project=project, busy_check=busy_check)

        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "conflicting_tool",
                "arguments": "{}"
            }
        })

        assert result["success"] is False
        assert "in progress" in result["error"].lower() or "waiting" in result["error"].lower()

    def test_no_conflict_flag_ignores_busy_check(self):
        """Test that tools without conflicts_with_workers ignore busy_check."""
        from unittest.mock import MagicMock

        registry = ToolRegistry()

        @registry.register(
            description="Non-conflicting tool",
            requires_project=True,
            conflicts_with_workers=False
        )
        def safe_tool(project):
            return {"success": True, "message": "ran fine"}

        busy_check = MagicMock(return_value=True)
        project = MockProject(name="Test Project")
        executor = ToolExecutor(registry=registry, project=project, busy_check=busy_check)

        result = executor.execute({
            "id": "test_call",
            "function": {
                "name": "safe_tool",
                "arguments": "{}"
            }
        })

        assert result["success"] is True
        busy_check.assert_not_called()
