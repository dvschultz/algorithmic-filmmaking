"""Safe tool execution with error handling and conflict detection.

The ToolExecutor bridges between LLM tool calls and the actual tool functions,
providing:
- Safe execution with error handling
- Project injection for tools that need it
- Conflict detection with running GUI workers (with wait-retry)
- Result formatting for LLM context
"""

import json
import logging
import time
from typing import Any, Callable, Optional

from core.chat_tools import ToolRegistry, tools as default_registry

logger = logging.getLogger(__name__)

# Default retry settings for busy operations
BUSY_RETRY_DELAY_SECONDS = 1.0  # Wait between retries
BUSY_MAX_WAIT_SECONDS = 30.0    # Maximum total wait time

# Tools that conflict with GUI workers
CONFLICTING_TOOLS = {
    "detect_scenes",      # Conflicts with DetectionWorker
    "analyze_colors",     # Conflicts with ColorAnalysisWorker
    "analyze_shots",      # Conflicts with ShotClassificationWorker
    "transcribe",         # Conflicts with TranscriptionWorker
    "download_video",     # Conflicts with DownloadWorker
}


class ToolExecutor:
    """Execute tools safely with error handling and conflict detection."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        project: Optional[Any] = None,
        busy_check: Optional[Callable[[str], bool]] = None
    ):
        """Initialize the executor.

        Args:
            registry: Tool registry (uses global default if not provided)
            project: Active Project instance for tools that require it
            busy_check: Callback to check if a conflicting operation is running.
                Signature: busy_check(tool_name) -> bool
        """
        self.registry = registry or default_registry
        self.project = project
        self.busy_check = busy_check

    def execute(self, tool_call: dict) -> dict:
        """Execute a tool call from the LLM.

        Args:
            tool_call: OpenAI-format tool call with structure:
                {
                    "id": "call_xxx",
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "arguments": '{"arg1": "value1"}'
                    }
                }

        Returns:
            Result dict with structure:
                {
                    "tool_call_id": "call_xxx",
                    "name": "tool_name",
                    "success": bool,
                    "result": Any,  # if success
                    "error": str,   # if not success
                }
        """
        # Parse tool call
        tool_call_id = tool_call.get("id", "unknown")
        function = tool_call.get("function", {})
        name = function.get("name", "")
        args_str = function.get("arguments", "{}")

        # Parse arguments
        try:
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str
        except json.JSONDecodeError as e:
            return self._error_result(tool_call_id, name, f"Invalid JSON arguments: {e}")

        # Get tool definition
        tool = self.registry.get(name)
        if not tool:
            return self._error_result(
                tool_call_id, name,
                f"Unknown tool: {name}. Available tools: {[t.name for t in self.registry.all_tools()]}"
            )

        # Check for conflicting operations with wait-retry
        if name in CONFLICTING_TOOLS and self.busy_check:
            wait_result = self._wait_for_busy_operation(name)
            if wait_result is not None:
                return self._error_result(tool_call_id, name, wait_result)

        # Check project requirement
        if tool.requires_project and not self.project:
            return self._error_result(
                tool_call_id, name,
                "This tool requires an active project. Please open or create a project first."
            )

        # Inject project if needed
        if tool.requires_project:
            args["project"] = self.project

        # Execute the tool
        try:
            logger.info(f"Executing tool: {name} with args: {list(args.keys())}")
            result = tool.func(**args)
            logger.info(f"Tool {name} completed successfully")

            return {
                "tool_call_id": tool_call_id,
                "name": name,
                "success": True,
                "result": result
            }

        except TypeError as e:
            # Likely wrong arguments
            return self._error_result(
                tool_call_id, name,
                f"Invalid arguments for {name}: {e}"
            )
        except Exception as e:
            logger.exception(f"Tool execution error: {name}")
            return self._error_result(
                tool_call_id, name,
                f"Tool execution failed: {e}"
            )

    def _error_result(self, tool_call_id: str, name: str, error: str) -> dict:
        """Create an error result."""
        logger.warning(f"Tool error ({name}): {error}")
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "success": False,
            "error": error
        }

    def _wait_for_busy_operation(self, tool_name: str) -> Optional[str]:
        """Wait for a conflicting operation to complete with retry.

        Args:
            tool_name: Name of the tool attempting to execute

        Returns:
            None if operation is clear to proceed, error message if timed out
        """
        if not self.busy_check:
            return None

        if not self.busy_check(tool_name):
            return None  # Not busy, proceed immediately

        logger.info(f"Tool {tool_name} waiting for conflicting operation to complete...")
        start_time = time.time()
        waited = False

        while time.time() - start_time < BUSY_MAX_WAIT_SECONDS:
            if not self.busy_check(tool_name):
                if waited:
                    logger.info(f"Conflicting operation completed, proceeding with {tool_name}")
                return None  # Conflict cleared

            waited = True
            time.sleep(BUSY_RETRY_DELAY_SECONDS)

        # Timed out waiting
        return (
            f"Cannot run {tool_name}: A similar operation is still in progress after "
            f"waiting {BUSY_MAX_WAIT_SECONDS:.0f} seconds. "
            "Please wait for it to complete or cancel it first."
        )

    def format_for_llm(self, result: dict) -> dict:
        """Format tool result as a message for LLM context.

        Args:
            result: Result from execute()

        Returns:
            Message dict in OpenAI format for tool results
        """
        if result["success"]:
            content = json.dumps(result["result"], indent=2, default=str)
        else:
            content = json.dumps({"error": result["error"]})

        return {
            "role": "tool",
            "tool_call_id": result["tool_call_id"],
            "name": result["name"],
            "content": content
        }

    def execute_and_format(self, tool_call: dict) -> tuple[dict, dict]:
        """Execute a tool call and return both raw result and LLM message.

        Args:
            tool_call: OpenAI-format tool call

        Returns:
            Tuple of (raw_result, llm_message)
        """
        result = self.execute(tool_call)
        message = self.format_for_llm(result)
        return result, message
