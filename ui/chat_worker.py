"""Chat worker thread for LLM interaction.

Runs the LLM agent loop in a separate thread to keep the UI responsive.
Handles streaming responses, tool execution, and emits signals for UI updates.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from core.chat_tools import tools as tool_registry
from core.llm_client import LLMClient, ProviderConfig, check_ollama_health
from core.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class ChatAgentWorker(QThread):
    """Worker thread for LLM chat with tool execution.

    Runs the agent loop asynchronously and emits signals for:
    - Streaming text chunks
    - Tool call start/end
    - Completion
    - Errors
    """

    # Signals
    text_chunk = Signal(str)  # Streaming text chunk
    tool_called = Signal(str, dict)  # tool_name, arguments
    tool_result = Signal(str, dict, bool)  # tool_name, result, success
    gui_tool_requested = Signal(str, dict, str)  # tool_name, args, tool_call_id (for main thread execution)
    gui_tool_completed = Signal(str)  # tool_call_id (set by main thread when done)
    complete = Signal(str, list)  # final response text, tool_history
    error = Signal(str)  # error message

    def __init__(
        self,
        config: ProviderConfig,
        messages: list[dict],
        project: Optional[Any] = None,
        busy_check: Optional[Callable[[str], bool]] = None,
        parent=None
    ):
        """Initialize the worker.

        Args:
            config: LLM provider configuration
            messages: Conversation history
            project: Active Project instance
            busy_check: Callback to check if operation is busy
            parent: Parent QObject
        """
        super().__init__(parent)
        self.config = config
        self.messages = messages.copy()
        self.project = project
        self.busy_check = busy_check
        self._stop_requested = False

        # For GUI tool synchronization
        import threading
        self._gui_tool_event = threading.Event()
        self._gui_tool_result: Optional[dict] = None

    def run(self):
        """Run the agent loop."""
        try:
            asyncio.run(self._async_run())
        except Exception as e:
            logger.exception("Agent worker error")
            self.error.emit(str(e))

    async def _async_run(self):
        """Async implementation of the agent loop."""
        from core.llm_client import ProviderType

        # Check Ollama health if using local provider
        if self.config.provider == ProviderType.LOCAL:
            api_base = self.config.get_api_base()
            healthy, error_msg = await check_ollama_health(api_base)
            if not healthy:
                self.error.emit(error_msg)
                return

        client = LLMClient(self.config)
        executor = ToolExecutor(tool_registry, self.project, self.busy_check)

        # Build system prompt with project context
        system_prompt = self._build_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + self.messages

        max_iterations = 10  # Prevent infinite tool loops
        iteration = 0
        tool_history = []  # Track tool interactions for history

        while iteration < max_iterations and not self._stop_requested:
            iteration += 1

            try:
                content, tool_calls = await self._stream_response(
                    client, full_messages
                )

                # Build assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                # Add to full messages
                full_messages.append(assistant_msg)

                # If no tool calls, we're done
                if not tool_calls:
                    self.complete.emit(content, tool_history)
                    return

                # Add assistant message with tool calls to history
                tool_history.append(assistant_msg)

                # Execute tools
                for tc in tool_calls:
                    if self._stop_requested:
                        break

                    name = tc.get("function", {}).get("name", "unknown")
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    tool_call_id = tc.get("id", "unknown")

                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}

                    # Emit tool call signal
                    self.tool_called.emit(name, args)

                    # Check if this is a GUI-modifying tool
                    tool_def = tool_registry.get(name)
                    if tool_def and tool_def.modifies_gui_state:
                        # Execute on main thread via signal/slot
                        self._gui_tool_event.clear()
                        self._gui_tool_result = None
                        self.gui_tool_requested.emit(name, args, tool_call_id)

                        # Wait for main thread to complete (with timeout)
                        completed = self._gui_tool_event.wait(timeout=30.0)
                        if not completed or self._stop_requested:
                            result = {
                                "tool_call_id": tool_call_id,
                                "name": name,
                                "success": False,
                                "error": "Tool execution timed out or was cancelled"
                            }
                        elif self._gui_tool_result:
                            result = self._gui_tool_result
                        else:
                            result = {
                                "tool_call_id": tool_call_id,
                                "name": name,
                                "success": False,
                                "error": "No result from GUI tool"
                            }
                    else:
                        # Execute non-GUI tool directly in worker thread
                        result = executor.execute(tc)

                    self.tool_result.emit(name, result, result.get("success", False))

                    # Add tool result to messages
                    tool_msg = executor.format_for_llm(result)
                    full_messages.append(tool_msg)
                    tool_history.append(tool_msg)

                # Continue loop for LLM to process tool results

            except Exception as e:
                logger.exception("Error in agent loop")
                self.error.emit(str(e))
                return

        if iteration >= max_iterations:
            self.error.emit("Maximum tool iterations reached. The agent may be stuck in a loop.")
        elif self._stop_requested:
            self.complete.emit("*Cancelled*", tool_history)

    async def _stream_response(
        self,
        client: LLMClient,
        messages: list[dict]
    ) -> tuple[str, list[dict]]:
        """Stream a single LLM response.

        Args:
            client: LLM client
            messages: Full message history

        Returns:
            Tuple of (content_text, tool_calls)
        """
        content = ""
        tool_calls = []

        async for chunk in client.stream_chat(
            messages,
            tools=tool_registry.to_openai_format()
        ):
            if self._stop_requested:
                break

            # Handle different response formats
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            # Accumulate content
            if hasattr(delta, 'content') and delta.content:
                content += delta.content
                self.text_chunk.emit(delta.content)

            # Accumulate tool calls
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    self._accumulate_tool_call(tool_calls, tc)

        return content, tool_calls

    def _accumulate_tool_call(self, buffer: list, delta_tc):
        """Accumulate partial tool call from streaming.

        Args:
            buffer: List to accumulate tool calls into
            delta_tc: Delta tool call from streaming chunk
        """
        idx = delta_tc.index if hasattr(delta_tc, 'index') else 0

        # Extend buffer if needed
        while len(buffer) <= idx:
            buffer.append({
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""}
            })

        # Update with delta
        if hasattr(delta_tc, 'id') and delta_tc.id:
            buffer[idx]["id"] = delta_tc.id

        if hasattr(delta_tc, 'function') and delta_tc.function:
            func = delta_tc.function
            if hasattr(func, 'name') and func.name:
                buffer[idx]["function"]["name"] = func.name
            if hasattr(func, 'arguments') and func.arguments:
                buffer[idx]["function"]["arguments"] += func.arguments

    def _build_system_prompt(self) -> str:
        """Build system prompt with project context.

        Returns:
            System prompt string
        """
        prompt = """You are an AI assistant for Scene Ripper, a video scene detection and editing tool.

You help users create video projects by:
- Detecting scenes in videos
- Analyzing clips (colors, shot types, transcription)
- Building sequences from clips
- Exporting clips and datasets

Available tools let you perform these operations. Always explain what you're doing before using tools.

When working with clips:
- Use filter_clips to find clips matching specific criteria
- Use list_clips to see all available clips
- Use add_to_sequence to add clips to the timeline
- Use get_project_state to check the current project status

When the user wants to work with videos:
- Use detect_scenes to analyze a video and create clips
- Use download_video to download from YouTube or Vimeo
- Use search_youtube to find videos

Be helpful and proactive. If the user's request is unclear, ask clarifying questions.
"""

        if self.project:
            # Add project context
            sources_info = []
            for s in self.project.sources:
                sources_info.append(f"  - {s.file_path.name} ({s.duration_seconds:.1f}s, {len(self.project.clips_by_source.get(s.id, []))} clips)")

            seq_length = 0
            if self.project.sequence and self.project.sequence.tracks:
                seq_length = len(self.project.sequence.tracks[0].clips)

            prompt += f"""

CURRENT PROJECT STATE:
- Name: {self.project.metadata.name}
- Path: {self.project.path or 'Unsaved'}
- Sources ({len(self.project.sources)} video(s)):
{chr(10).join(sources_info) if sources_info else "  (none)"}
- Total Clips: {len(self.project.clips)}
- Sequence Length: {seq_length} clips

You can reference existing clips by their IDs and build on this project.
"""
        else:
            prompt += """

NO PROJECT LOADED - The user should open or create a project first, or you can help them start by detecting scenes in a video.
"""

        return prompt

    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
        # Unblock any waiting GUI tool
        self._gui_tool_event.set()

    def set_gui_tool_result(self, result: dict):
        """Called by main thread to provide GUI tool result.

        Args:
            result: Tool execution result dict
        """
        self._gui_tool_result = result
        self._gui_tool_event.set()
