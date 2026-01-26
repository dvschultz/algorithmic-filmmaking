"""Chat worker thread for LLM interaction.

Runs the LLM agent loop in a separate thread to keep the UI responsive.
Handles streaming responses, tool execution, and emits signals for UI updates.
"""

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from core.chat_tools import tools as tool_registry
from core.llm_client import LLMClient, ProviderConfig, check_ollama_health
from core.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


def _parse_tool_calls_from_text(content: str, available_tools: list[str]) -> tuple[list[dict], str]:
    """Parse tool calls from text content when model doesn't use proper function calling.

    Some models (especially local ones via Ollama) output tool calls as JSON text
    instead of using the proper tool_calls response format. This function detects
    and extracts those tool calls.

    Args:
        content: The text content that may contain JSON tool calls
        available_tools: List of valid tool names to look for

    Returns:
        Tuple of (parsed_tool_calls, cleaned_content)
        - parsed_tool_calls: List of tool call dicts in OpenAI format
        - cleaned_content: Content with tool call JSON removed
    """
    if not content:
        return [], content

    tool_calls = []
    cleaned_content = content

    # Strategy 1: Direct tool name detection in JSON-like content
    # Check if content contains "name": "known_tool_name"
    for tool_name in available_tools:
        # Check for JSON-style tool reference
        name_patterns = [
            f'"name"\\s*:\\s*"{tool_name}"',
            f'"name":\\s*"{tool_name}"',
            f"'name'\\s*:\\s*'{tool_name}'",
        ]
        for pattern in name_patterns:
            if re.search(pattern, content):
                logger.info(f"Found tool call for '{tool_name}' in JSON-like content")

                # Try to extract arguments
                args = {}
                args_pattern = r'"arguments"\s*:\s*(\{[^{}]*\})'
                args_match = re.search(args_pattern, content)
                if args_match:
                    try:
                        args = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        args = {}

                tool_calls.append(_create_tool_call(tool_name, args))
                # Clean the entire JSON blob from content
                cleaned_content = ""  # Model only outputted JSON, nothing useful to show
                break
        if tool_calls:
            break

    # Strategy 2: Markdown code block with JSON
    if not tool_calls:
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        code_matches = re.findall(code_block_pattern, content, re.DOTALL)
        for match in code_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and "name" in parsed:
                    if parsed["name"] in available_tools:
                        tool_calls.append(_create_tool_call(parsed["name"], parsed.get("arguments", {})))
                        cleaned_content = cleaned_content.replace(f"```json\n{match}\n```", "")
                        cleaned_content = cleaned_content.replace(f"```\n{match}\n```", "")
                        break
            except json.JSONDecodeError:
                pass

    # Strategy 3: Model describes wanting to call a tool in plain English
    # Handles: "I should call get_project_state" or "The get_project_state function"
    if not tool_calls:
        for tool_name in available_tools:
            # Look for explicit mentions of calling/using the tool
            intent_patterns = [
                rf'\b(?:call|use|invoke|run|execute)\s+{re.escape(tool_name)}\b',
                rf'\b{re.escape(tool_name)}\s+(?:function|tool)\s+(?:is\s+)?available\b',
                rf'\bshould\s+(?:call|use)\s+{re.escape(tool_name)}\b',
                rf'\bI\'ll\s+(?:call|use)\s+{re.escape(tool_name)}\b',
            ]
            for pattern in intent_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    logger.info(f"Detected intent to call tool '{tool_name}' from text")
                    tool_calls.append(_create_tool_call(tool_name, {}))
                    # For intent-based detection, clear content since it's just reasoning
                    cleaned_content = ""
                    break
            if tool_calls:
                break

    # Deduplicate tool calls (some models repeat the same call)
    if len(tool_calls) > 1:
        seen = set()
        unique_calls = []
        for tc in tool_calls:
            key = (tc["function"]["name"], tc["function"]["arguments"])
            if key not in seen:
                seen.add(key)
                unique_calls.append(tc)
        tool_calls = unique_calls[:1]  # Only keep the first unique call

    return tool_calls, cleaned_content


def _format_tool_result_for_display(tool_name: str, result: dict) -> str:
    """Format a tool result for human-readable display.

    Args:
        tool_name: Name of the tool that was executed
        result: Tool result dict

    Returns:
        Human-readable summary string
    """
    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        return f"**Error:** {error}"

    data = result.get("data", result)

    if tool_name == "get_project_state":
        name = data.get("name", "Untitled")
        sources = data.get("sources", [])
        total_clips = data.get("total_clips", 0)
        sequence_clips = data.get("sequence_clips", 0)

        lines = [f"**Project:** {name}"]
        if sources:
            lines.append(f"**Videos:** {len(sources)}")
            for src in sources[:3]:  # Show first 3
                src_name = src.get("name", "Unknown")
                duration = src.get("duration", 0)
                clips = src.get("clips", 0)
                lines.append(f"  • {src_name} ({duration:.1f}s, {clips} clips)")
            if len(sources) > 3:
                lines.append(f"  • ...and {len(sources) - 3} more")
        lines.append(f"**Total Clips:** {total_clips}")
        if sequence_clips:
            lines.append(f"**Sequence:** {sequence_clips} clips")
        return "\n".join(lines)

    elif tool_name == "list_clips":
        clips = data if isinstance(data, list) else data.get("clips", [])
        if not clips:
            return "No clips available."
        lines = [f"**{len(clips)} clips:**"]
        for clip in clips[:5]:  # Show first 5
            clip_id = clip.get("id", "?")[:8]
            duration = clip.get("duration", 0)
            shot_type = clip.get("shot_type", "")
            shot_info = f" ({shot_type})" if shot_type else ""
            lines.append(f"  • {clip_id}... {duration:.1f}s{shot_info}")
        if len(clips) > 5:
            lines.append(f"  • ...and {len(clips) - 5} more")
        return "\n".join(lines)

    elif tool_name == "filter_clips":
        clips = data if isinstance(data, list) else data.get("clips", [])
        count = len(clips)
        if count == 0:
            return "No clips match the filter criteria."
        return f"Found **{count}** matching clips."

    elif tool_name == "detect_scenes":
        clip_count = data.get("clip_count", 0)
        source_name = data.get("source_name", "video")
        return f"Detected **{clip_count}** scenes in {source_name}."

    elif tool_name == "add_to_sequence":
        added = data.get("added", 0)
        return f"Added **{added}** clips to the sequence."

    # Default: show success message
    return f"✓ {tool_name} completed successfully."


def _create_tool_call(name: str, arguments: dict) -> dict:
    """Create a tool call dict in OpenAI format.

    Args:
        name: Tool name
        arguments: Tool arguments dict

    Returns:
        Tool call dict
    """
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
        }
    }


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
    clear_current_bubble = Signal()  # Clear the streaming bubble (for JSON suppression)
    tool_called = Signal(str, dict)  # tool_name, arguments
    tool_result = Signal(str, dict, bool)  # tool_name, result, success
    tool_result_formatted = Signal(str)  # Human-readable tool result summary
    gui_tool_requested = Signal(str, dict, str)  # tool_name, args, tool_call_id (for main thread execution)
    gui_tool_completed = Signal(str)  # tool_call_id (set by main thread when done)
    workflow_progress = Signal(str, int, int)  # step_name, current, total (for compound operations)
    complete = Signal(str, list)  # final response text, tool_history
    error = Signal(str)  # error message

    # GUI sync signals - emitted after tool execution to update GUI components
    youtube_search_completed = Signal(str, list)  # query, list of video dicts
    video_download_completed = Signal(str, dict)  # url, download result dict

    def __init__(
        self,
        config: ProviderConfig,
        messages: list[dict],
        project: Optional[Any] = None,
        busy_check: Optional[Callable[[str], bool]] = None,
        gui_state_context: Optional[str] = None,
        parent=None
    ):
        """Initialize the worker.

        Args:
            config: LLM provider configuration
            messages: Conversation history
            project: Active Project instance
            busy_check: Callback to check if operation is busy
            gui_state_context: Human-readable GUI state for system prompt
            parent: Parent QObject
        """
        super().__init__(parent)
        self.config = config
        self.messages = messages.copy()
        self.project = project
        self.busy_check = busy_check
        self.gui_state_context = gui_state_context
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

                # Execute tools with progress reporting
                total_tools = len(tool_calls)
                for tool_idx, tc in enumerate(tool_calls, start=1):
                    if self._stop_requested:
                        break

                    name = tc.get("function", {}).get("name", "unknown")
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    tool_call_id = tc.get("id", "unknown")

                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}

                    # Emit workflow progress for compound operations
                    if total_tools > 1:
                        self.workflow_progress.emit(name, tool_idx, total_tools)

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

                    # Emit GUI sync signals for specific tools
                    self._emit_gui_sync_signal(name, args, result)

                    # Emit human-readable summary
                    formatted = _format_tool_result_for_display(name, result)
                    if formatted:
                        self.tool_result_formatted.emit(formatted)

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
    ) -> tuple[list[dict], str]:
        """Stream a single LLM response.

        Args:
            client: LLM client
            messages: Full message history

        Returns:
            Tuple of (content_text, tool_calls)
        """
        from core.llm_client import ProviderType

        content = ""
        tool_calls = []
        display_buffer = ""  # Buffer for detecting JSON before displaying
        available_tools = [t.name for t in tool_registry.all_tools()]

        # Don't pass tools to Ollama - LiteLLM forces JSON mode which breaks responses
        # Instead, we rely on text-based tool parsing (the model outputs JSON in text)
        use_tools = None
        if self.config.provider != ProviderType.LOCAL:
            use_tools = tool_registry.to_openai_format()

        async for chunk in client.stream_chat(
            messages,
            tools=use_tools
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
                display_buffer += delta.content

                # Check if buffer looks like JSON tool call - don't display it
                if not self._looks_like_json_tool_call(display_buffer, available_tools):
                    # Safe to display - emit buffered content
                    if display_buffer:
                        self.text_chunk.emit(display_buffer)
                        display_buffer = ""
                # else: keep buffering, might be a tool call

            # Accumulate tool calls
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    self._accumulate_tool_call(tool_calls, tc)

        # Fallback: If no tool calls via API but content looks like JSON tool calls,
        # parse them from the text. This handles models that output tool calls as text.
        if not tool_calls and content:
            parsed_calls, cleaned_content = _parse_tool_calls_from_text(content, available_tools)

            if parsed_calls:
                logger.info(f"Parsed {len(parsed_calls)} tool call(s) from text content (model fallback)")
                tool_calls = parsed_calls
                content = cleaned_content
                # Don't display the JSON junk - it was a tool call
                display_buffer = ""
            elif display_buffer:
                # Not a tool call, display remaining buffer
                self.text_chunk.emit(display_buffer)

        return content, tool_calls

    def _looks_like_json_tool_call(self, text: str, available_tools: list[str]) -> bool:
        """Check if text looks like it might be a JSON tool call being streamed.

        Args:
            text: Text buffer to check
            available_tools: List of valid tool names

        Returns:
            True if text looks like JSON tool call (should suppress display)
        """
        stripped = text.strip()

        # Starts with { - might be JSON
        if stripped.startswith('{'):
            # Check if it contains tool name references
            for tool_name in available_tools:
                if f'"{tool_name}"' in stripped or f"'{tool_name}'" in stripped:
                    return True
            # Also check for generic tool call patterns
            if '"name"' in stripped or '"arguments"' in stripped:
                return True

        return False

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
        from core.llm_client import ProviderType
        from core.settings import load_settings

        # Get settings for default paths
        settings = load_settings()

        prompt = f"""You are an AI assistant for Scene Ripper, a video scene detection and editing tool.

DEFAULT PATHS (from user settings):
- Download directory: {settings.download_dir}
- Export directory: {settings.export_dir}

When downloading videos, use these defaults - do NOT ask the user for a path unless they specify one.

You help users create video projects by:
- Detecting scenes in videos
- Analyzing clips (colors, shot types, transcription)
- Building sequences from clips
- Exporting clips and datasets

IMPORTANT BEHAVIOR RULES:
1. Only perform the SPECIFIC task the user requests - nothing more
2. Do NOT automatically download, analyze, or process unless explicitly asked
3. After completing a task, STOP and report results - do not chain additional actions
4. If you think follow-up actions would help, SUGGEST them verbally - don't execute them
5. When in doubt, ask the user before taking action

WORKFLOW AUTOMATION (for multi-step requests):
When the user requests compound operations like "download 5 videos and detect scenes in each":
1. Process each step sequentially - complete one before starting the next
2. Report progress after each step (e.g., "Downloaded video 2/5, detecting scenes...")
3. If a step fails, continue with remaining items and report all failures at the end
4. Provide a summary when complete showing successes and failures
5. For large batches, confirm with the user before proceeding if the operation may take a long time

Example workflow response:
- "Processing 3 videos..."
- "Video 1/3: Downloaded 'example.mp4' - detecting scenes..."
- "Video 1/3: Complete - found 12 scenes"
- "Video 2/3: Downloaded 'demo.mp4' - detecting scenes..."
- "Video 2/3: Failed - file not found, continuing with remaining videos"
- "Video 3/3: Complete - found 8 scenes"
- "Summary: 2/3 videos processed successfully (20 total scenes), 1 failed"

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

If the user's request is unclear, ask clarifying questions.
"""

        # For Ollama, include tool schemas in the prompt since we don't pass them via API
        if self.config.provider == ProviderType.LOCAL:
            prompt += """
IMPORTANT: When you need to use a tool, output a JSON object with "name" and "arguments" fields.
Example: {"name": "get_project_state", "arguments": {}}

After receiving tool results, provide a well-structured response that:
- Summarizes the key information clearly
- Includes relevant details like durations, counts, and file names
- Uses markdown formatting for readability

CRITICAL: After completing the user's request, STOP. Do NOT automatically call more tools.
If the user might want to do something next, briefly mention it but do NOT execute it.

Available tools:
"""
            for tool in tool_registry.all_tools():
                params = []
                if tool.parameters.get("properties"):
                    for param, info in tool.parameters["properties"].items():
                        param_type = info.get("type", "string")
                        required = param in tool.parameters.get("required", [])
                        req_marker = " (required)" if required else ""
                        params.append(f"    - {param}: {param_type}{req_marker}")
                params_str = "\n".join(params) if params else "    (no parameters)"
                prompt += f"\n- {tool.name}: {tool.description}\n{params_str}\n"

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

        # Add GUI state context if available
        if self.gui_state_context:
            prompt += f"""

CURRENT GUI STATE:
{self.gui_state_context}
"""

        return prompt

    def _emit_gui_sync_signal(self, tool_name: str, args: dict, result: dict):
        """Emit GUI sync signals for tools that should update the GUI.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
            result: Tool execution result
        """
        # Only process tools we care about syncing
        if tool_name not in ("search_youtube", "download_video"):
            return

        if not result.get("success", False):
            return

        # ToolExecutor wraps the actual tool return value in "result" key
        data = result.get("result", result)

        if tool_name == "search_youtube":
            query = data.get("query", args.get("query", ""))
            videos = data.get("results", [])
            if videos:
                logger.info(f"GUI SYNC: youtube_search_completed ({len(videos)} videos)")
                self.youtube_search_completed.emit(query, videos)
            else:
                logger.warning(f"GUI SYNC: search_youtube had no videos. Keys: {list(data.keys())}")

        elif tool_name == "download_video":
            # Emit signal with URL and download result
            url = args.get("url", "")
            logger.info(f"Emitting video_download_completed for {url}")
            self.video_download_completed.emit(url, data)

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
