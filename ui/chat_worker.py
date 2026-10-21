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

try:
    from litellm.exceptions import RateLimitError
except ImportError:
    RateLimitError = None

# Per-turn project/GUI state rides in the user message it was sent with, so the
# system prompt stays fixed for the conversation. Claude thinking blocks are
# bound to everything before them; editing earlier turns invalidates them.
from core.chat_export import TURN_CONTEXT_CLOSE, TURN_CONTEXT_OPEN
from core.chat_tools import get_tool_timeout, tools as tool_registry
from core.llm_client import (
    LLMClient,
    ProviderConfig,
    check_ollama_health,
    normalize_provider_error,
)
from core.tool_executor import ToolExecutor
from ui.workers.gui_tool_mailbox import GuiToolMailbox

# Maximum iterations for the agent tool loop. Prevents infinite loops when the
# LLM repeatedly calls tools without converging on a final response.  Set high
# enough to allow complex multi-step workflows (e.g., downloading and processing
# many videos) but low enough to catch genuine infinite loops.
AGENT_MAX_ITERATIONS = 25


logger = logging.getLogger(__name__)


def strip_thinking_blocks(messages: list[dict]) -> list[dict]:
    """Return copies of messages without Claude thinking blocks."""
    return [
        {key: value for key, value in message.items() if key != "thinking_blocks"}
        for message in messages
    ]


def _is_thinking_replay_error(error: Exception) -> bool:
    # e.g. "messages.5.content.0: Invalid `signature` in `thinking` block."
    text = str(error)
    return "signature" in text and "thinking" in text


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
    gui_tool_requested = Signal(str, dict, str)  # tool_name, args, tool_call_id (for main thread execution)
    gui_tool_completed = Signal(str)  # tool_call_id (set by main thread when done)
    gui_tool_cancelled = Signal(str, str)  # tool_name, unique GUI request token
    workflow_progress = Signal(str, int, int)  # step_name, current, total (for compound operations)
    complete = Signal(str, list)  # final response text, tool_history
    error = Signal(str)  # error message
    auth_failed = Signal(str, str)  # category, user message

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
        system_prompt: Optional[str] = None,
        parent=None
    ):
        """Initialize the worker.

        Args:
            config: LLM provider configuration
            messages: Conversation history
            project: Active Project instance
            busy_check: Callback to check if operation is busy
            gui_state_context: Human-readable GUI state, sent with this turn
            system_prompt: The conversation's system prompt (built if None)
            parent: Parent QObject
        """
        super().__init__(parent)
        self.config = config
        self.messages = messages.copy()
        self.project = project
        self.busy_check = busy_check
        self.gui_state_context = gui_state_context
        self.system_prompt = system_prompt
        # Messages as sent, without the system prompt; becomes the chat history.
        self.conversation: list[dict] = []
        self._thinking_blocks: list[dict] = []
        self._thinking_replayable = True
        self._stop_requested = False

        # For GUI tool synchronization
        self._gui_tool_mailbox = GuiToolMailbox()

    def run(self):
        """Run the agent loop."""
        try:
            asyncio.run(self._async_run())
        except Exception as e:
            logger.exception("Agent worker error")
            self.error.emit(str(e))

    async def _async_run(self):
        """Async implementation of the agent loop."""
        from core.llm_client import ProviderType, is_claude_model

        # Check Ollama health if using local provider
        if self.config.provider == ProviderType.LOCAL:
            api_base = self.config.get_api_base()
            healthy, error_msg = await check_ollama_health(api_base)
            if not healthy:
                self.error.emit(error_msg)
                return

        client = LLMClient(self.config)
        executor = ToolExecutor(tool_registry, self.project, self.busy_check)

        if self.system_prompt is None:
            self.system_prompt = self._build_system_prompt()
        history = self._with_turn_context(self.messages)
        if not is_claude_model(self.config.model):
            history = strip_thinking_blocks(history)
        full_messages = [{"role": "system", "content": self.system_prompt}] + history

        iteration = 0
        tool_history = []  # Track tool interactions for history
        replay_retried = False

        while iteration < AGENT_MAX_ITERATIONS and not self._stop_requested:
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
                if self._thinking_blocks:
                    if self._thinking_replayable:
                        assistant_msg["thinking_blocks"] = self._thinking_blocks
                    else:
                        # LiteLLM sends thinking ahead of text and tool calls, so
                        # a block that arrived after them can't be replayed in
                        # place. Only a leading run of blocks may be dropped, so
                        # drop every earlier one as well.
                        full_messages[:] = strip_thinking_blocks(full_messages)

                # Add to full messages
                full_messages.append(assistant_msg)

                # If no tool calls, we're done
                if not tool_calls:
                    self.conversation = full_messages[1:]
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
                        # Provider call IDs may be reused. GUI transport gets a
                        # unique token; history keeps the original provider ID.
                        token = self._gui_tool_mailbox.begin(name)
                        if token is not None:
                            self.gui_tool_requested.emit(name, args, token)

                        # Wait for main thread to complete (with per-tool timeout)
                        tool_timeout = get_tool_timeout(name)
                        logger.info(f"Waiting for GUI tool '{name}' with timeout {tool_timeout}s ({tool_timeout/60:.1f} min)")
                        gui_result = self._gui_tool_mailbox.wait(tool_timeout)
                        if gui_result is None or self._stop_requested:
                            logger.warning(f"GUI tool '{name}' timed out after {tool_timeout}s or was cancelled (stop_requested={self._stop_requested})")
                            # Emit cancellation signal so MainWindow can stop orphaned workers
                            if token is not None:
                                self.gui_tool_cancelled.emit(name, token)
                            result = {
                                "tool_call_id": tool_call_id,
                                "name": name,
                                "success": False,
                                "error": "Tool execution timed out or was cancelled"
                            }
                        else:
                            result = {**gui_result, "tool_call_id": tool_call_id}
                    else:
                        # Execute non-GUI tool directly in worker thread
                        result = executor.execute(tc)

                    # Determine actual success: check both executor and inner tool
                    executor_success = result.get("success", False)
                    inner_result = result.get("result", {})
                    tool_success = executor_success
                    if executor_success and isinstance(inner_result, dict):
                        # If executor succeeded, check if tool itself reported failure
                        tool_success = inner_result.get("success", True)

                    self.tool_result.emit(name, result, tool_success)

                    # Emit GUI sync signals for specific tools
                    self._emit_gui_sync_signal(name, args, result)

                    # Add tool result to messages
                    tool_msg = executor.format_for_llm(result)
                    full_messages.append(tool_msg)
                    tool_history.append(tool_msg)

                # Continue loop for LLM to process tool results

            except Exception as e:
                if (
                    not replay_retried
                    and _is_thinking_replay_error(e)
                    and any("thinking_blocks" in m for m in full_messages)
                ):
                    logger.warning("Claude rejected replayed thinking blocks; retrying without them")
                    replay_retried = True
                    full_messages[:] = strip_thinking_blocks(full_messages)
                    iteration -= 1
                    continue
                logger.exception("Error in agent loop")
                # Provide user-friendly message for rate limit errors
                error_msg = str(e)
                from core.auth_errors import (
                    CATEGORY_NONE,
                    classify_subscription_error,
                    user_message_for_category,
                )

                auth_category = classify_subscription_error(e)
                if auth_category != CATEGORY_NONE:
                    user_message = user_message_for_category(auth_category, error_msg)
                    self.auth_failed.emit(auth_category, user_message)
                    self.error.emit(user_message)
                    return

                if RateLimitError and isinstance(e, RateLimitError):
                    error_msg = (
                        "Rate limit exceeded. The conversation has grown too large "
                        "for your API tier's token limit. Try clearing the chat history "
                        "or waiting a minute before trying again."
                    )
                elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                    error_msg = (
                        "Rate limit exceeded. The conversation has grown too large "
                        "for your API tier's token limit. Try clearing the chat history "
                        "or waiting a minute before trying again."
                    )
                else:
                    error_msg = normalize_provider_error(
                        self.config.provider.value,
                        e,
                    )
                self.error.emit(error_msg)
                return

        if iteration >= AGENT_MAX_ITERATIONS:
            self.error.emit("Maximum tool iterations reached. The agent may be stuck in a loop.")
        elif self._stop_requested:
            self.conversation = full_messages[1:]
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
        from core.llm_client import ProviderType, is_claude_model

        content = ""
        tool_calls = []
        display_buffer = ""  # Buffer for detecting JSON before displaying
        # Replay is verified on the direct Anthropic route only.
        claude = self.config.provider == ProviderType.ANTHROPIC and is_claude_model(self.config.model)
        thinking_blocks: list[dict] = []
        thinking_text = ""
        seen_output = False
        self._thinking_replayable = True
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

            # Thinking streams as text fragments; a signature closes each block.
            for block in (getattr(delta, "thinking_blocks", None) or []) if claude else []:
                if seen_output:
                    self._thinking_replayable = False
                if block.get("type") == "redacted_thinking":
                    thinking_blocks.append(dict(block))
                    continue
                thinking_text += block.get("thinking") or ""
                if block.get("signature"):
                    thinking_blocks.append(
                        {"type": "thinking", "thinking": thinking_text, "signature": block["signature"]}
                    )
                    thinking_text = ""
            if getattr(delta, "content", None) or getattr(delta, "tool_calls", None):
                seen_output = True

            # LiteLLM reports Claude's stop_reason "refusal" as content_filter;
            # discard any partial text so it isn't treated as an answer.
            if choice.finish_reason == "content_filter" and is_claude_model(self.config.model):
                raise RuntimeError(
                    f"{self.config.model} declined this request (Anthropic safety filter). "
                    "Rephrase it or choose a different model."
                )

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
            parsed_calls, cleaned_content = (
                _parse_tool_calls_from_text(content, available_tools)
                if self.config.provider == ProviderType.LOCAL
                else ([], content)
            )

            if parsed_calls:
                logger.info(f"Parsed {len(parsed_calls)} tool call(s) from text content (model fallback)")
                tool_calls = parsed_calls
                content = cleaned_content
                # Don't display the JSON junk - it was a tool call
                display_buffer = ""
            elif display_buffer:
                # Not a tool call, display remaining buffer
                self.text_chunk.emit(display_buffer)

        self._thinking_blocks = thinking_blocks
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
        """Build the system prompt, fixed for the whole conversation.

        Returns:
            System prompt string
        """
        from core.llm_client import ProviderType

        prompt = f"""You are an AI assistant for Scene Ripper, a video scene detection and editing tool.

When this is the first message in a new conversation, briefly introduce yourself and what you can help with (1-2 sentences), then answer the user's question. Do not repeat the introduction on subsequent messages.

Each user message starts with a {TURN_CONTEXT_OPEN} block: the default paths, project state and GUI state when that message was sent. The newest one is current. The user didn't type it, so don't quote it back.

When downloading videos, use the default paths from that block - do NOT ask the user for a path unless they specify one.

You help users create video projects by:
- Detecting scenes in videos
- Analyzing clips (colors, shot types, transcription, description, classification, object detection, person detection, face detection)
- Building sequences from clips
- Exporting clips and datasets

ANALYSIS TOOLS:
When analyzing clips, prefer the unified start_clip_analysis tool over individual tools:
- start_clip_analysis(clip_ids=[...], operations=["colors", "shots", "transcription"])
This runs multiple operations in a single call.

CUSTOM VISUAL QUERY RESULTS:
After custom_visual_query returns, summarize the matches from the tool result as one bullet
per match, containing the clip ID plus confidence and description/source if present.

BEHAVIOR RULES:
The user directs the edit, and downloads and analysis can take a long time, so do what was asked and no more:
1. Perform the specific task the user requests.
2. Download or process media only when asked, or when processing is needed to answer a question about missing properties (e.g. "what are the colors?" -> run start_clip_analysis with operations=["colors"]).
3. When the task is done, report the results and stop; don't chain additional actions.
4. If follow-up actions would help, suggest them instead of running them.
5. PROJECT NAMING: Before running a state-modifying tool (detect_scenes, download_video, add_to_sequence, etc.) on an unnamed project (Name: "Untitled Project", Path: Unsaved), first ask the user what to name the project, then set it with set_project_name. The project is auto-saved after each operation.

WORKFLOW AUTOMATION (for multi-step requests):
For compound operations like "download 5 videos and detect scenes in each", finish each step before starting the next. If one item fails, continue with the remaining items. End with a summary of successes and failures. For a large batch that may take a long time, confirm with the user before starting.

TAB NAVIGATION:
Tools do NOT auto-switch tabs. If you want the user to see the results of an action in a specific tab,
call navigate_to_tab explicitly AFTER the action completes. For example:
- After adding clips to sequence: navigate_to_tab("sequence")
- After sending clips to analyze: navigate_to_tab("analyze")
- After scene detection: navigate_to_tab("cut")
Background workers (detection, analysis, export) will auto-switch to the relevant tab.

BATCH SCENE DETECTION (detect_all_unanalyzed):
detect_all_unanalyzed queues every unanalyzed source and runs detection in the background. Each video takes 1-5 minutes, so a large batch can take hours, and you can't wait between tool calls. After queuing, call check_detection_status once to confirm detection started, then tell the user how many videos are queued, a time estimate (about 2-3 minutes per video), and that they can ask for a status update at any time. Check status once each time they ask. Detection has failed only if is_running=False and sources_analyzed < sources_total.

If the user's request is unclear, ask clarifying questions.

PLANNING MODE:
When the user asks you to "plan" something or describes a complex multi-step workflow (3+ steps), use the planning system:

1. DETECT PLANNING REQUESTS: Look for keywords like "plan", "create a workflow", or complex requests involving multiple operations (e.g., "download 10 videos, detect scenes in all, and analyze colors").

2. CLARIFY FIRST: Ask 2-3 focused questions to understand requirements:
   - What constraints matter? (quality, duration, count limits)
   - What's the desired outcome format?
   - Any preferences for how to handle edge cases?

3. BREAK DOWN: After getting answers, decompose into 3-10 clear steps.
   Each step should be a single logical action (search, download, detect, export, etc.)

4. PRESENT PLAN: Call present_plan to create the plan. Only the tool call creates a plan object; steps written in message text are not stored, and start_plan_execution then fails with "No plan exists".

   Parameters:
   - steps: List of human-readable step descriptions
   - summary: Brief description of what the plan accomplishes
   Example: present_plan(steps=["Search YouTube for 'mushroom documentary'", "Download top 10 results", "Detect scenes in all videos"], summary="Download and process mushroom documentaries")

5. WAIT FOR CONFIRMATION: After calling present_plan, STOP and wait for the user to confirm or edit the plan. Do NOT execute anything until confirmation.

6. EXECUTE AFTER CONFIRMATION: Once confirmed, execute the steps in order. If a step fails, report the error and wait for the user to choose [Retry] or [Stop]. End with a summary.

Example step descriptions (human-readable, not tool names):
- "Search YouTube for 'mushroom documentary' videos"
- "Download the top 10 search results"
- "Detect scenes in all downloaded videos"
- "Analyze clips for colors and shot types"
- "Filter clips by shot type (e.g., close-ups only)"
- "Export the clips as individual files"

PLANNING CONSTRAINTS:
Plans may not exceed 20 steps. If a workflow needs more, break it into multiple plans.

When creating plans, follow these tool dependency rules:

1. SEARCH → DOWNLOAD RULE: After a "search" step (search_youtube), you MUST include
   a "download" step for those results BEFORE any subsequent search step.
   Search results are ephemeral - a new search replaces previous results.

   WRONG:
   - Step 1: Search for "mushrooms"
   - Step 2: Search for "fungi"      ← Previous search results lost!
   - Step 3: Download all

   CORRECT:
   - Step 1: Search for "mushrooms"
   - Step 2: Download mushroom videos
   - Step 3: Search for "fungi"
   - Step 4: Download fungi videos

2. If the user wants to search multiple topics, create a plan that interleaves
   search and download steps for each topic.

3. SEQUENCE GENERATION: Use list_sorting_algorithms to check which algorithms exist and which are available for the current clips, then build the sequence with the tool its entry names (generate_remix for the direct sorting algorithms, generate_sequence for any registry algorithm). For matching clips to a reference video's structure, use generate_reference_guided.
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

        return prompt

    def _with_turn_context(self, messages: list[dict]) -> list[dict]:
        """Prefix the newest user message with this turn's context block."""
        messages = list(messages)
        if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str):
            context = self._build_turn_context().strip()
            messages[-1] = {
                **messages[-1],
                "content": f"{TURN_CONTEXT_OPEN}\n{context}\n{TURN_CONTEXT_CLOSE}\n\n{messages[-1]['content']}",
            }
        return messages

    def _build_turn_context(self) -> str:
        """Build the project and GUI state sent with the current user message."""
        from core.settings import load_settings

        settings = load_settings()

        prompt = f"""DEFAULT PATHS (from user settings):
- Download directory: {settings.download_dir}
- Export directory: {settings.export_dir}
"""

        if self.project:
            # Add project context
            sources_info = []
            for s in self.project.sources:
                clip_count = len(self.project.clips_by_source.get(s.id, []))
                status = "analyzed" if s.analyzed else "not analyzed"
                sources_info.append(f"  - {s.file_path.name} ({s.duration_seconds:.1f}s, {clip_count} clips, {status})")

            seq_length = 0
            seq_duration_seconds = 0.0
            seq_shot_types_line = ""
            if self.project.sequence and self.project.sequence.tracks:
                seq_length = len(self.project.sequence.tracks[0].clips)
                seq_duration_seconds = self.project.sequence.duration_seconds
                # Summarise per-clip shot types in the sequence
                seq_shot_counts: dict[str, int] = {}
                clips_by_id = self.project.clips_by_id
                for track in self.project.sequence.tracks:
                    for sc in track.clips:
                        src_clip = clips_by_id.get(sc.source_clip_id)
                        if src_clip and src_clip.shot_type:
                            seq_shot_counts[src_clip.shot_type] = seq_shot_counts.get(src_clip.shot_type, 0) + 1
                if seq_shot_counts:
                    sorted_seq_shots = sorted(seq_shot_counts.items(), key=lambda x: x[1], reverse=True)
                    seq_shot_strs = [f"{name}: {cnt}" for name, cnt in sorted_seq_shots]
                    seq_shot_types_line = f"\n  Sequence shot types: {', '.join(seq_shot_strs)}"

            # Compute analysis metadata counts
            total_clips = len(self.project.clips)
            analysis_counts = {}
            if total_clips > 0:
                for field_name in ("shot_type", "dominant_colors", "description",
                                   "transcript", "detected_objects", "person_count",
                                   "extracted_texts", "cinematography", "face_embeddings",
                                   "gaze_category"):
                    count = sum(
                        1 for c in self.project.clips
                        if getattr(c, field_name, None) is not None
                    )
                    if count > 0:
                        analysis_counts[field_name] = count

            analysis_lines = ""
            if analysis_counts:
                parts = [f"{k}: {v}/{total_clips}" for k, v in analysis_counts.items()]
                analysis_lines = f"\n- Analysis Coverage: {', '.join(parts)}"

            # Same catalog list_sorting_algorithms returns, so the two never disagree
            from core.spine.settings_io import list_sorting_algorithms
            algo_catalog = list_sorting_algorithms(self.project)["algorithms"]
            available_algos = [str(a["key"]) for a in algo_catalog if a.get("available")]
            unavailable_algos = [str(a["key"]) for a in algo_catalog if not a.get("available")]
            if available_algos:
                analysis_lines += f"\n- Available algorithms: {', '.join(available_algos)}"
            if unavailable_algos:
                analysis_lines += f"\n- Unavailable (need analysis): {', '.join(unavailable_algos)}"

            # Build analysis distribution summaries
            distribution_lines = ""
            if total_clips > 0:
                dist_parts = []

                # Shot type distribution
                shot_counts: dict[str, int] = {}
                for c in self.project.clips:
                    if c.shot_type:
                        shot_counts[c.shot_type] = shot_counts.get(c.shot_type, 0) + 1
                if shot_counts:
                    sorted_shots = sorted(shot_counts.items(), key=lambda x: x[1], reverse=True)
                    shot_strs = [f"{name}: {cnt}" for name, cnt in sorted_shots]
                    dist_parts.append(f"  Shot types: {', '.join(shot_strs)}")

                # Top 5 most common object labels
                object_counts: dict[str, int] = {}
                for c in self.project.clips:
                    if c.object_labels:
                        for label in c.object_labels:
                            object_counts[label] = object_counts.get(label, 0) + 1
                if object_counts:
                    top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    obj_strs = [f"{name}: {cnt}" for name, cnt in top_objects]
                    dist_parts.append(f"  Top objects: {', '.join(obj_strs)}")

                # Clips with transcripts
                clips_with_transcripts = sum(
                    1 for c in self.project.clips
                    if c.transcript
                )
                if clips_with_transcripts:
                    dist_parts.append(f"  Clips with transcripts: {clips_with_transcripts}/{total_clips}")

                if dist_parts:
                    distribution_lines = "\n- Analysis Distributions:\n" + "\n".join(dist_parts)

            prompt += f"""

CURRENT PROJECT STATE:
- Name: {self.project.metadata.name}
- Path: {self.project.path or 'Unsaved'}
- Sources ({len(self.project.sources)} video(s)):
{chr(10).join(sources_info) if sources_info else "  (none)"}
- Total Clips: {total_clips}
- Sequence Length: {seq_length} clips ({seq_duration_seconds:.1f}s){seq_shot_types_line}{analysis_lines}{distribution_lines}
- ACTIVE SETTINGS:
  Export quality: {settings.export_quality}
  Transcription model: {settings.transcription_model}
  Description model tier: {settings.description_model_tier}

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
        # Only process tools whose definition has emits_gui_sync=True
        tool_def = tool_registry.get(tool_name)
        if not tool_def or not tool_def.emits_gui_sync:
            return

        if not result.get("success", False):
            return

        # ToolExecutor wraps the actual tool return value in "result" key
        data = result.get("result", result)

        # Check if inner tool reported failure
        if isinstance(data, dict) and data.get("success") is False:
            error = data.get("error", "Unknown error")
            logger.warning(f"GUI SYNC: {tool_name} failed: {error}")
            return

        if tool_name == "search_youtube":
            query = data.get("query", args.get("query", ""))
            videos = data.get("results", [])
            if videos:
                logger.info(f"GUI SYNC: youtube_search_completed ({len(videos)} videos)")
                self.youtube_search_completed.emit(query, videos)
            else:
                logger.warning(f"GUI SYNC: search_youtube returned no videos for query: {query}")

        elif tool_name == "download_video":
            # Emit signal with URL and download result
            url = args.get("url", "")
            logger.info(f"Emitting video_download_completed for {url}")
            self.video_download_completed.emit(url, data)

    def stop(self) -> None:
        """Request the worker to stop."""
        self._stop_requested = True
        # Unblock any waiting GUI tool
        self._gui_tool_mailbox.cancel()

    def set_gui_tool_result(self, result: dict) -> bool:
        """Called by main thread to provide GUI tool result.

        Args:
            result: Tool execution result dict
        """
        return self._gui_tool_mailbox.submit(result)

    def is_gui_tool_pending(self, token: str, name: str) -> bool:
        """Allow GUI handoffs only while their original request is waiting."""
        return self._gui_tool_mailbox.is_pending(token, name)
