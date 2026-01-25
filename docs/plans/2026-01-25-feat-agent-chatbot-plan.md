# Agent Chatbot Feature Plan

**Date**: 2026-01-25
**Status**: Implementation Complete (Initial)
**Builds On**: Completed Phases 1-4 of Agent-Native Architecture

---

## Overview

Add an AI-powered chat interface to Scene Ripper that allows users to describe what they want to create in natural language. The agent executes operations using the CLI commands and Project class built in Phases 1-4, with changes appearing immediately in the GUI via the observer pattern.

### User Questions Answered

1. **What LLM to use?** Flexible multi-provider architecture supporting local (Ollama with Qwen3), self-hosted, and cloud (OpenAI, Anthropic, Gemini). LiteLLM provides unified interface.

2. **Can it build a project in the GUI?** Yes. Agent modifies the live Project instance directly. Changes trigger observer callbacks, which ProjectSignalAdapter bridges to Qt signals, updating UI components immediately.

3. **Can it continue existing projects?** Yes. Agent receives project context including sources, clips, and sequence state. Can add new content, modify sequences, or extend analysis on existing clips.

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      MainWindow                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   ChatPanel                           │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Message History (QScrollArea)                  │  │   │
│  │  │  - User messages                                │  │   │
│  │  │  - Assistant responses (streaming)              │  │   │
│  │  │  - Tool execution indicators                    │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Input Area + Send Button                       │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Provider/Model Selector (Settings)            │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ChatAgentWorker (QThread)                 │
│                                                              │
│  ┌────────────────┐    ┌────────────────┐                   │
│  │  LLMClient     │───▶│  ToolExecutor  │                   │
│  │  (LiteLLM)     │    │                │                   │
│  └────────────────┘    └────────────────┘                   │
│         │                      │                             │
│         ▼                      ▼                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Signals: text_chunk, tool_called, tool_result,     │    │
│  │           complete, error                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Tool Layer                            │
│                                                              │
│  ┌───────────────────────┐  ┌───────────────────────────┐   │
│  │  Core Functions       │  │  CLI Subprocess           │   │
│  │  (GUI state changes)  │  │  (Batch operations)       │   │
│  │                       │  │                           │   │
│  │  - project.add_source │  │  - scene-ripper export    │   │
│  │  - project.add_clips  │  │  - scene-ripper detect    │   │
│  │  - project.add_to_seq │  │  - scene-ripper download  │   │
│  └───────────────────────┘  └───────────────────────────┘   │
│              │                                               │
│              ▼                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Project (Observer Pattern)                          │    │
│  │  → ProjectSignalAdapter → Qt Signals → UI Update     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Files to Create/Modify

```
core/
├── llm_client.py          # NEW: Multi-provider LLM client via LiteLLM
├── chat_tools.py          # NEW: Tool definitions for agent
├── tool_executor.py       # NEW: Safe tool execution with error handling

ui/
├── chat_panel.py          # NEW: Chat UI panel
├── chat_widgets.py        # NEW: Message bubble widgets
├── chat_worker.py         # NEW: QThread worker for LLM calls
├── main_window.py         # MODIFY: Add chat panel toggle

core/settings.py           # MODIFY: Add LLM provider settings
```

---

## Critical Fixes (From Spec Analysis)

The following critical issues were identified through spec-flow analysis and MUST be addressed:

### Fix 1: Secure LLM API Key Storage (Security)

**Problem**: The original spec showed `llm_api_key: str = ""` stored directly in Settings, which would expose API keys in config.json.

**Solution**: Use keyring storage like the existing YouTube API key pattern.

Add to `core/settings.py`:

```python
# Constants
KEYRING_SERVICE = "scene-ripper"
KEYRING_LLM_API_KEY = "llm_api_key"

# Environment variables
ENV_LLM_PROVIDER = "SCENE_RIPPER_LLM_PROVIDER"
ENV_LLM_MODEL = "SCENE_RIPPER_LLM_MODEL"
ENV_LLM_API_KEY = "SCENE_RIPPER_LLM_API_KEY"
ENV_LLM_API_BASE = "SCENE_RIPPER_LLM_API_BASE"

def _get_llm_api_key_from_keyring() -> str:
    """Get LLM API key from system keyring."""
    try:
        import keyring
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_LLM_API_KEY)
        return key or ""
    except Exception:
        return ""

def _set_llm_api_key_in_keyring(api_key: str) -> bool:
    """Store LLM API key in system keyring."""
    try:
        import keyring
        if api_key:
            keyring.set_password(KEYRING_SERVICE, KEYRING_LLM_API_KEY, api_key)
        else:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_LLM_API_KEY)
        return True
    except Exception:
        return False
```

**Note**: Remove `llm_api_key` from the Settings dataclass. Load it separately via keyring or environment variable at runtime.

### Fix 2: Concurrent Message Handling (Thread Safety)

**Problem**: User can send new message while previous response is streaming, creating race conditions.

**Solution**: Disable input during streaming, add cancel button, properly handle worker lifecycle.

Update `ui/chat_panel.py`:

```python
class ChatPanel(QWidget):
    """Collapsible chat panel for agent interaction."""

    message_sent = Signal(str)
    cancel_requested = Signal()  # NEW: Cancel signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._is_streaming = False  # NEW: Streaming state
        self._response_finished_handled = False

    def _setup_ui(self):
        # ... existing code ...

        # Input area with cancel button
        input_layout = QHBoxLayout()

        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Describe what you want to create...")
        self.input_field.setMaximumHeight(80)
        input_layout.addWidget(self.input_field, 1)

        # Button container for send/cancel
        button_layout = QVBoxLayout()

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send_clicked)
        button_layout.addWidget(self.send_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setVisible(False)  # Hidden by default
        button_layout.addWidget(self.cancel_button)

        input_layout.addLayout(button_layout)
        layout.addLayout(input_layout)

    def _on_send_clicked(self):
        if self._is_streaming:
            return  # Ignore if already streaming

        message = self.input_field.toPlainText().strip()
        if not message:
            return

        self.input_field.clear()
        self._add_user_message(message)
        self._set_streaming_state(True)
        self.message_sent.emit(message)

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit()
        self._set_streaming_state(False)

    def _set_streaming_state(self, is_streaming: bool):
        """Update UI state for streaming."""
        self._is_streaming = is_streaming
        self._response_finished_handled = False  # Reset guard
        self.input_field.setEnabled(not is_streaming)
        self.send_button.setVisible(not is_streaming)
        self.cancel_button.setVisible(is_streaming)
```

Update `ui/main_window.py` to handle cancellation:

```python
def _on_chat_message(self, message: str):
    # ... existing code ...

    # Store message for history
    self._last_user_message = message  # FIX 5: Set before starting worker

    # Cancel any existing worker
    if hasattr(self, '_chat_worker') and self._chat_worker.isRunning():
        self._chat_worker.stop()
        self._chat_worker.wait(1000)  # Wait up to 1 second

    # Start new worker
    self._chat_worker = ChatAgentWorker(...)
    # ... rest of method ...

def _setup_chat_panel(self):
    # ... existing code ...

    # Connect cancel signal
    self.chat_panel.cancel_requested.connect(self._on_chat_cancel)

def _on_chat_cancel(self):
    """Handle chat cancellation."""
    if hasattr(self, '_chat_worker') and self._chat_worker.isRunning():
        self._chat_worker.stop()
        self.chat_panel.add_assistant_message("*Cancelled*")
```

### Fix 3: Tool/Worker Conflict Prevention (Concurrency)

**Problem**: Agent tools may conflict with GUI workers (e.g., agent calls detect_scenes while DetectionWorker is running).

**Solution**: Add busy-state checking to ToolExecutor.

Update `core/tool_executor.py`:

```python
class ToolExecutor:
    """Execute tools safely with error handling and conflict detection."""

    def __init__(
        self,
        registry: ToolRegistry,
        project: Optional[Project] = None,
        busy_check: Optional[Callable[[str], bool]] = None  # NEW
    ):
        self.registry = registry
        self.project = project
        self.busy_check = busy_check  # Callback to check if operation is busy

    def execute(self, tool_call: dict) -> dict:
        # ... existing parsing code ...

        # NEW: Check for conflicting operations
        if self.busy_check and tool.name in self._get_conflicting_tools():
            if self.busy_check(tool.name):
                return self._error_result(
                    tool_call_id, name,
                    f"Cannot run {name}: A similar operation is already in progress. "
                    "Please wait for it to complete or cancel it first."
                )

        # ... rest of execution code ...

    def _get_conflicting_tools(self) -> set:
        """Tools that conflict with GUI workers."""
        return {
            "detect_scenes",      # Conflicts with DetectionWorker
            "analyze_colors",     # Conflicts with ColorAnalysisWorker
            "analyze_shots",      # Conflicts with ShotClassificationWorker
            "transcribe",         # Conflicts with TranscriptionWorker
            "download_video",     # Conflicts with DownloadWorker
        }
```

Update MainWindow to provide busy_check callback:

```python
def _on_chat_message(self, message: str):
    # ... existing code ...

    # Create executor with busy check
    def check_busy(tool_name: str) -> bool:
        """Check if a conflicting worker is running."""
        worker_map = {
            "detect_scenes": "_detection_worker",
            "analyze_colors": "_color_worker",
            "analyze_shots": "_shot_worker",
            "transcribe": "_transcription_worker",
            "download_video": "_download_worker",
        }
        attr = worker_map.get(tool_name)
        if attr and hasattr(self, attr):
            worker = getattr(self, attr)
            return worker is not None and worker.isRunning()
        return False

    # Pass busy_check to worker (which passes to executor)
    self._chat_worker = ChatAgentWorker(
        config=config,
        messages=messages,
        project=self.project,
        busy_check=check_busy  # NEW
    )
```

### Fix 4: Initialize Missing Variables (Runtime Fix)

**Problem**: `_chat_history` and `_last_user_message` are referenced but never initialized.

**Solution**: Add initialization in `_setup_chat_panel()`:

```python
def _setup_chat_panel(self):
    """Initialize the chat panel dock."""
    from ui.chat_panel import ChatPanel

    # Initialize chat state
    self._chat_history: list[dict] = []  # FIX: Initialize history
    self._last_user_message: str = ""    # FIX: Initialize last message
    self._current_bubble = None          # FIX: Initialize bubble reference
    self._current_tool_indicator = None  # FIX: Initialize tool indicator
    self._chat_worker = None             # FIX: Initialize worker reference

    self.chat_panel = ChatPanel()
    # ... rest of method ...
```

### Fix 5: Include Tool Messages in History (Context Continuity)

**Problem**: Only user/assistant messages were added to history, losing tool execution context.

**Solution**: Include full conversation including tool calls/results:

```python
def _on_chat_complete(self, response: str, tool_history: list[dict]):
    """Handle chat completion with full history."""
    self._current_bubble.finish()
    self.chat_panel._set_streaming_state(False)

    # Add user message
    self._chat_history.append({"role": "user", "content": self._last_user_message})

    # Add tool interactions (from worker)
    for msg in tool_history:
        self._chat_history.append(msg)

    # Add final assistant response
    self._chat_history.append({"role": "assistant", "content": response})
```

Update ChatAgentWorker to emit tool history:

```python
class ChatAgentWorker(QThread):
    # Update signal to include tool history
    complete = Signal(str, list)  # final response text, tool_history

    async def _async_run(self):
        # ... existing code ...
        tool_history = []  # Track tool interactions

        while iteration < max_iterations and not self._stop_requested:
            # ... existing code ...

            if tool_calls:
                # Add assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls
                }
                tool_history.append(assistant_msg)

                for tc in tool_calls:
                    # ... execute tool ...
                    tool_msg = executor.format_for_llm(result)
                    tool_history.append(tool_msg)

        # Emit with history
        self.complete.emit(content, tool_history)
```

### Fix 6: CLI Tool Timeouts (Reliability)

**Problem**: CLI subprocess tools have no timeout, can hang indefinitely.

**Solution**: Add appropriate timeouts to all CLI tool calls:

```python
# Timeout values by operation type (in seconds)
TOOL_TIMEOUTS = {
    "detect_scenes": 600,      # 10 minutes for large videos
    "download_video": 1800,    # 30 minutes for long videos
    "search_youtube": 30,      # 30 seconds
    "analyze_colors": 300,     # 5 minutes
    "analyze_shots": 300,      # 5 minutes
    "transcribe": 1200,        # 20 minutes
    "export_clips": 600,       # 10 minutes
}

@tools.register(
    description="Detect scenes in a video file. Returns clip data.",
    requires_project=False,
    modifies_gui_state=False
)
def detect_scenes(video_path: str, sensitivity: float = 3.0) -> dict:
    """Run scene detection via CLI with timeout."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        result = subprocess.run(
            ["scene-ripper", "detect", video_path,
             "--sensitivity", str(sensitivity),
             "--output", output_path,
             "--format", "json"],
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUTS["detect_scenes"]  # ADD TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return {"error": "Scene detection timed out. The video may be too large."}
    finally:
        # Clean up temp file
        import os
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass

    if result.returncode != 0:
        return {"error": result.stderr}

    with open(output_path) as f:
        data = json.load(f)

    os.unlink(output_path)  # Clean up
    return data
```

---

## Phase 1: LLM Client Infrastructure

### 1.1 Multi-Provider Client

Create `core/llm_client.py`:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, AsyncIterator
import os

class ProviderType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"

@dataclass
class ProviderConfig:
    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def to_litellm_model(self) -> str:
        """Convert to LiteLLM model string."""
        prefixes = {
            ProviderType.OLLAMA: "ollama/",
            ProviderType.GEMINI: "gemini/",
            ProviderType.OPENROUTER: "openrouter/",
        }
        return f"{prefixes.get(self.provider, '')}{self.model}"

class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    async def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] = None
    ) -> AsyncIterator[dict]:
        """Stream chat completion with tool support."""
        from litellm import acompletion

        kwargs = {
            "model": self.config.to_litellm_model(),
            "messages": messages,
            "stream": True,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if tools:
            kwargs["tools"] = tools

        response = await acompletion(**kwargs)
        async for chunk in response:
            yield chunk
```

### 1.2 Default Provider Configurations

```python
# Recommended defaults in settings
DEFAULT_PROVIDERS = {
    "local": ProviderConfig(
        provider=ProviderType.OLLAMA,
        model="qwen3:8b",
        api_base="http://localhost:11434"
    ),
    "openai": ProviderConfig(
        provider=ProviderType.OPENAI,
        model="gpt-4o"
    ),
    "anthropic": ProviderConfig(
        provider=ProviderType.ANTHROPIC,
        model="claude-sonnet-4-20250514"
    ),
}
```

### 1.3 Settings Extension

Add to `core/settings.py`:

```python
@dataclass
class Settings:
    # ... existing fields ...

    # LLM Settings (API key stored in keyring, not here - see Fix 1)
    llm_provider: str = "local"  # local, openai, anthropic, gemini, openrouter
    llm_model: str = "qwen3:8b"
    llm_api_base: str = ""  # For local/custom endpoints
    llm_temperature: float = 0.7
    # NOTE: llm_api_key is NOT stored here - use keyring functions from Fix 1
```

**Important**: The `llm_api_key` is stored securely via system keyring (see Critical Fix 1). Use `_get_llm_api_key_from_keyring()` to retrieve it.

Environment variables to add:

| Variable | Purpose |
|----------|---------|
| `SCENE_RIPPER_LLM_PROVIDER` | Override LLM provider |
| `SCENE_RIPPER_LLM_MODEL` | Override model name |
| `SCENE_RIPPER_LLM_API_KEY` | API key for cloud providers |
| `SCENE_RIPPER_LLM_API_BASE` | Custom endpoint URL |

---

## Phase 2: Tool Definitions

### 2.1 Tool Registry

Create `core/chat_tools.py`:

```python
from dataclasses import dataclass
from typing import Callable, Any
import json
import inspect

@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable
    parameters: dict
    requires_project: bool = True  # Does this tool need active project?
    modifies_gui_state: bool = False  # Should use core func vs CLI?

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        description: str,
        requires_project: bool = True,
        modifies_gui_state: bool = False
    ):
        """Decorator to register a tool."""
        def decorator(func: Callable) -> Callable:
            schema = self._generate_schema(func)
            tool = ToolDefinition(
                name=func.__name__,
                description=description,
                func=func,
                parameters=schema,
                requires_project=requires_project,
                modifies_gui_state=modifies_gui_state
            )
            self._tools[tool.name] = tool
            return func
        return decorator

    def to_openai_format(self) -> list[dict]:
        """Convert all tools to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self._tools.values()
        ]
```

### 2.2 Tool Definitions

Tools split into two categories:

**GUI State Tools** (use core functions, trigger observer):

```python
tools = ToolRegistry()

@tools.register(
    description="Add clips to the timeline sequence. Clips appear in the Sequence tab.",
    requires_project=True,
    modifies_gui_state=True
)
def add_to_sequence(project: Project, clip_ids: list[str]) -> dict:
    """Add clips to sequence - updates GUI via observer."""
    valid_ids = [cid for cid in clip_ids if cid in project.clips_by_id]
    project.add_to_sequence(valid_ids)
    return {"added": valid_ids, "sequence_length": len(project.sequence)}

@tools.register(
    description="Get current project state including sources, clips, and sequence.",
    requires_project=True,
    modifies_gui_state=False
)
def get_project_state(project: Project) -> dict:
    """Get current project information."""
    return {
        "sources": [s.file_path.name for s in project.sources],
        "clip_count": len(project.clips),
        "sequence_length": len(project.sequence) if project.sequence else 0,
        "is_dirty": project.is_dirty
    }

@tools.register(
    description="Filter clips by criteria: shot_type, has_speech, color_palette, duration.",
    requires_project=True,
    modifies_gui_state=False
)
def filter_clips(
    project: Project,
    shot_type: str = None,
    has_speech: bool = None,
    min_duration: float = None,
    max_duration: float = None
) -> list[dict]:
    """Filter clips and return matching IDs with metadata."""
    results = []
    for clip in project.clips:
        # Apply filters
        if shot_type and getattr(clip, 'shot_type', None) != shot_type:
            continue
        if has_speech is not None:
            clip_has_speech = bool(getattr(clip, 'transcript', None))
            if clip_has_speech != has_speech:
                continue
        # Duration filtering
        duration = (clip.end_frame - clip.start_frame) / 30.0  # Assume 30fps
        if min_duration and duration < min_duration:
            continue
        if max_duration and duration > max_duration:
            continue

        results.append({
            "id": clip.id,
            "source": clip.source_id,
            "duration": duration,
            "shot_type": getattr(clip, 'shot_type', None),
            "has_speech": bool(getattr(clip, 'transcript', None))
        })
    return results
```

**CLI Tools** (subprocess execution, for batch/export operations):

```python
@tools.register(
    description="Detect scenes in a video file. Returns clip data.",
    requires_project=False,
    modifies_gui_state=False
)
def detect_scenes(video_path: str, sensitivity: float = 3.0) -> dict:
    """Run scene detection via CLI."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name

    result = subprocess.run(
        ["scene-ripper", "detect", video_path,
         "--sensitivity", str(sensitivity),
         "--output", output_path,
         "--format", "json"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {"error": result.stderr}

    with open(output_path) as f:
        return json.load(f)

@tools.register(
    description="Export clips from project to individual video files.",
    requires_project=True,
    modifies_gui_state=False
)
def export_clips(
    project_path: str,
    output_dir: str,
    clip_ids: list[str] = None
) -> dict:
    """Export clips via CLI."""
    cmd = ["scene-ripper", "export", "clips", project_path,
           "--output-dir", output_dir]
    if clip_ids:
        cmd.extend(["--clips", ",".join(clip_ids)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout}

@tools.register(
    description="Search YouTube for videos matching a query.",
    requires_project=False,
    modifies_gui_state=False
)
def search_youtube(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube via CLI."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name

    subprocess.run(
        ["scene-ripper", "youtube", "search", query,
         "--max-results", str(max_results),
         "--output", output_path],
        capture_output=True
    )

    with open(output_path) as f:
        return json.load(f)

@tools.register(
    description="Download a video from URL (YouTube, Vimeo, etc).",
    requires_project=False,
    modifies_gui_state=False
)
def download_video(url: str, output_dir: str = None) -> dict:
    """Download video via CLI."""
    cmd = ["scene-ripper", "youtube", "download", url]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout}

@tools.register(
    description="Run color palette extraction on project clips.",
    requires_project=True,
    modifies_gui_state=False
)
def analyze_colors(project_path: str) -> dict:
    """Run color analysis via CLI."""
    result = subprocess.run(
        ["scene-ripper", "analyze", "colors", project_path],
        capture_output=True,
        text=True
    )
    return {"success": result.returncode == 0, "output": result.stdout}

@tools.register(
    description="Run shot type classification (close-up, wide, etc) on project clips.",
    requires_project=True,
    modifies_gui_state=False
)
def analyze_shots(project_path: str) -> dict:
    """Run shot classification via CLI."""
    result = subprocess.run(
        ["scene-ripper", "analyze", "shots", project_path],
        capture_output=True,
        text=True
    )
    return {"success": result.returncode == 0, "output": result.stdout}

@tools.register(
    description="Transcribe speech in clips using Whisper.",
    requires_project=True,
    modifies_gui_state=False
)
def transcribe(project_path: str, model: str = "small.en") -> dict:
    """Run transcription via CLI."""
    result = subprocess.run(
        ["scene-ripper", "transcribe", project_path, "--model", model],
        capture_output=True,
        text=True
    )
    return {"success": result.returncode == 0, "output": result.stdout}
```

### 2.3 Tool Executor

Create `core/tool_executor.py`:

```python
from typing import Optional
import json
import logging

from core.chat_tools import ToolRegistry, ToolDefinition
from core.project import Project

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Execute tools safely with error handling."""

    def __init__(self, registry: ToolRegistry, project: Optional[Project] = None):
        self.registry = registry
        self.project = project

    def execute(self, tool_call: dict) -> dict:
        """Execute a tool call from the LLM."""
        name = tool_call.get("function", {}).get("name")
        args_str = tool_call.get("function", {}).get("arguments", "{}")
        tool_call_id = tool_call.get("id")

        # Parse arguments
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError as e:
            return self._error_result(tool_call_id, name, f"Invalid JSON: {e}")

        # Get tool definition
        tool = self.registry.get(name)
        if not tool:
            return self._error_result(tool_call_id, name, f"Unknown tool: {name}")

        # Check project requirement
        if tool.requires_project and not self.project:
            return self._error_result(
                tool_call_id, name,
                "This tool requires an active project. Please open or create a project first."
            )

        # Inject project if needed
        if tool.requires_project:
            args["project"] = self.project

        # Execute
        try:
            result = tool.func(**args)
            return {
                "tool_call_id": tool_call_id,
                "name": name,
                "success": True,
                "result": result
            }
        except TypeError as e:
            return self._error_result(tool_call_id, name, f"Invalid arguments: {e}")
        except Exception as e:
            logger.exception(f"Tool execution error: {name}")
            return self._error_result(tool_call_id, name, f"Execution failed: {e}")

    def _error_result(self, tool_call_id: str, name: str, error: str) -> dict:
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "success": False,
            "error": error
        }

    def format_for_llm(self, result: dict) -> dict:
        """Format tool result as message for LLM context."""
        if result["success"]:
            content = json.dumps(result["result"], indent=2)
        else:
            content = json.dumps({"error": result["error"]})

        return {
            "role": "tool",
            "tool_call_id": result["tool_call_id"],
            "name": result["name"],
            "content": content
        }
```

---

## Phase 3: Chat UI Components

### 3.1 Chat Panel Widget

Create `ui/chat_panel.py`:

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QTextEdit, QPushButton, QLabel, QComboBox, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot

class ChatPanel(QWidget):
    """Collapsible chat panel for agent interaction."""

    message_sent = Signal(str)  # User message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._response_finished_handled = False  # Signal guard

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with provider selector
        header = QHBoxLayout()
        header.addWidget(QLabel("Agent Chat"))
        header.addStretch()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Local (Ollama)", "OpenAI", "Anthropic"])
        header.addWidget(self.provider_combo)

        layout.addLayout(header)

        # Message history
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(8)
        self.scroll_area.setWidget(self.messages_widget)

        layout.addWidget(self.scroll_area, 1)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Describe what you want to create...")
        self.input_field.setMaximumHeight(80)
        input_layout.addWidget(self.input_field, 1)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send_clicked)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

    def _on_send_clicked(self):
        message = self.input_field.toPlainText().strip()
        if not message:
            return

        self.input_field.clear()
        self._add_user_message(message)
        self.message_sent.emit(message)

    def _add_user_message(self, text: str):
        bubble = MessageBubble(text, is_user=True)
        self.messages_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def add_assistant_message(self, text: str):
        """Add complete assistant message."""
        bubble = MessageBubble(text, is_user=False)
        self.messages_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def start_streaming_response(self) -> "StreamingBubble":
        """Start a streaming response bubble."""
        bubble = StreamingBubble()
        self.messages_layout.addWidget(bubble)
        self._scroll_to_bottom()
        return bubble

    def add_tool_indicator(self, tool_name: str, status: str = "running"):
        """Show tool execution status."""
        indicator = ToolIndicator(tool_name, status)
        self.messages_layout.addWidget(indicator)
        self._scroll_to_bottom()
        return indicator

    def _scroll_to_bottom(self):
        QTimer.singleShot(50, lambda:
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
        )

    @Slot(str)
    def on_stream_chunk(self, chunk: str):
        """Handle streaming chunk - with guard."""
        if self._current_bubble:
            self._current_bubble.append_text(chunk)

    @Slot(dict)
    def on_stream_complete(self, response: dict):
        """Handle stream completion - with guard."""
        if self._response_finished_handled:
            return
        self._response_finished_handled = True

        if self._current_bubble:
            self._current_bubble.finish()
        self.send_button.setEnabled(True)
```

### 3.2 Message Widgets

Create `ui/chat_widgets.py`:

```python
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt

class MessageBubble(QFrame):
    """Single message bubble."""

    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.setObjectName("userBubble" if is_user else "assistantBubble")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.MarkdownText)
        layout.addWidget(label)

        # Styling via object name in stylesheet

class StreamingBubble(QFrame):
    """Bubble that accumulates streaming text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantBubble")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.MarkdownText)
        layout.addWidget(self.label)

        self._text = ""

    def append_text(self, chunk: str):
        self._text += chunk
        self.label.setText(self._text)

    def finish(self):
        # Final render with markdown
        self.label.setText(self._text)

class ToolIndicator(QFrame):
    """Shows tool execution status."""

    def __init__(self, tool_name: str, status: str = "running", parent=None):
        super().__init__(parent)
        self.setObjectName("toolIndicator")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.status_icon = QLabel("⏳" if status == "running" else "✓")
        layout.addWidget(self.status_icon)

        self.name_label = QLabel(f"Running: {tool_name}")
        layout.addWidget(self.name_label)
        layout.addStretch()

    def set_complete(self, success: bool = True):
        self.status_icon.setText("✓" if success else "✗")
        self.name_label.setText(self.name_label.text().replace("Running:", "Completed:"))
```

### 3.3 Chat Worker Thread

Create `ui/chat_worker.py`:

```python
from PySide6.QtCore import QThread, Signal
import json
import asyncio
from typing import Optional

from core.llm_client import LLMClient, ProviderConfig
from core.tool_executor import ToolExecutor
from core.chat_tools import tools as tool_registry
from core.project import Project

class ChatAgentWorker(QThread):
    """Worker thread for LLM chat with tool execution."""

    # Signals
    text_chunk = Signal(str)
    tool_called = Signal(str, dict)  # tool_name, arguments
    tool_result = Signal(str, dict, bool)  # tool_name, result, success
    complete = Signal(str)  # final response text
    error = Signal(str)

    def __init__(
        self,
        config: ProviderConfig,
        messages: list[dict],
        project: Optional[Project] = None,
        parent=None
    ):
        super().__init__(parent)
        self.config = config
        self.messages = messages.copy()
        self.project = project
        self._stop_requested = False

    def run(self):
        """Run the agent loop."""
        asyncio.run(self._async_run())

    async def _async_run(self):
        client = LLMClient(self.config)
        executor = ToolExecutor(tool_registry, self.project)

        # Add system prompt with project context
        system_prompt = self._build_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + self.messages

        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations and not self._stop_requested:
            iteration += 1

            try:
                content, tool_calls = await self._stream_response(
                    client, full_messages
                )

                # Add assistant response to context
                full_messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls if tool_calls else None
                })

                # If no tool calls, we're done
                if not tool_calls:
                    self.complete.emit(content)
                    return

                # Execute tools
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])

                    self.tool_called.emit(name, args)
                    result = executor.execute(tc)
                    self.tool_result.emit(name, result, result["success"])

                    # Add to context
                    full_messages.append(executor.format_for_llm(result))

                # Continue loop for LLM to process results

            except Exception as e:
                self.error.emit(str(e))
                return

        if iteration >= max_iterations:
            self.error.emit("Maximum tool iterations reached")

    async def _stream_response(
        self,
        client: LLMClient,
        messages: list[dict]
    ) -> tuple[str, list[dict]]:
        """Stream a single LLM response."""
        content = ""
        tool_calls = []

        async for chunk in client.stream_chat(
            messages,
            tools=tool_registry.to_openai_format()
        ):
            if self._stop_requested:
                break

            delta = chunk.choices[0].delta

            if delta.content:
                content += delta.content
                self.text_chunk.emit(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    self._accumulate_tool_call(tool_calls, tc)

        return content, tool_calls

    def _accumulate_tool_call(self, buffer: list, delta_tc):
        """Accumulate partial tool call from streaming."""
        idx = delta_tc.index
        while len(buffer) <= idx:
            buffer.append({"id": None, "function": {"name": None, "arguments": ""}})

        if delta_tc.id:
            buffer[idx]["id"] = delta_tc.id
        if delta_tc.function:
            if delta_tc.function.name:
                buffer[idx]["function"]["name"] = delta_tc.function.name
            if delta_tc.function.arguments:
                buffer[idx]["function"]["arguments"] += delta_tc.function.arguments

    def _build_system_prompt(self) -> str:
        """Build system prompt with project context."""
        prompt = """You are an AI assistant for Scene Ripper, a video scene detection and editing tool.

You help users create video projects by:
- Detecting scenes in videos
- Analyzing clips (colors, shot types, transcription)
- Building sequences from clips
- Exporting clips and datasets

Available tools let you perform these operations. Always explain what you're doing before using tools.
"""

        if self.project:
            prompt += f"""

CURRENT PROJECT STATE:
- Sources: {len(self.project.sources)} video(s) loaded
- Clips: {len(self.project.clips)} detected clips
- Sequence: {len(self.project.sequence) if self.project.sequence else 0} clips in timeline
- Project path: {self.project.path or 'Unsaved'}

You can reference existing clips by their IDs and build on this project.
"""
        else:
            prompt += """

NO PROJECT LOADED - The user should open or create a project first, or you can help them start by detecting scenes in a video.
"""

        return prompt

    def stop(self):
        self._stop_requested = True
```

---

## Phase 4: MainWindow Integration

### 4.1 Add Chat Panel Toggle

Modify `ui/main_window.py`:

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... existing init ...

        self._setup_chat_panel()

    def _setup_chat_panel(self):
        """Initialize the chat panel dock."""
        from ui.chat_panel import ChatPanel

        self.chat_panel = ChatPanel()
        self.chat_panel.message_sent.connect(self._on_chat_message)

        # Create dock widget
        self.chat_dock = QDockWidget("Agent Chat", self)
        self.chat_dock.setWidget(self.chat_panel)
        self.chat_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)

        # Add toggle action to View menu
        self.chat_toggle_action = self.chat_dock.toggleViewAction()
        self.chat_toggle_action.setText("Show Agent Chat")
        self.view_menu.addAction(self.chat_toggle_action)

    def _on_chat_message(self, message: str):
        """Handle user message from chat panel."""
        from core.settings import load_settings
        from core.llm_client import ProviderConfig, ProviderType

        settings = load_settings()

        # Build provider config from settings
        config = ProviderConfig(
            provider=ProviderType(settings.llm_provider),
            model=settings.llm_model,
            api_key=settings.llm_api_key or None,
            api_base=settings.llm_api_base or None
        )

        # Build message history
        messages = self._chat_history + [{"role": "user", "content": message}]

        # Start worker
        self._chat_worker = ChatAgentWorker(
            config=config,
            messages=messages,
            project=self.project  # Pass live project instance
        )

        # Connect signals
        bubble = self.chat_panel.start_streaming_response()
        self._current_bubble = bubble

        self._chat_worker.text_chunk.connect(bubble.append_text)
        self._chat_worker.tool_called.connect(self._on_tool_called)
        self._chat_worker.tool_result.connect(self._on_tool_result)
        self._chat_worker.complete.connect(self._on_chat_complete)
        self._chat_worker.error.connect(self._on_chat_error)

        self._chat_worker.start()

    def _on_tool_called(self, name: str, args: dict):
        """Handle tool execution start."""
        self._current_tool_indicator = self.chat_panel.add_tool_indicator(name)

    def _on_tool_result(self, name: str, result: dict, success: bool):
        """Handle tool execution completion."""
        if self._current_tool_indicator:
            self._current_tool_indicator.set_complete(success)

    def _on_chat_complete(self, response: str):
        """Handle chat completion."""
        self._current_bubble.finish()
        self._chat_history.append({"role": "user", "content": self._last_user_message})
        self._chat_history.append({"role": "assistant", "content": response})

    def _on_chat_error(self, error: str):
        """Handle chat error."""
        self.chat_panel.add_assistant_message(f"Error: {error}")
```

---

## Phase 5: Error Handling & Recovery

### 5.0 Ollama Health Check (Important UX)

When "Local (Ollama)" is selected, check if Ollama is running before attempting to send messages:

```python
# In core/llm_client.py

import httpx

async def check_ollama_health(api_base: str = "http://localhost:11434") -> tuple[bool, str]:
    """Check if Ollama is running and accessible."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{api_base}/api/tags")
            if response.status_code == 200:
                return True, ""
            return False, f"Ollama returned status {response.status_code}"
    except httpx.ConnectError:
        return False, "Ollama is not running. Start Ollama with 'ollama serve' or switch to a cloud provider."
    except httpx.TimeoutException:
        return False, "Ollama connection timed out. It may be overloaded or not responding."
    except Exception as e:
        return False, f"Cannot connect to Ollama: {e}"
```

Use this in MainWindow before starting chat worker:

```python
async def _check_provider_health(self):
    """Check provider health before sending message."""
    settings = load_settings()
    if settings.llm_provider == "local":
        healthy, error = await check_ollama_health(settings.llm_api_base or "http://localhost:11434")
        if not healthy:
            self.chat_panel.add_assistant_message(f"**Provider Error**: {error}")
            self.chat_panel._set_streaming_state(False)
            return False
    return True
```

### 5.1 Retry Logic

```python
# In ChatAgentWorker
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30)
)
async def _call_with_retry(self, client, messages, tools):
    """Call LLM with automatic retry for transient errors."""
    return await client.stream_chat(messages, tools)
```

### 5.2 Provider Fallback

```python
class FallbackLLMClient:
    """Client with automatic fallback to secondary providers."""

    def __init__(self, primary: ProviderConfig, fallbacks: list[ProviderConfig]):
        self.primary = primary
        self.fallbacks = fallbacks

    async def stream_chat(self, messages, tools=None):
        providers = [self.primary] + self.fallbacks
        last_error = None

        for config in providers:
            try:
                client = LLMClient(config)
                async for chunk in client.stream_chat(messages, tools):
                    yield chunk
                return
            except Exception as e:
                last_error = e
                continue

        raise last_error
```

---

## Dependencies

Add to `requirements.txt`:

```
litellm>=1.40.0
tenacity>=8.2.0
```

---

## Success Criteria

### Functional Tests

1. **Basic Chat**: User can send message, receive streaming response
2. **Tool Execution**: Agent calls tools, results appear in chat
3. **GUI Updates**: When agent adds clips to sequence, UI updates immediately
4. **Project Context**: Agent knows about existing sources/clips and can build on them
5. **Multi-Provider**: Can switch between local Ollama and cloud providers
6. **Error Recovery**: Transient errors retry automatically, clear error messages for permanent failures

### Example Interactions

**Example 1: New Project**
```
User: "Find some old Soviet animation, download one video, detect scenes, and show me the close-ups"

Agent: I'll help you create a project from Soviet animation. Let me search YouTube first.
[Tool: search_youtube("Soviet animation 1980s")]
Found 10 results. I'll download "Wow! A Talking Fish" (1983) by Robert Saakiants.
[Tool: download_video("https://youtube.com/...")]
Downloaded successfully. Now detecting scenes...
[Tool: detect_scenes("/path/to/video.mp4")]
Detected 47 clips. Running shot classification...
[Tool: analyze_shots("project.json")]
Found 12 close-up shots. Adding them to your sequence.
[Tool: add_to_sequence(["clip-5", "clip-12", ...])]

Done! I've added 12 close-up shots to your sequence. You can see them in the Sequence tab.
```

**Example 2: Continue Existing Project**
```
User: "I have clips loaded. Find all the ones with speech and add them to the sequence"

Agent: I see you have 47 clips loaded. Let me find the ones with transcriptions.
[Tool: filter_clips(has_speech=True)]
Found 18 clips with detected speech. Adding to sequence...
[Tool: add_to_sequence(["clip-3", "clip-8", ...])]

Added 18 clips with speech to your sequence. The total sequence is now 23 clips.
```

---

## Implementation Order

**Pre-Implementation**: Apply all Critical Fixes before starting phases:
- Fix 1: Keyring storage for LLM API keys (in settings.py)
- Fix 4: Variable initialization pattern (MainWindow template)
- Fix 6: Define TOOL_TIMEOUTS constants

**Phase-by-Phase**:

1. **Phase 1**: LLM client infrastructure (core/llm_client.py)
   - Include Ollama health check (Fix 5.0)
   - Temperature parameter passthrough

2. **Phase 2**: Tool definitions and executor (core/chat_tools.py, core/tool_executor.py)
   - Include busy_check callback (Fix 3)
   - Include timeouts (Fix 6)
   - Include temp file cleanup

3. **Phase 3**: Chat UI components (ui/chat_panel.py, ui/chat_widgets.py)
   - Include cancel button and streaming state (Fix 2)
   - Include comprehensive signal guards

4. **Phase 4**: MainWindow integration + ChatAgentWorker
   - Include all variable initializations (Fix 4)
   - Include busy_check callback wiring (Fix 3)
   - Include tool history in signals (Fix 5)
   - Include cancellation handling (Fix 2)

5. **Phase 5**: Error handling, retry logic, provider fallback
   - Ollama health check integration
   - User-friendly error messages

---

## Spec Analysis Summary

A comprehensive spec-flow analysis identified **42 gaps** across 9 categories:

| Category | Critical | Important | Nice-to-Have |
|----------|----------|-----------|--------------|
| Message History | 2 | 2 | 1 |
| Tool Execution Safety | 2 | 2 | 1 |
| Chat Worker Thread | 2 | 1 | 1 |
| Security | 2 | 2 | 0 |
| State Synchronization | 1 | 3 | 0 |
| LLM Provider Config | 1 | 2 | 1 |
| Error Handling | 0 | 4 | 1 |
| UI Behavior | 0 | 2 | 4 |
| LLM Interaction | 0 | 3 | 2 |

All **10 critical issues** have been addressed in the "Critical Fixes" section above. The remaining important and nice-to-have items can be addressed during implementation or in follow-up iterations

---

## References

- [LiteLLM Documentation](https://github.com/berriai/litellm)
- [Qwen3 Tool Calling](https://github.com/QwenLM/Qwen)
- [Qt Asyncio](https://doc.qt.io/qtforpython-6/PySide6/QtAsyncio/index.html)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Agent-Native Architecture Plan](./agent-native-architecture-plan.md)
