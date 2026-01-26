# Agent-GUI Bidirectional Sync Plan

**Date**: 2026-01-25
**Status**: Implemented
**Builds On**: Agent Chatbot Implementation

---

## Overview

Currently, the chat agent and GUI operate independently. When the agent searches YouTube, downloads videos, or performs operations, the GUI doesn't update to reflect those changes (except for Project-modifying operations that use the observer pattern). This plan establishes full bidirectional synchronization so:

1. **Agent → GUI**: Agent operations update corresponding GUI components
2. **GUI → Agent context**: GUI state changes are available to the agent

### Problem Statement

User reported: "When I run a search, the search field in the Collect tab does not show the searched films."

This affects:
- YouTube search results don't appear in Collect tab
- Downloaded videos don't immediately appear in library
- Agent has no awareness of GUI state (current tab, selections, etc.)

---

## Architecture Analysis

### Current State

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MainWindow                                    │
│                                                                       │
│  ┌─────────────────────┐          ┌─────────────────────────────┐   │
│  │    ChatPanel        │          │     CollectTab               │   │
│  │    ↓                │          │     ↓                        │   │
│  │  ChatAgentWorker    │          │   YouTubeSearchPanel         │   │
│  │    ↓                │          │     ↓                        │   │
│  │  search_youtube()   │ (NO LINK)│   YouTubeSearchWorker        │   │
│  │  download_video()   │──────────│   DownloadWorker             │   │
│  └─────────────────────┘          └─────────────────────────────┘   │
│                                                                       │
│  Project ←── ProjectSignalAdapter ←── Observer (WORKS for clips/seq) │
└─────────────────────────────────────────────────────────────────────┘
```

### Target State

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MainWindow                                    │
│                                                                       │
│  ┌─────────────────────┐          ┌─────────────────────────────┐   │
│  │    ChatPanel        │          │     CollectTab               │   │
│  │    ↓                │   SYNC   │     ↓                        │   │
│  │  ChatAgentWorker    │←────────→│   YouTubeSearchPanel         │   │
│  │    ↓                │  SIGNALS │     ↓                        │   │
│  │  search_youtube()   │──────────│   display_results()          │   │
│  │  download_video()   │──────────│   add_source()               │   │
│  └─────────────────────┘          └─────────────────────────────┘   │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                     GUIStateManager                            │   │
│  │  - Current search results                                      │   │
│  │  - Download queue status                                       │   │
│  │  - Active tab / selection                                      │   │
│  │  - Exposes state for agent system prompt                       │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Operations to Sync

### Priority 1: Agent → GUI (User's Issue)

| Agent Tool | GUI Component | Sync Action |
|------------|---------------|-------------|
| `search_youtube` | YouTubeSearchPanel | Display results, update search input |
| `download_video` | CollectTab.source_browser | Add downloaded source to library |
| `detect_scenes` | Project (already works) | Clips appear via observer |
| `add_to_sequence` | SequenceTab (already works) | Timeline updates via observer |

### Priority 2: GUI → Agent Context

| GUI Action | Agent Context |
|------------|---------------|
| YouTube search results | Include in system prompt |
| Current tab selection | Contextual awareness |
| Selected clips | Reference in conversations |
| Active source | Targeted operations |

---

## Implementation

### Phase 1: Agent → GUI Sync for YouTube Search

**Goal**: When agent calls `search_youtube`, results appear in Collect tab's YouTube panel.

#### 1.1 Add YouTube sync signal to ChatAgentWorker

```python
# ui/chat_worker.py

class ChatAgentWorker(QThread):
    # Existing signals...

    # NEW: GUI sync signals
    youtube_search_completed = Signal(str, list)  # query, list of video dicts
    video_download_completed = Signal(str, dict)  # url, download result
```

#### 1.2 Modify tool execution to emit sync signals

In `_async_run`, after `search_youtube` tool executes:

```python
# After executing search_youtube
if name == "search_youtube" and result.get("success"):
    videos = result.get("data", {}).get("results", [])
    query = result.get("data", {}).get("query", "")
    self.youtube_search_completed.emit(query, videos)
```

#### 1.3 Connect signal in MainWindow

```python
# ui/main_window.py

def _on_chat_message(self, message: str):
    # ... existing setup ...

    # Connect sync signals
    self._chat_worker.youtube_search_completed.connect(
        self._on_agent_youtube_search
    )
    self._chat_worker.video_download_completed.connect(
        self._on_agent_video_downloaded
    )
```

#### 1.4 Implement sync handlers

```python
# ui/main_window.py

def _on_agent_youtube_search(self, query: str, videos: list[dict]):
    """Sync agent YouTube search to GUI."""
    # Convert dicts to YouTubeVideo objects
    from core.youtube_api import YouTubeVideo

    video_objects = []
    for v in videos:
        video_objects.append(YouTubeVideo(
            video_id=v.get("video_id", ""),
            title=v.get("title", ""),
            channel_title=v.get("channel", ""),
            duration_str=v.get("duration", ""),
            thumbnail_url=v.get("thumbnail", ""),
            view_count=v.get("view_count"),
        ))

    # Update YouTube panel
    self.collect_tab.youtube_search_panel.search_input.setText(query)
    self.collect_tab.youtube_search_panel.display_results(video_objects)

    # Expand panel if collapsed
    if not self.collect_tab.youtube_search_panel._expanded:
        self.collect_tab.youtube_search_panel.toggle_btn.click()

    # Switch to Collect tab
    self._switch_to_tab("collect")
```

### Phase 2: Agent → GUI Sync for Downloads

**Goal**: When agent downloads a video, it appears in the library.

#### 2.1 Modify download_video tool to emit signal

The download already works, but we need to add the source to the GUI. Since `download_video` doesn't have `modifies_gui_state=True`, we need to either:

**Option A**: Mark it as GUI-modifying and execute on main thread
**Option B**: Emit a sync signal after execution (less intrusive)

Choose **Option B** for consistency with YouTube search:

```python
# In ChatAgentWorker._async_run after download_video
if name == "download_video" and result.get("success"):
    self.video_download_completed.emit(
        args.get("url", ""),
        result.get("data", {})
    )
```

#### 2.2 Handle download completion in MainWindow

```python
def _on_agent_video_downloaded(self, url: str, result: dict):
    """Sync agent video download to GUI."""
    file_path = result.get("file_path")
    if not file_path:
        return

    from pathlib import Path
    path = Path(file_path)

    if path.exists():
        # Use existing video add logic
        self._add_video_to_library(path)
```

### Phase 3: GUI State Manager (Agent Context)

**Goal**: Agent knows about GUI state for better context.

#### 3.1 Create GUIStateManager

```python
# core/gui_state.py

from dataclasses import dataclass, field
from typing import Optional
from core.youtube_api import YouTubeVideo

@dataclass
class GUIState:
    """Current GUI state for agent context."""

    # YouTube search
    last_search_query: str = ""
    search_results: list[dict] = field(default_factory=list)
    selected_videos: list[str] = field(default_factory=list)

    # Tab state
    active_tab: str = "collect"  # collect, cut, arrange

    # Selection state
    selected_clip_ids: list[str] = field(default_factory=list)
    selected_source_id: Optional[str] = None

    def to_context_string(self) -> str:
        """Generate context string for agent system prompt."""
        lines = []

        if self.search_results:
            lines.append(f"RECENT YOUTUBE SEARCH: '{self.last_search_query}'")
            lines.append(f"  Found {len(self.search_results)} videos")
            for v in self.search_results[:3]:
                lines.append(f"  - {v.get('title', 'Unknown')} ({v.get('duration', '?')})")
            if len(self.search_results) > 3:
                lines.append(f"  - ...and {len(self.search_results) - 3} more")

        if self.selected_videos:
            lines.append(f"SELECTED FOR DOWNLOAD: {len(self.selected_videos)} videos")

        lines.append(f"ACTIVE TAB: {self.active_tab}")

        if self.selected_clip_ids:
            lines.append(f"SELECTED CLIPS: {len(self.selected_clip_ids)}")

        return "\n".join(lines) if lines else ""
```

#### 3.2 Track GUI state in MainWindow

```python
# ui/main_window.py

def __init__(self):
    # ... existing init ...
    self._gui_state = GUIState()

def _on_youtube_search_finished(self, result: YouTubeSearchResult):
    """Handle GUI YouTube search completion."""
    # Existing code...

    # Update GUI state
    self._gui_state.last_search_query = self.collect_tab.youtube_search_panel.get_search_query()
    self._gui_state.search_results = [
        {
            "video_id": v.video_id,
            "title": v.title,
            "duration": v.duration_str,
        }
        for v in result.videos
    ]

def _on_tab_changed(self, index: int):
    """Track active tab."""
    tab_names = ["collect", "cut", "arrange"]
    if 0 <= index < len(tab_names):
        self._gui_state.active_tab = tab_names[index]
```

#### 3.3 Include GUI state in agent context

```python
# ui/chat_worker.py

class ChatAgentWorker(QThread):
    def __init__(
        self,
        config: ProviderConfig,
        messages: list[dict],
        project: Optional[Any] = None,
        busy_check: Optional[Callable[[str], bool]] = None,
        gui_state: Optional[dict] = None,  # NEW
        parent=None
    ):
        # ... existing init ...
        self.gui_state = gui_state  # Dict representation of GUIState

    def _build_system_prompt(self) -> str:
        prompt = # ... existing prompt ...

        # Add GUI state context
        if self.gui_state:
            gui_context = self.gui_state.get("context_string", "")
            if gui_context:
                prompt += f"""

CURRENT GUI STATE:
{gui_context}
"""

        return prompt
```

#### 3.4 Pass GUI state when creating worker

```python
# ui/main_window.py

def _on_chat_message(self, message: str):
    # ... existing code ...

    self._chat_worker = ChatAgentWorker(
        config=config,
        messages=messages,
        project=self.project,
        busy_check=check_busy,
        gui_state={
            "context_string": self._gui_state.to_context_string()
        }
    )
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `ui/chat_worker.py` | Add sync signals, emit after tool execution, accept gui_state |
| `ui/main_window.py` | Connect sync signals, implement handlers, track GUI state |
| `core/gui_state.py` | NEW: GUIState dataclass |
| `core/chat_tools.py` | Return full data from tools (already does) |

---

## Signal Flow Diagram

```
Agent calls search_youtube()
         ↓
ToolExecutor.execute() returns result
         ↓
ChatAgentWorker emits youtube_search_completed(query, videos)
         ↓
MainWindow._on_agent_youtube_search()
         ↓
┌────────────────────────────────────────┐
│ 1. Update search_input.setText(query)  │
│ 2. Convert dicts → YouTubeVideo        │
│ 3. Call display_results(videos)        │
│ 4. Expand panel if collapsed           │
│ 5. Switch to Collect tab               │
│ 6. Update _gui_state                   │
└────────────────────────────────────────┘
```

---

## Success Criteria

### Functional Tests

1. **YouTube Search Sync**: Agent `search_youtube` → results appear in Collect tab
2. **Download Sync**: Agent `download_video` → video appears in library
3. **Context Awareness**: Agent mentions recent search when user asks "download those videos"
4. **Tab Awareness**: Agent knows which tab is active

### Example Interaction

```
User: "Search YouTube for drone footage mountains"

Agent: I'll search YouTube for you.
[Tool: search_youtube("drone footage mountains")]
Found 10 videos:
- "Epic Mountain Drone Footage 4K" (5:23)
- "Swiss Alps Aerial Views" (12:45)
...

[GUI automatically shows results in Collect tab]

User: "Download the first two"

Agent: I can see you want to download the first two results from your search.
Let me download those.
[Tool: download_video("https://youtube.com/...")]
[Tool: download_video("https://youtube.com/...")]

[Videos appear in library automatically]
```

---

## Implementation Order

1. **Phase 1**: YouTube search sync (addresses user's reported issue) ✅
   - [x] Add signal to ChatAgentWorker
   - [x] Connect in MainWindow
   - [x] Implement sync handler

2. **Phase 2**: Download sync ✅
   - [x] Add signal for download completion
   - [x] Add downloaded video to library

3. **Phase 3**: GUI state context ✅
   - [x] Create GUIState dataclass
   - [x] Track state changes
   - [x] Include in agent system prompt

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Signal threading issues | Use Qt's signal/slot mechanism (thread-safe) |
| Stale GUI state in agent | Pass state at message start, not during |
| UI flicker on sync | Batch updates, don't force tab switch for every operation |
| Performance with large results | Limit displayed results (already capped at 50) |

---

## Future Enhancements

- Bidirectional clip selection (agent selects → GUI highlights)
- Agent-driven tab navigation commands
- Undo/redo integration for agent operations
- Voice command integration
