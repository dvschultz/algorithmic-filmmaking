---
title: Export Agent Chat History
type: feat
date: 2026-01-28
---

# Export Agent Chat History

## Overview

Add the ability to export agent chat conversations to files for later review and error analysis. Supports both Markdown (human-readable) and JSON (machine-parseable) formats with configurable content options.

## Problem Statement / Motivation

Users need to:
1. **Review conversations later** - Reference past agent interactions for learning or documentation
2. **Report issues** - Provide chat history when reporting bugs or unexpected agent behavior
3. **Share workflows** - Export successful workflows to share with others or as templates

Currently, chat history only exists in memory and is lost when the app closes or chat is cleared.

## Proposed Solution

Add an "Export Chat" button to the chat panel header that opens an export dialog with format and content options. When exporting both formats, user selects a folder and files are auto-generated with timestamped names.

### User Flow

```
User clicks "Export Chat" button (next to Clear)
         │
         ▼
┌─────────────────────────────┐
│   Export Chat Dialog        │
├─────────────────────────────┤
│ Format:                     │
│ ○ Markdown (.md)            │
│ ○ JSON (.json)              │
│ ● Both formats              │
├─────────────────────────────┤
│ Include:                    │
│ ☑ User messages             │
│ ☑ Assistant responses       │
│ ☑ Tool calls & results      │
│ ☐ Tool arguments (verbose)  │
├─────────────────────────────┤
│        [Cancel] [Export]    │
└─────────────────────────────┘
         │
         ▼ (User clicks Export)
┌─────────────────────────────┐
│ Select Export Folder        │
│ (QFileDialog directory)     │
└─────────────────────────────┘
         │
         ▼
Files created:
  chat_2026-01-28_143022.md
  chat_2026-01-28_143022.json
         │
         ▼
Success toast + "Open Folder" option
```

## Technical Approach

### Architecture

```
ui/chat_panel.py
    │
    ├── _export_button (QPushButton)
    │       │
    │       └── clicked → _on_export_clicked()
    │                          │
    │                          ▼
    │                   ExportChatDialog
    │                          │
    │                          ▼
    │                   QFileDialog.getExistingDirectory()
    │                          │
    │                          ▼
    └──────────────────► core/chat_export.py
                              │
                              ├── export_chat_as_markdown()
                              └── export_chat_as_json()
```

### Files to Create/Modify

| File | Changes |
|------|---------|
| `core/chat_export.py` | **NEW** - Export logic with `ChatExportConfig`, `export_chat_as_markdown()`, `export_chat_as_json()` |
| `ui/chat_panel.py` | Add Export button, `_on_export_clicked()` handler |
| `ui/export_chat_dialog.py` | **NEW** - Export options dialog |
| `ui/main_window.py` | Pass chat history to export functions |

### Data Structures

**ChatExportConfig:**
```python
@dataclass
class ChatExportConfig:
    output_dir: Path
    format: str  # "markdown", "json", "both"
    include_user: bool = True
    include_assistant: bool = True
    include_tools: bool = True
    include_tool_args: bool = False  # Verbose mode
    project_name: str = ""
```

**Message Structure (existing in `_chat_history`):**
```python
# User message
{"role": "user", "content": "..."}

# Assistant message
{"role": "assistant", "content": "..."}

# Assistant with tool call
{"role": "assistant", "content": "", "tool_calls": [
    {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
]}

# Tool result
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

### Export Formats

**Markdown Format:**
```markdown
# Agent Chat Export

**Exported:** 2026-01-28 14:30:22
**Project:** My Video Project
**Messages:** 15

---

## User

Detect scenes in my video

---

## Assistant

I'll analyze your video for scenes.

> **Tool:** detect_scenes
> **Status:** Success

---

## Assistant

I detected 12 scenes in your video. Here's what I found...

---
```

**JSON Format:**
```json
{
  "version": "1.0",
  "exported_at": "2026-01-28T14:30:22",
  "project": {
    "name": "My Video Project",
    "path": "/path/to/project.sceneripper"
  },
  "message_count": 15,
  "messages": [
    {
      "role": "user",
      "content": "Detect scenes in my video"
    },
    {
      "role": "assistant",
      "content": "I'll analyze your video for scenes.",
      "tool_calls": [...]
    }
  ]
}
```

### Implementation Phases

#### Phase 1: Core Export Logic

Create `core/chat_export.py`:

- [x] Create `ChatExportConfig` dataclass
- [x] Implement `export_chat_as_markdown(messages, config) -> bool`
- [x] Implement `export_chat_as_json(messages, config) -> bool`
- [x] Implement `_format_tool_call_markdown(tool_call, include_args) -> str`
- [x] Implement `_sanitize_path_in_content(content) -> str` (replace home dir with `~`)
- [x] Add `generate_export_filename(format: str) -> str` helper

#### Phase 2: Export Dialog

Create `ui/export_chat_dialog.py`:

- [x] Create `ExportChatDialog(QDialog)` class
- [x] Add format radio buttons (Markdown / JSON / Both)
- [x] Add content checkboxes (user, assistant, tools, tool args)
- [x] Add preview count label ("X messages will be exported")
- [x] Implement `get_config() -> ChatExportConfig` method
- [x] Style dialog to match app theme

#### Phase 3: Chat Panel Integration

Modify `ui/chat_panel.py`:

- [x] Add "Export" button next to "Clear" button in header
- [x] Implement `_on_export_clicked()` slot
- [x] Disable Export button when chat is empty
- [x] Disable Export button during streaming
- [x] Add tooltip explaining disabled state

#### Phase 4: MainWindow Wiring

Modify `ui/main_window.py`:

- [x] Add `export_chat_requested` signal handling
- [x] Pass `_chat_history` to export dialog
- [x] Show folder selection dialog after options confirmed
- [x] Call export functions with config
- [x] Show success/error feedback
- [x] Offer "Open Folder" after successful export

## Acceptance Criteria

### Functional Requirements

- [x] Export button visible next to Clear button in chat panel header
- [x] Export button disabled when chat is empty (with tooltip)
- [x] Export button disabled during streaming
- [x] Export dialog shows format options (Markdown, JSON, Both)
- [x] Export dialog shows content checkboxes
- [x] Selecting "Both" exports two files to chosen folder
- [x] Exported Markdown is human-readable with clear formatting
- [x] Exported JSON is valid and parseable
- [x] User home directory paths replaced with `~` in exports
- [x] Success toast shown after export with "Open Folder" option
- [x] Error handling for write failures (permissions, disk full)

### Non-Functional Requirements

- [x] Export completes in < 1 second for typical conversations (< 100 messages)
- [x] Files use UTF-8 encoding
- [x] JSON is pretty-printed with 2-space indent
- [x] Filenames include timestamp to prevent overwrites

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Empty chat | Export button disabled, tooltip: "No messages to export" |
| Streaming in progress | Export button disabled, tooltip: "Wait for response to complete" |
| Write permission denied | Error dialog with specific message |
| Very long conversation (1000+ messages) | Export proceeds normally (no truncation) |
| Tool call with no result yet | Skip tool result, include call only |
| Cancelled response in history | Include with "(Cancelled)" indicator |

## Privacy Considerations

Content that gets sanitized:
- User home directory paths → replaced with `~`
- Absolute project paths → replaced with relative or `~` prefix

Content that is NOT filtered (user responsibility):
- Video URLs (may contain identifying info)
- Custom prompts
- Error messages (may contain paths)

## Dependencies & Prerequisites

- Existing `_chat_history` list in MainWindow
- Existing `QFileDialog` patterns in codebase
- Existing export patterns in `core/dataset_export.py`

## Success Metrics

- Users can successfully export chat to reviewable files
- Exported JSON can be loaded back for analysis
- No data loss in export (all selected content preserved)

## References

### Internal References

- Chat history storage: `ui/main_window.py:1252-1256`
- Existing export patterns: `core/dataset_export.py`, `core/edl_export.py`
- File dialog patterns: `ui/main_window.py:3961-3972`
- Chat panel header: `ui/chat_panel.py`

### Institutional Learnings

- Path sanitization security: `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`
- Export config pattern: `docs/plans/archive/2026-01-24-feat-edl-export-plan.md`
