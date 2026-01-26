---
title: "feat: Add LLM-Driven Planning Tool with Editable Plan UI"
type: feat
date: 2026-01-25
---

# feat: Add LLM-Driven Planning Tool with Editable Plan UI

## Overview

Add a planning capability to the agent chat that allows users to request complex multi-step workflows. The LLM detects "plan" requests, breaks down tasks into numbered steps, asks clarifying questions, and presents an editable plan widget. Users can modify steps before confirming execution.

**Example user request:**
> "Plan a video that downloads 100 films from YouTube on the subject of mushrooms. The final output should be random scenes totaling 20min."

**Agent response flow:**
1. LLM asks 2-3 clarifying questions (e.g., "What video quality?", "Should I filter by duration?")
2. User answers
3. LLM presents numbered plan steps in an editable widget
4. User can edit/reorder/delete steps
5. User confirms → LLM executes sequentially

## Problem Statement / Motivation

Currently, users must manually orchestrate complex workflows by sending multiple messages. For batch operations like "download 100 videos, detect scenes, and create a random cut," there's no way to:
- See the full plan before execution starts
- Modify the approach before committing
- Track progress across multiple steps

This creates friction for power users who want to automate complex video editing workflows.

## Proposed Solution

### Architecture

**LLM-driven approach** (not a state machine tool):
- System prompt guides LLM to detect "plan" requests
- LLM naturally breaks down tasks and asks clarifying questions in conversation
- A `present_plan` tool displays steps in an editable inline widget
- User confirms via button or chat message
- LLM executes tools sequentially, deciding specific tool calls at execution time

### Data Models

Add to `models/plan.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

@dataclass
class PlanStep:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""  # Human-readable step description
    status: str = "pending"  # pending, running, completed, failed, skipped
    error: Optional[str] = None
    result_summary: Optional[str] = None

@dataclass
class Plan:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    summary: str = ""  # Brief description of what plan accomplishes
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "draft"  # draft, confirmed, executing, completed, cancelled, failed
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
```

### New Tool: `present_plan`

Add to `core/chat_tools.py`:

```python
@tools.register(
    description="Display a plan to the user for review and editing. Call this after breaking down a complex request into steps. The user can edit, reorder, or delete steps before confirming.",
    requires_project=False,
    modifies_gui_state=True
)
def present_plan(main_window, steps: list[str], summary: str) -> dict:
    """
    Display an editable plan widget in the chat panel.

    Args:
        steps: List of human-readable step descriptions in execution order
        summary: Brief description of what the plan accomplishes

    Returns:
        Plan ID and instructions for the LLM to wait for confirmation
    """
```

### Plan Widget (Inline in Chat)

Add `PlanWidget` to `ui/chat_widgets.py`:

```
┌─────────────────────────────────────────────────────┐
│ 📋 Plan: Download mushroom videos and create edit   │
├─────────────────────────────────────────────────────┤
│ ☐ 1. Search YouTube for "mushroom documentary"  [≡] │
│ ☐ 2. Download top 100 results                   [≡] │
│ ☐ 3. Detect scenes in all downloaded videos     [≡] │
│ ☐ 4. Randomly select scenes totaling 20 min    [≡] │
│ ☐ 5. Add clips to sequence                      [≡] │
├─────────────────────────────────────────────────────┤
│        [Cancel]                    [✓ Confirm Plan] │
└─────────────────────────────────────────────────────┘
```

- Steps are editable (double-click to edit text)
- Drag handle `[≡]` for reordering
- Right-click or swipe to delete
- Status icons update during execution (☐ → ⏳ → ✓ or ✗)

### System Prompt Additions

Add to `ui/chat_worker.py` `_build_system_prompt()`:

```
PLANNING MODE:
When the user asks you to "plan" something or describes a complex multi-step workflow:

1. CLARIFY: Ask 2-3 focused questions to understand requirements:
   - What constraints matter? (quality, duration, count limits)
   - What's the desired outcome format?
   - Any preferences for how to handle edge cases?

2. BREAK DOWN: After getting answers, decompose into 3-10 clear steps.
   Each step should be a single logical action (search, download, detect, etc.)

3. PRESENT: Call present_plan with the steps and a summary.
   Wait for user confirmation before executing anything.

4. EXECUTE: After confirmation, execute steps sequentially.
   - Report progress after each step
   - If a step fails, log the error and continue with remaining steps
   - Provide a summary when complete

Example plan step descriptions (human-readable, not tool names):
- "Search YouTube for 'mushroom documentary' videos"
- "Download the top 100 search results"
- "Detect scenes in all downloaded videos"
- "Randomly select scenes totaling 20 minutes"
```

### Confirmation Detection

**Button click:** `PlanWidget` emits `confirmed` signal with current plan state (including edits)

**Chat message:** Detect phrases in `_should_confirm_plan()`:
- "confirm", "run it", "execute", "go ahead", "looks good", "start", "do it"
- NOT triggered by: "good question", "looks good so far" (partial matches)

### Execution Flow

1. User confirms → `plan_confirmed` signal emitted with edited steps
2. LLM receives context: "User confirmed plan. Execute steps in order."
3. LLM calls appropriate tools for each step
4. `PlanWidget` updates step status via `update_step_status(step_index, status)`
5. On failure: log error, mark step failed, continue
6. On completion: summary message with results

### Cancellation

**Before execution:** Cancel button dismisses widget, returns to normal chat

**During execution:**
- Cancel button sets `plan.status = "cancelled"`
- Current step completes (no mid-tool abort)
- Remaining steps skipped
- Summary shows what completed vs. skipped

## Technical Considerations

### Thread Safety
- `PlanWidget` lives on main thread
- `present_plan` tool uses `modifies_gui_state=True` pattern
- Step status updates via signals from worker thread

### State Synchronization
- Plan edits in widget must sync back to LLM context on confirm
- Use existing `gui_state` pattern to track current plan

### Conflict Detection
- Plan execution should respect existing `CONFLICTING_TOOLS` patterns
- Each step waits for previous to complete before starting

## Acceptance Criteria

- [ ] User can say "plan X" and LLM asks clarifying questions
- [ ] LLM calls `present_plan` with step list after getting answers
- [ ] Plan widget displays inline in chat with numbered steps
- [ ] User can edit step text by double-clicking
- [ ] User can reorder steps via drag-and-drop
- [ ] User can delete steps (right-click menu or delete key)
- [ ] Confirm button triggers sequential execution
- [ ] Cancel button before execution dismisses plan
- [ ] Step status indicators update during execution (pending → running → done/failed)
- [ ] Failed steps show error but execution continues
- [ ] Cancel during execution stops after current step
- [ ] Completion summary shows results per step

## Success Metrics

- Users can create and execute 5+ step plans without errors
- Plan widget renders correctly with 1-20 steps
- Edit operations (reorder, delete, text edit) work reliably
- Execution completes even when individual steps fail

## Dependencies & Risks

**Dependencies:**
- Existing chat tools infrastructure (`chat_tools.py`, `tool_executor.py`)
- Qt widget system for plan display
- LLM capability to follow system prompt guidance

**Risks:**
- LLM may not reliably detect "plan" intent → mitigate with clear system prompt examples
- Complex plans (20+ steps) may overwhelm UI → add scroll + step limit warning
- Long-running steps may timeout → leverage existing timeout handling

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `models/plan.py` | Create | Plan and PlanStep dataclasses | [x]
| `core/chat_tools.py` | Modify | Add `present_plan` tool | [x]
| `ui/chat_widgets.py` | Modify | Add `PlanWidget` class | [x]
| `ui/chat_panel.py` | Modify | Handle plan widget display/signals | [x]
| `ui/chat_worker.py` | Modify | Update system prompt, handle plan execution | [x]
| `core/gui_state.py` | Modify | Track current plan state | [x]

## References & Research

### Internal References
- Tool registration pattern: `core/chat_tools.py:171-262`
- GUI tool execution: `ui/main_window.py:1090-1097`
- System prompt: `ui/chat_worker.py:541-632`
- Existing widgets: `ui/chat_widgets.py` (MessageBubble, ToolIndicator patterns)

### Institutional Learnings
- Guard flags for signal handlers: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- State sync across threads: `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
- Single source of truth: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`

### External References
- Claude's planning approach: User provides request, agent breaks down into steps, asks clarifying questions, executes with progress updates
