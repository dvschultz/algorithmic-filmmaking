--- 
name: gui-aware-agent-context-sync
description: |
  Synchronizes GUI selection state with agent context in agent-native applications. Use when: (1) the chat agent reports "no items selected" despite visual selections in the UI, (2) the agent is unaware of the active tab or current focus, (3) building applications where the agent needs to act on the user's current visual context. Uses a central state object and signal/slot mechanism to propagate UI changes to the agent's system prompt.
author: Claude Code
version: 1.0.0
date: 2026-01-25
---

# GUI-Aware Agent Context Synchronization

## Problem
In agent-native GUI applications (like Scene Ripper), the chat agent often lives in a separate thread or context from the main UI. Without explicit synchronization, the agent cannot "see" what the user has selected or which tab they are viewing. This leads to user frustration when they ask the agent to "analyze these selected clips" and the agent responds that nothing is selected.

## Context / Trigger Conditions
- User performs a selection in the GUI and asks the agent to act on it
- Agent response indicates it lacks context about the current UI state
- UI components (tabs, grids, lists) have their own internal state not shared with the agent's system prompt

## Solution
1. **Define a central State Object**: Create a dataclass (e.g., `GUIState`) that tracks relevant UI properties (active tab, selected IDs, active source).
2. **Implement Context Serialization**: Add a method to the state object (e.g., `to_context_string()`) that generates a human-readable summary for the agent's system prompt.
3. **Connect UI Signals**: Use the framework's signal/slot mechanism (e.g., Qt Signals) to update the central state object whenever selection or navigation occurs.
4. **Inject into Prompt**: Ensure the latest state context is included in the system prompt sent to the LLM for every user message.

## Verification
1. Select items in the UI.
2. Ask the agent: "Which items do I have selected?" or "What am I looking at right now?"
3. The agent should correctly report the count and specific IDs of selected items.

## Example
**GUIState Dataclass:**
```python
@dataclass
class GUIState:
    active_tab: str = "main"
    selected_ids: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        return f"ACTIVE TAB: {self.active_tab}\nSELECTED: {len(self.selected_ids)} items"
```

**Connecting Signal in MainWindow:**
```python
# In MainWindow._connect_signals
self.list_view.selection_changed.connect(self._on_selection_changed)

def _on_selection_changed(self, ids):
    self._gui_state.selected_ids = ids
```

## Notes
- Be careful with large selections; truncate the list of specific IDs in the context string to avoid token bloat.
- Ensure state updates are thread-safe if the agent context is read from a different thread.
- This pattern creates a "bidirectional" feel where the agent is truly native to the application.

## References
- [Agent-User Interaction Protocol (AG-UI) 2026 Trends](https://thesys.dev)
- [CopilotKit Real-time Sync Patterns](https://copilotkit.ai)
