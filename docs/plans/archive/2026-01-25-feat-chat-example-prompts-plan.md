---
title: "feat: Add Example Prompts to Chat Agent"
type: feat
date: 2026-01-25
---

# Add Example Prompts to Chat Agent

Add clickable example prompts that appear in the chat area when empty. These help users discover what the chat agent can do and provide a starting point for interaction.

## Overview

When the chat panel has no messages, display 3-4 example prompts as clickable chips/buttons. Clicking a prompt fills the input box with that text (editable before sending). Once any message is sent, hide the example prompts permanently for that session.

## Acceptance Criteria

- [x] Example prompts appear in the chat area when message history is empty
- [x] Clicking a prompt fills the input text box with that prompt's text
- [x] User can edit the prompt text before sending
- [x] All example prompts hide after the first message is sent
- [x] Prompts reappear when `clear_messages()` is called (reset to empty state)
- [x] Prompts are disabled/hidden during streaming state
- [x] Styling matches existing chat panel aesthetics (colors, border-radius)
- [x] Prompts are responsive and wrap appropriately

## Edge Cases & Behavior

| Scenario | Behavior |
|----------|----------|
| Click prompt with text already in input | Replace existing text entirely |
| Click prompt during streaming | Prompts disabled while streaming |
| Call `clear_messages()` | Prompts reappear (reset to empty state) |
| Rapid multiple clicks | Guard pattern prevents duplicate signals |
| No project loaded | Show all prompts anyway (agent handles gracefully) |
| Input focus after click | Focus input field, cursor at end of text |

**Source of truth for "empty state":** Visual state (`messages_layout.count() == 1`, accounting for the prompts widget itself)

## Example Prompts

Based on the chat agent's tool capabilities:

| Prompt Text | Showcases |
|-------------|-----------|
| "Show me all clips in this project" | `list_clips` - basic query |
| "Find close-up shots with speech" | `filter_clips` - filtering |
| "Analyze colors in the first 5 clips" | `analyze_colors` - analysis |
| "Add all wide shots to the sequence" | `filter_clips` + `add_to_sequence` - workflow |

## Technical Approach

### New Widget: ExamplePromptsWidget

Create a new widget in `ui/chat_widgets.py`:

```python
class ExamplePromptsWidget(QWidget):
    """Displays clickable example prompts when chat is empty."""

    prompt_clicked = Signal(str)  # Emits the prompt text when clicked
```

**Layout:**
- Centered container with header text ("Try asking...")
- FlowLayout or QGridLayout with 2 columns for prompt buttons
- Each prompt is a styled QPushButton

**Styling:**
- Match existing chat styling:
  - Background: `#f0f2f5` (light gray, like ToolIndicator)
  - Border: `1px solid #d0d0d0`
  - Border-radius: `12px` (match message bubbles)
  - Text: `#333333`
  - Hover: Slight background darkening

### Integration in ChatPanel

**chat_panel.py modifications:**

1. Add `ExamplePromptsWidget` as first item in `messages_layout`
2. Track visibility: `self._example_prompts_visible = True`
3. Connect `prompt_clicked` signal to populate input field:
   ```python
   self.example_prompts.prompt_clicked.connect(self._on_example_prompt_clicked)
   ```
4. Hide prompts in `_add_user_message()`:
   ```python
   if self._example_prompts_visible:
       self.example_prompts.hide()
       self._example_prompts_visible = False
   ```
5. Disable prompts during streaming in `_set_streaming_state()`:
   ```python
   self.example_prompts.setEnabled(not streaming)
   ```
6. Restore prompts in `clear_messages()`:
   ```python
   self.example_prompts.show()
   self._example_prompts_visible = True
   self.example_prompts.reset_guard()  # Reset click guard
   ```

### Signal Flow

```
User clicks prompt button
    → ExamplePromptsWidget.prompt_clicked(str) emitted
    → ChatPanel._on_example_prompt_clicked() receives text
    → self.input_field.setText(text)
    → self.input_field.setFocus()
```

### Guard Pattern (from learnings)

Apply the duplicate signal guard pattern:

```python
# In ExamplePromptsWidget
self._click_handled = False

def _on_button_clicked(self, prompt_text: str):
    if self._click_handled:
        return
    self._click_handled = True
    self.prompt_clicked.emit(prompt_text)
```

Reset the guard when showing prompts again (new session/clear chat).

## Files to Modify

| File | Changes |
|------|---------|
| `ui/chat_widgets.py` | Add `ExamplePromptsWidget` class |
| `ui/chat_panel.py` | Integrate widget, handle signals, manage visibility |

## Testing

**Happy path:**
1. Start app with no chat history → prompts should appear centered in chat area
2. Click a prompt → input field should fill with that text, focused, cursor at end
3. Edit the text → should be editable
4. Send message → prompts should disappear

**Edge cases:**
5. Type text in input, then click prompt → input replaced with prompt text
6. Click prompt during streaming response → click should be ignored (disabled)
7. Call `clear_messages()` → prompts should reappear
8. Rapidly click multiple prompts → only one should register
9. Resize chat panel → prompts should wrap appropriately

## References

- Existing styling patterns: `ui/chat_panel.py:89-120` (bubble colors, borders)
- Signal patterns: `ui/chat_widgets.py:1-50` (MessageBubble signals)
- Guard pattern: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
