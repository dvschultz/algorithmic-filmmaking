---
title: "feat: Delete Pending Imports in Intention Flow Modal"
type: feat
status: completed
date: 2026-03-18
---

# feat: Delete Pending Imports in Intention Flow Modal

Allow users to remove individual items from the pending imports list in the `IntentionImportDialog` by selecting item(s) and pressing Delete or Backspace.

## Acceptance Criteria

- [x] Selecting one or more items in `pending_list` and pressing Delete or Backspace removes them
- [x] Removing a local file item removes it from `_local_paths` and `_durations`
- [x] Removing a URL item removes the corresponding line from `url_input` QTextEdit (so it doesn't reappear on next `_update_pending_list()` call)
- [x] Cost estimates refresh after deletion
- [x] Start Import button disables when list becomes empty
- [x] Multi-select deletion works (select several items, delete all at once)
- [x] Tests cover: single file deletion, single URL deletion, multi-select, delete-all-then-empty-state

## Context

The `IntentionImportDialog` (`ui/dialogs/intention_import_dialog.py`) has a `QListWidget` at line 316 (`self.pending_list`) that shows pending imports. Items are either local files (from drag-drop/browse, stored in `self._local_paths`) or URLs (parsed live from `self.url_input` QTextEdit). Currently there is no way to remove items once added.

### Key Complication: URL Sync

URLs aren't stored independently — they're re-parsed from the `url_input` QTextEdit each time `_update_pending_list()` runs. Deleting a URL item from the list widget alone would cause it to reappear on the next refresh. The fix must also remove the corresponding line from `url_input`.

### Data stored per item

Each `QListWidgetItem` stores `Qt.UserRole` data:
- Files: `("file", Path)` — match against `_local_paths`
- URLs: `("url", str)` — match against line in `url_input`

## Implementation

### `ui/dialogs/intention_import_dialog.py`

1. **Override `keyPressEvent`** on the dialog (or install an event filter on `pending_list`) to catch `Qt.Key_Delete` and `Qt.Key_Backspace` when `pending_list` has focus and selected items.

2. **Add `_remove_selected_pending()` method:**

```python
def _remove_selected_pending(self):
    """Remove selected items from the pending imports list."""
    selected = self.pending_list.selectedItems()
    if not selected:
        return

    urls_to_remove = []
    for item in selected:
        data = item.data(Qt.UserRole)
        if not data:
            continue
        kind, value = data
        if kind == "file":
            if value in self._local_paths:
                self._local_paths.remove(value)
                self._durations.pop(str(value), None)
        elif kind == "url":
            urls_to_remove.append(value)

    # Remove URL lines from text input to prevent reappearance
    if urls_to_remove:
        lines = self.url_input.toPlainText().split("\n")
        remaining = [l for l in lines if l.strip() not in urls_to_remove]
        self.url_input.blockSignals(True)
        self.url_input.setPlainText("\n".join(remaining))
        self.url_input.blockSignals(False)

    self._update_pending_list()
```

3. **Wire up the key press** — either in `keyPressEvent` or via event filter on `pending_list`:

```python
# In _create_import_view, after creating pending_list:
self.pending_list.installEventFilter(self)

# Add eventFilter method:
def eventFilter(self, obj, event):
    if obj is self.pending_list and event.type() == event.Type.KeyPress:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._remove_selected_pending()
            return True
    return super().eventFilter(obj, event)
```

### Edge Cases

- **blockSignals on url_input**: Prevents `textChanged` → `_update_pending_list()` from firing mid-edit, avoiding double-refresh
- **Deleting all items**: `_update_pending_list()` already disables Start button when empty — no extra work
- **Duplicate URLs**: If user pasted the same URL twice, only exact line matches are removed
- **Running duration probe**: If a probe is in progress for a file being removed, the result is harmless — `_on_durations_probed` updates `_durations` dict but the file is already gone from `_local_paths`

### `tests/test_intention_import_delete.py`

```python
# Test cases:
# - test_delete_local_file_removes_from_pending
# - test_delete_url_removes_from_pending_and_text_input
# - test_delete_multiple_items
# - test_delete_all_disables_start_button
# - test_delete_refreshes_cost_estimate
```

## Sources

- Dialog implementation: `ui/dialogs/intention_import_dialog.py:207-928`
- Existing tests: `tests/test_intention_import_heuristic.py`
- Similar key handling patterns: `ui/clip_browser.py` (context menu), `ui/frame_browser.py` (selection)
