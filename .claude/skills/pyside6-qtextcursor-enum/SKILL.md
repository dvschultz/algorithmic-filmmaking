---
name: pyside6-qtextcursor-enum
description: |
  Fix AttributeError when using QTextCursor movement operations in PySide6. Use when:
  (1) "'QTextCursor' object has no attribute 'End'" error, (2) cursor.movePosition
  fails with enum access, (3) migrating PyQt5 code to PySide6 involving text cursors.
author: Claude Code
version: 1.0.0
date: 2026-01-25
---

# PySide6 QTextCursor Enum Access

## Problem

When calling `cursor.movePosition(cursor.End)` in PySide6, you get:
```
AttributeError: 'PySide6.QtGui.QTextCursor' object has no attribute 'End'
```

## Context / Trigger Conditions

- Working with QTextEdit or QPlainTextEdit cursor positioning
- Code that worked in PyQt5 but fails in PySide6
- Any `cursor.movePosition()` call using cursor instance enum access

## Solution

In PySide6, movement operations must be accessed via the class enum, not the instance.

### Wrong (PyQt5 style)
```python
cursor = text_edit.textCursor()
cursor.movePosition(cursor.End)  # AttributeError!
```

### Correct (PySide6 style)
```python
from PySide6.QtGui import QTextCursor

cursor = text_edit.textCursor()
cursor.movePosition(QTextCursor.MoveOperation.End)
```

## Common Move Operations

| Operation | PySide6 Enum |
|-----------|--------------|
| End of document | `QTextCursor.MoveOperation.End` |
| Start of document | `QTextCursor.MoveOperation.Start` |
| End of line | `QTextCursor.MoveOperation.EndOfLine` |
| Start of line | `QTextCursor.MoveOperation.StartOfLine` |
| Next character | `QTextCursor.MoveOperation.NextCharacter` |
| Previous character | `QTextCursor.MoveOperation.PreviousCharacter` |

## Verification

After fixing, the cursor should move without raising AttributeError.

## Notes

- This is a PySide6-specific issue due to stricter enum handling
- PyQt5 was more lenient and allowed instance-based enum access
- Always import QTextCursor explicitly when using move operations
