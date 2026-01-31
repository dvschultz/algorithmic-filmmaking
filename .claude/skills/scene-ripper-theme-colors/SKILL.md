---
name: scene-ripper-theme-colors
description: |
  Fix AttributeError when using theme colors in Scene Ripper PySide6 UI code.
  Use when: (1) "'ThemeColors' object has no attribute" error appears,
  (2) writing new UI components with custom styling, (3) applying theme
  colors in stylesheets. Lists all valid theme color attributes.
author: Claude Code
version: 1.0.0
date: 2026-01-31
---

# Scene Ripper Theme Color Reference

## Problem
When writing PySide6 UI code with theme styling, using non-existent theme
color attributes causes runtime `AttributeError` that only surfaces when
the UI is actually rendered.

## Context / Trigger Conditions
- Error: `AttributeError: 'ThemeColors' object has no attribute 'input_background'`
- Writing `theme().some_color` in stylesheet strings
- Creating new dialogs, widgets, or UI components
- The code compiles but fails at runtime when the widget is instantiated

## Solution

### Valid Theme Color Attributes

**Backgrounds:**
- `background_primary` - Main background
- `background_secondary` - Cards, panels
- `background_tertiary` - Nested elements, hover states, input fields
- `background_elevated` - Popovers, menus, hover states

**Text:**
- `text_primary` - Main text
- `text_secondary` - Subtitles, labels
- `text_muted` - Placeholders, disabled

**Borders:**
- `border_primary` - Default borders
- `border_secondary` - Subtle dividers
- `border_focus` - Focus rings

**Accents:**
- `accent_blue` - Primary accent (selection, links, progress bars)
- `accent_blue_hover` - Blue hover state
- `accent_red` - Errors, destructive, playhead
- `accent_green` - Success, completion indicators
- `accent_orange` - Warnings
- `accent_purple` - Special highlights

**Cards:**
- `card_background` - Card/panel background
- `card_hover` - Card hover state

**Timeline (specialized):**
- `timeline_background`, `timeline_ruler`, `timeline_track`
- `thumbnail_background`

### Common Mappings for UI Elements

| UI Element | Use This |
|------------|----------|
| Input field background | `background_tertiary` |
| Button background | `background_tertiary` |
| Button hover | `background_elevated` |
| Button text | `text_primary` |
| Progress bar fill | `accent_blue` |
| Error/warning text | `accent_orange` |
| Success indicator | `accent_green` |
| Active/selected | `accent_blue` |

## Verification
After fixing, test by instantiating the widget:
```python
python -c "
from ui.dialogs.your_dialog import YourDialog
from PySide6.QtWidgets import QApplication
app = QApplication([])
dialog = YourDialog(parent=None)
print('Dialog created successfully')
"
```

## Example

**Wrong:**
```python
self.setStyleSheet(f"""
    QTextEdit {{
        background-color: {theme().input_background};  # DOESN'T EXIST
    }}
    QPushButton {{
        background-color: {theme().button_background};  # DOESN'T EXIST
    }}
    QProgressBar::chunk {{
        background-color: {theme().accent_primary};  # DOESN'T EXIST
    }}
""")
```

**Correct:**
```python
self.setStyleSheet(f"""
    QTextEdit {{
        background-color: {theme().background_tertiary};
    }}
    QPushButton {{
        background-color: {theme().background_tertiary};
    }}
    QProgressBar::chunk {{
        background-color: {theme().accent_blue};
    }}
""")
```

## Notes
- Theme colors are defined in `ui/theme.py` in the `ThemeColors` dataclass
- Both dark and light themes use the same attribute names with different values
- When adding new color attributes, update both `DARK_COLORS` and `LIGHT_COLORS`
- Use `theme().changed.connect(self._apply_theme)` to respond to theme switches

## References
- Source: `ui/theme.py` lines 30-90 for ThemeColors definition
- Source: `ui/theme.py` lines 120-240 for color values
