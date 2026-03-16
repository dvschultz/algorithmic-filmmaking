---
paths:
  - "ui/**"
---

# UI Consistency Standards

Use constants from `ui/theme.py:UISizes`:

| Constant | Value | Use |
|----------|-------|-----|
| `COMBO_BOX_MIN_HEIGHT` | 32 | All combo boxes |
| `LINE_EDIT_MIN_HEIGHT` | 32 | Text inputs |
| `BUTTON_MIN_HEIGHT` | 32 | Buttons |
| `FORM_LABEL_WIDTH` | 140 | Standard label width |
| `FORM_LABEL_WIDTH_NARROW` | 120 | Compact layouts |
| `FORM_LABEL_WIDTH_WIDE` | 180 | Wide labels |
| `COMBO_BOX_MIN_WIDTH` | 200 | Minimum combo width |

Wrap long form content in `QScrollArea` with `setWidgetResizable(True)` and `setFrameShape(QScrollArea.NoFrame)`.

**QSS vs QPainter**: A global `QWidget { background-color: X }` rule covers custom `paintEvent` glow effects. Add `WidgetClassName { background-color: transparent; }` for custom-painted widgets.
