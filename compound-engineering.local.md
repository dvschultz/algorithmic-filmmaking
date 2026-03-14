---
review_agents:
  - kieran-python-reviewer
  - security-sentinel
  - performance-oracle
  - architecture-strategist
  - code-simplicity-reviewer
---

## Review Context

This is a PySide6 (Qt 6) desktop application for algorithmic video editing. Key conventions:

- **UI thread safety**: Heavy operations must run in QThread workers (CancellableWorker pattern)
- **FFmpeg safety**: Always use argument arrays, never shell interpolation
- **Project state**: Use Project methods to modify state, never directly append to lists
- **Agent parity**: Any action a user can take, the chat agent should also be able to take
- **Settings**: Always use paths from `core/settings.py`, never hardcode
- **Theme constants**: Use `UISizes` from `ui/theme.py` for consistent widget sizing
