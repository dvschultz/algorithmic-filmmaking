---
name: pyside6-shiboken-object-setattr-frozen-crash
description: |
  Fix "TypeError: can't apply this __setattr__ to FilterState object" (or
  any QObject-subclass name) raised at startup of a PyInstaller-frozen
  PySide6 app. Use when: (1) dev tests pass but the frozen .app crashes at
  instantiation of a QObject subclass that overrides __setattr__ or calls
  `object.__setattr__(self, ...)` in __init__, (2) the traceback points at
  a line like `object.__setattr__(self, name, value)` inside a QObject
  subclass, (3) the issue only reproduces in the PyInstaller bundle, never
  in pytest under offscreen Qt. Symptom often caught by macOS CI smoke test.
author: Claude Code
version: 1.0.0
date: 2026-04-22
---

# PySide6 Shiboken object.__setattr__ frozen crash

## Problem

A QObject subclass that performs attribute assignment via
`object.__setattr__(self, name, value)` — either directly in `__init__` to
bypass a custom `__setattr__`, or inside the custom `__setattr__` itself —
works perfectly in regular Python but crashes at startup of a
PyInstaller-frozen build with:

```
TypeError: can't apply this __setattr__ to FilterState object
```

Shiboken (PySide6's binding layer) installs a custom metaclass on QObject
subclasses. In the frozen runtime, this metaclass rejects `object.__setattr__`
calls — it expects all attribute assignments to flow through the QObject
`__setattr__` so Shiboken can track lifetime and property bindings. Regular
CPython is more permissive, which is why dev tests pass and the bug only
shows up after packaging.

## Context / Trigger Conditions

- PySide6 QObject subclass that defines a custom `__setattr__` for dirty
  checking, batched change signals, or proxy properties.
- The class uses `object.__setattr__(self, ...)` either to:
  - **Bootstrap** instance attributes in `__init__` without triggering the
    custom setter
  - **Fall through** for attributes that shouldn't be dirty-tracked
- Dev tests (`pytest`, `python -m unittest`) pass under offscreen Qt.
- The frozen `.app` / `.exe` bundle crashes at startup with the TypeError
  above, specifically at a line doing `object.__setattr__(self, ...)`.
- PyInstaller build log shows no warning — the crash surfaces only when the
  frozen binary runs and Qt initializes.

Scene Ripper hit this at `core/filter_state.py` after extracting `FilterState`
as a `QObject` with a custom `__setattr__` for batched emission dirty checks.

## Solution

Replace every `object.__setattr__(self, name, value)` call inside a QObject
subclass with `QObject.__setattr__(self, name, value)`.

`QObject.__setattr__` is the descriptor Shiboken has installed, so the
metaclass accepts the call. Behavior is identical under CPython — you'd
normally never know the difference.

```python
from PySide6.QtCore import QObject

class FilterState(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        # BAD: object.__setattr__(self, "_internal", 0)
        # GOOD:
        QObject.__setattr__(self, "_internal", 0)

    def __setattr__(self, name, value):
        # ... dirty check logic ...
        # BAD: object.__setattr__(self, name, value)
        # GOOD:
        QObject.__setattr__(self, name, value)
```

Keeping a local alias cuts the noise:

```python
def __init__(self, parent=None):
    super().__init__(parent)
    _init = QObject.__setattr__
    _init(self, "_internal", 0)
    _init(self, "field_a", set())
    # ...
```

## Verification

1. Unit tests under `QT_QPA_PLATFORM=offscreen` still pass unchanged.
2. Run the macOS CI smoke test (or equivalent local PyInstaller bundle run)
   and confirm the `MainWindow` / containing app reaches its "startup
   complete" log line.

For Scene Ripper specifically, the `.github/workflows/build-macos.yml` smoke
test runs the frozen `.app` with `SCENE_RIPPER_STARTUP_SMOKE_TEST=1` and
greps for "Startup smoke test completed successfully" in the app log.

## Example

Scene Ripper fix commit: `99330ff` (`fix(filter-state): use
QObject.__setattr__ for Shiboken compatibility`).

Original crash in v0.3.12 CI run 24761972737:
```
Traceback (most recent call last):
  File "ui/main_window.py", line 1082, in _setup_ui
    self._filter_state = FilterState()
  File "core/filter_state.py", line 73, in __init__
    self._batching = False
  File "core/filter_state.py", line 120, in __setattr__
    object.__setattr__(self, name, value)
TypeError: can't apply this __setattr__ to FilterState object
```

Lines 73 and 120 both did `object.__setattr__`. Swapping to
`QObject.__setattr__` made the fix trivial and atomic.

## Notes

- The failure mode doesn't reproduce in any unit test, including tests
  constructing the QObject under `QT_QPA_PLATFORM=offscreen`. It's a frozen-
  runtime-only hazard. Budget a CI smoke test cycle to catch it.
- `super().__setattr__(name, value)` works equivalently if the class is a
  direct QObject child. Prefer `QObject.__setattr__` for clarity when the
  class hierarchy might grow or when you're in `__init__` before `super()`
  is conceptually "settled".
- Related Shiboken gotchas: same restriction applies to `object.__delattr__`
  and direct `self.__dict__[name] = value` is also risky for Shiboken-tracked
  attributes. Stick with `QObject.__setattr__`.
- If you need to suppress the custom `__setattr__` during batched operations,
  use an internal flag rather than `object.__setattr__` — e.g.,
  `self._batching = True` (via `QObject.__setattr__`) and check the flag
  inside `__setattr__`.

## References

- PySide6 docs on Shiboken metaclass: https://doc.qt.io/qtforpython-6/shiboken6/index.html
- Example commit: https://github.com/dvschultz/algorithmic-filmmaking/commit/99330ff
- Scene Ripper plan that triggered this: `docs/plans/2026-04-21-001-feat-comprehensive-clip-filter-system-plan.md`
