---
title: "Circular Import When Consolidating Duplicate Configuration Dictionaries"
date: 2026-02-07
category: logic-errors
tags:
  - circular-import
  - config-consolidation
  - pyside6
  - module-extraction
  - single-source-of-truth
symptoms:
  - Two modules contain identical configuration dictionaries that can drift out of sync
  - Attempting to import from one module into the other raises ImportError (circular import)
  - Adding a new algorithm requires updating both files
module: ui/algorithm_config.py
severity: P3
---

# Circular Import When Consolidating Duplicate Configuration Dictionaries

## Problem

Two UI modules each maintained their own copy of algorithm metadata:

- **`ui/tabs/sequence_tab.py`** had `ALGORITHM_CONFIG` (dict with `label`, `description`, `allow_duplicates`)
- **`ui/widgets/sorting_card_grid.py`** had `ALGORITHMS` (dict of tuples with `icon`, `label`, `description`)

Both contained the same 13 algorithms with identical labels and descriptions. Adding or renaming an algorithm required updating both files, and the two dicts used different data structures (dict-of-dicts vs dict-of-tuples).

## The Naive Fix That Fails

The obvious consolidation — import `ALGORITHM_CONFIG` from `sequence_tab.py` into `sorting_card_grid.py` — creates a **circular import**:

```
sequence_tab.py
  → imports SortingCardGrid from ui.widgets.sorting_card_grid
    → imports ALGORITHM_CONFIG from ui.tabs.sequence_tab  # CIRCULAR
```

Python raises `ImportError` because `sequence_tab` hasn't finished loading when `sorting_card_grid` tries to import from it.

## Root Cause

The circular dependency exists because both modules serve different roles in the same UI layer:
- `sequence_tab.py` is a tab controller that uses `SortingCardGrid` as a child widget
- `sorting_card_grid.py` is a widget that needs algorithm metadata to render cards

The configuration data has no inherent dependency on either module's Qt code, but it was embedded inside both.

## Solution

Extract the shared configuration to a **neutral intermediate module** that neither consumer imports from:

```
ui/algorithm_config.py  (NEW — zero dependencies, no PySide6 imports)
  ↑                    ↑
  |                    |
sequence_tab.py    sorting_card_grid.py
```

### The extracted module (`ui/algorithm_config.py`)

```python
"""Algorithm configuration — single source of truth for all sequencer algorithms."""

ALGORITHM_CONFIG = {
    "color": {
        "icon": "\U0001f3a8",
        "label": "Chromatic Flow",
        "description": "Arrange clips along a color gradient",
        "allow_duplicates": False,
    },
    # ... 12 more algorithms
}

def get_algorithm_config(algorithm: str) -> dict:
    return ALGORITHM_CONFIG.get(algorithm.lower(), {
        "icon": "",
        "label": algorithm.replace("_", " ").title(),
        "description": "",
        "allow_duplicates": False,
    })

def get_algorithm_label(algorithm: str) -> str:
    config = ALGORITHM_CONFIG.get(algorithm.lower())
    if config:
        return config["label"]
    return algorithm.replace("_", " ").title()
```

### Consumer updates

**`sorting_card_grid.py`** — replaced 66-line `ALGORITHMS` dict:
```python
from ui.algorithm_config import ALGORITHM_CONFIG

# In the grid setup:
cfg = ALGORITHM_CONFIG[key]
icon, title, description = cfg["icon"], cfg["label"], cfg["description"]
```

**`sequence_tab.py`** — replaced inline `ALGORITHM_CONFIG` and helper functions:
```python
from ui.algorithm_config import ALGORITHM_CONFIG, get_algorithm_config, get_algorithm_label
```

## Key Design Decisions

1. **Zero dependencies**: `ui/algorithm_config.py` imports nothing — no PySide6, no core modules. This guarantees it can never participate in a circular import chain.

2. **Unified schema**: The old `ALGORITHMS` dict used tuples `(icon, label, description)` while `ALGORITHM_CONFIG` used nested dicts. The consolidated version uses dicts with all four fields (`icon`, `label`, `description`, `allow_duplicates`).

3. **Fallback for unknown algorithms**: `get_algorithm_config()` returns a sensible default for unrecognized algorithm keys, making the system forward-compatible when new algorithms are added to the backend before the config is updated.

## Prevention

### When you see duplicate data structures across modules

1. Check if a direct import between them would create a circular dependency
2. If yes, extract the shared data to a new module that lives "below" both consumers in the import hierarchy
3. The extracted module should have **zero dependencies** on its consumers
4. Use a single data structure that serves all consumers' needs

### Warning signs

- Two modules with similar-looking dicts/lists containing the same domain data
- Adding a feature requires updating multiple files with the same information
- `ImportError` when trying to consolidate by importing from one module into another

## Impact

- Net reduction: ~70 lines of duplicate configuration eliminated
- Single place to add/modify algorithm metadata
- Both `sequence_tab.py` and `sorting_card_grid.py` always agree on labels and descriptions

## Files Modified

- `ui/algorithm_config.py` — NEW: single source of truth
- `ui/tabs/sequence_tab.py` — removed inline config, imports from shared module
- `ui/widgets/sorting_card_grid.py` — removed duplicate dict, imports from shared module
