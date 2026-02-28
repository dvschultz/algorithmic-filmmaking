---
title: "Sequence Cost Estimate Panel"
type: feat
date: 2026-02-07
---

# Sequence Cost Estimate Panel

## Overview

Add an inline cost estimate panel that shows users the analysis time and dollar cost before generating a sequence. The panel appears in two contexts:

1. **Clips-ready flow** (card click with selected clips): A gatekeeper panel between algorithm selection and generation, showing what analysis is needed and letting the user confirm before proceeding.
2. **Intention flow** (card click with no clips): Cost estimates integrated into the existing `IntentionImportDialog` parameter area.

The panel includes per-operation local/cloud tier dropdowns (per-sequence overrides, not global), and replaces the existing `MissingDescriptionsDialog` / `ExquisiteCorpusDialog` missing-analysis handling with a unified experience.

## Problem Statement

Users have no visibility into analysis costs before generating a sequence. Some algorithms require analysis that takes time (local inference) or money (cloud API calls). Users discover this only after clicking a card, when they either get an error about missing metadata or silently wait for auto-compute. The intention flow already gates on parameter selection but doesn't surface cost information.

## Proposed Solution

### New Files

- **`core/cost_estimates.py`** — Hardcoded pricing constants + estimation logic
- **`ui/widgets/cost_estimate_panel.py`** — Reusable cost panel widget (used in both flows)

### Modified Files

- **`ui/tabs/sequence_tab.py`** — Insert gatekeeper panel between card click and generation
- **`ui/dialogs/intention_import_dialog.py`** — Embed cost estimates in the existing parameter area
- **`ui/algorithm_config.py`** — Add `required_analysis` field to algorithm config

## Technical Approach

### Phase 1: Data Model — Algorithm-to-Analysis Mapping

Add a `required_analysis` field to `ALGORITHM_CONFIG` in `ui/algorithm_config.py`. This formalizes the implicit mapping currently spread across `_update_card_availability()` and `generate_sequence()`.

```python
# ui/algorithm_config.py — new field per algorithm
"storyteller": {
    "icon": "📖",
    "label": "Storyteller",
    "description": "Create a narrative from clip descriptions",
    "allow_duplicates": False,
    "required_analysis": ["describe"],  # NEW: analysis operation keys
},
"color": {
    ...
    "required_analysis": ["colors"],
},
"shot_type": {
    ...
    "required_analysis": ["shots"],
},
```

Mapping (from codebase analysis):

| Algorithm | `required_analysis` | Notes |
|-----------|-------------------|-------|
| shuffle, sequential, duration | `[]` | No analysis needed |
| color, color_cycle | `["colors"]` | Always local, always free |
| brightness | `["brightness"]` | Auto-computed, always local |
| volume | `["volume"]` | Auto-computed, always local |
| shot_type | `["shots"]` | Has local/cloud tier |
| proximity | `["shots"]` | Can use `shot_type` OR `cinematography` |
| similarity_chain | `["embeddings"]` | Auto-computed CLIP, always local |
| match_cut | `["boundary_embeddings"]` | Auto-computed, always local |
| exquisite_corpus | `["extract_text"]` | Has local/cloud tier |
| storyteller | `["describe"]` | Has local/cloud tier |

### Phase 2: Cost Estimation Engine

Create `core/cost_estimates.py` with hardcoded pricing and time estimates.

```python
# core/cost_estimates.py

from dataclasses import dataclass

@dataclass
class OperationEstimate:
    """Cost and time estimate for a single analysis operation."""
    operation: str           # analysis operation key
    clips_needing: int       # clips that still need this analysis
    clips_total: int         # total clips in selection
    tier: str                # "local" or "cloud"
    time_seconds: float      # estimated wall-clock time
    cost_dollars: float      # estimated dollar cost

# Per-clip time estimates (conservative, CPU-only)
TIME_PER_CLIP = {
    "colors":              {"local": 0.3,  "cloud": None},
    "shots":               {"local": 1.0,  "cloud": 1.5},
    "extract_text":        {"local": 0.5,  "cloud": 0.8},
    "describe":            {"local": 3.0,  "cloud": 0.8},
    "brightness":          {"local": 0.1},
    "volume":              {"local": 0.2},
    "embeddings":          {"local": 0.8},
    "boundary_embeddings": {"local": 1.5},
    "transcribe":          {"local": 2.0},
    "cinematography":      {"cloud": 1.0},
}

# Per-clip dollar costs (cloud only)
COST_PER_CLIP = {
    "shots":          {"cloud": 0.005},   # Replicate VideoMAE
    "extract_text":   {"cloud": 0.001},   # Gemini VLM
    "describe":       {"cloud": 0.001},   # Gemini Flash
    "cinematography": {"cloud": 0.002},   # Gemini
}

# Operations that support tier switching
TIERED_OPERATIONS = {
    "shots": {"local": "Local (Free)", "cloud": "Cloud (Paid)"},
    "extract_text": {"local": "Local (Free)", "cloud": "Cloud (Paid)"},
    "describe": {"local": "Local (Free)", "cloud": "Cloud (Paid)"},
}

def estimate_sequence_cost(
    algorithm: str,
    clips_with_sources: list,
    tier_overrides: dict[str, str] | None = None,
    settings = None,
) -> list[OperationEstimate]:
    """Calculate cost estimates for a sequence algorithm.

    Args:
        algorithm: Algorithm key from ALGORITHM_CONFIG
        clips_with_sources: List of (Clip, Source) tuples
        tier_overrides: Per-operation tier overrides {"describe": "cloud"}
        settings: Settings object for parallelism values

    Returns:
        List of OperationEstimate, one per required operation.
        Empty list if no analysis needed.
    """
    ...
```

The function:
1. Looks up `required_analysis` from `ALGORITHM_CONFIG[algorithm]`
2. For each required operation, counts clips missing that metadata
3. Applies tier (from override or global settings)
4. Divides total time by parallelism setting for wall-clock estimate
5. Returns estimates (may be empty if all clips are ready)

**Metadata check functions** (mapping operation key to clip field check):

```python
METADATA_CHECKS = {
    "colors":              lambda clip: bool(clip.dominant_colors),
    "shots":               lambda clip: bool(clip.shot_type) or bool(clip.cinematography),
    "extract_text":        lambda clip: bool(getattr(clip, 'extracted_texts', None)),
    "describe":            lambda clip: bool(clip.description),
    "brightness":          lambda clip: clip.average_brightness is not None,
    "volume":              lambda clip: clip.rms_volume is not None,
    "embeddings":          lambda clip: bool(clip.embedding),
    "boundary_embeddings": lambda clip: bool(clip.first_frame_embedding),
}
```

### Phase 3: Cost Estimate Panel Widget

Create `ui/widgets/cost_estimate_panel.py` — a standalone widget usable in both the sequence tab and the intention dialog.

```
+-----------------------------------------------+
| Analysis Estimate                      [v Hide]|
+-----------------------------------------------+
| Operation      | Clips  | Tier        | Est.  |
| -------------- | ------ | ----------- | ----- |
| Describe       | 24/50  | [Local  v]  | Free  |
|                |        |             | ~72s  |
+-----------------------------------------------+
| Total: 24 clips need analysis | Free | ~72s  |
+-----------------------------------------------+
```

**Widget API:**

```python
class CostEstimatePanel(QWidget):
    tier_changed = Signal(str, str)  # (operation_key, new_tier)

    def __init__(self, parent=None):
        ...

    def set_estimates(self, estimates: list[OperationEstimate]):
        """Update the panel with new estimates."""
        ...

    def get_tier_overrides(self) -> dict[str, str]:
        """Return current tier selections as {operation: tier} dict."""
        ...

    def set_collapsed(self, collapsed: bool):
        """Collapse or expand the panel."""
        ...
```

**Behavior:**
- Hidden when `estimates` is empty (no analysis needed)
- Collapsed state shows single summary line: "24 clips need analysis | Free | ~72s"
- Expanded state shows per-operation table with tier dropdowns
- Tier dropdowns only appear for operations in `TIERED_OPERATIONS`
- Tier dropdowns initialize from global settings, user can override
- `tier_changed` signal triggers recalculation in the parent

**Styling:**
- Use `QGroupBox` or styled `QFrame` matching the render tab pattern
- `UISizes.COMBO_BOX_MIN_HEIGHT` for tier dropdowns
- Theme colors from `theme()` for text and borders

### Phase 4: Clips-Ready Flow Integration

Modify `_on_card_clicked()` in `sequence_tab.py` to insert a gatekeeper step:

**Current flow:**
```
Card click → check clips → _apply_algorithm() → generate
```

**New flow:**
```
Card click → check clips → estimate cost → show panel + "Generate" button
                                          → user clicks Generate → _apply_algorithm()
```

Implementation approach:

1. Add a new `STATE_CONFIRM` (index 2) to the `QStackedWidget` that shows the cost estimate panel + a "Generate Sequence" button
2. When a card is clicked and clips are available, compute estimates and transition to `STATE_CONFIRM`
3. If estimates are empty (no analysis needed), skip directly to `_apply_algorithm()` — no gatekeeper for zero-cost algorithms
4. The confirmation view shows: algorithm name, cost panel, direction dropdown (if applicable), and Generate/Back buttons

**For dialog-based algorithms** (storyteller, exquisite_corpus): The existing dialog workflows (`_show_storyteller_dialog`, `_show_exquisite_corpus_dialog`) are replaced. The cost estimate panel now handles their missing-analysis concerns. The `MissingDescriptionsDialog` and similar dialogs are no longer needed in the card-click path.

### Phase 5: Intention Flow Integration

Add cost estimates to the existing `IntentionImportDialog`:

1. Add a `CostEstimatePanel` instance to the import view, below the existing parameter dropdowns (direction, shot type, poem length, etc.)
2. The panel shows estimates based on the selected algorithm and tier preferences
3. Since clips don't exist yet in the intention flow, the estimate assumes ALL clips will need analysis (clips_needing = clips_total = "all imported clips")
4. Show a note: "Estimates based on all imported clips needing analysis"
5. The panel updates when the user changes tier dropdowns

### Phase 6: Live Updates

Connect signals for recalculation:

**In sequence_tab.py (STATE_TIMELINE header):**
- `algorithm_dropdown.currentTextChanged` → recalculate
- `set_available_clips()` called → recalculate
- `clips_data_changed` signal → recalculate (after auto-compute mutates clips)
- `on_tab_activated()` → recalculate (clips may have been analyzed on another tab)

**In CostEstimatePanel:**
- `tier_changed` signal → parent recalculates and calls `set_estimates()` again

## Acceptance Criteria

### Functional Requirements

- [x] Cost panel appears after clicking an algorithm card when clips are selected
- [x] Panel shows required analysis operations for the chosen algorithm
- [x] Panel shows "N/M clips need [operation]" counts
- [x] Panel shows estimated time (conservative) and dollar cost per operation
- [x] Tiered operations (describe, shots, extract_text) show local/cloud dropdown
- [x] Tier dropdowns default to global settings value
- [x] Changing tier dropdown updates estimates immediately
- [x] Tier selections are per-sequence (don't modify global settings)
- [x] Algorithms needing no analysis skip the panel and generate immediately
- [x] "Generate" button in confirmation view triggers sequence generation
- [x] "Back" button returns to card grid
- [x] Panel integrates into IntentionImportDialog for the intention flow
- [x] Storyteller and Exquisite Corpus use the cost panel instead of their separate missing-analysis dialogs
- [x] Panel updates when switching algorithms in the STATE_TIMELINE header
- [x] Panel updates when returning to Sequence tab after analyzing clips elsewhere

### Non-Functional Requirements

- [x] Panel renders in < 50ms (synchronous metadata check is fast for 3600+ clips)
- [x] No new dependencies
- [x] Follows existing theme/styling patterns (UISizes, theme colors)
- [x] Works in both light and dark themes

## Dependencies & Risks

**Dependencies:**
- `ui/algorithm_config.py` needs new `required_analysis` field (backward-compatible — provide default)
- Existing dialog workflows (Storyteller, Exquisite Corpus) need refactoring to remove duplicate missing-analysis handling

**Risks:**
- **Stale pricing**: Hardcoded cloud pricing will become inaccurate over time. Mitigated by making prices easy to update in one file (`core/cost_estimates.py`).
- **API key validation**: If user selects cloud tier but has no API key, generation will fail. The panel should show a warning when cloud tier is selected without a configured key. Use existing `settings.get_*_api_key()` functions to check.
- **Hybrid text extraction**: The global setting supports "hybrid" mode but the panel only shows local/cloud. Keep it simple — hybrid is a settings-level concern, not a per-sequence concern.

## Implementation Phases

### Phase 1: Foundation (core/cost_estimates.py + algorithm_config update)
- Add `required_analysis` to `ALGORITHM_CONFIG`
- Create cost estimation engine with hardcoded pricing
- Write tests for estimation logic

### Phase 2: Widget (ui/widgets/cost_estimate_panel.py)
- Build the standalone panel widget
- Collapsible behavior, tier dropdowns, summary line
- Theme integration

### Phase 3: Clips-Ready Integration (sequence_tab.py)
- Add `STATE_CONFIRM` view
- Wire card click → cost estimate → generate flow
- Replace storyteller/exquisite_corpus dialog missing-analysis handling

### Phase 4: Intention Flow Integration (intention_import_dialog.py)
- Embed cost panel in existing dialog
- Wire tier changes to estimate updates

### Phase 5: Live Updates & Polish
- Connect all update signals
- Add API key warning for cloud tiers
- Test with 0 clips, partial analysis, full analysis scenarios

## References

- Brainstorm: `docs/brainstorms/2026-02-07-sequence-cost-estimate-brainstorm.md`
- Algorithm config: `ui/algorithm_config.py`
- Analysis operations registry: `core/analysis_operations.py`
- Sequence tab: `ui/tabs/sequence_tab.py` (lines 232-297 for card click flow)
- Intention dialog: `ui/dialogs/intention_import_dialog.py`
- Settings tiers: `core/settings.py` (lines 399-435)
- Render tab GroupBox pattern: `ui/tabs/render_tab.py` (lines 71-188)
- Theme constants: `ui/theme.py` (lines 184-208 for UISizes)
