---
title: "feat: Add category pill bar to Sequence tab algorithm grid"
type: feat
status: completed
date: 2026-04-04
origin: docs/brainstorms/2026-04-04-sequencer-tab-categories-requirements.md
---

# feat: Add category pill bar to Sequence tab algorithm grid

## Overview

Add a tag-based category system and horizontal pill bar to the Sequence tab's algorithm card grid, allowing users to filter 19 algorithms by intent: All, Arrange, Find, Connect, Audio, Text. Algorithms can appear in multiple categories. The selected category persists across sessions.

## Problem Frame

The Sequence tab shows 19 algorithm cards in a flat 5-row grid. Users must scan every card to find what they want. A pill-bar filter above the grid lets users narrow by intent while preserving the full grid via the "All" category. (see origin: `docs/brainstorms/2026-04-04-sequencer-tab-categories-requirements.md`)

## Requirements Trace

- R1. Tag-based category system in `ALGORITHM_CONFIG` — each algorithm belongs to one or more categories
- R2. Categories: All, Arrange, Find, Connect, Audio, Text
- R3. "All" shows every algorithm
- R4. Selecting a category filters the card grid
- R5. Tag assignments per the origin document's table (19 algorithms, 4 multi-tagged)
- R6. Horizontal pill/chip bar above the card grid
- R7. Pill bar replaces the subheader ("Select how you want to arrange your clips"); the main header ("Choose a Sorting Method") remains. [Origin R7 was ambiguous ("replaces or sits below"); resolved in Key Technical Decisions.]
- R8. Active pill has a visually distinct selected state
- R9. Selected category persists across sessions via settings
- R10. First-time default is "All"
- R11. Card grid reflows when category changes
- R12. Algorithm availability (enabled/disabled) works the same within category views

## Scope Boundaries

- No changes to algorithm behavior, dialogs, or timeline view
- No search/filter-by-text
- No drag-to-reorder cards
- Categories are hardcoded, not user-configurable
- No agent/chat tool changes (agent bypasses the card grid entirely)

## Context & Research

### Relevant Code and Patterns

- `ui/algorithm_config.py` — Zero-dependency module, single source of truth for algorithm metadata. 19 entries with `icon`, `label`, `description`, `allow_duplicates`, `required_analysis`, optional `is_dialog`. Both `sequence_tab.py` and `sorting_card_grid.py` import from it.
- `ui/widgets/sorting_card_grid.py` (160 lines) — Builds cards in `_setup_ui()` with hardcoded `(key, row, col)` positions in a `QGridLayout`. Header ("Choose a Sorting Method"), subheader ("Select how you want to arrange your clips"), then grid. `grid_layout` is currently a local variable, not an instance attribute.
- `ui/widgets/sorting_card.py` (232 lines) — Fixed 200x150px card with custom `paintEvent`. Signals: `clicked(str)`. API: `set_selected()`, `set_enabled()`, `is_enabled()`.
- `ui/tabs/sequence_tab.py` (2152 lines) — `QStackedWidget` with STATE_CARDS/STATE_TIMELINE/STATE_CONFIRM. Cards view created in `_create_cards_view()`. Availability updated in `_update_card_availability()`.
- `core/settings.py` — Settings persistence requires 3 locations: `Settings` dataclass field, `_load_from_json()` reader, `_settings_to_json()` writer. Pattern: `analysis_selected_operations` field.
- `ui/dialogs/signature_style_dialog.py` line 296 — Existing `QButtonGroup` + checkable `QPushButton` pattern for exclusive toggle. Closest precedent for pill bar.
- `ui/theme.py` — `Radii.FULL = 9999` for pill shapes. `accent_blue` for selected state. `background_secondary`/`background_tertiary` for inactive/hover. `TypeScale.SM = 11` for pill text.

### Institutional Learnings

- **Circular import risk** (`docs/solutions/logic-errors/circular-import-config-consolidation.md`): Algorithm metadata must live exclusively in `ui/algorithm_config.py`. Do not duplicate category mappings in the pill bar or sequence tab.
- **Duplicate state risk** (`docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`): Single owner for filter state. `SortingCardGrid` owns the selected category; the pill bar emits user-interaction signals, and the grid re-emits `category_changed` for the parent tab.
- **QSS vs QPainter** (project memory): If pills use custom painting, add `CategoryPillBar { background-color: transparent; }` in the global stylesheet. For this feature, QSS-only styling avoids the issue entirely.

## Key Technical Decisions

- **Pill bar widget**: `QButtonGroup` + checkable `QPushButton` with pill QSS styling. Follows existing `signature_style_dialog.py` pattern. No custom `paintEvent` needed — QSS handles all states. This avoids the QSS-vs-QPainter gotcha.
- **Grid reflow via rebuild**: Clear `QGridLayout` and re-add visible cards in sequential positions (4-column grid, left-aligned partial rows). `setVisible(False)` on cards would leave gaps since `QGridLayout` doesn't reflow — rebuilding is cleaner.
- **4-column fixed width**: All categories use a 4-column grid. Small categories (Text: 2 cards) show a partially filled first row, consistent with the current grid's row 4 (3 of 4 slots filled).
- **R7 resolved — replace subheader**: The pill bar replaces the subheader ("Select how you want to arrange your clips") since the category pills make it redundant. The main header ("Choose a Sorting Method") remains.
- **Category field structure**: `"categories": ["find", "connect"]` (e.g., eyes_without_a_face) — a list of lowercase strings, consistent with the existing `"required_analysis"` list pattern.
- **Filter state ownership**: `SortingCardGrid` owns the pill bar and selected category. It exposes a `category_changed(str)` signal for the parent tab to persist to settings.
- **Card order within categories**: Cards maintain their relative order from the full grid positions list when filtered.
- **Category display order**: All, Arrange, Find, Connect, Audio, Text — matching R2.

## Open Questions

### Resolved During Planning

- **Pill bar widget choice**: `QButtonGroup` + checkable `QPushButton` — standard Qt pattern with QSS pill styling. No custom widget needed.
- **Grid reflow strategy**: Clear and rebuild `QGridLayout` on category change. Promote `grid_layout` to instance attribute.
- **Tag field structure**: `"categories"` list in each `ALGORITHM_CONFIG` entry.
- **R7 placement**: Replace subheader, keep header.
- **Within-session persistence**: Selected category survives STATE_CARDS → STATE_TIMELINE → STATE_CARDS transitions (the pill bar widget retains its state since the cards view is not destroyed).

### Deferred to Implementation

- **Exact QSS for pill hover/active transitions**: Fine-tune colors once visible in the actual theme context.
- **Fallback for unrecognized persisted key**: Implementation should fall back to "All" if the stored category key is not in the current category list.

## Implementation Units

- [x] **Unit 1: Add categories field to ALGORITHM_CONFIG**

  **Goal:** Add a `"categories"` list to each of the 19 algorithm entries in the config.

  **Requirements:** R1, R2, R5

  **Dependencies:** None

  **Files:**
  - Modify: `ui/algorithm_config.py`
  - Create: `tests/test_algorithm_config.py`

  **Approach:**
  - Add `"categories": [...]` to each entry using the tag assignments from the origin document's R5 table
  - Add a module-level `CATEGORY_ORDER` list: `["All", "Arrange", "Find", "Connect", "Audio", "Text"]` — this is the display-order constant consumed by the pill bar
  - "All" is not stored in any algorithm's categories list — it's a virtual category meaning "no filter"

  **Patterns to follow:**
  - Existing `"required_analysis"` field (list of lowercase strings)
  - `ui/algorithm_config.py` as zero-dependency module (no imports from ui/ or core/)

  **Test scenarios:**
  - Happy path: Every algorithm in `ALGORITHM_CONFIG` has a `"categories"` key containing a non-empty list of strings
  - Happy path: All category values in algorithm entries are members of `CATEGORY_ORDER` (excluding "All")
  - Happy path: Multi-tagged algorithms (volume, gaze_sort, eyes_without_a_face, reference_guided) have exactly the expected categories
  - Edge case: No algorithm has an empty categories list
  - Happy path: `CATEGORY_ORDER` contains exactly 6 entries in the expected order

  **Verification:**
  - All 19 algorithms have categories matching the origin document table
  - `CATEGORY_ORDER` is importable and matches R2

- [x] **Unit 2: Add settings persistence for selected category**

  **Goal:** Persist the user's selected category key across sessions.

  **Requirements:** R9, R10

  **Dependencies:** Unit 1 (for `CATEGORY_ORDER` to validate against)

  **Files:**
  - Modify: `core/settings.py`
  - Test: `tests/test_settings.py`

  **Approach:**
  - Add `sequence_selected_category: str = "All"` to the `Settings` dataclass
  - Add reader in `_load_from_json()` under a `"sequence"` section: validate it's a string, fall back to "All" if missing or unrecognized
  - Add writer in `_settings_to_json()`: `"sequence": {"selected_category": settings.sequence_selected_category}`

  **Patterns to follow:**
  - `analysis_selected_operations` 3-location pattern in `core/settings.py`

  **Test scenarios:**
  - Happy path: Default value is "All" on fresh Settings instance
  - Happy path: Round-trip — set category to "Audio", save to JSON, load from JSON, value is "Audio"
  - Edge case: Loading settings JSON with missing `sequence` section falls back to "All"
  - Edge case: Loading settings JSON with unrecognized category value (e.g., "deleted_category") falls back to "All"
  - Edge case: Loading settings JSON with non-string category value falls back to "All"

  **Verification:**
  - `load_settings().sequence_selected_category` returns "All" by default
  - Value survives save/load cycle

- [x] **Unit 3: Build CategoryPillBar widget**

  **Goal:** Create a reusable horizontal pill bar widget that displays category buttons and emits a signal when the selection changes.

  **Requirements:** R6, R8

  **Dependencies:** Unit 1 (imports `CATEGORY_ORDER`)

  **Files:**
  - Create: `ui/widgets/category_pill_bar.py`
  - Create: `tests/test_category_pill_bar.py`

  **Approach:**
  - `CategoryPillBar(QWidget)` containing a `QHBoxLayout` with centered `QPushButton` widgets in a `QButtonGroup` (exclusive mode)
  - Each button is checkable, pill-shaped via QSS (`border-radius: Radii.FULL`)
  - Populate from `CATEGORY_ORDER`
  - Signal: `category_changed(str)` — emitted when user clicks a different pill
  - Method: `set_category(str)` — programmatically select a pill (used for restoring from settings)
  - Connect to `theme().changed` for theme refresh
  - QSS states: default (`background_secondary`, `text_secondary`), hover (`background_tertiary`), checked/active (`accent_blue`, `text_inverted`)
  - Spacing between pills: `Spacing.SM` (8px). Pill padding: `Spacing.SM` horizontal, `Spacing.XS` vertical. Font: `TypeScale.SM`.

  **Patterns to follow:**
  - `QButtonGroup` + checkable `QPushButton` from `ui/dialogs/signature_style_dialog.py` line 296
  - Theme refresh pattern from `SortingCardGrid._refresh_theme()`

  **Test scenarios:**
  - Happy path: Widget creates 6 buttons matching `CATEGORY_ORDER`
  - Happy path: Clicking a pill emits `category_changed` with the category name
  - Happy path: `set_category("Audio")` selects the Audio pill without emitting `category_changed` (programmatic set does not emit — use `blockSignals` or guard flag to distinguish from user clicks)
  - Edge case: Clicking the already-selected pill does not re-emit `category_changed`
  - Happy path: Only one pill is checked at a time (exclusive selection)
  - Edge case: `set_category("nonexistent")` does not crash — falls back to "All"

  **Verification:**
  - Widget renders 6 pills in the correct order
  - Selection is mutually exclusive
  - Signal emits correctly on user interaction and programmatic set

- [x] **Unit 4: Add category filtering and grid reflow to SortingCardGrid**

  **Goal:** Integrate the pill bar into the grid widget and implement dynamic card filtering with grid reflow.

  **Requirements:** R3, R4, R7, R11, R12

  **Dependencies:** Unit 1, Unit 3

  **Files:**
  - Modify: `ui/widgets/sorting_card_grid.py`
  - Create: `tests/test_sorting_card_grid.py`

  **Approach:**
  - Promote `grid_layout` and `grid_container` to instance attributes (`self._grid_layout`, `self._grid_container`)
  - Remove the subheader label (replaced by pill bar per R7 resolution)
  - Add `CategoryPillBar` between the header and the grid container
  - Store the master positions list as `self._positions` (promote from local variable) so `_rebuild_grid` can derive card order
  - Add `_rebuild_grid(category: str)` method:
    - Determine visible cards: if "All", all cards; otherwise filter `self._cards` by checking each algorithm's `categories` list in `ALGORITHM_CONFIG`
    - If the currently selected card (`_selected_key`) is not in the visible set, call `clear_selection()` before rebuilding
    - Drain the layout using `takeAt(0)` in a while loop until empty. Do NOT call `deleteLater()` — cards are persistent objects reused across rebuilds. This differs from `cost_estimate_panel._rebuild_grid` which destroys its widgets.
    - Re-add visible cards in sequential positions: `row = i // 4`, `col = i % 4`, maintaining relative order from `self._positions`
    - Show filtered cards (`setVisible(True)`), hide others (`setVisible(False)`) as a defensive measure
  - Connect `CategoryPillBar.category_changed` → `_rebuild_grid`
  - Expose `category_changed(str)` signal on `SortingCardGrid` (re-emit from pill bar)
  - Expose `set_category(str)` method (delegates to pill bar)
  - `set_algorithm_availability()` must update ALL cards (visible or not) so switching categories shows correct enabled/disabled state (per R12)
  - `_refresh_theme()` must: remove the stale `self._subheader` reference (since the subheader is deleted), and add pill bar theme refresh call

  **Patterns to follow:**
  - Current `_setup_ui()` card creation loop
  - `set_algorithm_availability()` iterating `self._cards`

  **Test scenarios:**
  - Happy path: "All" category shows all 19 cards
  - Happy path: "Arrange" category shows exactly 9 cards (shuffle, sequential, duration, color, brightness, volume, shot_type, proximity, gaze_sort)
  - Happy path: "Text" category shows exactly 2 cards (exquisite_corpus, storyteller)
  - Happy path: Multi-tagged algorithm (volume) appears in both "Arrange" and "Audio" views
  - Happy path: Cards within a filtered category maintain their relative order from the full grid
  - Edge case: Switching from "Text" (2 cards) to "All" (19 cards) correctly reflows the full grid
  - Edge case: Selecting a card then switching to a category that doesn't contain it — selection is cleared
  - Integration: `set_algorithm_availability()` disables a card, then switching categories still shows it as disabled
  - Happy path: Selecting a card in a filtered view emits `algorithm_selected` with the correct key
  - Edge case: Category with all algorithms disabled (e.g., Text before analysis) shows disabled cards, not empty state

  **Verification:**
  - Grid dynamically reflows on category change
  - No visual gaps in the grid after filtering
  - Availability state is preserved across category switches

- [x] **Unit 5: Wire SequenceTab integration**

  **Goal:** Connect the category pill bar to settings persistence and ensure the category survives tab state transitions.

  **Requirements:** R9, R10

  **Dependencies:** Unit 2, Unit 4

  **Files:**
  - Modify: `ui/tabs/sequence_tab.py`
  - Create: `tests/test_sequence_tab.py`

  **Approach:**
  - In `_create_cards_view()`: after creating `self.card_grid`, call `self.card_grid.set_category(settings.sequence_selected_category)` to restore the persisted category
  - Connect `self.card_grid.category_changed` to a handler that saves the new category to settings: `settings.sequence_selected_category = category; save_settings(settings)`
  - The category naturally persists within a session because the cards view widget (and thus the pill bar) is not destroyed during STATE_CARDS → STATE_TIMELINE transitions — QStackedWidget preserves all child widgets

  **Patterns to follow:**
  - `save_settings` calls in `ui/main_window.py` (SequenceTab does not currently call `save_settings` — this is a new pattern for this tab; add `save_settings` import)
  - `_create_cards_view()` initialization flow

  **Test scenarios:**
  - Happy path: On tab activation with a persisted category of "Audio", the pill bar shows "Audio" selected and only Audio cards are visible
  - Happy path: Changing category triggers a settings save
  - Integration: Switch to STATE_TIMELINE, switch back to STATE_CARDS — selected category is preserved
  - Edge case: First launch with no persisted setting defaults to "All"

  **Verification:**
  - Category persists across app restarts
  - Category survives STATE_CARDS ↔ STATE_TIMELINE transitions

## System-Wide Impact

- **Interaction graph:** `SequenceTab` → `SortingCardGrid` → `CategoryPillBar`. The pill bar emits `category_changed`, the grid rebuilds, the tab persists to settings. No other tabs or systems are affected.
- **Error propagation:** Invalid persisted category key → fall back to "All". No other failure modes.
- **State lifecycle risks:** The pill bar widget state survives QStackedWidget page switches because the widget is not destroyed. No cache or cleanup concerns.
- **API surface parity:** The agent/chat system bypasses the card grid entirely (uses `generate_sequence` directly), so no changes needed there.
- **Unchanged invariants:** `SortingCard` API, `algorithm_selected` signal, card click → algorithm routing flow, dialog workflows, timeline view — all unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| QGridLayout rebuild may cause visual flicker | Qt batches layout changes within a single event loop tick — add/remove operations complete before repaint. If needed, wrap in `setUpdatesEnabled(False/True)` |
| Global QPushButton stylesheet conflicts with pill styling | Use class-specific QSS selector (`CategoryPillBar QPushButton`) to override ALL global QPushButton properties: background-color, color, border, padding, border-radius, AND min-height (global sets 32px via UISizes.BUTTON_MIN_HEIGHT) |
| Stale comments in sorting_card_grid.py | Fix as part of Unit 4: class docstring says "2x2 grid" and line 65 comment says "14 algorithms, 4 columns" — both should reflect 19 algorithms |

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-04-sequencer-tab-categories-requirements.md](docs/brainstorms/2026-04-04-sequencer-tab-categories-requirements.md)
- Related code: `ui/algorithm_config.py`, `ui/widgets/sorting_card_grid.py`, `ui/tabs/sequence_tab.py`
- Related learnings: `docs/solutions/logic-errors/circular-import-config-consolidation.md`, `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
