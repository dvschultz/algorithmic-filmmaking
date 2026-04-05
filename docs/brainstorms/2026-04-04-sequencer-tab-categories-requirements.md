---
date: 2026-04-04
topic: sequencer-tab-categories
---

# Sequencer Tab: Algorithm Category Tabs

## Problem Frame

The Sequence tab displays 19 algorithm cards in a 5-row grid. As the algorithm count has grown, the grid has become overwhelming — users must scan all 19 cards to find what they want. Categorized sub-navigation would let users filter by intent and reduce cognitive load.

## Requirements

**Category System**

- R1. Add a tag-based category system to `ALGORITHM_CONFIG` where each algorithm can belong to one or more categories
- R2. Categories are: **All**, **Arrange**, **Find**, **Connect**, **Audio**, **Text**
- R3. The "All" category shows every algorithm (equivalent to the current flat grid)
- R4. Selecting a category filters the card grid to show only algorithms tagged with that category

**Tag Assignments**

- R5. Tag assignments for all 19 algorithms:

| Algorithm | Arrange | Find | Connect | Audio | Text |
|---|---|---|---|---|---|
| shuffle | x | | | | |
| sequential | x | | | | |
| duration | x | | | | |
| color | x | | | | |
| brightness | x | | | | |
| volume | x | | | x | |
| shot_type | x | | | | |
| proximity | x | | | | |
| gaze_sort | x | x | | | |
| gaze_consistency | | x | | | |
| rose_hobart | | x | | | |
| eyes_without_a_face | | x | x | | |
| similarity_chain | | | x | | |
| match_cut | | | x | | |
| reference_guided | | | x | x | |
| signature_style | | | x | | |
| staccato | | | | x | |
| exquisite_corpus | | | | | x |
| storyteller | | | | | x |

**UI Presentation**

- R6. Categories appear as a horizontal pill/chip bar above the card grid (not a secondary tab bar or sidebar)
- R7. Pill bar replaces or sits below the existing "Choose a Sorting Method" header area
- R8. Active category pill has a visually distinct selected state

**Persistence**

- R9. Remember the user's last-selected category across sessions (via settings)
- R10. First-time default is "All"

**Grid Behavior**

- R11. Card grid reflows when category changes — cards fill available columns naturally for the filtered count
- R12. Algorithm availability (enabled/disabled based on analysis state) works the same within category views as it does today

## Success Criteria

- Users can find algorithms faster by filtering to a category instead of scanning all 19
- No algorithm becomes harder to discover than it is today (All tab preserves full visibility)
- Category preference persists across app restarts

## Scope Boundaries

- No changes to algorithm behavior, dialogs, or the timeline view
- No search/filter-by-text within categories (could be a future addition)
- No drag-to-reorder of cards within categories
- Category definitions are hardcoded, not user-configurable

## Key Decisions

- **Tags, not exclusive categories**: Algorithms can appear in multiple tabs (e.g., volume in both Arrange and Audio). This avoids forced categorization for cross-cutting algorithms.
- **Pill/chip bar, not nested tabs**: Since the Sequence tab is already a top-level app tab, sub-navigation uses a lighter pill/chip pattern to avoid visual competition with the main tab bar.
- **No "People" category**: Rejected because many visual algorithms may involve people — the gaze/face algorithms are better described as "Find" (filter/isolate) rather than a people-specific group.

## Dependencies / Assumptions

- `core/settings.py` supports persisting the selected category key (straightforward addition to existing settings)
- `ui/algorithm_config.py` is the single source of truth for algorithm metadata including new tags

## Outstanding Questions

### Deferred to Planning

- [Affects R6][Needs research] What's the best PySide6 widget for the pill/chip bar — custom-styled `QButtonGroup` with `QPushButton`, or a custom widget?
- [Affects R11][Technical] Should the grid maintain a fixed column count (4) when fewer cards are shown, or reflow to fewer columns for small categories like Text (2 cards)?
- [Affects R5][Technical] What field structure should tags use within `ALGORITHM_CONFIG` entries — a `categories` list, a set, or comma-separated string?

## Next Steps

-> `/ce:plan` for structured implementation planning
