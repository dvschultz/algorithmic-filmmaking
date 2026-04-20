---
title: "feat: Show custom query text on matching clip cards"
type: feat
status: active
date: 2026-04-20
---

# feat: Show custom query text on matching clip cards

## Overview

Replace the generic "Query Match" / "Query No Match" badge on Analyze-tab clip cards with one green badge per matching query, labeled with the actual query text. Non-matching clips show no badge. The clip detail sidebar keeps its current display.

## Problem Frame

The current Analyze-tab clip card shows "Query Match" or "Query No Match" after a Custom Query analysis. This tells you *something* matched but not *what*, and non-match badges add noise without carrying useful information. Running multiple queries compounds the problem — the badge doesn't distinguish which query matched.

## Requirements Trace

- R1. On each Analyze-tab clip card, show one badge per custom query that matched (query text as the label, green background).
- R2. Clips with no matching queries show no badge (the "Query No Match" badge is removed).
- R3. If a clip matches multiple queries, show each as a separate badge side-by-side (not comma-separated).
- R4. Badge visual style matches the Collect tab's CUT/ANALYZED badges — same padding, radius, font size, `badge_analyzed` color tokens.
- R5. The clip detail sidebar's custom query display is unchanged.
- R6. Tooltip still shows the full per-query result summary (YES/NO, confidence, model) so users can see non-match results by hovering.

## Scope Boundaries

- No changes to `ui/clip_details_sidebar.py`
- No changes to the custom query data model (`Clip.custom_queries`)
- No changes to how custom query analysis runs
- Badge on the clip browser in the **Analyze** tab only — not introducing custom query badges elsewhere

## Context & Research

### Relevant Code and Patterns

- `ui/clip_browser.py:47-55` — `get_latest_custom_query_results()` collapses the append-only history to the latest result per query. Keep as-is.
- `ui/clip_browser.py:231-233` — current `self.custom_query_label` (single QLabel) added to `info_layout`. Will be replaced with a flow container for multiple badges.
- `ui/clip_browser.py:637-655` — `_update_custom_query_badge()`. Current logic: any_match → "Query Match" green, else → "Query No Match" orange. Replace with: iterate matching queries, emit one badge each; hide container if zero matches.
- `ui/clip_browser.py:618-635` — `_custom_query_tooltip()` builds the hover summary. Reuse on each badge and/or the container.
- `ui/source_thumbnail.py:130-148` — reference style for the preferred badge design (`badge_style` string with `TypeScale.XS`, `Spacing.XXS`/`Spacing.SM` padding, `Radii.SM`, `badge_analyzed` text/background). The clip browser should match this.

### Institutional Learnings

None directly relevant — this is a small UI polish change.

## Key Technical Decisions

- **Replace the single QLabel with a flow container** (`QHBoxLayout` holding N badge QLabels). Each badge is one matched query.
- **Reuse Collect-tab badge styling** by factoring the style string used in `source_thumbnail._update_badge()` — or inline the same declaration in clip_browser. Keep tokens (`TypeScale.XS`, `Radii.SM`, `Spacing.XXS`/`SM`, `theme().badge_analyzed`) identical so the two surfaces stay visually consistent.
- **Truncate long queries** in the badge text to keep cards from ballooning. Full query text remains in the tooltip. Threshold to be decided during implementation based on card width (not critical for v1).
- **Hide the container entirely when zero matches** — matches R2 and avoids empty whitespace.

## Open Questions

### Resolved During Planning

- Multi-match layout: separate badges side-by-side (user confirmed).
- Styling source: Collect tab's CUT/ANALYZED badges (user confirmed).

### Deferred to Implementation

- Badge truncation threshold: the right max-chars depends on observed card width and font metrics. Pick a value when implementing and refine if it clips common queries.
- Whether long-running queries wrap or horizontally scroll when many matches exist: start with flow-wrap or hidden overflow; revisit if a clip commonly matches 5+ queries.

## Implementation Units

- [ ] **Unit 1: Replace single badge with flow container of per-query badges**

  **Goal:** Swap `self.custom_query_label` for a badge container that shows one green badge per matching query, hides when zero matches, and uses Collect-tab badge styling.

  **Requirements:** R1, R2, R3, R4, R6

  **Dependencies:** None

  **Files:**
  - Modify: `ui/clip_browser.py` — `ClipThumbnail.__init__` (widget creation, ~line 231), `_update_custom_query_badge()` (~line 637)
  - Test: `tests/test_custom_query_ui.py` (extend existing tests)

  **Approach:**
  - Replace the single `self.custom_query_label = QLabel()` with `self.custom_query_container = QWidget()` + `QHBoxLayout` (right-aligned, matching current label alignment).
  - In `_update_custom_query_badge()`:
    - Clear existing badge widgets from the container
    - From `get_latest_custom_query_results()`, filter to entries where `bool(result.get("match"))` is True
    - If zero: hide the container, clear tooltip, return
    - For each matching result: create a QLabel with the query text, apply the Collect-tab badge style (`TypeScale.XS`, `Radii.SM`, `Spacing.XXS`/`Spacing.SM` padding, `badge_analyzed` text/background), set per-badge tooltip (reuse `_custom_query_tooltip()` or a scoped variant for the specific query), add to layout
    - Show the container
  - Keep `_custom_query_tooltip()` for hover-over-container and/or per-badge hover — on hover the user still needs to see non-match results, so applying it to the container is fine; it's not lost just because no non-match badge renders.

  **Patterns to follow:**
  - Style declaration mirrors `ui/source_thumbnail.py:_update_badge()` exactly (tokens and order) so the two surfaces look identical.
  - Widget cleanup pattern: see `ui/source_browser.py` for how existing widgets are removed from a layout before re-adding (avoid memory leaks in repeated updates).

  **Test scenarios:**
  - Happy path: clip with custom_queries `[{"query": "dog", "match": True}]` → container visible with 1 badge showing "dog", green background
  - Happy path: clip with two matching queries `[{"query": "dog", "match": True}, {"query": "animal", "match": True}]` → container shows 2 badges side-by-side
  - Happy path: clip with mixed results `[{"query": "dog", "match": True}, {"query": "car", "match": False}]` → container shows 1 badge ("dog") only
  - Edge case: clip with all non-match queries `[{"query": "car", "match": False}]` → container hidden, no badges rendered
  - Edge case: clip with no custom_queries → container hidden (regression of existing behavior)
  - Edge case: clip with empty custom_queries list → container hidden
  - Edge case: clip with query where "match" key is missing or None → treated as non-match, not rendered
  - Edge case: repeated `_update_custom_query_badge()` calls (clip selected, re-selected) do not leak widgets — badge count equals current match count after each call
  - Integration: badge tooltip on hover still shows the full YES/NO summary including non-matches (so users can discover hidden non-matches without opening the sidebar)
  - Integration: clip details sidebar's custom query display is unaffected by this change (assert sidebar renders the same text before/after the update)

  **Verification:**
  - Running a custom query "dog" on a set of clips displays a green "dog" badge on matching clip cards and no badge on non-matching ones.
  - Running a second custom query "car" alongside adds a second badge only where that clip matched "car".
  - The Analyze tab and Collect tab badges are visually consistent (same padding, radius, font size, color tokens).

## System-Wide Impact

- **Interaction graph:** Only the Analyze-tab clip card badge render changes. No observer/callback changes, no data model changes.
- **Error propagation:** `get_latest_custom_query_results()` already handles malformed entries by skipping them; no new error paths.
- **State lifecycle risks:** Re-rendering the badge container must clean up the previous badge widgets to avoid memory leaks across repeated updates (scroll, filter change, re-analysis). Covered by the repeat-update test scenario.
- **API surface parity:** Clip details sidebar and any other clip-visualization surfaces remain unchanged.
- **Unchanged invariants:** `Clip.custom_queries` schema, `_custom_query_tooltip()` content, sidebar rendering, analysis pipeline.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Multiple badges overflow narrow cards | Start with simple horizontal flow; if overflow is visible in practice, add wrapping or truncate per-badge text (planning-time decision deferred to implementation). |
| Memory leak from re-adding badge widgets on every update | Explicitly remove/deleteLater existing children before re-adding; covered by test scenario. |
| Visual drift between Collect and Analyze badges over time | Factoring: consider a shared `badge_style()` helper (optional, not required for v1). |

## Documentation / Operational Notes

- `docs/user-guide/analysis.md` mentions Custom Query — update the description if it references the "Query Match" label wording (check during implementation; may be untouched).
- No changelog entry needed (lightweight UI polish).

## Sources & References

- Related code: `ui/clip_browser.py` (target), `ui/source_thumbnail.py` (style reference), `models/clip.py` (Clip.custom_queries data model), `tests/test_custom_query_ui.py` (existing tests)
- User-provided screenshots describing current state and preferred design
