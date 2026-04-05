---
title: "feat: Add semantic clip search and enhanced filtering"
type: feat
status: completed
date: 2026-04-04
origin: docs/brainstorms/2026-04-04-semantic-clip-search-requirements.md
---

# feat: Add semantic clip search and enhanced filtering

## Overview

Expand the ClipBrowser with 4 new filters (gaze, object, brightness, description), add visual similarity search via DINOv2 embeddings, and enhance the agent's `filter_clips` tool to cover all analysis dimensions. This makes every computed analysis dimension searchable via at least one path (UI or agent).

## Problem Frame

Scene Ripper computes 30+ analysis dimensions per clip but only 6 are filterable in the UI and ~10 via the agent. Users cannot find specific clips across sources without manual scanning. (see origin: `docs/brainstorms/2026-04-04-semantic-clip-search-requirements.md`)

## Requirements Trace

- R1. Gaze direction filter dropdown in ClipBrowser
- R2. Object detection text filter in ClipBrowser (substring match against `object_labels` + `detected_objects`)
- R3. Brightness range filter (RangeSlider 0.0-1.0) in ClipBrowser
- R4. Description search text filter in ClipBrowser
- R5. All new filters combine with existing via AND logic
- R6. Filters disabled with tooltip when analysis data missing
- R7. "Find Similar" action on clip thumbnails — shows all clips sorted by DINOv2 embedding similarity
- R8. Similarity mode visual indicator + "Clear Similarity" button
- R9. Exclude clips without valid embeddings (check `norm > 0` for zero-vector fallbacks)
- R10. Expand `filter_clips` with: `gaze_category`, `min_brightness`/`max_brightness`, `search_ocr_text`, `min_volume`/`max_volume`, `search_tags`, `search_notes`, curated `cinematography_*` fields
- R11. Add `similar_to_clip_id` to `filter_clips` for agent-driven similarity search
- R12. Expand `filter_clips` return fields: `average_brightness`, `rms_volume`, `tags`, `notes`, `extracted_texts`, cinematography summary
- R13. Cross-source filtering preserved (existing behavior)
- R14. Source identification in filtered grid (already handled by `SourceGroupHeader`)

## Scope Boundaries

- No new tab — all UI in existing ClipBrowser
- No drag-drop to Sequence
- AND-only filter logic
- No saved searches / smart collections
- No CLIP text-to-image — DINOv2 visual similarity only
- No full-text indexing — substring matching sufficient
- Cinematography sub-fields in agent tool only, not ClipBrowser UI

## Context & Research

### Relevant Code and Patterns

- `ui/clip_browser.py` (~1900 lines) — Two-zone filter layout: header row (shot type, color, transcript, custom query, sort) + collapsible panel (duration slider, aspect ratio, clear button). Filter pattern: state variable → `_matches_filter()` gate → `_do_rebuild_grid()`. Public API: `apply_filters(dict)`, `clear_all_filters()`, `get_active_filters()`.
- `ClipThumbnail` (inline in `clip_browser.py`, lines 107-644) — Clip card widget with context menu (`_build_context_menu()` at line 417) containing "View Details" and "Export Clip..." actions. No existing "Find Similar" action.
- `core/chat_tools.py` — `filter_clips` (lines 586-729) has 11 filter parameters. Returns list of dicts with ~15 fields. Does NOT currently filter by gaze, brightness, volume, OCR, tags, notes, or cinematography.
- `ui/widgets/range_slider.py` — Production `RangeSlider(QWidget)` with `set_range()`, `set_values()`, `values()`, `reset()`, `set_suffix()`. Already used for duration filter in ClipBrowser.
- `core/analysis/gaze.py` — `GAZE_CATEGORY_DISPLAY` maps internal keys (`at_camera`, `looking_left`, etc.) to display labels. Import directly, don't hardcode.
- `core/analysis/embeddings.py` — DINOv2 768-dim, L2-normalized. Zero vector returned for failed extractions (line 161). Similarity = `np.dot(a, b)`.

### Institutional Learnings

- **Duplicate state risk** (`docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`): Similarity mode state must have a single owner (ClipBrowser) — don't track it in both ClipBrowser and ClipThumbnail.
- **Circular import risk** (`docs/solutions/logic-errors/circular-import-config-consolidation.md`): Import gaze constants from `core.analysis.gaze`, not from UI modules.

## Key Technical Decisions

- **New filters go in collapsible panel, not header row**: Header row already has 8 controls (label, filters toggle, expand/collapse, shot, color, transcript search, custom query, sort). Adding 4 more would overflow. Gaze dropdown, object search, description search, and brightness slider all go in the collapsible filter panel alongside the existing duration slider and aspect ratio filter.
- **Similarity shows all clips sorted, not top-N**: A fixed N (e.g., 10) is arbitrary and forces a cutoff that users may disagree with. Showing all clips sorted by similarity score lets users decide where the meaningful boundary is. The existing grid already handles variable clip counts.
- **Similarity mode is orthogonal to filters**: When similarity mode is active, clips are pre-sorted by similarity score, but other active filters still apply (AND logic). Clearing similarity restores the previous filter state and sort order.
- **Object search matches both label sources**: `object_labels` (ImageNet classification) and `detected_objects[].label` (YOLO) use different vocabularies but both contain useful object names. Searching both gives broader coverage. This matches the existing `has_object` behavior in `filter_clips`.
- **Upgrade existing `has_object` to substring matching**: The existing agent tool parameter does exact label membership. Change to case-insensitive substring to match R2's UI behavior. This is a minor behavioral change but makes the UI and agent tool consistent.
- **Curated cinematography sub-fields for agent**: Expose 7 most actionable fields: `shot_size`, `camera_angle`, `camera_movement`, `lighting_style`, `subject_count`, `emotional_intensity`, `suggested_pacing`. The remaining ~13 fields are too granular for typical agent queries.
- **R14 already satisfied**: `SourceGroupHeader` already groups clips by source filename in the grid. No additional work needed. Similarity mode should maintain source grouping.
- **Filter disabled state by data presence**: A filter control is enabled if ANY clip in the project has the relevant field populated. This avoids disabling filters that would still match some clips.

## Open Questions

### Resolved During Planning

- **Similarity result count**: Show all clips sorted by similarity (not top-N). Grid naturally handles any count.
- **Object search sources**: Match both `object_labels` and `detected_objects[].label` (matches existing `has_object` behavior).
- **RangeSlider for brightness**: Reuse existing `RangeSlider(0.0, 1.0)` from `ui/widgets/range_slider.py` with no suffix.
- **Cinematography sub-fields**: 7 curated fields: `shot_size`, `camera_angle`, `camera_movement`, `lighting_style`, `subject_count`, `emotional_intensity`, `suggested_pacing`.
- **R14 source identification**: Already handled by `SourceGroupHeader` — no new work.
- **New filter placement**: Collapsible panel (Zone 2), not header row.
- **Similarity + active filters interaction**: Similarity is an additional sort/rank overlay. Existing filters still apply as AND gates. Clearing similarity restores previous sort order.

### Deferred to Implementation

- **Exact similarity score display**: Whether to show a numeric score or percentage on each card during similarity mode.
- **Performance of all-clip similarity**: For projects with 500+ clips, computing pairwise dot products may need optimization (numpy vectorized). Profile during implementation.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
ClipBrowser state additions:
  _gaze_filter: str = "All"
  _object_search: str = ""
  _description_search: str = ""
  _min_brightness: Optional[float] = None
  _max_brightness: Optional[float] = None
  _similarity_anchor_id: Optional[str] = None
  _similarity_scores: dict[str, float] = {}  # clip_id -> score

_matches_filter(thumb) additions:
  if _gaze_filter != "All": check clip.gaze_category
  if _object_search: check object_labels + detected_objects
  if _description_search: check clip.description
  if _min/_max_brightness: check clip.average_brightness
  if _similarity_anchor_id: check clip_id in _similarity_scores

_do_rebuild_grid() modification:
  if _similarity_anchor_id:
    sort visible_thumbs by _similarity_scores[clip.id] descending
    (maintain source grouping within similarity mode)

filter_clips tool additions:
  new params: gaze_category, min/max_brightness, search_ocr_text,
              min/max_volume, search_tags, search_notes,
              cinematography_* (7 fields), similar_to_clip_id
  new return fields: average_brightness, rms_volume, tags, notes,
                     combined_text, cinematography dict
```

## Implementation Units

### Phase 1: ClipBrowser Filter Expansion

- [ ] **Unit 1: Add 4 new filters to ClipBrowser collapsible panel**

  **Goal:** Add gaze direction dropdown, object search text input, description search text input, and brightness range slider to the collapsible filter panel. Wire them through the existing `_matches_filter` → `_do_rebuild_grid` pipeline.

  **Requirements:** R1, R2, R3, R4, R5

  **Dependencies:** None

  **Files:**
  - Modify: `ui/clip_browser.py`
  - Create: `tests/test_clip_browser_filters.py`

  **Approach:**
  - Add 4 new state variables: `_gaze_filter`, `_object_search`, `_description_search`, `_min_brightness`/`_max_brightness`
  - Add UI controls to the collapsible filter panel (`_setup_filter_panel` or wherever Zone 2 is built):
    - Gaze: `QLabel("Gaze:") + QComboBox` populated from `GAZE_CATEGORY_DISPLAY` + "All Gaze" default. Import display names from `core.analysis.gaze`.
    - Object: `QLabel("Object:") + QLineEdit` with placeholder "Search objects..."
    - Description: `QLabel("Description:") + QLineEdit` with placeholder "Search descriptions..."
    - Brightness: `QLabel("Brightness:") + RangeSlider(0.0, 1.0)` with no suffix
  - Add 4 new gates to `_matches_filter()`:
    - Gaze: if not "All Gaze", compare `clip.gaze_category` against selected internal key
    - Object: case-insensitive substring against `clip.object_labels` (join) + `clip.detected_objects[].label`
    - Description: case-insensitive substring against `clip.description`
    - Brightness: compare `clip.average_brightness` against min/max bounds
  - Update `apply_filters()` to accept new keys: `gaze`, `object_search`, `description_search`, `min_brightness`, `max_brightness`
  - Update `clear_all_filters()` to reset new state and controls
  - Update `get_active_filters()` to include new filters
  - Connect control signals (combo `currentTextChanged`, line edit `textChanged`, slider `range_changed`) to `_do_rebuild_grid` via existing pattern

  **Patterns to follow:**
  - Shot type filter: `_current_filter` state + `filter_combo` QComboBox + gate in `_matches_filter` (lines 1558-1568)
  - Transcript search: `_current_search_query` state + `search_input` QLineEdit + gate (lines 1575-1580)
  - Duration slider: `_min_duration`/`_max_duration` state + `duration_slider` RangeSlider + gate (lines 1590-1599)

  **Test scenarios:**
  - Happy path: Set gaze filter to "At Camera" → only clips with `gaze_category == "at_camera"` shown
  - Happy path: Type "person" in object search → clips with "person" in object_labels or detected_objects shown
  - Happy path: Set brightness range to 0.5-1.0 → only clips with `average_brightness >= 0.5` shown
  - Happy path: Type "sunset" in description search → clips with "sunset" in description shown
  - Integration: Set gaze="At Camera" AND brightness 0.5-1.0 → only clips matching BOTH shown
  - Edge case: Clip with no `gaze_category` doesn't match any gaze filter except "All Gaze"
  - Edge case: Clip with no `average_brightness` is excluded when brightness filter has non-default range
  - Edge case: Clear all filters resets all 4 new controls and shows all clips
  - Happy path: `get_active_filters()` includes new filter keys when set

  **Verification:**
  - All 4 new filters narrow the visible clip set
  - Filters combine with existing filters via AND logic
  - `clear_all_filters()` resets everything

- [ ] **Unit 2: Add disabled state for filters requiring unanalyzed data**

  **Goal:** Disable filter controls when no clip has the required analysis data, with a tooltip explaining what's needed.

  **Requirements:** R6

  **Dependencies:** Unit 1

  **Files:**
  - Modify: `ui/clip_browser.py`
  - Modify: `tests/test_clip_browser_filters.py`

  **Approach:**
  - Add a `_update_filter_availability()` method that checks whether ANY clip in the current project has the relevant field populated:
    - Gaze dropdown: enabled if any clip has `gaze_category`
    - Object search: enabled if any clip has `object_labels` or `detected_objects`
    - Brightness slider: enabled if any clip has `average_brightness`
    - Description search: enabled if any clip has `description`
  - When disabled: `setEnabled(False)` + `setToolTip("Requires [analysis name] — run it in the Analyze tab")`
  - Call `_update_filter_availability()` when clips change (connect to the same signals that trigger `_update_card_availability` in the sequence tab, or call after `add_clip`/`remove_clip`)

  **Patterns to follow:**
  - `SortingCard.set_enabled(enabled, reason)` for the tooltip pattern
  - `SortingCardGrid.set_algorithm_availability()` for the batch-update pattern

  **Test scenarios:**
  - Happy path: No clips have gaze data → gaze dropdown disabled with tooltip
  - Happy path: One clip gets gaze data → gaze dropdown becomes enabled
  - Edge case: All clips removed → all analysis-dependent filters disabled
  - Happy path: Brightness slider disabled when no clip has `average_brightness`

  **Verification:**
  - Filter controls are disabled when no clip has the required analysis
  - Tooltip explains which analysis is needed

### Phase 2: Visual Similarity Search

- [ ] **Unit 3: Add "Find Similar" to ClipThumbnail and similarity mode to ClipBrowser**

  **Goal:** Add a "Find Similar" context menu action on clip thumbnails that activates similarity mode in ClipBrowser — sorting all clips by visual similarity to the anchor clip, with a clear button to exit.

  **Requirements:** R7, R8, R9

  **Dependencies:** None (parallel with Phase 1)

  **Files:**
  - Modify: `ui/clip_browser.py` (both `ClipThumbnail` and `ClipBrowser` classes)
  - Create: `tests/test_clip_browser_similarity.py`

  **Approach:**
  - **ClipThumbnail changes:**
    - Add `find_similar_requested = Signal(object)` signal (emits Clip)
    - Add "Find Similar" action to `_build_context_menu()` after "Export Clip..."
    - Connect to emit `find_similar_requested`
  - **ClipBrowser state additions:**
    - `_similarity_anchor_id: Optional[str] = None`
    - `_similarity_scores: dict[str, float] = {}` — clip_id → similarity score
  - **`_activate_similarity(clip: Clip)` method:**
    - Validate source clip has valid embedding (`embedding is not None` and `np.linalg.norm(embedding) > 0`)
    - Compute dot product of anchor embedding against all clips with valid embeddings
    - Store scores in `_similarity_scores`
    - Set `_similarity_anchor_id`
    - Show "Clear Similarity" button (add to filter panel, hidden by default)
    - Highlight anchor clip card (e.g., accent border or badge)
    - Call `_do_rebuild_grid()`
  - **`_matches_filter` modification:**
    - If `_similarity_anchor_id` is set, exclude clips not in `_similarity_scores` (those without valid embeddings)
  - **`_do_rebuild_grid` modification:**
    - If `_similarity_anchor_id` is set, sort `visible_thumbs` within each source group by `_similarity_scores[clip.id]` descending (most similar first)
  - **`_clear_similarity()` method:**
    - Reset `_similarity_anchor_id` and `_similarity_scores`
    - Hide "Clear Similarity" button
    - Remove anchor highlight
    - Call `_do_rebuild_grid()` to restore normal sort order
  - **Integration:** Connect `thumb.find_similar_requested` signal in the thumbnail creation path (wherever ClipBrowser creates `ClipThumbnail` instances)

  **Patterns to follow:**
  - `_build_context_menu()` for adding actions (line 417)
  - `_matches_filter()` for the AND gate pattern
  - `_do_rebuild_grid()` for sort logic (currently sorts by timeline/color/duration)

  **Test scenarios:**
  - Happy path: "Find Similar" on a clip with embeddings → all clips sorted by similarity, most similar first
  - Happy path: "Clear Similarity" restores previous sort order
  - Edge case: "Find Similar" on a clip without embedding → no-op or show message
  - Edge case: Zero-vector embedding clips excluded from results (norm check)
  - Integration: Active filters + similarity mode — only clips matching filters AND having embeddings shown, sorted by similarity
  - Edge case: Clearing similarity while other filters are active preserves those filters
  - Happy path: Anchor clip visually indicated in the grid

  **Verification:**
  - "Find Similar" computes and displays similarity ranking
  - Similarity mode combines with existing filters
  - "Clear Similarity" exits cleanly

### Phase 3: Agent Tool Enhancement

- [ ] **Unit 4: Expand filter_clips with new parameters**

  **Goal:** Add filter parameters for all missing analysis dimensions: gaze, brightness, volume, OCR text, tags, notes, cinematography, and embedding similarity.

  **Requirements:** R10, R11

  **Dependencies:** None (parallel with Phases 1-2)

  **Files:**
  - Modify: `core/chat_tools.py`
  - Modify: `tests/test_chat_tools.py` (or create `tests/test_filter_clips.py`)

  **Approach:**
  - Add new parameters to `filter_clips`:
    - `gaze_category: Optional[str]` — exact match on `clip.gaze_category`
    - `min_brightness: Optional[float]`, `max_brightness: Optional[float]` — range gate on `clip.average_brightness`
    - `search_ocr_text: Optional[str]` — case-insensitive substring on `clip.combined_text`
    - `min_volume: Optional[float]`, `max_volume: Optional[float]` — range gate on `clip.rms_volume`
    - `search_tags: Optional[str]` — case-insensitive substring against `clip.tags` (join with space)
    - `search_notes: Optional[str]` — case-insensitive substring against `clip.notes`
    - `cinematography_shot_size: Optional[str]` — exact match on `clip.cinematography.shot_size`
    - `cinematography_camera_angle: Optional[str]` — exact match
    - `cinematography_camera_movement: Optional[str]` — exact match
    - `cinematography_lighting_style: Optional[str]` — exact match
    - `cinematography_subject_count: Optional[str]` — exact match
    - `cinematography_emotional_intensity: Optional[str]` — exact match
    - `cinematography_suggested_pacing: Optional[str]` — exact match
    - `similar_to_clip_id: Optional[str]` — returns clips ranked by embedding similarity (R11)
  - Upgrade existing `has_object` from exact label membership to case-insensitive substring matching (aligns with R2 UI behavior)
  - Update the tool's `@tools.register` description to list all new parameters
  - For `similar_to_clip_id`: look up anchor clip, validate embedding, compute dot products, sort results by score descending. Combinable with other filters (filter first, then rank).

  **Patterns to follow:**
  - Existing `shot_type`, `search_description`, `has_object` filter logic in `filter_clips`
  - Existing `_ASPECT_RATIO_RANGES` pattern for validated enum values

  **Test scenarios:**
  - Happy path: `gaze_category="at_camera"` returns only clips with matching gaze
  - Happy path: `min_brightness=0.5` filters out dark clips
  - Happy path: `search_ocr_text="EXIT"` finds clips with "EXIT" in OCR text
  - Happy path: `min_volume=-20` filters out quiet clips
  - Happy path: `search_tags="landscape"` finds clips tagged "landscape"
  - Happy path: `cinematography_lighting_style="dramatic"` filters on cinematography
  - Happy path: `similar_to_clip_id="abc123"` returns clips sorted by similarity
  - Edge case: `similar_to_clip_id` with invalid clip ID returns error
  - Edge case: `similar_to_clip_id` combined with `gaze_category` — filters by gaze first, then ranks by similarity
  - Edge case: Clip with no cinematography data passes through cinematography filters (filter only applies to clips that have the data)
  - Happy path: Upgraded `has_object` uses substring matching (searching "car" matches "car" and "racecar")

  **Verification:**
  - All new parameters filter correctly
  - Similarity ranking works and combines with filters
  - Backward compatible — existing parameters unchanged

- [ ] **Unit 5: Expand filter_clips return fields**

  **Goal:** Add missing metadata fields to `filter_clips` results so the agent can reason about clips without follow-up calls.

  **Requirements:** R12

  **Dependencies:** Unit 4

  **Files:**
  - Modify: `core/chat_tools.py`
  - Modify: `tests/test_chat_tools.py` (or `tests/test_filter_clips.py`)

  **Approach:**
  - Add to each result dict:
    - `"average_brightness"`: `clip.average_brightness` (float or None)
    - `"rms_volume"`: `clip.rms_volume` (float or None)
    - `"tags"`: `clip.tags` (list or [])
    - `"notes"`: `clip.notes` (str or "")
    - `"extracted_text"`: `clip.combined_text` (str or None)
    - `"cinematography"`: dict of 7 curated fields from `clip.cinematography` (or None)
    - `"similarity_score"`: float (only when `similar_to_clip_id` is used)
  - Do NOT include raw embedding vectors (768 floats per clip would bloat responses)

  **Patterns to follow:**
  - Existing result dict construction in `filter_clips` (lines 706-727)
  - Conditional field inclusion pattern for gaze fields (lines 721-726)

  **Test scenarios:**
  - Happy path: Result includes `average_brightness` when clip has it
  - Happy path: Result includes `cinematography` dict with 7 fields
  - Happy path: Result includes `tags` as list
  - Edge case: Clip with no cinematography returns `cinematography: None`
  - Happy path: `similarity_score` included only when `similar_to_clip_id` is used

  **Verification:**
  - Agent can read brightness, volume, tags, notes, OCR text, and cinematography from results
  - No embedding vectors in results

## System-Wide Impact

- **Interaction graph:** ClipBrowser → ClipThumbnail (new signal), ClipBrowser → `_matches_filter` (4 new gates + similarity gate), `filter_clips` tool (new params + return fields). No other tabs, workers, or systems affected.
- **Error propagation:** Invalid embeddings (zero-vector) → excluded from similarity results via norm check. Missing analysis fields → filter treated as non-matching (clip excluded from results for that filter dimension).
- **State lifecycle risks:** Similarity mode state (`_similarity_anchor_id`, `_similarity_scores`) lives solely on ClipBrowser. Cleared on `clear_all_filters()` and when the anchor clip is deleted.
- **API surface parity:** Agent `filter_clips` gains parity with UI filters. Cinematography goes beyond UI (agent-only per scope).
- **Unchanged invariants:** ClipBrowser selection, drag behavior, clip card rendering, source grouping, sort options — all unchanged. Existing `filter_clips` parameters backward compatible.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Collapsible filter panel gets crowded with 4 new controls | Use 2-column grid layout within the panel instead of single row. Group related controls. |
| Embedding similarity computation slow for 500+ clips | numpy vectorized dot product is O(n) and fast for thousands of vectors. Profile during implementation. |
| `has_object` behavior change (exact → substring) could break agent workflows | Substring is strictly more permissive — existing queries that worked with exact match still work. |
| Zero-vector embeddings silently included in similarity results | R9 explicitly requires norm check. Test scenario covers this. |

## Future Considerations

- **CLIP text-to-image search**: Add a text encoder to enable natural language → embedding similarity. DINOv2 similarity is visual-only.
- **Saved searches / smart collections**: Persist filter combinations for one-click access.
- **Face identity search**: "Find all clips with this person" using face embeddings. Complex due to multi-face handling.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-04-semantic-clip-search-requirements.md](docs/brainstorms/2026-04-04-semantic-clip-search-requirements.md)
- Related code: `ui/clip_browser.py`, `core/chat_tools.py`, `ui/widgets/range_slider.py`, `core/analysis/gaze.py`, `core/analysis/embeddings.py`
- Related learnings: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`, `docs/solutions/logic-errors/circular-import-config-consolidation.md`
