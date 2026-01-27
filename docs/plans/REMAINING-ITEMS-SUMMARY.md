---
title: "Remaining Items Summary"
date: 2026-01-26
---

# Remaining Items Summary

This document consolidates all unchecked items from incomplete planning documents. Review and prioritize as needed.

---

## 1. CLI Interface (Phase 1)

**Source:** `2026-01-25-feat-cli-interface-phase1-plan.md`

### Quality Gates (3 items)

- [ ] Ctrl+C cleanly terminates with partial progress saved
- [ ] `mypy cli/` passes with no errors
- [ ] Integration test: full pipeline from detect to export

---

## 2. Agent-Native Phases 2-3-4

**Source:** `2026-01-25-feat-agent-native-phases-2-3-4-plan.md`

**Status:** Core implementation complete, integration tests pending

### Integration Tests (4 items)

- [ ] CLI can create project, GUI can open it
- [ ] GUI can create project, CLI can modify it
- [ ] Settings changed in CLI are reflected in GUI (after restart)
- [ ] Environment variables work in both CLI and GUI

---

## 3. Agent-Accessible GUI Features

**Source:** `2026-01-25-feat-agent-accessible-gui-features-plan.md`

### Phase 2: Content-Aware Tools (4 items)

- [ ] `search_transcripts(query: str)` - Find clips by spoken content
- [ ] `find_similar_clips(clip_id: str, criteria: list[str])` - Visual similarity search
- [ ] `group_clips_by(criteria: str)` - Group by color/shot_type/duration
- [ ] Extended `filter_clips` with transcript content filtering

### Testing Requirements (3 items)

- [ ] Unit tests for each new tool
- [ ] Integration tests for GUI state synchronization
- [ ] Test duplicate signal delivery guards

---

## 4. Sequence Tab Card-Based Redesign

**Source:** `2026-01-25-feat-sequence-tab-card-based-redesign-plan.md`

**Status:** ✅ COMPLETE - All phases implemented, ready for archive

### Phase 1: Core Card Infrastructure ✅ COMPLETE

- [x] Create `SortingCard` widget following `ClipThumbnail` patterns (`ui/widgets/sorting_card.py`)
- [x] Create `SortingCardGrid` with 2x2 layout for 4 MVP algorithms (`ui/widgets/sorting_card_grid.py`)
- [x] Add card icons and descriptions (color 🎨, duration ⏱️, shuffle 🎲, sequential 📋)
- [x] Implement hover/selected/disabled states with theme support
- [x] Update `SequenceTab` to show card grid when clips available (`ui/tabs/sequence_tab.py`)

### Phase 2: Parameter Panel + Preview ✅ COMPLETE

- [x] Create `SortingParameterPanel` with algorithm-specific controls (`ui/widgets/sorting_parameter_panel.py`)
- [x] Create `TimelinePreview` (simplified read-only timeline) (`ui/widgets/timeline_preview.py`)
- [x] Wire parameter changes to preview updates (debounced 300ms)
- [x] Add "Back" button to return to card grid
- [x] Add "Apply to Timeline" button

### Phase 3: Agent Integration ✅ COMPLETE

- [x] Add `list_sorting_algorithms` tool (`core/chat_tools.py:2726`)
- [x] Add `generate_remix` tool with full parameter support (`core/chat_tools.py:2791`)
- [x] Add `get_remix_state` tool for agent state awareness (`core/chat_tools.py:2865`)
- [x] Agent can: select algorithm → set params → apply via `generate_and_apply()` method
- [x] Guard flags for GUI-modifying operations (`_apply_in_progress` flag in SequenceTab)

### Phase 4: Polish + Edge Cases ✅ COMPLETE

- [x] Handle no-clips state gracefully (STATE_NO_CLIPS with EmptyStateWidget)
- [x] Disable color card when no clips have `dominant_colors` (`_update_card_availability()`)
- [x] Add loading spinner during preview computation (`set_loading()` in TimelinePreview)
- [x] Add keyboard navigation (Tab/Arrow/Enter in SortingCard via `keyPressEvent`)
- [x] Persist selected algorithm when switching tabs (`on_tab_activated/deactivated` handlers)

### Acceptance Criteria ✅ ALL MET

- [x] Sequence Tab shows card grid when clips are available
- [x] Four sorting cards displayed: Color, Duration, Shuffle, Sequential
- [x] Clicking card shows parameter panel and timeline preview
- [x] Parameters include clip count (all algorithms) + algorithm-specific options
- [x] Timeline preview updates within 500ms of parameter change (300ms debounce)
- [x] "Apply" button commits sequence to main timeline
- [x] "Back" button returns to card grid
- [x] Agent can select algorithm via `generate_remix` tool
- [x] Agent can query state via `get_remix_state` tool

### Quality Gates ✅ ALL MET

- [x] No duplicate state objects (`_preview_clips` single source of truth)
- [x] All signal handlers have guard flags (`_apply_in_progress`, `_preview_update_pending`)
- [x] Model/view sync verified in init
- [x] Worker IDs synced before use

**Recommendation:** Move `2026-01-25-feat-sequence-tab-card-based-redesign-plan.md` to `archive/`

---

## 5. Agent Planning Tool

**Source:** `2026-01-25-feat-agent-planning-tool-plan.md`

**Status:** Files marked as complete, but acceptance criteria unchecked

### Acceptance Criteria (12 items)

- [ ] User can say "plan X" and LLM asks clarifying questions
- [ ] LLM calls `present_plan` with step list after getting answers
- [ ] Plan widget displays inline in chat with numbered steps
- [ ] User can edit step text by double-clicking
- [ ] User can reorder steps via drag-and-drop
- [ ] User can delete steps (right-click menu or delete key)
- [ ] Confirm button triggers sequential execution
- [ ] Cancel button before execution dismisses plan
- [ ] Step status indicators update during execution (pending → running → done/failed)
- [ ] Failed steps show error but execution continues
- [ ] Cancel during execution stops after current step
- [ ] Completion summary shows results per step

---

## 6. MCP Server (Phase 5)

**Source:** `2026-01-26-feat-mcp-server-phase-5-plan.md`

**Status:** Core implementation complete (33 tools), quality gates pending

### Functional Requirements

- [ ] Export operations produce expected file outputs

### Non-Functional Requirements

- [ ] Tool timeouts configurable via environment
- [ ] Memory efficient for large video operations

### Quality Gates

- [ ] MCP Inspector shows all tools with correct schemas
- [ ] Integration tests cover happy path for each tool category
- [ ] Claude Desktop integration documented with example config

---

## 7. Clip Details Sidebar (Original Plan)

**Source:** `2026-01-26-feat-clip-details-sidebar-plan.md`

**Note:** The editable version (`2026-01-26-feat-editable-clip-details-sidebar-plan.md`) is complete and archived. This original plan may be superseded.

### Functional Requirements (14 items)

- [ ] Sidebar opens via right-click context menu "View Details"
- [ ] Sidebar opens via double-click on clip card (ClipThumbnail, SortingCard)
- [ ] Sidebar opens via keyboard (Enter or 'i' when clip selected)
- [ ] Video preview plays clip range (start_frame to end_frame)
- [ ] Displays clip title as "source_filename - HH:MM:SS"
- [ ] Displays duration, frame range, source resolution
- [ ] Displays dominant colors as swatches (if analyzed)
- [ ] Displays shot type badge (if analyzed)
- [ ] Displays transcript text (if transcribed)
- [ ] Sidebar content updates when different clip is selected
- [ ] Dismissable via X button
- [ ] Dismissable via Escape key
- [ ] Sidebar persists across tab changes

### Non-Functional Requirements (5 items)

- [ ] Uses theme colors (light/dark mode support)
- [ ] Sidebar width: 350px default, resizable via dock widget
- [ ] Video preview maintains aspect ratio
- [ ] Empty states for missing analysis data ("Not analyzed")
- [ ] Error state for missing source file

---

## Reference Documents (Kept in docs/plans/)

These are reference/roadmap documents, not implementation checklists:

- `agent-native-architecture-plan.md` - Master architecture plan
- `2026-01-24-feat-course-feature-mapping-plan.md` - Course-to-feature roadmap
- `2026-01-25-agent-native-architecture-review.md` - Architecture review (score: 35/50)

---

## Archived Plans (Completed)

Moved to `docs/plans/archive/`:

1. `2026-01-24-feat-project-save-load-plan.md`
2. `2026-01-24-feat-whisper-transcription-plan.md`
3. `2026-01-24-feat-youtube-search-bulk-download-plan.md`
4. `2026-01-24-refactor-split-cut-analyze-tabs-plan.md`
5. `2026-01-25-feat-agent-chatbot-plan.md`
6. `2026-01-25-feat-agent-gui-bidirectional-sync-plan.md`
7. `2026-01-25-feat-chat-example-prompts-plan.md`
8. `2026-01-25-feat-gui-aware-agent-tools-plan.md`
9. `2026-01-25-refactor-move-analysis-to-on-demand-plan.md`
10. `2026-01-26-feat-editable-clip-details-sidebar-plan.md`
11. `2026-01-25-feat-sequence-tab-card-based-redesign-plan.md`

---

## Summary by Priority

| Priority | Plan | Remaining Items |
|----------|------|-----------------|
| Low | CLI Interface | 3 quality gates |
| Low | Agent-Native Phases 2-3-4 | 4 integration tests |
| Medium | Agent-Accessible GUI Features | 7 items (Phase 2 tools + tests) |
| Medium | MCP Server Phase 5 | 6 quality gates |
| Medium | Agent Planning Tool | 12 acceptance criteria |
| ✅ Done | Sequence Tab Redesign | 0 items (ready for archive) |
| Review | Clip Details Sidebar (original) | May be superseded |

**Total remaining items: ~32** (reduced from ~57 after validating Sequence Tab is complete)
