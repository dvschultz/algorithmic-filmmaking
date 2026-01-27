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

**Status:** ✅ COMPLETE (with minor UX variations from original spec)

### Implementation Details

- `present_plan` tool: `core/chat_tools.py:298`
- `PlanStepWidget`: `ui/chat_widgets.py:737` (268 lines)
- `PlanWidget`: `ui/chat_widgets.py:1007` (150+ lines)
- `Plan` model: `models/plan.py` (201 lines)
- Execution flow: `ui/main_window.py:1456-1655`

### Acceptance Criteria (12 items)

- [x] User can say "plan X" and LLM asks clarifying questions
- [x] LLM calls `present_plan` with step list after getting answers
- [x] Plan widget displays inline in chat with numbered steps
- [x] User can edit step text by double-clicking (`_on_double_click` in PlanStepWidget)
- [x] User can reorder steps via up/down buttons (buttons instead of drag-drop)
- [x] User can delete steps via delete button (button instead of right-click/keyboard)
- [x] Confirm button triggers sequential execution (`_on_confirm` → `_on_plan_confirmed`)
- [x] Cancel button before execution dismisses plan (`_on_plan_cancelled`)
- [x] Step status indicators update during execution (`update_step_status`)
- [x] Failed steps show error with retry option (`retry_btn`, `_on_plan_retry_requested`)
- [x] Cancel during execution stops after current step (`_on_plan_stop_requested`)
- [x] Completion summary shows results per step (progress tracking in Plan model)

**Note:** Reordering uses ▲/▼ buttons instead of drag-and-drop. Delete uses button instead of right-click menu/keyboard. Core functionality complete.

**Recommendation:** Move `2026-01-25-feat-agent-planning-tool-plan.md` to `archive/`

---

## 6. MCP Server (Phase 5)

**Source:** `2026-01-26-feat-mcp-server-phase-5-plan.md`

**Status:** ✅ COMPLETE - 33 tools, full test coverage, documentation complete

### Implementation Details

- Server: `scene_ripper_mcp/server.py`
- Tools: `scene_ripper_mcp/tools/` (6 modules: project, analyze, clips, sequence, export, youtube)
- Tests: `scene_ripper_mcp/tests/test_integration.py` (479 lines)
- Docs: `docs/mcp-claude-desktop-setup.md` (230 lines)

### Functional Requirements

- [x] Export operations produce expected file outputs (`TestExportTools.test_export_edl`, `test_export_dataset`)

### Non-Functional Requirements

- [x] Tool timeouts configurable via environment (`MCP_TOOL_TIMEOUT`, documented in setup guide)
- [x] Memory efficient for large video operations (progress reporting, streaming FFmpeg)

### Quality Gates

- [x] MCP Inspector shows all tools with correct schemas (`TestToolSchemas.test_all_tools_have_schemas` verifies 33 tools)
- [x] Integration tests cover happy path for each tool category (Project, Clips, Sequence, YouTube, Export, Security)
- [x] Claude Desktop integration documented with example config (`docs/mcp-claude-desktop-setup.md`)

**Recommendation:** Move `2026-01-26-feat-mcp-server-phase-5-plan.md` to `archive/`

---

## 7. Clip Details Sidebar (Original Plan)

**Source:** `2026-01-26-feat-clip-details-sidebar-plan.md`

**Status:** ✅ SUPERSEDED - Editable version implemented all criteria plus editing capabilities

The editable version (`2026-01-26-feat-editable-clip-details-sidebar-plan.md`) was implemented and archived. It includes all features from the original plan plus editable name, shot type, and transcript fields.

### Implementation Details

- Sidebar: `ui/clip_details_sidebar.py` (519 lines)
- Supporting widgets: `ui/widgets/editable_label.py`, `ui/widgets/shot_type_dropdown.py`, `ui/widgets/editable_transcript.py`
- Integration: `ui/main_window.py:1105` (`show_clip_details` method)
- Triggers: `ui/clip_browser.py` (right-click:268, double-click:261, keyboard:1095)

### Functional Requirements ✅ ALL MET

- [x] Sidebar opens via right-click context menu "View Details" (`clip_browser.py:268-275`)
- [x] Sidebar opens via double-click on ClipThumbnail (`clip_browser.py:261-266`)
- [x] Sidebar opens via keyboard Enter/'i' when clip selected (`clip_browser.py:1095-1112`)
- [x] Video preview plays clip range with `set_clip_range()` (`clip_details_sidebar.py:351-359`)
- [x] Displays clip title as "source_filename at HH:MM:SS" (`clip_details_sidebar.py:316-322`)
- [x] Displays duration, frame range, source resolution (`clip_details_sidebar.py:324-332`)
- [x] Displays dominant colors as swatches (`clip_details_sidebar.py:419-450`)
- [x] Displays shot type via editable dropdown (exceeds original plan)
- [x] Displays transcript via editable widget (exceeds original plan)
- [x] Sidebar content updates when different clip selected (`show_clip` method)
- [x] Dismissable via X button (QDockWidget built-in)
- [x] Dismissable via Escape key (`clip_details_sidebar.py:514-516`)
- [x] Sidebar persists across tab changes (QDockWidget at main window level)

### Non-Functional Requirements ✅ ALL MET

- [x] Uses theme colors with light/dark mode (`_refresh_theme` connected to `theme().changed`)
- [x] Sidebar width: 400-550px, resizable via dock widget (`setMinimumWidth(400)`, `setMaximumWidth(550)`)
- [x] Video preview maintains aspect ratio (Qt VideoPlayer handles this)
- [x] Empty states for missing analysis data ("Not analyzed" in `_update_colors`)
- [x] Error state for missing source file (`_show_missing_file_state`)

**Note:** Original plan mentioned SortingCard for double-click, but SortingCard is for sorting algorithms, not clips. ClipThumbnail has full support.

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
7. `2026-01-25-feat-agent-planning-tool-plan.md`
8. `2026-01-25-feat-chat-example-prompts-plan.md`
9. `2026-01-25-feat-gui-aware-agent-tools-plan.md`
10. `2026-01-25-feat-sequence-tab-card-based-redesign-plan.md`
11. `2026-01-25-refactor-move-analysis-to-on-demand-plan.md`
12. `2026-01-26-feat-editable-clip-details-sidebar-plan.md`
13. `2026-01-26-feat-mcp-server-phase-5-plan.md`

---

## Summary by Priority

| Priority | Plan | Remaining Items |
|----------|------|-----------------|
| Low | CLI Interface | 3 quality gates |
| Low | Agent-Native Phases 2-3-4 | 4 integration tests |
| Medium | Agent-Accessible GUI Features | 7 items (Phase 2 tools + tests) |
| ✅ Done | MCP Server Phase 5 | 0 items |
| ✅ Done | Agent Planning Tool | 0 items |
| ✅ Done | Sequence Tab Redesign | 0 items |
| ✅ Done | Clip Details Sidebar | 0 items (superseded by editable version) |

**Total remaining items: ~14** (CLI quality gates + integration tests + agent-accessible features)
