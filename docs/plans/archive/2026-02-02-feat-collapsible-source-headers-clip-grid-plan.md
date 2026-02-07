---
title: "feat: Collapsible Source Headers in Clip Grid"
type: feat
date: 2026-02-02
---

# Collapsible Source Headers in Clip Grid

## Overview

Add collapsible source video headers to the Cut and Analyze tabs' clip grids. Clips will be organized by their parent video, with each group having a clickable header showing the source filename. Filmmakers can collapse groups to focus on specific sections of their material.

## Problem Statement / Motivation

When working with clips from multiple source videos, the current flat grid layout makes it difficult to:
- Identify which clips belong to which source
- Focus on clips from a specific video
- Navigate large collections efficiently

Filmmakers often want to work on clips from one source at a time while keeping other sources accessible but minimized.

## Proposed Solution

Add source-based grouping to ClipBrowser with collapsible headers:

```
┌─────────────────────────────────────────────────────────┐
│ [Filters ▼] [Sort ▼]                                    │
├─────────────────────────────────────────────────────────┤
│ ▼ interview_day1.mp4 (24 clips)                         │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                        │
│ │ C01 │ │ C02 │ │ C03 │ │ C04 │                        │
│ └─────┘ └─────┘ └─────┘ └─────┘                        │
│ ┌─────┐ ┌─────┐ ...                                    │
│ │ C05 │ │ C06 │                                        │
│ └─────┘ └─────┘                                        │
├─────────────────────────────────────────────────────────┤
│ ▶ broll_footage.mp4 (12 clips) [collapsed]              │
├─────────────────────────────────────────────────────────┤
│ ▶ drone_shots.mp4 (8 clips) [collapsed]                 │
└─────────────────────────────────────────────────────────┘
```

## Technical Approach

### Architecture

The implementation modifies `ClipBrowser` to support grouped display while maintaining backward compatibility with existing features.

**Key Insight from Codebase**: ClipBrowser already has a collapsible filter panel pattern using `setVisible()` toggles. We'll extend this pattern for source groups.

**Pattern from Institutional Learnings**: Use `source.id` as the stable grouping key (not object identity) to avoid state mismatch bugs.

### Components

#### 1. New Widget: `SourceGroupHeader`

Location: `ui/widgets/source_group_header.py`

```python
class SourceGroupHeader(QFrame):
    """Clickable header for a source group in ClipBrowser."""

    toggled = Signal(str, bool)  # (source_id, is_expanded)

    def __init__(self, source_id: str, filename: str, clip_count: int):
        self.source_id = source_id
        self.is_expanded = True
        # Visual: chevron icon + filename + clip count badge
```

Styling follows existing card patterns from `ui/widgets/sorting_card.py`:
- Background: subtle contrast from grid background
- Hover state: slight highlight
- Click feedback: pressed state

#### 2. Modified: `ClipBrowser._rebuild_grid()`

Current flow:
```
1. Clear grid
2. Filter thumbnails by _matches_filter()
3. Add thumbnails in sort order (4-column grid)
```

New flow:
```
1. Clear grid
2. Group thumbnails by source_id
3. For each source group (sorted by filename):
   a. Add SourceGroupHeader (spans 4 columns)
   b. If expanded: add thumbnails in 4-column grid
   c. If collapsed: skip thumbnails (header only)
4. Apply filters within each group
```

#### 3. State Management

Add to `ClipBrowser`:

```python
# New instance variables
self._group_expanded_state: dict[str, bool] = {}  # source_id -> is_expanded
self._clips_by_source: dict[str, list[ClipThumbnail]] = {}

# Public API
def toggle_source_group(self, source_id: str) -> None:
    """Toggle collapse state for a source group."""

def expand_all_groups(self) -> None:
    """Expand all source groups."""

def collapse_all_groups(self) -> None:
    """Collapse all source groups."""
```

### File Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `ui/widgets/source_group_header.py` | **NEW** | Collapsible header widget |
| `ui/clip_browser.py` | Modify | Add grouping logic to `_rebuild_grid()`, state tracking |
| `ui/theme.py` | Modify | Add `SOURCE_HEADER_HEIGHT`, header colors |
| `ui/tabs/cut_tab.py` | Minor | No changes required (ClipBrowser handles internally) |
| `ui/tabs/analyze_tab.py` | Minor | No changes required |

### Integration Points

**Existing signal flow preserved:**
```
ClipThumbnail.clicked → ClipBrowser._on_thumbnail_clicked → clip_selected signal → Tab
```

**New signal flow for headers:**
```
SourceGroupHeader.clicked → ClipBrowser._on_header_toggled → _rebuild_grid()
```

## Acceptance Criteria

### Functional Requirements

- [x] Clips in Cut tab are grouped by source video filename
- [x] Clips in Analyze tab are grouped by source video filename
- [x] Each group has a header showing: chevron icon, filename, clip count
- [x] Clicking a header toggles the group between expanded/collapsed
- [x] Collapsed groups show only the header row (no clip thumbnails)
- [x] Expanded groups show all clips in 4-column grid
- [x] Groups are sorted alphabetically by filename
- [x] Collapse state persists while navigating between tabs in same session

### Filter/Sort Integration

- [x] Filters apply within each group (matching clips shown, non-matching hidden)
- [x] Clip count in header updates to reflect filtered count (e.g., "3 of 24")
- [x] Empty groups after filtering show "(0 matching)" but remain visible
- [x] Sort order applies within each group, not across groups

### Selection Behavior

- [x] Selection state is preserved when collapsing a group
- [x] Collapsed header shows selection indicator if any clips selected (e.g., "24 clips • 3 selected")
- [x] Multi-select across groups continues to work

### Edge Cases

- [x] Single source: shows one group header (not redundant)
- [x] Empty grid: no headers shown
- [x] Source with 1 clip: shows header with single clip
- [x] Source removal: group disappears, no stale state

### Non-Functional Requirements

- [x] Collapse/expand is instant (no animation for MVP)
- [x] Grid rebuild with 100+ clips and 10 sources completes in <100ms
- [x] Keyboard accessible: Tab to headers, Enter/Space to toggle

## Implementation Phases

### Phase 1: Core Grouping (MVP)

**Deliverables:**
- `SourceGroupHeader` widget
- `_rebuild_grid()` grouping logic
- Basic expand/collapse toggle
- Theme constants

**Files:**
- `ui/widgets/source_group_header.py` (new)
- `ui/clip_browser.py` (modify)
- `ui/theme.py` (modify)

### Phase 2: Filter Integration

**Deliverables:**
- Update `_matches_filter()` to work with groups
- Header clip count reflects filtered count
- Empty group handling

**Files:**
- `ui/clip_browser.py` (modify)

### Phase 3: Selection & Polish

**Deliverables:**
- Selection indicator on collapsed headers
- Expand All / Collapse All buttons (optional)
- Keyboard navigation

**Files:**
- `ui/widgets/source_group_header.py` (modify)
- `ui/clip_browser.py` (modify)

## Design Decisions

### Decision 1: Collapsed State Appearance

**Chosen**: Header bar only (no mini-thumbnails)

Collapsed group shows only the header row with:
- Chevron icon (▶) pointing right
- Source filename
- Clip count badge

**Rationale**: Maximizes space savings, simplest implementation, matches existing filter panel pattern.

### Decision 2: Sort Within Groups

**Chosen**: Sort applies within each group, groups sorted alphabetically by filename

**Rationale**: Maintains source organization as the primary grouping while allowing secondary sorting within each source's clips.

### Decision 3: State Persistence

**Chosen**: Session-only (not persisted to project file)

Collapse state resets when:
- Application closes
- Project is closed/opened

**Rationale**: Simplifies implementation. Can add persistence later if users request it.

### Decision 4: Cut Tab Sidebar Interaction

**Chosen**: Grouping is independent of sidebar source selection

The sidebar in Cut tab selects which source to detect scenes from. The grouped grid shows ALL detected clips from ALL sources, organized by their origin.

**Rationale**: These are different use cases. Sidebar = "which video to process". Grid grouping = "how to organize processed clips".

## Dependencies & Prerequisites

- None - uses existing ClipBrowser infrastructure

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance with many groups | Low | Medium | Lazy-load collapsed groups (hide widgets, not destroy) |
| Conflicts with existing sort | Medium | High | Clear documentation that sort is within-group only |
| User confusion with Cut tab sidebar | Low | Low | Tooltip explaining difference |

## References

### Internal References

- Collapsible pattern: `ui/clip_browser.py:985-989` (filter panel toggle)
- Clip-source relationship: `models/clip.py:158-185`
- Grid layout: `ui/clip_browser.py:868-893`
- Theme sizing: `ui/theme.py` (UISizes)
- Card styling: `ui/widgets/sorting_card.py`

### Institutional Learnings

- Source ID stability: `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
- Single state ownership: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`

### Project Conventions

From CLAUDE.md:
- Use `UISizes` constants for consistent sizing
- Follow existing signal/slot patterns
- Wrap long content in `QScrollArea` with `setWidgetResizable(True)`
