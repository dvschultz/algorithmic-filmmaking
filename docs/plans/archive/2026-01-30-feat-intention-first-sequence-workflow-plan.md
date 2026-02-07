---
title: "feat: Intention-First Sequence Workflow"
type: feat
date: 2026-01-30
---

# Intention-First Sequence Workflow

## Overview

Enable filmmakers to start from the Sequence tab with an intention (e.g., "I want a color-sorted montage") and be guided through importing, detecting, and analyzing video—reversing the traditional Collect → Cut → Analyze → Sequence flow.

When a user clicks any sequence card (Color, Duration, Shuffle, Sequential) and no clips exist in the project, an import modal appears. After import, videos are automatically cut into clips, analyzed based on the sequence type's requirements, and populated into the timeline.

## Problem Statement / Motivation

**Current pain point:** Users must navigate through 3-4 tabs before seeing their creative output. The workflow assumes users start with video and discover what they can make. But many filmmakers start with an *intention* ("I want to create a color-sorted montage") and want the app to guide them there.

**User story:** As a filmmaker, I want to select "Color" sequence first, then be prompted to import my footage, so I can start with my creative goal and let the app handle the technical steps.

## Proposed Solution

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     SEQUENCE TAB                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │  Color  │ │Duration │ │ Shuffle │ │Sequential│           │
│  │   🌈    │ │   ⏱️    │ │   🎲    │ │   📋    │           │
│  └────┬────┘ └─────────┘ └─────────┘ └─────────┘           │
│       │                                                     │
│       ▼ (no clips exist)                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              IMPORT MODAL                            │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │     Drag & drop videos here                 │    │   │
│  │  │         or click to browse                  │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                                                      │   │
│  │  Or enter YouTube/Vimeo URLs (one per line):        │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │ https://youtube.com/watch?v=...            │    │   │
│  │  │ https://vimeo.com/...                       │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                                                      │   │
│  │  [Cancel]                        [Start Import]      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING (same modal)                   │
│                                                             │
│   Creating your Color sequence...                          │
│                                                             │
│   ✓ Downloading videos (2 of 2)                            │
│   ● Detecting scenes... 45%                                │
│   ○ Generating thumbnails                                   │
│   ○ Analyzing colors                                        │
│   ○ Building sequence                                       │
│                                                             │
│   [Cancel]                                                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                   TIMELINE VIEW                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Video Player                                        │   │
│   └─────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Timeline with auto-populated clips                 │   │
│   │  [clip1][clip2][clip3][clip4][clip5]...             │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Condition | Action |
|-----------|--------|
| Clips exist with required analysis | Generate sequence immediately |
| Clips exist but missing analysis (e.g., no colors for Color) | Run required analysis, then generate |
| No clips but sources exist | Run detection on existing sources, then analysis, then generate |
| No clips, no sources | Show import modal |

### Analysis Requirements by Sequence Type

| Sequence Type | Required Analysis | Optional Analysis |
|---------------|-------------------|-------------------|
| Color | `dominant_colors` | None |
| Duration | None (duration from detection) | None |
| Shuffle | None | None |
| Sequential | None | None |

## Technical Considerations

### Architecture

**New Components:**
- `IntentionImportDialog` - Modal dialog for file/URL import
- `IntentionWorkflowCoordinator` - Orchestrates the multi-step flow

**Modified Components:**
- `sequence_tab.py` - Handle card click with no clips
- `sorting_card_grid.py` - Emit signal with algorithm info on click
- `main_window.py` - Connect workflow coordinator to existing workers

### State Management

**Critical learnings from `docs/solutions/`:**

1. **Single Source of Truth** (from `timeline-widget-sequence-mismatch-20260124.md`)
   - The `IntentionWorkflowCoordinator` owns workflow state
   - UI components read via properties, don't maintain copies

2. **Source ID Synchronization** (from `pyside6-thumbnail-source-id-mismatch.md`)
   - After detection, sync `clip.source_id` to match existing `Source.id`
   - Workers may create new Source objects with different UUIDs

3. **Duplicate Signal Guards** (from `qthread-destroyed-duplicate-signal-delivery-20260124.md`)
   - Use guard flags for each step transition
   - Reset guards when starting new workflow
   - Use `Qt.UniqueConnection` for all signal connections

### Worker Chain

```python
# Sequential processing for simplicity
for source in sources:
    download_if_url(source)      # URLDownloadWorker
    detect_scenes(source)         # DetectionWorker
    generate_thumbnails(source)   # ThumbnailWorker
    if needs_analysis(algorithm):
        run_analysis(clips)       # ColorWorker / etc.

# After all sources processed
generate_sequence(algorithm, all_clips)
```

### Error Handling Strategy

- **Download failure:** Log error, skip source, continue with others
- **Detection failure:** Log error, skip source, continue with others
- **Analysis failure:** Log warning, use clips without analysis, degrade gracefully
- **All sources failed:** Show error message, return to card grid

### Key Files to Modify

| File | Changes |
|------|---------|
| `ui/tabs/sequence_tab.py` | Add workflow trigger on card click |
| `ui/widgets/sorting_card_grid.py` | Emit algorithm info with click signal |
| `ui/main_window.py` | Add `IntentionWorkflowCoordinator` |
| `ui/dialogs/intention_import_dialog.py` | **NEW** - Import modal UI |
| `core/intention_workflow.py` | **NEW** - Workflow state machine |

## Acceptance Criteria

### Core Flow
- [x] Clicking sequence card with no clips shows import modal
- [x] Import modal accepts local files via drag-drop
- [x] Import modal accepts local files via file browser
- [x] Import modal accepts YouTube/Vimeo URLs (multiple, one per line)
- [x] Progress shows current step and overall progress
- [x] Cancel button stops workflow at any point
- [x] Successful completion shows populated timeline

### Conditional Analysis
- [x] Color sequence triggers color analysis before population
- [x] Duration/Shuffle/Sequential skip analysis step
- [x] Clips missing required analysis are analyzed before use

### Error Handling
- [x] Failed downloads show error but continue with other sources
- [x] Failed detections show error but continue with other sources
- [x] Partial success shows summary: "Created sequence with 45 clips from 2 of 3 videos"
- [x] Complete failure returns to card grid with error message

### State Management
- [x] Imported sources appear in Collect tab
- [x] Detected clips appear in Cut tab
- [x] Project auto-saves after workflow completion
- [ ] Navigation to other tabs works correctly mid-flow

### Edge Cases
- [x] Clicking card when clips already exist generates sequence immediately
- [ ] Clicking card when clips exist but need analysis runs analysis first
- [ ] Clicking card when sources exist but no clips runs detection first

## Success Metrics

- Users can create their first sequence in under 2 minutes (vs current ~5 min)
- 70% of new users start from Sequence tab (measure via analytics)
- Workflow completion rate > 80% (users don't abandon mid-flow)

## Dependencies & Risks

### Dependencies
- Existing `DetectionWorker`, `ThumbnailWorker`, `ColorWorker` infrastructure
- `generate_sequence()` algorithm implementations
- yt-dlp for URL downloads

### Risks

| Risk | Mitigation |
|------|------------|
| Long processing time discourages users | Show granular progress, estimated time for large files |
| Worker thread crashes leave inconsistent state | Guard flags, cleanup on cancel, auto-save checkpoints |
| Memory issues with many large videos | Process sequentially, not in parallel |
| Source ID mismatch breaks clip lookups | Apply documented ID sync pattern from learnings |

## References & Research

### Internal References
- Sorting card click handling: `ui/widgets/sorting_card_grid.py:65-85`
- Sequence generation: `core/remix/__init__.py:40-171`
- Detection worker pattern: `ui/main_window.py:82-122`
- Empty state widget: `ui/widgets/empty_state.py:1-45`
- URL download worker: `ui/main_window.py` (`URLBulkDownloadWorker`)

### Documented Learnings
- State duplication gotcha: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- Source ID mismatch: `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
- Signal delivery duplication: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`

### Related Agent Tools
- `download_videos(urls)` - Bulk URL download
- `detect_all_unanalyzed(sensitivity)` - Batch detection
- `analyze_colors_live(clip_ids)` - Color analysis
- `generate_sequence(algorithm, ...)` - Sequence creation

## Implementation Outline

### Phase 1: Import Modal
1. Create `IntentionImportDialog` with drag-drop + URL input
2. Add "no clips" detection to `_on_card_clicked` in `sequence_tab.py`
3. Show modal and collect sources

### Phase 2: Workflow Coordinator
1. Create `IntentionWorkflowCoordinator` state machine
2. Connect to existing workers (download, detection, thumbnail, analysis)
3. Implement progress reporting

### Phase 3: Integration
1. Wire coordinator to `MainWindow`
2. Handle cancel and error cases
3. Populate timeline on completion

### Phase 4: Polish
1. Add step-by-step progress indicators
2. Handle edge cases (existing clips, partial analysis)
3. Add error summaries
