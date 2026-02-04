---
title: "feat: Add Storyteller Sequence Card"
type: feat
date: 2026-02-04
---

# feat: Add Storyteller Sequence Card

## Overview

Add a new "Storyteller" sequence card that creates narrative-driven video sequences from clip descriptions. The LLM analyzes all clip descriptions, selects clips that fit a coherent story, and reorders them based on the chosen narrative structure. Users can target specific video durations (10min, 30min, 1hr, 90min).

## Problem Statement / Motivation

Currently, sequence cards sort clips by technical attributes (color, duration, shot type) or randomize them (shuffle). There's no way to create a sequence based on **meaning** - the actual content of the clips.

Filmmakers often have footage with varied content and need to assemble it into a coherent story. Manually reviewing descriptions and arranging clips is tedious. The Storyteller card automates this creative process by using an LLM to:
1. Understand each clip's content from its description
2. Select clips that contribute to a narrative
3. Arrange them in a structure that tells a story

## Proposed Solution

### High-Level Approach

1. Add a new "Storyteller" card to the sequence card grid
2. Create a multi-step dialog similar to Exquisite Corpus:
   - Configuration page: theme, structure, duration
   - Progress page: LLM processing
   - Preview page: view and reorder before applying
3. Implement LLM-based narrative generation in `core/remix/storyteller.py`
4. Integrate with both sequence tab and intention-first workflow

### User Flow

```
Sequence Tab → Storyteller Card Click →
  Check for descriptions →
    If missing: Prompt (Exclude OR Analyze) →
  Configuration Dialog (Theme, Structure, Duration) →
  LLM Processing (Progress) →
  Preview & Reorder →
  Apply to Timeline
```

## Technical Approach

### Architecture

Following existing patterns from `exquisite_corpus.py` and `sequence_tab.py`:

```
┌─────────────────────────────────────────────────────────┐
│                    UI Layer                             │
├─────────────────────────────────────────────────────────┤
│  sequence_tab.py                                        │
│    └─ ALGORITHM_CONFIG["storyteller"]                   │
│    └─ _on_card_clicked() → _show_storyteller_dialog()   │
│                                                         │
│  dialogs/storyteller_dialog.py (NEW)                    │
│    └─ Page 1: Configuration (theme, structure, duration)│
│    └─ Page 2: Progress (LLM call)                       │
│    └─ Page 3: Preview (reorderable list)                │
│                                                         │
│  dialogs/intention_import_dialog.py                     │
│    └─ storyteller_container (duration, theme, structure)│
├─────────────────────────────────────────────────────────┤
│                    Core Layer                           │
├─────────────────────────────────────────────────────────┤
│  core/remix/storyteller.py (NEW)                        │
│    └─ generate_narrative()                              │
│    └─ NarrativeLine dataclass                           │
│    └─ sequence_by_narrative()                           │
│                                                         │
│  models/clip.py                                         │
│    └─ clip.description (already exists)                 │
└─────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Core Narrative Generation

**Files to create/modify:**

1. **`core/remix/storyteller.py`** (NEW)

```python
@dataclass
class NarrativeLine:
    """A single element in the generated narrative."""
    clip_id: str
    description: str
    narrative_role: str  # e.g., "opening", "rising_action", "climax"
    line_number: int

def generate_narrative(
    clips_with_descriptions: list[tuple],  # [(Clip, description_text), ...]
    target_duration_minutes: int,
    narrative_structure: str,  # "three_act", "chronological", "thematic", "auto"
    theme: str | None = None,
    model: str | None = None,
) -> list[NarrativeLine]:
    """Generate a narrative sequence using LLM."""
    ...

def sequence_by_narrative(
    narrative_lines: list[NarrativeLine],
    clips_by_id: dict,
    sources_by_id: dict,
) -> list[tuple]:
    """Create clip sequence from narrative order."""
    ...
```

2. **LLM Prompt Design:**

```
SYSTEM: You are a film editor creating a narrative sequence from video clips.

RULES:
1. SELECT clips that fit a coherent narrative (you may exclude clips that don't fit)
2. ARRANGE clips in {structure} order
3. Target approximately {duration} minutes of content
4. Each clip can only be used once
5. Consider pacing - vary clip lengths and intensity

NARRATIVE STRUCTURE:
- three_act: Setup (25%) → Confrontation (50%) → Resolution (25%)
- chronological: Arrange by time references in descriptions
- thematic: Group similar themes, build through contrast
- auto: Choose the best structure for these clips

{theme_section}

OUTPUT FORMAT:
Return JSON array of clip IDs in narrative order:
["c1", "c5", "c3", ...]
```

#### Phase 2: Storyteller Dialog

**Files to create/modify:**

1. **`ui/dialogs/storyteller_dialog.py`** (NEW)
   - Three-page stacked widget (like ExquisiteCorpusDialog)
   - Configuration page with:
     - Optional theme text input
     - Narrative structure dropdown
     - Target duration dropdown
   - Progress page with status
   - Preview page with drag-drop reordering

2. **`ui/tabs/sequence_tab.py`**
   - Add to `ALGORITHM_CONFIG`:
     ```python
     "storyteller": {
         "label": "Storyteller",
         "description": "Create narrative from clip descriptions",
         "allow_duplicates": False,
     }
     ```
   - Add `_show_storyteller_dialog()` method
   - Add availability check for clips with descriptions

#### Phase 3: Intention-First Integration

**Files to modify:**

1. **`ui/dialogs/intention_import_dialog.py`**
   - Add storyteller configuration widgets (similar to poem_length for exquisite_corpus)
   - Duration dropdown
   - Structure dropdown
   - Theme input (optional)

2. **`ui/main_window.py`**
   - Add `_intention_pending_storyteller_*` attributes
   - Add `_show_storyteller_dialog_for_intention()`

### Data Model

**Clip requirements:**
- `clip.description` must be non-empty string
- Clips without descriptions are either excluded or analyzed first

**Duration targets:**
| Option | Target | Acceptable Range |
|--------|--------|------------------|
| ~10 min | 10 min | 6-14 min |
| ~30 min | 30 min | 20-40 min |
| ~1 hour | 60 min | 45-75 min |
| ~90 min | 90 min | 70-110 min |

**Narrative structures:**
| Structure | Description | Pacing |
|-----------|-------------|--------|
| Three-act | Setup → Confrontation → Resolution | 25% / 50% / 25% |
| Chronological | Time-based ordering from descriptions | Linear |
| Thematic | Group by theme, build contrast | Clustered |
| Auto | LLM chooses best fit | Variable |

### Missing Description Handling

When clips lack descriptions, show a prompt dialog:

```
Some clips don't have descriptions yet.

Storyteller needs descriptions to create a narrative.

[ Exclude clips without descriptions ]  [ Run description analysis first ]
                                                    ↓
                                        Navigate to Analyze tab
                                        Run Describe on missing clips
                                        Return to Storyteller
```

### Duration Shortfall Handling

If total clip duration < target:

```
Created 15:32 sequence using all 12 clips.
(Target was 90 minutes - not enough clips available)

[ Apply to Timeline ]  [ Add More Clips ]
```

## Acceptance Criteria

### Functional Requirements

- [x] Storyteller card appears in sequence card grid
- [ ] Card shows disabled state with reason if no clips have descriptions
- [x] Configuration dialog allows setting:
  - [x] Optional theme (free text, max 500 chars)
  - [x] Narrative structure (dropdown: Three-act, Chronological, Thematic, Auto)
  - [x] Target duration (dropdown: ~10min, ~30min, ~1hr, ~90min, Use all clips)
- [x] LLM generates narrative sequence from clip descriptions
- [x] Preview shows proposed sequence with drag-drop reordering
- [x] Sequence applies to timeline correctly
- [x] Works from sequence tab with existing clips
- [x] Works from intention-first workflow (import dialog)
- [x] Handles missing descriptions gracefully (prompt user)
- [x] Handles duration shortfall gracefully (use available clips)

### Non-Functional Requirements

- [ ] LLM call completes within 30 seconds for typical clip pools (<100 clips)
- [x] Dialog is responsive during LLM processing (shows progress)
- [x] Uses configured LLM model from settings

### Quality Gates

- [x] Unit tests for `generate_narrative()` with mocked LLM
- [x] Unit tests for `sequence_by_narrative()`
- [ ] Integration test for full dialog flow
- [ ] Manual testing of all four narrative structures

## Success Metrics

- Users can create narrative sequences from described clips
- Sequences make logical sense when reviewed
- Duration targets are approximated within acceptable ranges

## Dependencies & Prerequisites

- Clips must have descriptions (`clip.description` field populated)
- LLM API configured and working (same as chat agent)
- Existing sequence card infrastructure (already in place)

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM returns malformed output | Medium | High | Retry with prompt adjustment; fallback to original order |
| LLM times out | Low | Medium | 60-second timeout; show error with retry option |
| Very large clip pools hit token limits | Medium | Medium | Batch clips or truncate descriptions |
| Narrative quality is poor | Medium | Medium | Allow user to regenerate; provide preview/reorder |

## Implementation Checklist

### Files to Create

- [x] `core/remix/storyteller.py` - Core narrative generation logic
- [x] `ui/dialogs/storyteller_dialog.py` - Multi-page configuration dialog
- [x] `tests/test_storyteller.py` - Unit tests for storyteller module

### Files to Modify

- [x] `ui/tabs/sequence_tab.py` - Add card config, click handler, availability check
- [x] `ui/dialogs/intention_import_dialog.py` - Add storyteller configuration widgets
- [x] `ui/main_window.py` - Add intention workflow handlers
- [x] `ui/widgets/sorting_card_grid.py` - Add Storyteller card to grid
- [x] `ui/dialogs/__init__.py` - Export StorytellerDialog

### Key Patterns to Follow

From institutional learnings:
- Use `Qt.UniqueConnection` when connecting signals
- Use guard flags for signal handlers that create workers
- Single source of truth for state (don't duplicate)
- Sync worker-created object IDs with existing UI state

## References & Research

### Internal References

- Exquisite Corpus implementation: `core/remix/exquisite_corpus.py`
- Sequence card configuration: `ui/tabs/sequence_tab.py:31-62`
- Card click handler: `ui/tabs/sequence_tab.py:268-328`
- Import dialog options: `ui/dialogs/intention_import_dialog.py:286-311`
- Clip description field: `models/clip.py:177`

### Related Work

- Exquisite Corpus uses similar LLM + preview + reorder pattern
- Poem length options added to both dialogs (model for duration)
