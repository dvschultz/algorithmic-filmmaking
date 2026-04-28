---
title: "feat: Comprehensive clip filter system for Cut and Analyze tabs"
type: feat
status: draft
date: 2026-04-21
---

# Comprehensive clip filter system for Cut and Analyze tabs

## Problem

The current clip filter surface on the Cut and Analyze tabs doesn't cover
the full set of analysis dimensions the project produces. The visible top
row shows only shot type, color palette, custom query, and name search;
many richer filters (duration, aspect ratio, gaze, object search,
description search, brightness) live behind a "Filters" button in a
collapsing panel and are easy to miss. Several analysis dimensions —
notably ImageNet classification, YOLO object counts, OCR, cinematography,
person count, audio volume, and transcript/analysis presence — have no
filter exposure at all. Most enum filters are single-select, which blocks
common combinations like "close-up OR extreme close-up".

Primary use case driving this work: **shot hunting** — finding clips that
match multiple precise criteria at once (e.g., *"close-ups of exactly one
person looking at camera, no on-screen text, between 2–5 seconds"*). The
current UI makes that workflow difficult because filters are partly
hidden, partly missing, and partly too coarse.

## Users

- Video editors curating raw clip sets for a project
- Sequencer users pre-filtering clips before running an algorithm
- Power users with large projects (hundreds of clips) who rely on
  analysis metadata to find needles in haystacks

## Requirements

### Layout

- R1. Replace the current top-row filter strip + "Filters" panel with a
  **toggleable sidebar** that can be shown/hidden as the user works.
  Shown by default; hidden state preserved per-tab across app restarts.
- R2. Sidebar organizes filter dimensions into **named sections** so
  related filters group visually. Sections are collapsible so users can
  focus on the dimensions they're actively using.
- R3. When the sidebar is hidden, **active filter chips** appear above
  the clip grid as read-only indicators so the user can see what's
  applied without re-opening the sidebar. Each chip has a small `×` for
  quick removal. This is the minimum visibility guarantee for hidden
  state.
- R4. Filter state is **shared by default between Cut and Analyze tabs**.
  Each tab has its own "Reset filters" button that clears the shared
  state without forcing the other tab to also reset. Switching tabs
  pulls the current shared state; resetting on one tab does not
  retroactively affect what the other tab was doing until the next tab
  switch.

### Filter dimensions

Each filter must be exposed with the control type noted. Dimensions
marked **[new]** don't exist today; others are expanding or changing
control type.

**Shot properties**
- R5. Duration — range slider (continuous). *Existing.*
- R6. Aspect ratio — **multi-select chips** (was single-select).
- R7. **[new]** Has audio — boolean toggle.

**Visual / cinematography**
- R8. Shot type — **multi-select chips** (was single-select).
- R9. Color palette — **multi-select chips** (was single-select).
- R10. Brightness — range slider. *Existing.*
- R11. **[new]** Cinematography (camera movement, angle, etc. as
  produced by `cinematography` analysis) — multi-select chips.

**People & gaze**
- R12. **[new]** Person count — operator picker (`>`, `=`, `<`) with
  numeric input. Discrete integer, so operators rather than range.
- R13. Gaze direction — **multi-select chips** (was single-select,
  buried). *Existing, promoted.*

**ImageNet classification** (1000 classes)
- R14. **[new]** Has ImageNet class — typeahead search that autocompletes
  against classes **actually present in the loaded project**, sorted by
  frequency (e.g., `dog (42 clips)`, `golden retriever (8 clips)`).
  Selections become a multi-select chip list. A flat dropdown is not
  viable at 1000 classes.
- R15. **[new]** Mode toggle on the ImageNet class list: **Any** of the
  selected classes vs **All** of the selected classes.

**YOLO object detection** (~80 classes, separate data source from
ImageNet)
- R16. **[new]** Has object label — multi-select chips over all detected
  labels in the project.
- R17. **[new]** Total object count per clip — operator picker
  (`>`, `=`, `<`).
- R18. **[new]** Per-label count — compound rule: "at least N of
  [label]". Multiple rules can be stacked (ANDed). Removable like a
  chip.

**Text / OCR / speech**
- R19. **[new]** Has on-screen text — boolean.
- R20. **[new]** On-screen text search — text input over
  `extracted_texts`.
- R21. **[new]** Has transcript — boolean.
- R22. Description / transcript text search — text input. *Existing
  partially; expand to cover transcripts explicitly.*

**Audio**
- R23. **[new]** RMS volume — range slider.

**Custom queries**
- R24. Custom query match — multi-select chips. *Existing, no change.*

**Meta / housekeeping**
- R25. **[new]** Has analysis of type X — multi-select chips over
  analysis operation names (e.g., "describe", "objects", "colors"). Lets
  users hide un-analyzed clips for a specific dimension.
- R26. **[new]** Enabled / disabled — boolean toggle. Lets the user
  hide clips they've marked disabled.
- R27. Tag / note text search — text input.

### Interaction behavior

- R28. All filters combine with **AND** across dimensions. Within a
  multi-select chip list, selected values combine with **OR** by default
  (ImageNet has an explicit Any/All mode toggle per R15).
- R29. **Negation is implicit only in v1** — boolean filters have Yes/No
  options. Chip-list filters do not have per-chip or per-filter NOT.
  (Revisit if shot-hunting workflows demand it.)
- R30. Every filter has a visible "clear" affordance that resets only
  that filter.
- R31. When no clips match the active filters, the grid shows an
  explicit empty state with a "Clear filters" CTA (not just a blank
  area).

## Scope Boundaries

- **No saved filter presets in v1.** Revisit after the richer filter
  surface ships and usage data shows whether preset reuse is a real
  pattern.
- **No negation beyond boolean Yes/No.** Per-chip exclude and
  per-filter-mode switches are out of scope; their absence should be
  acceptable for the shot-hunting use case.
- **No changes to sequencer-tab filter behavior.** This is Cut and
  Analyze only. Sequencer uses its own card-based UI and is out of
  scope.
- **No new analysis operations.** This proposal only surfaces existing
  analysis outputs; it does not introduce new detection models or
  inference.
- **No filter state persisted across projects.** Sidebar visibility
  persists, but filter values reset when switching projects.

## Success Criteria

- A user can construct a multi-criterion shot-hunt filter (e.g., "close-up
  OR extreme close-up" × "1 person" × "at-camera gaze" × "no on-screen
  text" × "≥ 2s") in a single continuous flow without opening or closing
  multiple panels.
- Every analysis operation the project runs (from
  `core/analysis_operations.py`) has at least one filter exposure in the
  sidebar.
- Users can filter by both ImageNet classes (via typeahead across 1000
  labels) and YOLO object detections (via chip list of ~80 labels), and
  can express object-count constraints with `>`, `=`, `<` operators.
- Active filters are visible whether the sidebar is shown or hidden.
- Cut and Analyze tabs share filter state by default so switching
  between them doesn't force reconfiguration.

## Relevant Code

- `ui/clip_browser.py` — current filter top row, collapsing filter
  panel, and filter state. Most changes concentrate here.
- `models/clip.py` — analysis fields that back each filter
  (`object_labels` = ImageNet, `detected_objects` = YOLO,
  `person_count`, `extracted_texts`, `rms_volume`, `transcript`,
  `description`, `cinematography`, etc.).
- `core/analysis/classification.py` — ImageNet source of truth for label
  vocabulary and per-clip labels.
- `core/analysis_operations.py` — canonical analysis operation list used
  by R25 ("Has analysis of type X").
- `ui/theme.py` — `UISizes` constants for sidebar width, chip height,
  input heights (see `.claude/rules/ui-consistency.md`).

## Open Questions — Deferred to Planning

- **Sidebar width and location** — left rail or right rail? Fixed or
  resizable?
- **Filter state persistence scope** — reset on project switch, but
  persist during a single project session including app restart?
- **Active-filter chip overflow** — what happens when 10+ filters are
  applied and the chip bar above the grid runs out of horizontal room?
- **Typeahead performance** — 1000 ImageNet classes is small; no
  anticipated concerns, but worth confirming during implementation for
  large projects.
- **Per-label count compound rule UI (R18)** — exact layout for stacking
  multiple "at least N of [label]" rules deserves a design pass during
  planning.
