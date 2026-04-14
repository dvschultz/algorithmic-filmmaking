---
date: 2026-04-13
topic: multi-sequence-projects
---

# Multi-Sequence Projects

## Problem Frame

Scene Ripper now has 20 sequencer algorithms, but projects can only hold one sequence at a time. Every algorithm run overwrites the previous result, forcing users to either export before experimenting or lose their work. This makes the Sequence tab feel destructive rather than exploratory. Users should be able to try multiple sequencers on their clip pool and keep every result for comparison, without needing to save/reload between experiments.

## Requirements

**Sequence Storage**
- R1. A project can hold any number of sequences. Each sequence has a name, and all sequences persist through project save/load. The R2 invariant (at least one sequence) is enforced at the Project model level (`Project.__init__()`, `Project.new()`, `Project.clear()`) — not just the UI.
- R2. There is always at least one sequence in a project. New projects start with one empty sequence.

**Creation**
- R3. Clicking an algorithm card (intentional new run) auto-creates a new sequence rather than overwriting the active one. The sequence is auto-named after the algorithm using a monotonic counter: scan existing names for the highest N matching `"{Algorithm} #{N}"`, use N+1. First run of an algorithm omits the suffix (e.g., "Chromatics"); second run produces "Chromatics #2". Renamed and deleted sequences do not reset the counter.
- R3a. Parameter tweaks on the current algorithm (changing direction dropdown, re-running the same algorithm with different settings) prompt the user: "Replace current sequence or create new?" This avoids sequence sprawl from casual exploration while keeping the choice explicit.
- R4. A "New Sequence" button (replacing the current "Clear Sequence" button) creates a blank empty sequence and switches to it. Switching to the new sequence follows the same behavior defined in R6 (all related UI state updates).

**Switching**
- R5. A dropdown menu in the Sequence tab lets the user switch between sequences. The dropdown shows each sequence's name.
- R6. Switching sequences first auto-persists any unsaved edits (drag-reorder, clip removal, etc.) from the current timeline back to the departing sequence. Then it swaps the timeline, preview, and all related UI state (algorithm label, chromatic bar setting, etc.) to reflect the selected sequence. If the departing sequence has unsaved manual edits, a prompt appears: "Save changes to '[name]'?" with Save / Discard / Cancel.

**Deletion**
- R7. Users can delete a sequence. Empty sequences (0 SequenceClips) delete without confirmation. Populated sequences (1+ SequenceClips, regardless of render state) show a confirmation dialog: "Delete '[name]'? This cannot be undone."
- R8. Deleting the active sequence switches to the sequence at index 0 (the oldest/first-created) of the remaining list. If it was the last sequence, a fresh empty sequence is created automatically (R2 invariant holds). Deleting a non-active sequence removes it from the list without affecting the active view.

**Naming**
- R9. Sequence names are editable — users can rename any sequence to something meaningful (e.g., "Final Cut", "Warm Version"). Duplicate names are permitted (no uniqueness constraint).

## Success Criteria

- Running 5 different sequencers on the same clips results in 5 named sequences accessible via the dropdown, all preserved on save/reload.
- Users never lose a previous sequence result by running a new algorithm.
- Parameter tweaks (direction change, re-run same algorithm) give the user a choice rather than silently creating sequences they didn't intend.
- The mental model matches NLE software (Premiere, Resolve): sequences are lightweight experiments you accumulate.

## Scope Boundaries

- No sequence comparison/diff view (side-by-side). Users compare by switching via the dropdown.
- No sequence duplication (clone an existing sequence to modify). Can be added later.
- No drag-drop reordering of the sequence list in the dropdown. Order is creation order.
- No per-sequence render settings. The render tab exports whichever sequence is active.
- No batch-delete for sequences. Single deletion only (R7). Bulk cleanup can be added later if sprawl proves to be a problem in practice.

### Agent/MCP Compatibility (In Scope)

- A backward-compatible `project.sequence` property is added that proxies to the active sequence (getter returns `project.sequences[active_index]`, setter replaces that entry). This allows the 78+ references across `chat_tools.py`, MCP server, CLI, and tests to continue working without modification. Full multi-sequence agent awareness is deferred to a separate follow-up.

## Key Decisions

- **Auto-create on new algorithm, prompt on parameter tweaks**: Clicking an algorithm card always creates a new sequence. Changing direction/re-running the same algorithm prompts "Replace or create new?" This avoids sequence sprawl from casual exploration while preserving the experiment-log mental model for intentional algorithm changes.
- **Replace "Clear Sequence" with "New Sequence"**: Clears don't make sense when sequences accumulate. "New Sequence" is the explicit action for starting fresh, and it creates a blank sequence rather than erasing the current one.
- **At least one sequence invariant, enforced at model level**: Simplifies all code paths — there's always an active sequence to render, export, or display. Enforced in `Project.__init__()`, `Project.new()`, and `Project.clear()` so UI code never needs null-state handling.
- **Delete confirmation only for populated sequences**: Empty sequences (0 SequenceClips) are disposable experiments that didn't go anywhere. No friction to clean them up. Populated sequences represent invested work and get a safety dialog.
- **Switch to index 0 on delete**: Predictable landing point after deletion (the oldest/first-created sequence). Combined with the at-least-one invariant, the user always sees a valid sequence after any delete.
- **Auto-persist edits on switch with prompt**: Switching sequences saves the departing sequence's timeline state. If the user has made manual edits, they're prompted to save or discard. This prevents silent data loss while keeping the flow lightweight when sequences haven't been manually edited.
- **Monotonic naming counter**: Uses `max existing N + 1` per algorithm key, ignoring renames and deletions. "Chromatics #1" deleted → next run produces "Chromatics #2", not "#1" again. Simple, predictable, no collision risk.
- **Compatibility property for `project.sequence`**: Rather than migrating 78+ call sites across agent tools, MCP server, and CLI, a property accessor proxies to the active sequence. This decouples the multi-sequence feature from a codebase-wide migration and keeps the blast radius contained to the Sequence tab and Project model.

## Dependencies / Assumptions

- The `Sequence` model already has `name` and `id` fields, so no schema additions are needed on the model itself.
- The `Project` data model will change from `sequence: Optional[Sequence]` to `sequences: list[Sequence]` with an `active_sequence_index`. This is a project file format change that requires a schema version bump (1.3 → 1.4) and backward-compatible loading (old `"sequence"` key → wrap in a single-element list; new format writes `"sequences"` list + `"active_sequence_index"`).
- The `save_project()` and `load_project()` standalone functions in `core/project.py` have their own signatures that take/return a single `Sequence` — these must be updated alongside the `Project` class. The `_prepare_prerendered_clips` helper and `_validate_project_structure` also need multi-sequence handling.
- The timeline widget (`TimelineWidget`) currently holds a single `Sequence` reference that will need to be swappable when the user switches sequences.
- A backward-compatible `project.sequence` property proxies to the active sequence, so agent tools, MCP server, CLI, and tests continue working without changes.

## Outstanding Questions

### Deferred to Planning
- [Affects R5][Technical] Where exactly the dropdown goes in the Sequence tab layout — above the timeline, in the toolbar area, or as part of the existing algorithm dropdown row.
- [Affects R6][Technical] How to swap the timeline widget's active sequence efficiently without rebuilding the entire widget (may need a `set_sequence()` method that re-renders).
- [Affects R9][Technical] Where rename UI lives — inline editing in the dropdown, a separate rename action in a context menu, or double-click on the dropdown item.
- [Affects R3][Technical] The ~12 sequence-apply handlers in `sequence_tab.py` all call `clear_timeline()` before populating. Each needs to be refactored to a shared "create-and-activate-sequence" pattern. The plan should enumerate them.
- [Affects R6][Technical] What constitutes "unsaved manual edits" for the switch prompt — any divergence from the last-persisted state, or only explicit user actions like drag-reorder? Needs a lightweight dirty-tracking mechanism.

## Next Steps

-> `/ce:plan` for structured implementation planning
