---
date: 2026-02-28
topic: expanding-cinema-integration
---

# Expanding Cinema Integration Brainstorm

## What We're Building
We are defining a clip-centric evolution of Scene Ripper inspired by the Expanding Cinema interview: treat video material as searchable, recombinable data while preserving the app's existing clip-first workflow.

The roadmap has three directions in order:
1. Annotation-first sequencing improvements
2. Clip-browser structure views (not timeline-first)
3. Prompt-only recombination controls

Direction 1 is the main immediate value: automatic, source-anchored annotations projected onto clips and used to improve sequencing quality and control. Direction 2 improves discovery in the clip browser so users can read structure at a glance without forcing timeline-heavy workflows. Direction 3 keeps interaction simple by using prompt-only recombination on top of richer metadata.

## Why This Approach
The repository is already strong in clip metadata, analysis pipelines, and remix algorithms. The lowest-risk, highest-leverage move is to enrich metadata semantics and use them in sequencing rather than introducing a new paradigm first.

This also fits YAGNI:
- Start with one annotation layer that directly improves sequence outcomes.
- Keep structure exploration in the clip browser where users already work.
- Keep recombination UX simple (prompt-only) before adding extra controls.

The interview's core ideas (time-based annotation, machine-assisted viewing, recombination) are adopted in a practical way that matches Scene Ripper's architecture.

## Key Decisions
- Project framing: clip-centric, not timeline-centric.
- Direction 1 approach: Source-timeline annotation layer + sequencer scoring (Approach A).
- Annotation anchoring: source-anchored annotations with clip projections.
- Annotation generation: fully automatic, then editable by users.
- Sequencing objective: optimize annotation layer for better sequencing first.
- Sequencing behavior: preference-based ranking (soft influence), not mandatory hard filters.
- Annotation schema: hybrid vocabulary.
- Core fixed fields for V1: `subject`, `action`, `setting`.
- Flexible field: open tags.
- V1 success criteria: both improved coherence and reliable steering with minimal prompt tweaking.
- Direction 2 approach: clip-browser structure views.
- Direction 3 approach: prompt-only recombination.
- Rollout order: full roadmap in sequence (1 -> 2 -> 3).

## Open Questions
- Which auto-annotation sources should be primary for `subject/action/setting` in V1 (description, OCR, transcript, object detection, or weighted blend)?
- How should confidence be surfaced to users when auto-annotations are noisy?
- For Direction 2, which first clip-browser structure view is most valuable: picture strip, signal bars, or tag-density heatmap?
- Should sequencing expose user-facing weighting presets (e.g., "action-heavy", "setting-consistent") in V1 or later?

## Next Steps
-> `/workflows:plan` for implementation details and phased delivery.
