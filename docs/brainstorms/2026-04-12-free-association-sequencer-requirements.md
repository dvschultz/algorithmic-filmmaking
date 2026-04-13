---
date: 2026-04-12
topic: free-association-sequencer
---

# Free Association Sequencer

## Problem Frame

Scene Ripper's existing sequencers operate on fixed axes — similarity chains use embeddings, storyteller follows narrative arcs, shuffle is random. None let the user leverage the full breadth of clip metadata through an LLM that makes practical, grounded associative decisions one clip at a time. Editors want a sequencer that acts like a knowledgeable collaborator: it sees everything about each clip and proposes connections the editor can accept, reject, or re-roll.

## Requirements

**Core Sequencing**
- R1. The sequencer accepts a user-selected first clip and iteratively builds a sequence by proposing one next clip at a time.
- R2. Each proposal is informed by a tiered metadata strategy to keep per-step token cost bounded (~800 tokens input):
  - **Current clip**: full metadata (description, shot type, dominant colors, brightness, volume, objects, faces, transcript, cinematography, gaze, extracted text).
  - **Candidate pool**: a shortlist of ~10-15 clips pre-filtered locally using embedding similarity, each represented as a compact metadata digest (~20-30 tokens per clip, e.g., "CU | warm tones | bright | 2 people | outdoor | dialogue | slow pan"). The shortlist is built from structured fields with no LLM cost.
  - **Sequence history**: a rolling theme summary (~50-100 tokens) updated after each acceptance, not the full metadata of every placed clip.
- R3. The rolling sequence summary captures motifs, patterns, and variety so the LLM avoids repetitive transitions without needing the full history of every placed clip.

**Rationale**
- R4. Each proposed clip includes a rationale explaining the metadata-driven connection (e.g., "Both clips share close-up framing with warm dominant colors and similar motion energy").
- R5. Rationales are displayed in a scrollable side panel log that accumulates as the sequence is built, showing each transition decision.

**Accept / Reject Interaction**
- R6. The user can accept the proposed clip (it's added to the sequence, LLM proposes the next one) or reject it.
- R7. On rejection, the LLM proposes a different clip, knowing which clips have been rejected for this position. The user can re-roll as many times as needed.
- R8. Rejected clips return to the available pool for future positions — they're only excluded from the current position's proposals.

**Completion**
- R9. Sequencing ends when all clips are placed or the user explicitly stops early.
- R10. The accumulated rationale log persists with the sequence so it can be reviewed after the dialog is closed.

## Success Criteria

- User can build a full sequence one clip at a time with meaningful, metadata-grounded rationales for each transition.
- Reject/re-roll produces a different clip with a different rationale.
- Rationale log is readable and useful as editorial notes.

## Scope Boundaries

- No abstract/poetic mode — rationales are grounded in actual metadata values.
- No multi-candidate selection (showing ranked alternatives). Single proposal with re-roll.
- No "regenerate from midpoint" — this is a step-by-step sequencer, not batch-with-rewind.
- No new metadata fields or analysis types — uses whatever metadata already exists on clips. The compact digest and rolling summary are ephemeral prompt artifacts, not persisted data.

## Key Decisions

- **Step-by-step over one-shot**: Gives the editor creative control at each transition rather than reviewing a fait accompli. More engaging and educational (you learn what metadata connections exist).
- **Single re-roll over ranked alternatives**: Keeps the UI simple. Showing 3 candidates with rationales would be noisy and slow (3x LLM output per step).
- **Side panel log over inline card annotations**: Keeps the sequence grid clean while providing full transparency. The log reads like editor's notes.
- **Tiered metadata over full dump**: The current clip gets full detail, candidates get compact digests, and history is a rolling summary. This keeps per-step cost at ~800 tokens regardless of total clip count — local embedding similarity absorbs the scaling, not the LLM context. The LLM still sees all metadata dimensions for the current clip and can weight what matters per transition.
- **Local pre-filtering over LLM pool scanning**: Using existing embeddings to shortlist ~10-15 candidates keeps the LLM focused on making a good choice from strong options rather than scanning 100+ clips. This also bounds latency and cost per step.

## Dependencies / Assumptions

- Clips must have at least `description` populated for meaningful associations. Other metadata fields enrich the output but aren't strictly required.
- Existing LLM infrastructure (`core/llm_client.py`, LiteLLM) handles the model calls. Each LLM response must be validated for None/empty content before parsing — this is a documented codebase bug pattern (existing sequencers don't guard against it). A failed call mid-sequence must not discard the user's accepted clips.
- Dialog-based sequencer pattern (QDialog + QThread worker) is established by Storyteller and Exquisite Corpus. However, the iterative accept/reject loop is a novel interaction pattern — existing dialogs run to completion in one go. Planning must design a new worker lifecycle (e.g., single-step worker invoked per proposal, not a long-running worker).

## Outstanding Questions

### Deferred to Planning
- [Affects R5, R10][Technical] Whether the rationale log should be a new field on SequenceClip or a separate data structure on the sequence, and its serialization format for project save/load.
- [Affects R2][Technical] Exact format of the compact metadata digest — which fields to include, field ordering, and how to handle missing metadata gracefully.
- [Affects R2][Technical] Shortlist size tuning — 10-15 is the starting target, but the right number may depend on how diverse the candidate pool is.
- [Affects R1][Technical] Worker lifecycle design for the iterative accept/reject loop — single-step worker per proposal vs. long-lived worker with inter-thread signaling.

## Next Steps

-> `/ce:plan` for structured implementation planning
