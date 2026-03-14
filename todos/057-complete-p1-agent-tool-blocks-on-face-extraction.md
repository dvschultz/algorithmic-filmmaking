---
status: complete
priority: p1
issue_id: "057"
tags: [code-review, performance, agent-tools, rose-hobart]
dependencies: []
---

# Agent Tool Blocks Synchronously on Face Extraction

## Problem Statement

The `generate_rose_hobart` chat tool calls `extract_faces_from_clip` synchronously for every clip lacking cached embeddings. InsightFace inference is ~2 seconds/clip. For 500 clips, this blocks the LLM worker thread for ~17 minutes with zero progress reporting and no cancellation. The GUI dialog path correctly uses a QThread worker; the agent path does not.

## Findings

**Python Reviewer (Critical)**: Every other heavy operation in this codebase uses a QThread worker. This is a functional regression for agent-driven workflows.

**Performance Oracle (Critical)**: The chat tool infrastructure expects tools to return within a reasonable timeframe.

## Proposed Solutions

### Option A: Require Pre-computed Embeddings (Recommended)

The chat tool should only work with clips that already have `face_embeddings`. If any clips lack them, return an error directing the agent to run face detection first via `start_clip_analysis`.

```python
missing = [c for c in clips if not c.face_embeddings]
if missing:
    return {
        "success": False,
        "error": f"{len(missing)} clips lack face embeddings. "
                 "Run face detection first: start_clip_analysis(clip_ids=[...], operations=['face_detection'])"
    }
```

**Pros:** Simple, no blocking, clear guidance for agent
**Cons:** Requires two-step workflow for agent
**Effort:** Small
**Risk:** Low

### Option B: Delegate to RoseHobartWorker via Signal

Emit a signal that the GUI orchestrates through the existing worker pattern with progress.

**Pros:** Full parity with GUI
**Cons:** Complex signal wiring, harder to return results to agent
**Effort:** Large
**Risk:** Medium

## Technical Details

- **File:** `core/chat_tools.py` lines 3937-3968
- Also need to add `face_detection` / `face_embeddings` to `start_clip_analysis` valid operations

## Acceptance Criteria

- [ ] Agent tool does not block for more than a few seconds
- [ ] Clear error message when embeddings are missing
- [ ] Agent can trigger face detection via start_clip_analysis

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer + Performance Oracle | Agent tools must not run heavy inference synchronously |
