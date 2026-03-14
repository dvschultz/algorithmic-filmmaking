---
status: pending
priority: p2
issue_id: "076"
tags: [code-review, performance, thread-safety]
dependencies: []
---

# Agent generate_rose_hobart Still Runs Reference Image Face Extraction on Main Thread

## Problem Statement

Todo 057 (complete) fixed the agent tool to require pre-computed clip embeddings instead of running extraction inline. However, `extract_faces_from_image()` for reference images (1-3 images) still runs synchronously on the main thread because the tool is marked `modifies_gui_state=True`. This loads the InsightFace model (~500MB, 3-10s on first load) and runs neural network inference on the main thread, freezing the UI.

The dialog path correctly uses `_RefImageExtractWorker` on a background thread.

## Findings

- **Python Reviewer**: Critical issue #1
- **Performance Oracle**: Critical issue #1
- **Agent-Native Reviewer**: Warning #2
- Location: `core/chat_tools.py` line ~3978

## Proposed Solutions

### Option A: Split tool into non-GUI and GUI phases
Run face extraction in the ChatAgentWorker thread (non-GUI), then dispatch only the sequence application to the main thread.

- **Pros**: Clean separation, matches dialog pattern
- **Cons**: Requires restructuring the tool function
- **Effort**: Medium
- **Risk**: Low

### Option B: Document the delay and accept it
1-3 reference images take 2-6s total. Model load is amortized after first use.

- **Pros**: No code change
- **Cons**: UI freeze is noticeable, inconsistent with project conventions
- **Effort**: None
- **Risk**: Low (known UX degradation)

## Acceptance Criteria

- [ ] `extract_faces_from_image` does not run on the Qt main thread
- [ ] UI remains responsive during agent Rose Hobart generation

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Todo 057 fixed clip extraction but not reference image extraction |
