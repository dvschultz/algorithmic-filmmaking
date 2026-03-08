---
status: complete
priority: p2
issue_id: "064"
tags: [code-review, agent-tools, rose-hobart]
dependencies: ["057"]
---

# Agent Parity Gaps for Face Features

## Problem Statement

Several face-related capabilities available in the GUI are not accessible to the agent:

1. `list_clips` does not expose `has_face_embeddings` field — agent can't see which clips have face data
2. `filter_clips` has no `has_faces` filter parameter — agent can't filter by face presence
3. `generate_rose_hobart` accepts only a single `reference_image_path` — GUI allows 1-3 reference images
4. `start_clip_analysis` doesn't include `face_embeddings` / `face_detection` in its valid operations set
5. System prompt analysis capabilities list (chat_worker.py line 519) omits "face detection"
6. System prompt analysis coverage tuple (chat_worker.py lines 710-712) omits `face_embeddings` — agent can't passively see face analysis status
7. `analyze_all_live` tool description doesn't list `face_embeddings` as an available operation
8. `generate_rose_hobart` missing `sampling_interval` parameter (GUI exposes 0.25-5.0 sec)

## Findings

**Agent-Native Reviewer (Critical)**: The `start_clip_analysis` tool rejects `face_embeddings` because its `valid_operations` set on line 3390 is hard-coded without it. This creates a **dead-end workflow** — `list_sorting_algorithms` reports Rose Hobart as unavailable when clips lack face embeddings, but the agent has no way to fix this via the recommended `start_clip_analysis` tool.

**Agent-Native Reviewer**: System prompt's analysis coverage enumeration at line 710-712 only tracks 8 fields — `face_embeddings` is absent. The agent cannot passively know when face detection has already been run.

## Proposed Solutions

### Option A: Add Face Fields to All Agent Touchpoints (Recommended)

1. Add `"has_face_embeddings": bool(clip.face_embeddings)` and `"face_count"` to `list_clips` output
2. Add `has_faces: Optional[bool]` filter to `filter_clips`
3. Add `reference_image_paths: list[str]` parameter to `generate_rose_hobart` (accept 1-3)
4. Add `"face_embeddings"` to `start_clip_analysis` valid_operations, op_to_worker, worker_checks, and pipeline_op_map
5. Add "face detection" to system prompt analysis capabilities (chat_worker.py line 519)
6. Add `"face_embeddings"` to analysis coverage tuple (chat_worker.py line 710-712)
7. Add `face_embeddings` to `analyze_all_live` tool description
8. Add `sampling_interval: float = 1.0` parameter to `generate_rose_hobart`

**Pros:** Full parity with GUI, agent can discover and use all face features
**Cons:** Multiple file changes
**Effort:** Medium
**Risk:** Low

## Technical Details

- **Files:**
  - `core/chat_tools.py` — list_clips, filter_clips, generate_rose_hobart, start_clip_analysis, analyze_all_live
  - `ui/chat_worker.py` — system prompt (lines 519, 710-712)

## Acceptance Criteria

- [ ] Agent can see which clips have face embeddings
- [ ] Agent can filter clips by face presence
- [ ] Agent can provide multiple reference images
- [ ] Agent can trigger face detection via start_clip_analysis
- [ ] System prompt mentions face detection in capabilities
- [ ] Analysis coverage includes face_embeddings
- [ ] Agent can adjust sampling interval

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Agent-Native Reviewer | Every GUI capability must have an agent-accessible equivalent |
