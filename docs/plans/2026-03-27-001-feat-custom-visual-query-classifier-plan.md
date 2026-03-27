---
title: "feat: Add Custom Visual Query Classifier"
type: feat
status: completed
date: 2026-03-27
---

# feat: Add Custom Visual Query Classifier

## Overview

Add a new analysis operation that lets filmmakers type a natural language visual query (e.g., "blue flower", "person wearing a red hat", "outdoor scene at sunset") and have a vision model evaluate each clip's thumbnail, returning a True/False match with confidence score. Multiple queries accumulate on each clip so users can build up rich custom metadata.

## Problem Frame

Filmmakers working with large clip libraries need to find specific visual elements across hundreds of clips. The existing analysis features (shot classification, object detection, content labels) use fixed taxonomies — none support arbitrary visual queries. Users need a way to ask "does this clip contain X?" for any X they can describe.

## Requirements Trace

- R1. User can type an arbitrary natural language visual query
- R2. Every clip in the Analyze tab scope is evaluated True/False against the query
- R3. Results include a confidence score and the model that produced them
- R4. Multiple queries accumulate — running a second query does not erase the first
- R5. Results persist across save/load (project serialization)
- R6. Works with both cloud VLMs (via LiteLLM) and local VLMs (Moondream, Qwen3-VL)
- R7. Feature is accessible from the Quick Run dropdown in the Analyze tab
- R8. Results are visible in the clip details sidebar

## Scope Boundaries

- No custom model training or fine-tuning — uses existing VLM inference
- No real-time/streaming classification — batch processing like other analysis ops
- No image region annotation — whole-frame True/False only
- No query history management UI (clearing individual queries can be a follow-up)
- No sequence filtering by custom query results (natural follow-up, not in scope)

## Context & Research

### Relevant Code and Patterns

- **Analysis operation registry**: `core/analysis_operations.py` — `AnalysisOperation` dataclass with key, label, tooltip, phase, default_enabled
- **Analysis pipeline dispatch**: `ui/main_window.py:_launch_analysis_worker()` — switch on op_key to launch typed worker
- **Worker pattern**: `ui/workers/description_worker.py` — `DescriptionWorker(CancellableWorker)` with `ThreadPoolExecutor`, frozen `@dataclass` task objects, progress/ready/error/completed signals
- **Cloud VLM path**: `core/analysis/description.py:describe_frame_cloud()` — base64 image + LiteLLM completion with provider-aware key routing
- **Local VLM path**: `core/analysis/description.py:describe_frame_local()` — Qwen3-VL (Apple Silicon) / Moondream (fallback)
- **Clip model**: `models/clip.py:Clip` — dataclass with `to_dict()`/`from_dict()` serialization, analysis fields are `Optional` and only serialized when non-None
- **Operation availability**: `core/analysis_availability.py:operation_is_complete_for_clip()` — per-op completion check
- **Sidebar display**: `ui/clip_details_sidebar.py` — section-based layout with headers and editable labels per analysis field
- **Feature registry**: `core/feature_registry.py` — `FeatureDeps` for on-demand dependency installation
- **QInputDialog**: Already used in `ui/tabs/collect_tab.py` and `ui/main_window.py` for text input

### Institutional Learnings

- **QThread duplicate signal delivery** (`docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`): Use guard flags on completion handlers, `Qt.UniqueConnection`, and `@Slot()` decorators. Applies to the new worker's signal connections.
- **Source ID mismatch** (`docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`): Always look up clips by ID from the project's canonical `clips_by_id`, not from worker-internal copies.

## Key Technical Decisions

- **Dedicated field, not tags**: Custom query results get their own `custom_queries` field on `Clip` (list of result dicts) rather than overloading the `tags` field. Tags are simple strings for user organization; query results need structured data (query text, boolean match, confidence, model name). This keeps the data model clean and makes filtering straightforward.

- **Reuse existing VLM infrastructure**: The cloud path reuses `describe_frame_cloud`'s LiteLLM + base64 pattern with a different prompt. The local path reuses the Moondream/Qwen3-VL loading from `description.py`. No new model dependencies needed.

- **Structured yes/no prompt**: The VLM receives a prompt like: `"Does this image contain: {query}? Answer with exactly YES or NO, followed by a confidence percentage (0-100%)."` Response parsing extracts the boolean and confidence.

- **Cloud phase**: The operation runs in the "cloud" phase since it's I/O-bound for cloud VLMs and benefits from parallelism. Local tier runs through the same phase but with parallelism=1 since local models aren't thread-safe.

- **Query text stored with each result**: Each result dict stores the original query text, so multiple queries are self-describing and don't require a separate query registry.

## Open Questions

### Resolved During Planning

- **Where to store results?** → New `custom_queries` field on Clip (list of dicts), not tags. Rationale: structured data with confidence/model, accumulation semantics, clean separation from user tags.
- **How to get query text from user?** → `QInputDialog.getText()` intercept in `_on_quick_run_from_tab` when op_key is `"custom_query"`. Simple, follows existing pattern, no new dialog class needed.
- **What if user runs same query twice?** → Append a new result (potentially with different model/confidence). Accumulation is the principle — deduplication is a UI concern for later.

### Deferred to Implementation

- **Exact prompt wording for local vs cloud VLMs**: Local models may need simpler prompts. Tune during implementation based on response quality.
- **Response parsing edge cases**: VLMs may return unexpected formats. Implementation should handle gracefully and fall back to match=False with low confidence.
- **Parallelism setting**: Whether to expose a setting or hardcode (cloud=3, local=1). Can be decided during implementation.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
User selects "Custom Query" from Quick Run dropdown
  → QInputDialog.getText() prompts for query string
  → MainWindow stores query string, launches CustomQueryWorker
  → Worker iterates clips with ThreadPoolExecutor:
      For each clip thumbnail:
        → evaluate_custom_query(image_path, query, tier)
        → Constructs yes/no prompt with query text
        → Routes to cloud (LiteLLM) or local (Moondream/Qwen3-VL)
        → Parses VLM response → (match: bool, confidence: float)
        → Emits query_result_ready(clip_id, query, match, confidence, model)
  → MainWindow._on_custom_query_ready():
      → Appends {"query": str, "match": bool, "confidence": float, "model": str}
        to clip.custom_queries
  → Sidebar displays accumulated query results
```

## Implementation Units

- [ ] **Unit 1: Data Model — Add custom_queries field to Clip**

  **Goal:** Add structured storage for custom query results on the Clip model with full serialization support.

  **Requirements:** R3, R4, R5

  **Dependencies:** None

  **Files:**
  - Modify: `models/clip.py`
  - Test: `tests/test_clip_model.py` (or existing clip serialization test)

  **Approach:**
  - Add `custom_queries: Optional[list[dict]] = None` field to the `Clip` dataclass
  - Each dict: `{"query": str, "match": bool, "confidence": float, "model": str}`
  - Add serialization in `to_dict()` — only serialize when non-None, same pattern as `detected_objects`
  - Add deserialization in `from_dict()` — parse from saved data with validation
  - Add to Frame model as well if Frame-level custom queries are desired (mirror Clip pattern)

  **Patterns to follow:**
  - `detected_objects` field: same Optional[list[dict]] pattern, same serialization approach
  - `face_embeddings` field: validation pattern in `from_dict()`

  **Test scenarios:**
  - Round-trip serialization: Clip with custom_queries → to_dict → from_dict → identical data
  - Empty/None serialization: custom_queries=None should not appear in to_dict output
  - Backward compatibility: Loading a project saved before this feature (no custom_queries key) produces None

  **Verification:**
  - Existing clip serialization tests still pass
  - New tests confirm round-trip fidelity

---

- [ ] **Unit 2: Core Analysis — Create custom query evaluation module**

  **Goal:** Implement the core yes/no VLM evaluation logic with cloud and local tier support.

  **Requirements:** R1, R2, R3, R6

  **Dependencies:** Unit 1

  **Files:**
  - Create: `core/analysis/custom_query.py`
  - Modify: `core/feature_registry.py` (add feature entry)
  - Test: `tests/test_custom_query.py`

  **Approach:**
  - Main function: `evaluate_custom_query(image_path, query, tier=None) → (match, confidence, model_name)`
  - Tier routing follows `describe_frame()` pattern: check settings for tier, normalize legacy names
  - Cloud path: construct yes/no prompt, encode image as base64, call LiteLLM via the same pattern as `describe_frame_cloud()` (provider-aware key routing, model normalization)
  - Local path: use existing `describe_frame_local()` infrastructure to load Moondream/Qwen3-VL, but with the yes/no prompt
  - Response parser: extract YES/NO and confidence percentage from VLM text output. Handle variations ("Yes, 85%", "YES (confidence: 90%)", "no", etc.)
  - Feature registry: add `"custom_query"` entry — same deps as `"describe"` since it uses the same VLM infrastructure
  - Include the certifi SSL fix pattern for Windows (same as shots.py/description.py)

  **Patterns to follow:**
  - `core/analysis/description.py:describe_frame_cloud()` — LiteLLM call pattern, API key routing, error formatting
  - `core/analysis/description.py:describe_frame()` — tier routing and settings lookup
  - `core/analysis/shots.py:classify_shot_type_tiered()` — tiered dispatch pattern

  **Test scenarios:**
  - Response parsing: "YES, 92%" → (True, 0.92), "No (15%)" → (False, 0.15), "yes" → (True, 1.0)
  - Malformed response handling: "I'm not sure" → (False, 0.0) or similar fallback
  - Tier routing: verify cloud vs local dispatch based on settings

  **Verification:**
  - Response parser handles at least 5 common VLM response formats
  - Feature registry entry allows dependency check and install

---

- [ ] **Unit 3: Worker — Create CustomQueryWorker**

  **Goal:** Background worker that evaluates a custom query across multiple clips with progress reporting.

  **Requirements:** R2, R3

  **Dependencies:** Unit 2

  **Files:**
  - Create: `ui/workers/custom_query_worker.py`
  - Test: `tests/test_custom_query_worker.py`

  **Approach:**
  - Subclass `CancellableWorker` from `ui/workers/base.py`
  - Constructor takes: clips, query string, sources_by_id, tier, parallelism, skip_existing flag
  - Frozen `@dataclass` task object: `CustomQueryTask(clip_id, thumbnail_path, query)`
  - `skip_existing` logic: skip clips that already have a result for this exact query string
  - Signals: `progress(int, int)`, `query_result_ready(str, str, bool, float, str)` (clip_id, query, match, confidence, model), `analysis_completed()`, `error(str)`
  - `_process_task()` calls `evaluate_custom_query()` and returns result tuple
  - `run()` uses `ThreadPoolExecutor` with configurable parallelism (cloud=3, local=1)
  - Include retry logic for cloud rate limits, same pattern as DescriptionWorker

  **Patterns to follow:**
  - `ui/workers/description_worker.py` — exact structural template (task dataclass, build_tasks, process_task, run with executor)
  - `ui/workers/shot_type_worker.py` — simpler variant for reference

  **Test scenarios:**
  - Worker processes multiple clips and emits progress signals
  - Cancellation stops processing mid-batch
  - Skip-existing correctly filters clips with prior results for same query

  **Verification:**
  - Worker follows CancellableWorker lifecycle (start → progress → ready → completed)
  - Error in one clip doesn't stop processing of others

---

- [ ] **Unit 4: Pipeline Integration — Register operation and wire up MainWindow**

  **Goal:** Make "Custom Query" available as an analysis operation and wire the full dispatch/result flow in MainWindow.

  **Requirements:** R2, R4, R7

  **Dependencies:** Unit 3

  **Files:**
  - Modify: `core/analysis_operations.py`
  - Modify: `core/analysis_availability.py`
  - Modify: `ui/main_window.py`
  - Modify: `ui/tabs/analyze_tab.py`
  - Test: `tests/test_analysis_operations.py` (if exists), `tests/test_analysis_availability.py`

  **Approach:**
  - Add `AnalysisOperation("custom_query", "Custom Query", "Search clips for specific visual content using a VLM", "cloud", False)` to `ANALYSIS_OPERATIONS`
  - Add `custom_query` case to `operation_is_complete_for_clip()` — returns True if clip has any custom_queries results (note: this is a simplification; "complete" means "has been queried at least once")
  - In `_on_quick_run_from_tab()`: intercept `op_key == "custom_query"` to show `QInputDialog.getText()` before dispatching. Store the query text on `self._custom_query_text`.
  - Add `_launch_custom_query_worker()` in MainWindow following `_launch_description_worker()` pattern
  - Add `_on_custom_query_ready()` handler: look up clip by ID from `project.clips_by_id`, append result dict to `clip.custom_queries` (initializing list if None)
  - Add `_on_custom_query_finished()` handler with guard flag pattern (per learnings)
  - Wire worker signals with `Qt.UniqueConnection` where appropriate

  **Patterns to follow:**
  - `_launch_description_worker()` — worker creation, signal wiring, start
  - `_on_description_ready()` — result application to clip model
  - `_on_pipeline_colors_finished()` — phase completion with guard flag

  **Test scenarios:**
  - Operation appears in ANALYSIS_OPERATIONS list
  - Availability check: clip with no custom_queries → not complete; clip with results → complete
  - Quick run with "custom_query" triggers text input dialog (manual QA)

  **Verification:**
  - "Custom Query" appears in Quick Run dropdown
  - Selecting it prompts for query text
  - Results are written to clip.custom_queries after worker completes

---

- [ ] **Unit 5: Sidebar Display — Show custom query results**

  **Goal:** Display accumulated custom query results in the clip details sidebar.

  **Requirements:** R4, R8

  **Dependencies:** Unit 4

  **Files:**
  - Modify: `ui/clip_details_sidebar.py`
  - Test: Manual QA (sidebar is tightly coupled to Qt rendering)

  **Approach:**
  - Add a "Custom Queries" section header below existing analysis sections
  - For each query result in `clip.custom_queries`, display: query text, match indicator (checkmark/X), confidence percentage, model name
  - Use a compact layout — each result is one row: `✓ "blue flower" (92%, gpt-4o)` or `✗ "red car" (15%, moondream-2b)`
  - Section is hidden when `custom_queries` is None or empty
  - Multi-clip selection: show count of queries across selected clips or "N clips selected" placeholder

  **Patterns to follow:**
  - Object labels display section in sidebar — section header + content pattern
  - Description display section — optional visibility based on data presence

  **Test scenarios:**
  - Single clip with 3 custom queries: all 3 display correctly
  - Clip with no custom queries: section is hidden
  - Multi-select: shows appropriate placeholder

  **Verification:**
  - Sidebar correctly renders accumulated queries with match status
  - Section visibility toggles based on data presence

---

- [ ] **Unit 6: Agent Tool — Expose custom query to the chat agent**

  **Goal:** Allow the integrated chat agent to run custom visual queries programmatically.

  **Requirements:** R1, R2 (agent parity)

  **Dependencies:** Unit 4

  **Files:**
  - Modify: `core/chat_tools.py`
  - Modify: `core/tool_executor.py`
  - Test: `tests/test_chat_tools.py` (if exists)

  **Approach:**
  - Add a `custom_query` tool definition that accepts a query string and optional clip scope
  - Tool executor dispatches to the same `evaluate_custom_query()` function or triggers the worker via MainWindow
  - Returns structured results: list of clip IDs with match/confidence
  - Follow existing agent tool pattern with `{"success": True/False, "result": data}` return format

  **Patterns to follow:**
  - Existing analysis agent tools in `core/chat_tools.py`
  - `core/tool_executor.py` dispatch pattern

  **Test scenarios:**
  - Agent can invoke custom query with a query string
  - Results include clip IDs, match status, and confidence

  **Verification:**
  - Agent tool returns structured results matching the GUI behavior

## System-Wide Impact

- **Interaction graph:** Quick Run dropdown → QInputDialog → MainWindow worker dispatch → CustomQueryWorker → Clip model update → sidebar refresh. Analysis pipeline phase system handles ordering.
- **Error propagation:** Worker errors emit per-clip error signals (same as DescriptionWorker). Pipeline-level errors show in status bar. SSL/network errors on Windows handled by certifi pattern.
- **State lifecycle risks:** Multiple concurrent custom query runs should be prevented (worker guard pattern). Clip model mutations happen on main thread via signal handler.
- **API surface parity:** Agent tool (Unit 6) ensures agent can do everything the GUI can. MCP server may need a matching tool in a follow-up.
- **Integration coverage:** End-to-end flow (query → VLM call → result on clip → sidebar display → project save/load) needs manual QA since it spans UI, network, and serialization.

## Risks & Dependencies

- **VLM response format variability**: Different VLMs (GPT-4o, Claude, Gemini, Moondream) may format yes/no answers differently. The response parser needs to be robust. Mitigated by testing against multiple response formats.
- **Local VLM accuracy**: Moondream and small local models may not be accurate enough for nuanced visual queries ("blue flower" vs. "purple flower"). This is a known limitation, not a bug — confidence scores help users assess reliability.
- **Cost awareness**: Each clip requires one VLM API call per query. Running a query across 500 clips could be expensive for cloud tier. Consider adding a confirmation dialog showing estimated cost/count before running.

## Sources & References

- Related code: `core/analysis/description.py` (VLM infrastructure), `ui/workers/description_worker.py` (worker pattern)
- Related code: `core/analysis_operations.py` (operation registry), `core/analysis_availability.py` (completion checks)
- Related code: `ui/clip_details_sidebar.py` (result display)
- Learnings: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- Learnings: `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
