---
title: "feat: Add Embeddings as a user-triggerable analysis operation"
type: feat
status: active
date: 2026-04-13
deepened: 2026-04-13
---

# feat: Add Embeddings as a user-triggerable analysis operation

## Overview

Add **Generate Embeddings** as a first-class checkbox in the Analyze tab's analysis picker dialog, alongside other operations like Extract Colors / Classify Shots / Detect Gaze. Embeddings already exist as a Clip field and are auto-computed as a side effect of running similarity_chain or staccato — but there is no UI affordance to compute them directly. This makes the new Free Association sequencer (which uses embeddings for local candidate shortlisting) harder to get good results from on a fresh project.

## Problem Frame

Embeddings (DINOv2 visual vectors, 768-dim) are a building block used by multiple sequencers — similarity_chain, staccato, the new free_association — and by reference_guided's embedding dimension. Today, getting them onto your clips requires running one of those sequencers, which has visible side effects (you get an unwanted sequence). Several other analysis operations follow the same shape (run an extractor, write back to the Clip, persist) and are already exposed as checkboxes via `core/analysis_operations.py`. Embeddings is the conspicuous missing entry. Adding it requires no new infrastructure — just registration, a worker, and a dispatch entry.

## Requirements Trace

- R1. The analysis picker dialog shows a "Generate Embeddings" checkbox in the appropriate phase group.
- R2. Selecting the checkbox and running analysis populates `clip.embedding` and `clip.embedding_model` for clips that don't already have embeddings.
- R3. Already-computed embeddings are skipped (no redundant work).
- R4. The operation respects cancellation — partial results persist on cancel (mutate-in-place pattern, mirroring gaze).
- R5. The dependency check (torch + transformers) prompts the user to install if missing, using the existing feature registry pattern.
- R6. Embeddings are persisted via the existing `Clip.to_dict`/`from_dict` path (no schema change needed).
- R7. The auto-compute path used by similarity_chain / staccato / free_association continues to work as a fallback when the user hasn't explicitly run embeddings.

## Scope Boundaries

- **No** boundary embeddings (`first_frame_embedding` / `last_frame_embedding`) — those are computed only by match_cut via `extract_boundary_embeddings`, on a different code path. Adding them would be a separate operation.
- **No** model selection UI — DINOv2 is the hardcoded default and there's no existing setting for picking CLIP vs DINOv2.
- **No** GPU/MPS device selector — torch handles device automatically; existing `core/analysis/embeddings.py` doesn't expose this.
- **No** new Clip schema fields — `embedding` and `embedding_model` already exist.
- **No** changes to cost estimation infrastructure — `TIME_PER_CLIP`, `OPERATION_LABELS`, `METADATA_CHECKS`, and `local_model_parallelism` for embeddings are already wired.
- **No** changes to the existing `_auto_compute_embeddings` fallback path — it already checks `clip.embedding is None` so it becomes a no-op when the user has run embeddings explicitly.

## Context & Research

### Relevant Code and Patterns

- `core/analysis_operations.py` — `AnalysisOperation` dataclass and the `ANALYSIS_OPERATIONS` registry. Phase one of: `local | sequential | cloud`. Embeddings should be `sequential` (model load/unload lifecycle, not parallelizable across processes).
- `ui/workers/gaze_worker.py` — closest reference pattern. Extends `CancellableWorker`, signals `progress(int, int)`, `<op>_ready(...)`, `<op>_completed()`, plus inherited `error(str)` and `finished()`. Mutates clips in place.
- `core/analysis/embeddings.py` — `extract_clip_embeddings_batch(thumbnail_paths) -> list[list[float]]`, `is_model_loaded()`, `unload_model()`, model tag `"dinov2-vit-b-14"`.
- `core/remix/__init__.py:473` — `_auto_compute_embeddings(clips)` — the synchronous fallback. Shows the exact "skip already-computed" + "call batch function" + "assign back" pattern. Worker should mirror this logic.
- `ui/main_window.py:3280` — `_launch_analysis_worker(op_key, clips)` if/elif chain. Add `elif op_key == "embeddings":` branch and a `_launch_embeddings_worker(clips)` method modeled after `_launch_gaze_worker` (line 3399).
- `ui/main_window.py:3688` — `_on_pipeline_gaze_finished` shows the guard-flag + `_on_analysis_phase_worker_finished(op_key)` completion handler pattern. Mirror it for embeddings.
- `core/cost_estimates.py` — already has all four entries for embeddings (TIME_PER_CLIP, OPERATION_LABELS, METADATA_CHECKS, parallelism). No changes needed.
- `core/feature_registry.py` — `FEATURE_DEPS["embeddings"]` already declares `torch + transformers`, ~450 MB. The dialog flow at `ui/dialogs/reference_guide_dialog.py:437` shows the install-prompt pattern.

### Institutional Learnings

- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` — Qt's `finished` signal can be delivered twice. Use a guard flag (`self._embeddings_finished_handled = False`) plus `Qt.UniqueConnection` on the completion signal. This is the same pattern already used by gaze (`_gaze_finished_handled`) and other workers in `main_window.py`.

## Key Technical Decisions

- **Phase = `"sequential"`**: Matches gaze. The DINOv2 model is loaded once into memory and reused across all clips in a batch — running this in parallel with other model-loading operations would compete for GPU/RAM. Sequential phase ensures it runs alone in its slot.
- **Default enabled = `False`**: Heavy optional dependency (~450 MB). Users opt in. Matches the convention for other heavy operations like `face_embeddings` and `gaze`.
- **Operation key = `"embeddings"`**: Matches the key already used by `METADATA_CHECKS`, `FEATURE_DEPS`, `TIME_PER_CLIP`, and `algorithm_config.py`'s `required_analysis: ["embeddings"]` references. Reusing the existing key avoids any ripple changes.
- **Batch processing inside the worker**: `extract_clip_embeddings_batch` already batches internally (size 32). For UX, the worker chunks the input into windows of N (default 16) so progress can update mid-job and cancellation is checked between chunks. The chunk size is a tradeoff between overhead (small chunks = repeated model warm-up signals) and responsiveness; 16 keeps progress visible without much overhead since the model stays loaded.
- **Skip-existing default = `True`**: Match gaze. Re-run requires either the user clearing the field manually or a separate "force regenerate" affordance (out of scope). The auto-compute path uses the same skip logic, so they stay consistent.
- **Mutate clips in place**: Same pattern as every other analysis worker. Partial results survive cancellation.
- **Fail soft on missing thumbnails**: Skip clips without `thumbnail_path` and log a warning, rather than aborting the whole run. Matches the auto-compute path.

## Open Questions

### Resolved During Planning

- **Where to insert the operation in the picker order**: After `face_embeddings` and `gaze` (the other heavy sequential ML operations). Keeps related operations grouped.
- **Should we expose the chunk size as a setting?** No — it's an implementation tuning parameter, not a user-facing concern. Hardcode at 16 with a module-level constant for easy adjustment.
- **Should we handle cases where the model is already loaded from a previous run?** Yes — `is_model_loaded()` exists for this. Skip the load if true; only unload at the end if we loaded it.
- **Cost confirmation gate**: All three phases (`local`, `sequential`, `cloud`) pass through the same cost-estimation code path — the phase only affects ordering/parallelism, not whether the cost gate runs. Verify during implementation that the gate displays the embeddings line correctly using the existing `OPERATION_LABELS["embeddings"]` and `TIME_PER_CLIP["embeddings"]` entries.

### Deferred to Implementation

- **Exact wording of the tooltip and label**: Draft "Generate Embeddings" / "Extract DINOv2 visual feature vectors for similarity-based sequencing". Adjust if it reads awkwardly next to other operations.
- **Whether to emit a per-batch or per-clip `embedding_ready` signal**: Per-clip is more uniform with other workers but requires the worker to map batch outputs back to individual clips (already done — same logic as `_auto_compute_embeddings`). Per-batch is simpler. Decide based on whether any UI consumer needs per-clip notification (no current consumer does, but a future "thumbnails go green when analyzed" UI would).

## Implementation Units

- [ ] **Unit 1: Register embeddings in the analysis operations catalog**

  **Goal:** Add the operation entry so it appears in the analysis picker.

  **Requirements:** R1

  **Dependencies:** None

  **Files:**
  - Modify: `core/analysis_operations.py`
  - Test: `tests/test_analysis_operations.py` (extend if it exists, otherwise add focused assertions to an existing analysis-config test)

  **Approach:**
  - Add `AnalysisOperation("embeddings", "Generate Embeddings", "Extract DINOv2 visual feature vectors for similarity-based sequencing", "sequential", False)` to the `ANALYSIS_OPERATIONS` list, ordered after `face_embeddings` and `gaze`.
  - The operation is automatically picked up by `OPERATIONS_BY_KEY`, `SEQUENTIAL_OPS`, and the picker dialog (which iterates `ANALYSIS_OPERATIONS`).

  **Patterns to follow:**
  - `gaze` and `face_embeddings` entries in the same file.

  **Test scenarios:**
  - Happy path: `OPERATIONS_BY_KEY["embeddings"]` returns an `AnalysisOperation` with `key="embeddings"`, `phase="sequential"`, `default_enabled=False`.
  - Happy path: `"embeddings" in SEQUENTIAL_OPS`.
  - Edge case: `"embeddings" not in DEFAULT_SELECTED` (it's opt-in).

  **Verification:**
  - Launching the app and opening the analysis picker shows a "Generate Embeddings" checkbox under the sequential phase group.

- [ ] **Unit 2: Add EmbeddingAnalysisWorker**

  **Goal:** Background worker that runs DINOv2 batch extraction on selected clips, mutating `clip.embedding` and `clip.embedding_model` in place.

  **Requirements:** R2, R3, R4 (the install-prompt R5 is enforced by the launch site in Unit 3, not inside the worker)

  **Dependencies:** Unit 1

  **Files:**
  - Create: `ui/workers/embedding_worker.py`
  - Modify: `ui/workers/__init__.py` (export `EmbeddingAnalysisWorker`)
  - Test: `tests/test_embedding_worker.py`

  **Approach:**
  - Class `EmbeddingAnalysisWorker(CancellableWorker)` mirroring `GazeAnalysisWorker`.
  - Constructor: `(clips, sources_by_id=None, skip_existing=True, chunk_size=16, parent=None)`. `sources_by_id` is accepted for launch-site API parity with other workers but is **never read inside the worker** — embeddings only need each clip's `thumbnail_path`. Document this in the docstring.
  - Signals: `progress(int, int)`, `embedding_ready(str)` (clip_id), `analysis_completed()`, plus inherited `error(str)` and `finished()`.
  - `run()`:
    1. Filter clips: skip those with `clip.embedding is not None` if `skip_existing`, skip those with no `thumbnail_path` (log warning). `extract_clip_embeddings_batch` itself handles unreadable image files internally (returns zero vectors), so per-thumbnail try/except inside the worker is redundant — the only worker-level filter needed is the `thumbnail_path` existence check.
    2. If 0 clips remain → emit `analysis_completed` and return early.
    3. Always-load-then-unload pattern (matches `gaze_worker.py`): track a single `model_loaded` boolean. If `is_model_loaded()` returns False, call the embeddings module's lazy loader and set `model_loaded = True`. **Don't** try to detect "someone else loaded it and skip the unload" — the underlying `is_model_loaded()` reads `_model` without holding `_model_lock`, so the check has a TOCTOU window. Sequential phase guarantees no concurrent embeddings consumer is active, so simpler is correct.
    4. Process in chunks of `chunk_size`. Maintain a per-chunk list of `(clip, thumbnail_path)` pairs so the returned vectors map back 1:1 by index (per-chunk equivalent of `_auto_compute_embeddings`'s `zip(needs_embedding, embeddings)`). For each chunk: check `is_cancelled()`, call `extract_clip_embeddings_batch(thumbnail_paths)`, zip results back to the chunk's clips, assign `clip.embedding` and `clip.embedding_model = _EMBEDDING_MODEL_TAG` (import the constant from `core.analysis.embeddings` — it exists at module scope, line 31; do not duplicate the literal string). Emit `embedding_ready(clip.id)` per clip and `progress(processed_so_far, total)` after each chunk.
    5. In `finally`: unload the model if `model_loaded` is True; always emit `analysis_completed`.
    6. **Chunk-level error policy: abort with `error(str)` signal.** A model-level exception (OOM, model corruption, library mismatch) usually means subsequent chunks will also fail. Catch the exception around each chunk's `extract_clip_embeddings_batch` call, emit `error` once with a descriptive message, break the chunk loop, let `finally` unload and emit `analysis_completed`. Partial results from already-processed chunks are preserved on the clips (mutate-in-place).

  **Patterns to follow:**
  - `ui/workers/gaze_worker.py` — overall structure, model load/unload lifecycle, mutate-in-place + signal emission, `try/finally` for completion.
  - `core/remix/__init__.py:473` `_auto_compute_embeddings` — exact filter/call/assign sequence for embeddings specifically.

  **Test scenarios:**
  - Happy path: Worker with mocked `extract_clip_embeddings_batch` returning N vectors processes N clips, sets `clip.embedding` and `clip.embedding_model` on each, emits `embedding_ready` per clip, emits final `analysis_completed`.
  - Happy path: `skip_existing=True` and clips already have embeddings → `extract_clip_embeddings_batch` is never called, `analysis_completed` still emits.
  - Edge case: Empty clip list → no model load, immediate `analysis_completed`.
  - Edge case: Mix of clips with and without thumbnails → only clips with thumbnails are processed; missing-thumbnail clips logged and skipped.
  - Edge case: Cancellation between chunks (set `_cancel_event` before second chunk) → second chunk does not run, partial results persist on already-processed clips, `analysis_completed` still emits.
  - Error path: `extract_clip_embeddings_batch` raises in the middle of a chunk → worker emits `error` exactly once, breaks the chunk loop, partial results from prior chunks are preserved on the clips, `analysis_completed` still emits in `finally`.
  - Edge case: Model already loaded when worker starts (`is_model_loaded() == True`) → worker does NOT call the loader, but DOES still call `unload_model()` in `finally` (always-unload pattern; matches gaze).

  **Verification:**
  - All clips with thumbnails get `embedding` and `embedding_model` set.
  - The worker never crashes on bad input (missing thumbnail, bad clip, empty list).
  - Cancellation produces partial results, never corrupted state.

- [ ] **Unit 3: Wire the worker into the analysis dispatch**

  **Goal:** Hook the new worker into the `_launch_analysis_worker` dispatch chain in `MainWindow` so selecting the checkbox actually runs the worker.

  **Requirements:** R2, R3, R5, R7

  **Dependencies:** Unit 2

  **Files:**
  - Modify: `ui/main_window.py`
  - Test: `tests/test_main_window_analysis_dispatch.py` (or extend existing test if there is one — search for any existing test that exercises `_launch_analysis_worker`)

  **Approach:**
  - Add `elif op_key == "embeddings": self._launch_embeddings_worker(clips)` in `_launch_analysis_worker` (around line 3313, alongside the other `elif` entries).
  - Add `_launch_embeddings_worker(self, clips: list)` method modeled directly on `_launch_gaze_worker` (line 3399). Set guard flag `self._embeddings_finished_handled = False`, build `sources_by_id`, instantiate `EmbeddingAnalysisWorker`, connect signals (`progress` → `_on_embeddings_progress`, `embedding_ready` → `_on_embedding_ready`, `analysis_completed` → `_on_pipeline_embeddings_finished` with `Qt.UniqueConnection`, `error` → `_on_embeddings_error`, `finished` → `deleteLater` and clearing the worker reference).
  - Add the four `@Slot()` handlers: `_on_embeddings_progress(current, total)` updates the progress bar; `_on_embedding_ready(clip_id)` updates clip-browser visual state if applicable (look at `_on_gaze_ready` for the precedent — may be a no-op for embeddings since there's no visible per-clip indicator yet); `_on_pipeline_embeddings_finished` uses the guard flag and calls `_on_analysis_phase_worker_finished("embeddings")`; `_on_embeddings_error` logs and shows a status-bar message.
  - **No additional `check_feature_ready` guard inside `_launch_embeddings_worker`.** The pipeline already filters via `_filter_available_analysis_operations` (`ui/main_window.py:3209`), which calls `_ensure_analysis_operation_available` → `get_operation_feature_candidates` → `check_feature_ready`, and `core/analysis_dependencies.py:43-44` already maps `embeddings` to the `embeddings` feature. By the time a worker is launched, the user has already been prompted (and either installed or declined). Adding a second check duplicates pipeline-level filtering — the gaze worker (the reference template) does not do this. R5 is satisfied by the existing pipeline plumbing, not by a launcher-level guard.

  **Patterns to follow:**
  - `_launch_gaze_worker` and its handlers (`_on_pipeline_gaze_finished`, `_on_gaze_error`, etc.) — exact structural template.
  - `ui/dialogs/reference_guide_dialog.py:434-457` — the `check_feature_ready` + install-prompt pattern.

  **Test scenarios:**
  - Happy path: `_launch_analysis_worker("embeddings", clips)` instantiates an `EmbeddingAnalysisWorker` and starts it.
  - Happy path: Worker `analysis_completed` signal triggers `_on_pipeline_embeddings_finished` exactly once even if the signal fires twice (guard flag works).
  - Integration: Worker `error` signal logs and shows a status-bar message but does not block subsequent operations.
  - Integration: When the user declines installing torch+transformers at the pipeline-filter step (existing path, not new code), the embeddings operation is filtered out before any launcher runs — the launcher itself never executes.

  **Verification:**
  - Toggling the "Generate Embeddings" checkbox in the picker and clicking Run triggers a worker that processes the selected clips end to end.
  - The progress bar updates during processing and clears on completion.

- [ ] **Unit 4: Verify and document**

  **Goal:** Confirm the new operation is visible to the rest of the app correctly (cost gate, sequence-tab card availability, auto-compute fallback). This unit performs no code changes — it is a docs + manual-verification pass to catch any regression in the surfaces R5/R6/R7 implicitly cover.

  **Requirements:** R7 (verifies the auto-compute fallback in `core/remix/__init__.py` is unaffected; R5 was implemented in Unit 3 via the existing pipeline-level filter, R6 was already satisfied by existing serialization code per Scope Boundaries — both are verified in passing here but not newly implemented)

  **Dependencies:** Unit 3

  **Files:**
  - Modify: `docs/user-guide/` — add a brief note about the new analysis option (locate the appropriate file, e.g., `analyze.md` or similar).
  - No code changes expected; this unit is verification + docs.

  **Approach:**
  - Open the cost gate by selecting embeddings + a few clips and verify the line item shows "Embeddings" with a sensible time estimate. If the line is missing or labeled wrong, debug — but research shows `OPERATION_LABELS["embeddings"]` and `TIME_PER_CLIP["embeddings"]` already exist.
  - Run embeddings on a few clips, then go to the Sequence tab and confirm the `similarity_chain` algorithm (UI label "Human Centipede") is now available without the install prompt or warning.
  - Open and re-load a project that has clips with `clip.embedding` populated; verify the embeddings survived the round-trip (this is already covered by existing tests, but worth a manual pass).
  - Confirm the existing `_auto_compute_embeddings` fallback in `core/remix/__init__.py` still runs for clips without embeddings (didn't accidentally break the auto path).
  - Add a section to the user guide explaining how to run embeddings explicitly and why a user might want to.

  **Test scenarios:**
  - Test expectation: none — this unit is documentation + manual verification. Behavioral correctness is covered by Units 2 and 3.

  **Verification:**
  - User-facing docs mention the new operation.
  - Manual smoke test confirms end-to-end: pick clips → check embeddings → Run → embeddings populated → Free Association / Human Centipede now use them automatically.

## System-Wide Impact

- **Interaction graph:** `MainWindow._launch_analysis_worker` adds one new branch; the new worker emits to four new handler methods (progress / ready / completed / error). No new cross-component callbacks beyond the established analysis-pipeline pattern.
- **Error propagation:** Worker errors route to `_on_embeddings_error` → status bar message + log. Pipeline advances via `_on_analysis_phase_worker_finished("embeddings")` so a single failure doesn't stall subsequent operations.
- **State lifecycle risks:** Mutate-in-place pattern means partial results persist on cancel — same as gaze. No risk of corrupted state.
- **API surface parity:** Embeddings are produced by both the new explicit operation AND the existing `_auto_compute_embeddings` path used by similarity_chain / staccato / free_association. Both paths must produce embeddings in the same shape (768-dim list, same `embedding_model` tag). Since both call `extract_clip_embeddings_batch` directly, this is guaranteed.
- **Integration coverage:** A clip processed by the explicit operation should subsequently NOT be re-processed by the auto-compute fallback. Guaranteed by the `clip.embedding is None` check at both call sites.
- **Unchanged invariants:** Existing `Clip.embedding` / `Clip.embedding_model` schema, serialization, and validation are unchanged. The `_auto_compute_embeddings` fallback is unchanged. Cost-estimate infrastructure is unchanged. Feature registry is unchanged. Sequence-tab card-availability mapping is unchanged (it already checks for embeddings the right way).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| QThread `finished` signal duplicate delivery causes pipeline-advance handler to fire twice | Apply the guard flag pattern from the documented learning (`_embeddings_finished_handled = False` on launch, set True in handler, return early on second invocation). Use `Qt.UniqueConnection` on the `analysis_completed` connection. |
| Heavy dependency (torch + transformers, ~450 MB) not installed | Use the existing `check_feature_ready("embeddings")` + `install_for_feature("embeddings")` flow (proven in `reference_guide_dialog.py`). On decline, advance the pipeline without aborting it. |
| Model already loaded from a prior call (e.g., similarity_chain ran during this session), then unloaded by the worker, leaving subsequent algorithms slow | Track a `we_loaded_it` boolean — only call `unload_model()` if we were the ones who loaded it. |
| Single bad thumbnail crashes the whole batch | Wrap `extract_clip_embeddings_batch` per-chunk in `try/except`; on chunk failure, log and abort with `error` signal (a chunk-level failure usually indicates a systemic issue like OOM). |
| Cancellation latency feels poor for large jobs | Chunk processing (default 16 clips/chunk) gives cancellation checkpoints between chunks. Acceptable for most use cases; large projects may want a smaller chunk size. |

## Documentation / Operational Notes

- Update `docs/user-guide/` to mention "Generate Embeddings" as an analysis option, and explain why a user might want to run it explicitly (Free Association, Human Centipede, Reference Guide all use embeddings; running it once up-front is faster than letting each sequencer auto-compute).
- No migration concerns — existing projects' clips that already have `embedding` set are unaffected.
- No changes to release notes / CI / build pipeline.

## Sources & References

- Related code:
  - `core/analysis_operations.py`
  - `ui/workers/gaze_worker.py`
  - `core/analysis/embeddings.py`
  - `core/remix/__init__.py` (`_auto_compute_embeddings`)
  - `ui/main_window.py` (`_launch_analysis_worker`, `_launch_gaze_worker`)
  - `core/cost_estimates.py`
  - `core/feature_registry.py`
- Related PRs/issues: #87 (Free Association sequencer — the immediate consumer that motivated this work)
- Institutional learnings: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
