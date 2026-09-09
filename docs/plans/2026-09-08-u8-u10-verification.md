# U8–U10 verification

Scope: U8, U9, and U10 in [the shared editing engine plan](2026-09-06-1218-refactor-shared-editing-engine-plan.md). U11–U17 remain separate work. This is unit acceptance evidence, not a packaged release certification.

## Runtime evidence

The full command `HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen python -m pytest tests/ scene_ripper_mcp/tests/ -q` passed at `c1f689a`: **6,108 passed, 2 skipped**, in 718.99 seconds. The interpreter was `/Users/derrickschultz/miniconda3/bin/python` on macOS. Log: `/tmp/u8-u10-full-c1f689a.log`.

The subsequent typing cleanup changes local variable names, annotations, enum qualification, and explicit narrowing of already-validated values. Its affected checks passed:

- 156 checkpoint, import recovery, boundary embedding, gaze, and object detection tests (`/tmp/u8-u10-type-cleanup-tests.log`).
- 332 scalar, playback mapping, runtime smoke, audio transcription recovery, and MCP tests (`/tmp/u8-u10-final-static-regressions.log`).
- 10 verified agent summary tests (`/tmp/u8-u10-summary-types-tests.log`).

The two skips were independently identified with `pytest tests/test_word_llm_composer.py tests/test_word_sequencer_e2e.py -q -rs`: the opt-in real Ollama latency test and the opt-in Word Sequencer MP4 smoke test. The decoded FFmpeg tests in `test_media_render_e2e.py` ran without skips. No remote inference service was needed for this verification.

## Requirement evidence

All test files below ran in the full suite. Assertions and the corresponding implementation seams were inspected; the table identifies what each check proves.

| Requirement | Implementation and evidence |
|---|---|
| U8: One source/timeline conversion seam | Production `SequenceClip(...)` construction is confined to `core/sequence_time.py`. `models/media_time.py` distinguishes video frames, audio samples, still holds, and rational timeline ranges. `test_media_time.py` checks source frame 240 and 24/25/30/30000/1001 rates. |
| U8: Shared adjacent boundaries | `test_long_mixed_rate_sequence_rounds_shared_boundaries_once` checks 20,000 one-frame entries against the cumulative rational duration, including half-frame rounding. |
| U8: Distinct still, audio, and VFR coordinates | `test_still_audio_and_vfr_have_distinct_time_coordinates` checks sample-based audio and timestamp-based video. Decoded VFR and nonzero-PTS tests in `test_media_render_e2e.py` verify actual output. |
| U8: Preserve ambiguous legacy data | `test_legacy_sequence_migration.py` covers all v1.0–v1.4 fixtures, malformed values, both sequence keys, unknown fields, and ambiguous ranges. Original values remain in diagnostics; unresolved ranges refuse rendering. |
| U8: Resolution through every interface | `test_legacy_sequence_resolution.py` exercises GUI, CLI, MCP, and built-in agent resolution, undo/redo, exact original-file backup, stale-target rejection, and blocking all render outputs until resolved. |
| U9: Shared render validation | Export, preview, EDL, and GUI mapping call `compile_render_plan`. `test_render_plan.py` checks immutable plans, references, gaps, missing media, and overlapping tracks. GUI mapping may inspect offline media; output rendering retains media preflight. |
| U9: Exact decoded frame selection | `test_mixed_rate_preview_and_export_match_independent_frame_oracle` uses four distinct source-ID ranges trimmed from frame 240, with hardcoded output cuts 0/29/56/79/102. Both outputs decode to the independently specified frame list. Other decoded tests check first/last frames and one-frame upsampling. |
| U9: Transforms, stills, gaps, and music agree | `test_nonzero_offset_and_transforms_agree_in_preview_and_export` covers reverse and flips. `test_still_gap_and_short_music_match_preview` decodes visual frames and PCM, asserting the sound interval and silence after the music ends. Silent/video-audio cut alignment is separately checked. |
| U9: Atomic output and cancellation | `test_export_publication_is_atomic` injects encoder failure and cancellation, verifies no partial final output, preserves existing output bytes, and decodes successful output. `test_cancel_terminates_an_active_encoder` checks active process cancellation. |
| U9: EDL limitations explicit | `test_render_plan.py` checks unsupported transformations, stills, and gaps; failed EDL publication preserves an earlier file. |
| U10: Typed records and correct invalidation | `test_analysis_records.py` checks successful empty results versus failed/missing records, source content, trim, model, operation/schema versions, parameters, sampling, and prompts. The operation and completion suites exercise these decisions through the migrated analysis implementations. |
| U10: Explicit legacy policy for U7 analyses | `core/operations/legacy_reuse.py` validates saved values and inputs before acceptance, retaining unknown provenance. `test_legacy_analysis_reuse.py`, `test_legacy_reuse_agent.py`, `test_legacy_reuse_dialog.py`, and the legacy audio tests cover explicit reuse and invalidation across interfaces. `test_legacy_face_recomputation.py` proves faces recompute across all four interfaces and preserve old vectors on failure. |
| U10: Stage before publishing references | `test_staged_artifact_is_pinned_before_manifest_publication` collects during staging and after manifest publication. Interrupted copy/publication and database failure tests verify conservative recovery and cleanup of temporary files. |
| U10: Targeted missing-artifact recovery | `test_missing_embedding_recomputes_only_its_referencing_clip` removes one artifact from a saved two-clip project and asserts one recomputation, one reuse, intact notes, and both sequence entries. Missing/corrupt artifact tests also verify saved record state and references survive. |
| U10: Closed projects, jobs, exports, and undo retain content | `test_artifact_store.py` tests closed project manifests and actual project undo/redo. `test_job_referenced_artifacts.py` uses a real queued/running runtime while retiring the producer. `core/project_export.py` holds a lease for the entire export. Receipt retention/pruning tests check saved copies, historical owners, interrupted saves, and concurrent readers. |
| U10: Never reclaim sources or uncertain files | The collector visits only indexed, unreferenced, unchanged, single-link regular files. `test_cleanup_never_deletes_source_unknown_or_replaced_files` checks source media, untracked content, and replacement hard links. Pending-manifest tests retain missing, corrupt, future-schema, and concurrently changed projects. |
| U10: Portable reconstruction | `test_project_export.py` exports managed arrays, loads the bundle into a separate artifact store, checks exact values and notes, and verifies the closed imported project retains its artifacts. Unsaved inline arrays are staged before bundle manifest publication. |
| Cross-cutting regressions | The full run includes the plan's named project, multi-sequence, spine import, operation contract, CLI/MCP integration, job lifecycle/recovery, build-support, and runtime-smoke files. Startup import boundaries remain tested. No new packaged-platform build or performance benchmark is claimed. |

## Review closure

The CE review of `11bfdf6..9b798a5` identified five actionable findings. `c2379a7` fixes failed brightness seeks, CLI cache isolation, dimensions influenced by unused media, and filesystem preflight in GUI mapping. `c1f689a` moves artifact hydration off the GUI thread. The latter has native GUI heartbeat, cancellation, close, stale-result, and failure tests, plus a visually inspected progress dialog. The review's original report remains historical evidence of the pre-fix state.

## Static checks and baseline exceptions

The migrated shared scope passes mypy in **141 files**:

```sh
python -m mypy --follow-imports=silent core/operations core/jobs core/spine models core/analysis_records.py core/analysis_availability.py core/analysis_target.py core/artifacts.py core/sequence_time.py core/render_plan.py core/sequence_export.py core/sequence_preview.py core/edl_export.py core/project_export.py core/remix/prerender.py core/legacy_sequence_time.py core/project_migrations.py
```

Adding `core/paths.py` produces its existing `sys._MEIPASS` diagnostic. The frozen-path expression is unchanged from `11bfdf6`; this is an explicit baseline exception.

`ruff check .` reports six existing unused imports in `ui/main_window.py`: `calculate_download_timeout`, `CancellableWorker`, `CinematographyWorker`, `FaceDetectionWorker`, `GazeAnalysisWorker`, and `EmbeddingAnalysisWorker`. No other Ruff findings remain.

Full-tree mypy reports **781 errors in 115 files**, compared with **876 errors in 129 files** when the same command and interpreter check an isolated archive of `11bfdf6`. The required shared scope is clean; full-tree typing is not claimed clean. Logs are `/tmp/u8-u10-signoff-types.log` and `/tmp/u8-u10-baseline-types.log`. Existing full-tree exceptions remain owned by the plan's later cleanup/gate work.

Matching diagnostics by file and message initially left eight apparent differences after the fixes. Mapping unchanged source lines back to `11bfdf6` confirms all eight already had diagnostics at those exact original lines:

| Current location | Baseline location | Existing diagnostic |
|---|---|---|
| `cli/commands/analyze.py:831` | `:734` | Optional CLI cache directory |
| `scene_ripper_mcp/tools/sequence.py:262` | `:245` | Context parameter defaults to `None` |
| `scene_ripper_mcp/tools/jobs.py:905` | `:682` | Reused `resolve_options` import name, cinematography |
| `scene_ripper_mcp/tools/jobs.py:923` | `:700` | Reused `resolve_options` import name, custom query |
| `scene_ripper_mcp/tools/jobs.py:1880` | `:1633` | Context parameter defaults to `None` |
| `ui/main_window.py:8354` | `:8185` | Legacy QMessageBox enum alias |
| `ui/main_window.py:8357` | `:8188` | Legacy QMessageBox enum alias |
| `ui/main_window.py:8800` | `:8615` | Legacy QMessageBox enum alias |

The resolver diagnostic wording changed with richer function signatures; the conflicting imports already existed. Baseline diagnostics were not treated as new U8–U10 work merely because their line numbers or message formatting changed.
