---
title: Shared Editing Engine — Remaining Work
type: refactor
date: 2026-09-09
status: planned
source_plan: 2026-09-06-1218-refactor-shared-editing-engine-plan.md
---

# Shared Editing Engine — Remaining Work

Implement U11–U17 of the [original architecture plan](2026-09-06-1218-refactor-shared-editing-engine-plan.md). This document is the checklist for the remaining work; it preserves the original scope, unit IDs, requirements, and acceptance checks. Completed prerequisites are omitted from the dependency lists. See the [U8–U10 verification record](2026-09-08-u8-u10-verification.md) for the current foundation and static-check baseline.

## Execution order and progress

Start with U11. Use the order below by default, delivering one algorithm or runtime family at a time. U13 and U15 have no remaining prerequisite units and may move earlier if sequencing work encounters an external dependency. U14 must wait for U13's installed-platform proof; U16 requires both U12 and U15; final U17 sign-off requires U12, U14, and U16.

- [x] **U11:** Algorithm registry and recipe model; prove shuffle and color first. Evidence recorded below.
- [x] **U12:** Migrate every sequencer and expose variation commands. Evidence recorded below.
- [ ] **U13:** Prove managed native-worker isolation with transcription on all supported packaged platforms. Source-mode proof and CI gates landed; packaged evidence pending the next release builds (see evidence below).
- [ ] **U14:** Migrate the remaining native features and unify runtime install/repair.
- [ ] **U15:** Shared clip/frame item models with measured large-library performance.
- [ ] **U16:** Recipe inspection and A/B variation comparison in the existing workspace.
- [ ] **U17:** Enforce quality gates, dependency contracts, and remove superseded paths.

Check off a unit only after its acceptance evidence is recorded. For each completed family, record the commit, exact validation commands/results, applicable interface coverage, and remaining compatibility wrappers with removal conditions.

## Rules carried into this work

The original plan's R1–R14 requirements and technical decisions remain authoritative. In particular:

- Use the existing project session, reversible commands, shared jobs, spine, media-time/render compiler, analysis records, and artifact store. Workers compute from snapshots; the owning session validates and applies results.
- Keep one Qt-free algorithm definition per algorithm. Separate input selection and prerequisite work from generation. Persist both a versioned recipe and realized choices, trims, transforms, rationale, and provider outputs. Reconstruction performs no provider calls; regeneration creates a new variation by default.
- Launch native inference through a managed interpreter and a bounded, versioned JSON pipe protocol. Keep logs separate; validate staging paths; resolve install requests through allowlisted runtime profiles and manifests. Do not persist credentials, pickle objects, or accept arbitrary executable/package paths.
- Project shared data into Qt item models while keeping selections and filters local to each workspace. Apply Qt model changes on the owning thread. Establish numeric performance budgets from recorded workloads before replacing rendering.
- Preserve the six tabs, current algorithm names, public defaults and payloads, project compatibility, and startup ordering. Retain prior working runtimes for rollback. Remove Torch/MLX startup safeguards only after packaged evidence proves no GUI path needs them.
- Update the [surface compatibility matrix](../architecture/surface-compatibility.md) at every capability cutover. Remove the superseded implementation once parity passes; do not run both implementations against live state.

This work does not add new creative algorithms, a new frontend, general compositing, a plugin marketplace, distributed jobs, collaborative editing, or live-desktop MCP RPC.

## U11. Define the algorithm registry and recipe model

**Goal:** Make algorithm execution independent of dialogs and preserve generation inputs.

**Requirements:** R1, R9, R10. **Remaining dependencies:** None.

**Files:** New `core/remix/registry.py`, `models/recipe.py`, `core/spine/sequences.py`, `tests/test_recipe_model.py`, `tests/test_algorithm_registry.py`; existing `core/remix/__init__.py`, `ui/algorithm_config.py`, `core/cost_estimates.py`, `models/sequence.py`.

**Approach:** Apply KTD12 to shuffle and color first. Separate candidate selection and prerequisite computation from pure generation. Use typed sequence proposals for simple ordering, timed/audio edits, and provider-assisted algorithms. Keep UI labels and custom-control hints in adapters; the engine owns algorithm parameters and prerequisite meaning.

**Test scenarios:**

1. Same inputs, algorithm version, and seed reconstruct deterministic output.
2. Seed zero is an explicit seed if the new contract defines it so; legacy random-seed behavior is translated at compatibility adapters.
3. Recipe round-trip retains input selection, analysis identities, parameters, and realized transforms.
4. Registry import is Qt-free and exposes schemas to both agent surfaces without UI imports.

**Verification:** Pilot algorithms execute through the registry from every applicable surface and persist reconstructable recipes.

**Decisions (recorded 2026-09-09):**

- *Recipe schema/version policy.* `models/recipe.py` stores `schema_version` 1 on each recipe. The project schema stays at 1.7: `Sequence.recipe` is an optional `"recipe"` key, so older builds open newer projects and drop only the recipe on re-save. Future or malformed recipe documents load as `UnreadableRecipe` and are written back verbatim; readers see `Sequence.readable_recipe is None` with an explicit "newer build" result.
- *Parameter normalization.* Each definition declares `ParameterSpec`s. `normalize_parameters` fills defaults, checks type/choices/bounds, rejects unknown keys, and stores keys sorted. Stored parameters are plain JSON; recipe equality and `generation_fingerprint` use canonical JSON.
- *Algorithm versions.* Each definition carries an integer `version` (both pilots: 1) recorded on the recipe. Reconstruction replays realized entries and needs no algorithm, so version drift never blocks replay; regeneration (U12) must check the version.
- *Seed contract.* Seeds are explicit non-negative integers and zero is a valid seed. `seed=None` on a seeded algorithm draws a fresh seed with `secrets.randbelow(2**31)` and records it. Unseeded algorithms reject a seed. Legacy `seed=0`/negative ("random") is translated by `core.remix.engine.legacy_seed` inside `core.remix.generate_sequence`, `run_registry_algorithm`, and `SequenceTab._run_generation`; new surfaces (spine `generate_sequence`, MCP) use the explicit contract directly.

**Evidence (2026-09-09):**

- Files: `core/remix/engine.py`, `core/remix/registry.py`, `core/remix/chromatics.py`, `core/remix/shuffle.py` (`ShuffleDefinition`), `models/recipe.py`, `models/sequence.py` (`recipe`), `core/spine/sequences.py` (`generate_sequence`, `publish_recipe`, `get_sequence_recipe`, `reconstruct_sequence`, `list_algorithms`), `core/spine/settings_io.py` (`engine` schema), `ui/workers/sequence_worker.py`, `ui/dialogs/dice_roll_dialog.py`, `ui/tabs/sequence_tab.py`, `scene_ripper_mcp/tools/sequence.py`, `scene_ripper_mcp/schemas/inputs.py`.
- Tests: `tests/test_recipe_model.py` (15), `tests/test_algorithm_registry.py` (19, covering scenarios 1–4 plus GUI worker, dice-roll worker, sequence-tab chat path, spine generate/inspect/reconstruct/undo), `scene_ripper_mcp/tests/test_sequence_recipes.py` (4), and `tests/test_spine_imports.py` now imports `models.recipe`, `core.remix.engine`, `core.remix.registry` in the Qt-free subprocess.
- Validation: `python -m pytest tests/test_recipe_model.py tests/test_algorithm_registry.py tests/test_sequencer_algorithms.py tests/test_sequence_generation_history.py tests/test_sequence_tab_chromatic_bar.py tests/test_sequence_tab_direction_dropdown.py tests/test_multi_sequence_tab.py tests/test_multi_sequence_project.py tests/test_project.py tests/test_spine_imports.py tests/test_operation_contracts.py tests/test_cli_integration.py tests/test_legacy_sequence_migration.py tests/test_sequence_history.py tests/test_sequence_management_history.py tests/test_sequencer_embedding_inputs.py tests/test_sequence_embedding_recovery.py scene_ripper_mcp/tests/ -q` → 542 passed. Scoped mypy (U8–U10 scope plus the four new remix modules) → clean, 146 files. `ruff check .` → only the six pre-existing `ui/main_window.py` unused imports.
- Surfaces: GUI (worker + Hatchet Job dialog), chat (`generate_remix` → `generate_and_apply`), MCP (`generate_sequence`, `get_sequence_recipe`, `reconstruct_sequence`, `list_sequence_algorithms`). CLI generation is deferred to U12's `cli/commands/sequence.py`.
- Compatibility wrappers: `core.remix.generate_sequence` for `shuffle`/`color` (remove when U12 migrates all callers); `SequenceTab._run_generation` legacy-keyword adapter (remove with U12's tab migration). MCP `shuffle_sequence` is unchanged and is a reorder, not a generation.
- Behavior change: an unknown Chromatics `direction` now returns a validation error instead of silently sorting by hue.

## U12. Migrate all sequencers and add variation commands

**Goal:** Complete algorithm parity and preserve creative alternatives.

**Requirements:** R1, R3, R4, R10, R11. **Remaining dependencies:** U11.

**Files:** `core/remix/`, `core/spine/sequences.py`, `ui/dialogs/`, `ui/workers/sequence_worker.py`, `ui/tabs/sequence_tab.py`, `core/chat_tools.py`, `scene_ripper_mcp/tools/sequence.py`, new `cli/commands/sequence.py`, `tests/test_recipe_reconstruction.py`, `tests/test_sequence_variations.py`, `scene_ripper_mcp/tests/test_integration.py`.

**Approach:** Migrate registry groups independently: arrange; similarity/reference/gaze/face; audio/word timing; text/LLM/drawing. Custom dialogs collect validated parameters and observe jobs. Provide list, inspect recipe, duplicate, regenerate variation, reconstruct, and activate commands. Drawing/reference inputs become asset references rather than GUI objects. Preserve realized provider decisions for offline reconstruction.

**Test scenarios:**

1. Every algorithm in the surface compatibility matrix is discoverable and executable without instantiating its dialog.
2. Reconstructing an LLM-generated edit performs no provider call; regeneration records a new run.
3. Generating a variation does not modify the previous sequence; undo removes the new insertion as one edit.
4. Missing recipe inputs or unavailable algorithm versions produce an actionable result without overwriting existing edits.

**Verification:** All current sequencers use the registry, and cross-surface tests cover each family plus algorithm-specific existing regressions.

**Evidence (2026-09-09):**

- Commits: `cc4af8b` (arrange family, variation commands, CLI group), `67c31ee` (gaze, reference, Rose Hobart), `7a143bf` (audio and word families), `95b9fec` (text, drawing, free association; cleanup). All 23 `ui/algorithm_config.py` keys are registered (`tests/test_recipe_reconstruction.py::test_every_matrix_algorithm_is_registered`).
- Engine additions: object/array parameters, `Prepared` context, `sequence_settings`, `source_parameters`/`asset_parameters`, `provider` flag, `progress`/`resources` hooks, `resolve_prerequisites` for GUI-owned prerequisite jobs, definition-owned `legacy_parameters`.
- Scenario 1 (discoverable/executable without dialogs): parametrized over every key with provider fakes in `tests/test_recipe_reconstruction.py`.
- Scenario 2 (reconstruct without provider; regenerate records a new run): same file, provider call counters for Storyteller, Exquisite Corpus, Free Association, LLM Word Composer.
- Scenario 3 (variation does not modify the previous sequence; undo removes it as one edit): `tests/test_sequence_variations.py`, plus MCP and CLI round trips.
- Scenario 4 (missing inputs / unavailable versions are actionable and non-destructive): `tests/test_sequence_variations.py::test_missing_inputs_and_version_drift_are_actionable_without_overwriting`.
- Surfaces: chat tools `generate_remix`, `generate_eyes_without_a_face`, `generate_reference_guided`, `generate_rose_hobart` (dialog-driven), `generate_staccato`, `generate_cassette_tape`, `generate_exquisite_corpus`, `generate_storyteller`, `generate_signature_style` plus `get_sequence_recipe`, `reconstruct_sequence`, `regenerate_sequence`, `duplicate_sequence`, `list_sequences`, `activate_sequence`; MCP tools of the same names plus `generate_sequence`/`list_sequence_algorithms`; CLI `scene_ripper sequence …`. Word Sequencer, LLM Word Composer and Free Association gain headless parity through the generic tools (previously GUI-only).
- Removed: per-algorithm branches in `core.remix.generate_sequence`, `_is_dialog_only_algorithm`, `core.spine.sequences.apply_generated_order`, direct `generate_poem`/`generate_narrative`/`reference_guided_match`/`match_phrases` sequencing in dialogs and chat tools. Rose Hobart matching moved to `core/remix/rose_hobart.py`; Signature Style gained a Qt-free `DrawingImage` and saves the canvas as a PNG asset.
- Behavior changes: an unknown `gaze_sort` direction now fails parameter validation (message changed); Free Association has a headless autonomous mode; `generate_signature_style` accepts `total_duration_seconds` (the previous tool passed a PIL image to a QImage sampler and always failed).
- Static checks: scoped mypy is clean for the new modules; four pre-existing diagnostics in unchanged lines of `reference_match.py`, `free_association.py`, and `staccato.py` remain for U17.
- Review closure: the CE review of `6a9f025..3773366` (run `20260909-030011-18ffc50a`) found one P0 (caller resources dropped when prerequisites were pre-resolved), two P1s (reconstruction lost music/reference settings; regeneration replayed a stale manual reorder), and security/reliability items (headless Rose Hobart package install, provider work on the serial MCP editorial path). `7cb6b2f` fixes all applied findings and adds `start_generate_sequence` / `start_regenerate_sequence` MCP jobs, a generic chat `generate_sequence` tool, and the Eyes Without a Face dialog recipe. 1,168 affected tests pass.

## U13. Prove native worker isolation with transcription

**Goal:** Demonstrate crash containment in source and frozen applications.

**Requirements:** R5, R7, R14. **Remaining dependencies:** None.

**Files:** New `core/runtime_worker/`, `core/runtime_supervisor.py`, `tests/test_runtime_worker_protocol.py`, `tests/test_runtime_supervisor.py`; existing `core/transcription.py`, `core/dependency_manager.py`, `core/paths.py`, `core/runtime_smoke.py`, `packaging/build_support.py`, platform staging manifests/scripts.

**Approach:** Implement KTD8 with one transcription backend before migrating other models. Validate protocol/version handshake, bounded messages, worker readiness, cancellation acknowledgement, and abnormal exits. Use explicit managed interpreter paths. Drain pipes while work runs. Process-tree cleanup must handle child FFmpeg processes and platform-specific termination; [Python subprocess documentation](https://docs.python.org/3.11/library/subprocess.html#popen-objects) informs this distinction.

**Test scenarios:**

1. A native worker exits or crashes while the GUI remains able to edit and save.
2. Cancellation of a blocked task terminates its process tree after a bounded grace period and preserves committed items.
3. Protocol mismatch, truncated output, excessive output, and malformed results fail the job without applying data.
4. Worker startup succeeds from installed macOS, Windows, and Linux packages without relying on a development Python installation.
5. A result pointing outside assigned staging is rejected, and install requests cannot name arbitrary packages or executables.

**Verification:** Platform smoke evidence proves managed interpreter launch and real inference. Do not remove existing startup safeguards or expand the migration until this gate passes.

**Decisions (recorded 2026-09-09):**

- *Managed executable layout.* Workers run as ``<interpreter> -X utf8 -m runtime_worker`` with ``PYTHONPATH`` = worker root + the package directories built for that interpreter. Frozen apps stage `core/runtime_worker/*.py` as plain source at `<resources>/runtime_worker_src/runtime_worker` (`packaging/build_support.collect_runtime_worker_datas`, wired into both PyInstaller specs) and launch the managed python-build-standalone interpreter from `get_managed_python_dir()`, never the frozen executable. Source runs use `sys.executable`; `SCENE_RIPPER_WORKER_PYTHON` overrides. The Linux AppImage runs from source, so its own `usr/bin/python3` is the worker interpreter.
- *Compatible runtime families.* Managed package directories are compiled for the managed Python; they join the worker path only when that interpreter runs the worker (`default_launch`). A developer interpreter never has those wheels prepended (this shadowing broke NumPy on the first local run).
- *Protocol/message bounds.* Protocol 1, newline-delimited JSON, ASCII-only, 1 MiB per line both ways; results reference files instead of inlining data; logs on stderr only. Unknown message types, oversized or malformed lines, and version mismatches are protocol violations; a violating worker is retired.
- *Cancellation.* The host sends `cancel`; cooperative tasks answer `cancelled` and the warm worker survives. After `cancel_grace` (3 s default) the host terminates the whole process tree (POSIX session `killpg` SIGTERM then SIGKILL; Windows `taskkill /T /F`). Task timeouts terminate the same way.
- *Worker idle policy.* One warm worker per runtime family; tasks per family are serialized (accelerator admission = one task per family). Workers live until supervisor shutdown or failure; no idle reaper yet (U14 may add one with the profile work).
- *Result validation.* Every `*_path`/`path` value in a result must resolve (following symlinks) inside the task's staging directory.
- *Install requests.* `core/runtime_profiles.py` maps allowlisted profile ids (`transcription-whisper`) to feature-registry names; package pins come from `core/package_manifest.json`. Callers cannot name packages, URLs, or executables. Headless Rose Hobart no longer installs packages on demand (fixed in U12 review); worker features must be installed explicitly.
- *Credentials.* Worker environments drop `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV` and any `*_API_KEY`/`*_TOKEN`/`*_SECRET` variables; provider calls stay in the host.

**Evidence (2026-09-09, source mode on macOS):**

- Files: `core/runtime_worker/{__init__,protocol,tasks,__main__}.py`, `core/runtime_supervisor.py`, `core/runtime_profiles.py`, `core/transcription.py` (`native_worker_enabled`, `_transcribe_in_worker`; faster-whisper video and clip paths route through the worker; `SCENE_RIPPER_NATIVE_WORKERS` and the `native_worker_isolation` setting control it), `core/runtime_smoke.py` (`native-worker` target), `packaging/build_support.py` + both `.spec` files, `.github/workflows/{build-macos,build-windows,linux-build}.yml` (smoke gate with `SCENE_RIPPER_SMOKE_INSTALL_PROFILES=1`).
- Scenario 1 (crash containment): `tests/test_runtime_supervisor.py::test_worker_crash_is_contained_and_project_edits_continue` and `::test_worker_exit_during_a_task_surfaces_as_a_crash_not_a_hang`.
- Scenario 2 (blocked-task cancellation with tree teardown): `::test_cancelling_a_blocked_task_kills_the_worker_tree_after_grace` (child process recorded and verified dead), `::test_cooperative_cancellation_returns_promptly`, `::test_task_timeout_terminates_the_worker`.
- Scenario 3 (protocol mismatch, truncated, excessive, malformed): `::test_protocol_mismatch_is_refused`, `::test_excessive_and_malformed_output_fail_the_task_without_a_result`, `::test_truncated_output_from_a_dying_worker_is_a_crash`, and `tests/test_runtime_worker_protocol.py` (10 tests incl. version refusal, oversize, malformed host lines, pre-task cancel).
- Scenario 5 (staging escape, install allowlist): `::test_result_paths_outside_staging_are_rejected` (traversal, relative, symlink), `::test_install_requests_cannot_name_packages_or_executables`, credential stripping and explicit-interpreter tests.
- Real inference: `tests/test_runtime_smoke.py::test_native_worker_runtime_smoke_passes_in_source_mode` runs `native-worker` (handshake + tiny.en transcription of a synthetic tone through the worker) with the developer interpreter; the same target was also run by hand under the managed interpreter at `~/Library/Application Support/Scene Ripper/python` for the handshake/echo half (its faster-whisper profile is not installed locally).
- Scenario 4 (installed macOS/Windows/Linux packages): **not yet proven.** The `native-worker` smoke target is wired into all three release workflows and fails the build if the worker resolves to the frozen executable, the staged package is missing, the handshake fails, or transcription cannot run. Evidence must be recorded from the next `build-macos.yml`, `build-windows.yml`, and `linux-build.yml` runs before U14 begins; existing Torch/MLX startup safeguards stay in place until then.
- Review closure: the CE review of `91a6bf3..6fd3d1a` (run `20260909-034551-456bc93c`) found an environment leak from the smoke target, per-clip model reloads in the worker, a source-mode interpreter/package mismatch for on-demand installs, lost critical error classes, lock ordering after crashes, handshake-failure leaks, and staging cleanup; `93fe589` applies 19 fixes with regression tests (`tests/test_runtime_supervisor.py` now 23 tests).
- Validation: `python -m pytest tests/test_runtime_supervisor.py tests/test_runtime_worker_protocol.py tests/test_runtime_smoke.py tests/test_build_support.py -q` → passes; transcription and settings suites pass with `SCENE_RIPPER_NATIVE_WORKERS=0` defaulted in `tests/conftest.py` (the existing suites patch in-process models). Scoped mypy clean for the new modules; ruff clean.

## U14. Migrate native features and runtime installation

**Goal:** Complete isolation and make installation/repair consistent across surfaces.

**Requirements:** R1, R7, R14. **Remaining dependencies:** U13.

**Files:** `core/analysis/`, `core/transcription.py`, `core/feature_registry.py`, `core/dependency_manager.py`, `core/package_manifest.json`, `core/settings.py`, `main.py`, `core/runtime_smoke.py`, `ui/widgets/dependency_widgets.py`, `core/spine/`, `cli/commands/`, `scene_ripper_mcp/tools/`, `tests/test_analysis_dependency_gates.py`, `tests/test_runtime_smoke.py`, `tests/test_build_support.py`.

**Approach:** Migrate transcription variants, embeddings, local VLMs, OCR, faces/objects/gaze, alignment, and audio/stem dependencies by compatible runtime family. Expose capability status and explicit install/repair jobs through shared operations; no silent install on analysis. Create locked runtime profiles and stage replacements separately, switching only after health checks. Retain the previous working runtime for rollback. Remove Torch/MLX GUI startup workarounds only after no GUI path imports those runtimes.

**Test scenarios:**

1. Missing features present the same install requirement to UI and automation, with consistent progress and failure states.
2. An interrupted install or incompatible repair leaves the previous runtime usable.
3. Switching runtime profiles does not reinstall native packages into a live interpreter.
4. Accelerator contention queues work; concurrent jobs do not load conflicting models without resource admission.
5. Credentials reach only the required worker/provider and are absent from persistent arguments and logs.

**Verification:** Every optional native feature has packaged smoke coverage or an explicit unsupported-platform capability result. Startup import tests prove GUI isolation.

## U15. Replace browser bookkeeping with shared item models

**Goal:** Reduce duplicated data and preserve responsive large libraries.

**Requirements:** R2, R6, R11, R13. **Remaining dependencies:** None.

**Files:** New `ui/models/clip_model.py`, `ui/models/frame_model.py`, `tests/test_library_models.py`; existing `ui/clip_browser.py`, `ui/frame_browser.py`, `ui/tabs/cut_tab.py`, `ui/tabs/analyze_tab.py`, `ui/project_adapter.py`, `tests/test_clip_browser_selection.py`, `tests/test_clip_browser_filters.py`, `tests/test_analyze_tab_clip_sync.py`.

**Approach:** Apply KTD13 while retaining current virtualization and shared theme primitives. Move data access into item models before replacing card rendering. Preserve workspace-specific selection and filters. Agent context queries session data and explicit selection state rather than maintaining another project copy.

**Test scenarios:**

1. Edits update visible cards in both workspaces without rebuilding unrelated cards or resetting selection.
2. Filtered-out and offscreen selections retain their documented semantics across model updates.
3. Thumbnail results for removed items are ignored, and all Qt model mutations occur on the owning thread.
4. Recorded 1,000- and 10,000-clip fixtures meet the responsiveness and memory budgets established before cutover.

**Verification:** Existing selection/filter regressions pass, view behavior is manually checked, and measured large-library behavior meets the recorded budget.

## U16. Add sequence variation comparison to the existing workspace

**Goal:** Expose recipes and alternatives through a compact creative workflow.

**Requirements:** R10, R11. **Remaining dependencies:** U12, U15.

**Files:** `ui/tabs/sequence_tab.py`, `ui/dialogs/intention_import_dialog.py`, `ui/widgets/cost_estimate_panel.py`, new `ui/widgets/sequence_comparison.py`, `tests/test_sequence_comparison.py`, `tests/test_multi_sequence_tab.py`, `docs/user-guide/sequencers.md`, `docs/user-guide/agent-tools.md`.

**Approach:** Add recipe inspection, duplicate/regenerate actions, and an A/B comparison panel with two named sequence selectors. Show duration, clip count, changed recipe parameters, and preview switching at the same elapsed timeline time when both previews exist; clamp to the shorter sequence's end when needed. Deleting a compared sequence clears that selector without changing the surviving sequence. Missing analysis/cost and generation progress use shared operation/job state. Keep the existing tabs and timeline controls.

**Test scenarios:**

1. Duplicate and change one parameter, generate B, and compare while A remains unchanged.
2. A missing preview offers rendering without blocking access to recipe differences.
3. Cancel prerequisite analysis or generation without creating a misleading completed variation.
4. Keyboard controls, empty states, unavailable capabilities, and agent-created variations behave consistently.

**Verification:** A complete import-to-variation-to-export walkthrough works through UI and automation; include screenshots or a short recording for review.

## U17. Enforce quality gates and remove completed compatibility paths

**Goal:** Finish the migration with a smaller supported execution surface.

**Requirements:** R1, R4, R13, R14. **Remaining dependencies:** U12, U14, U16. Scoped gate improvements can land earlier as each area becomes clean.

**Files:** `pyproject.toml`, `requirements-core.txt`, `requirements-optional.txt`, new `requirements-engine.txt` and platform lock inputs under `packaging/`, `.github/workflows/quality.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml`, `.github/workflows/linux-build.yml`, release build workflows, `packaging/build_support.py`, `AGENTS.md`, `README.md`, `docs/releases.md`, `docs/user-guide/headless-mcp.md`, `tests/test_build_support.py`, `tests/test_spine_imports.py`.

**Approach:** Define engine, desktop, and optional runtime dependency contracts; retain `requirements-core.txt` as the frozen desktop contract. Make supported source-install extras resolve those same definitions. Enforce checks for migrated modules first, then retire informational baseline exceptions as their owning areas become clean. Include the separate MCP suite. Remove old execution loops, redundant state mirrors, expired wrappers, and main-window workflow code after matrix coverage proves replacement.

**Test scenarios:**

1. A clean engine/MCP install imports and runs a synthetic headless workflow without Qt or ML packages.
2. Frozen builds stage the correct worker/runtime assets and survive startup, preview, update checks, and one optional feature installation.
3. Intentional lint, typing, contract, or runtime regressions in required areas fail CI.
4. Old project fixtures, public tool payloads, CLI defaults, and job history remain supported after cleanup.

**Verification:** Required CI includes both test roots, scoped strict typing, dependency/import checks, and platform runtime evidence. No replacement is considered complete while its duplicate implementation remains active.

## Verification and delivery gates

For each unit, add failing regression tests before defect corrections, run its existing affected regressions, verify supported UI/chat/CLI/MCP contracts, and update user-facing documentation. Keep implementation changes small enough to review by capability. Review the registry pilot before expanding U12, and review installed worker isolation before expanding U14. These are evidence gates within the authorized work.

| Area | Required verification |
|---|---|
| Project and import boundaries | `python -m pytest tests/test_project.py tests/test_multi_sequence_project.py tests/test_spine_imports.py -v` |
| Interface parity | `python -m pytest tests/test_operation_contracts.py tests/test_cli_integration.py scene_ripper_mcp/tests/test_integration.py -v` |
| Jobs and recovery | `python -m pytest tests/test_job_lifecycle.py tests/test_job_recovery.py scene_ripper_mcp/tests/ -v` |
| Media output | `python -m pytest tests/test_media_time.py tests/test_render_plan.py tests/test_media_render_e2e.py tests/test_sequence_playback_mapping.py -v`; inspect decoded frames and audio |
| Packaging and native workers | `python -m pytest tests/test_build_support.py tests/test_runtime_smoke.py -v`, plus installed macOS, Windows, and Linux worker/inference smoke evidence; run updater regressions when startup or updater metadata changes |
| Full regression | `python -m pytest tests/ scene_ripper_mcp/tests/ -v`; explain every skip |
| Static checks | `ruff check .` and `python -m mypy cli core models ui scene_ripper_mcp`; enforce migrated scopes and explicitly retire baseline exceptions through U17 |
| Library performance | Record hardware, fixture size, initial population, filter/update latency, scrolling responsiveness, and peak memory for 1,000 and 10,000 clips; set budgets before cutover |
| Creative workflow | Record an import → variation → comparison → export walkthrough through UI and automation, with screenshots or a short recording |

A unit is complete only when its scenarios pass, public contracts remain compatible, documentation is current, and the replaced implementation is removed or reduced to a wrapper with an explicit removal condition. File-existence checks do not substitute for runtime or installed-platform evidence.

## Remaining execution decisions

Resolve and record these before their corresponding implementation expands:

- **U11:** Recipe schema/version policy, parameter normalization, algorithm versions, and explicit seed-zero behavior, including legacy adapter translation.
- **U13:** Managed executable layout, protocol/message bounds, cancellation grace periods, process-tree handling, worker idle policy, and accelerator admission budgets. Prove one transcription backend before broadening the protocol.
- **U14:** Compatible runtime families, locked profiles and package pins, health checks, atomic profile switching, and rollback behavior.
- **U15:** Recorded baseline workloads and numeric latency/memory budgets before renderer cutover.

Release by capability. Preserve pre-upgrade project backups and previous working runtimes; verify rollback before switching routing. Do not assume older binaries can write newer project schemas. Stop an affected cutover if project preservation, public compatibility, or packaged runtime operation cannot be proved, and continue independent ready work where possible.

## Final completion criteria

The remaining program is complete when every existing sequencer runs through the registry from its applicable interfaces, saves a reconstructable recipe, and creates variations without overwriting earlier edits; optional native inference and runtime repair are isolated and verified on supported packages; library models meet measured budgets; A/B comparison works in the existing workspace; headless installs do not require Qt or local ML packages; and required CI enforces the contracts with duplicate execution paths removed. Preserve all previously verified editing, persistence, recovery, media-time, and artifact behavior throughout.
