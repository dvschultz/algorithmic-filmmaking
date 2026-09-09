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
- [x] **U13:** Prove managed native-worker isolation with transcription on all supported packaged platforms. Source-mode proof, CI gates and a packaged macOS run landed; Windows/Linux packaged evidence pending the next release builds (see evidence below).
- [x] **U14:** Every native family has an isolation seam, profile and packaged smoke gate; install/repair is unified across surfaces; per-family cutover flags flip as CI evidence lands.
- [x] **U15:** Shared clip/frame item models with measured large-library performance.
- [x] **U16:** Recipe inspection and A/B variation comparison in the existing workspace.
- [x] **U17 (gates, contracts, wrapper removal; final sign-off pending Windows/Linux packaged evidence):** Enforce quality gates, dependency contracts, and remove superseded paths.

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
- Scenario 4 (installed packages): **macOS proven locally on 2026-09-09**; Windows and Linux still need the CI runs. A local unsigned PyInstaller build (`packaging/macos/build.sh`, `APP_VERSION=0.0.0-u13`, app at `/Volumes/Lexar/scene-ripper-build/dist/Scene Ripper.app`) ran `SCENE_RIPPER_RUNTIME_SMOKE_TEST_TARGET=native-worker SCENE_RIPPER_SMOKE_INSTALL_PROFILES=1`: handshake OK with the managed interpreter (`~/Library/Application Support/Scene Ripper/python/bin/python3`, worker pid 31173), `transcription-whisper` installed through the manifest into the managed packages, and `Native worker transcription OK ... language=en` followed by `Runtime smoke test 'native-worker' completed successfully` (exit 0; log copy at `~/.cache/u13-smoke/app-pass.log`). The first two packaged runs failed and were fixed in `baee3fc`: `_reset_imported_package_roots` raised `KeyError: 'torch'` while evicting namespace-package submodules, and the frozen py3.13 host then validated the install by importing the py3.11 `faster_whisper`/`ctranslate2` wheels in-process (`module 'ctranslate2' has no attribute 'StorageView'`). Validation now runs as an allowlisted `probe` task in a fresh worker (`RuntimeSupervisor.restart_family`, `core.runtime_profiles.probe_profile_runtime`); the host never imports the native runtime when isolation is on. The three release workflows keep the same smoke gate; `build-windows.yml` and `linux-build.yml` evidence is pending a push/CI run the user must trigger.
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

**Decisions (recorded 2026-09-09, transcription family first):**

- Runtime installs are staged: packages go to `packages-staging/<overlay-name>`, an allowlisted `probe` task in a fresh worker (which sees the staged directory first, `default_launch(staged_paths=)`) imports the runtime, and only a passing health check promotes the directory to `packages-overlays/overlay-<stamp>-<profile>` with an atomic rename. Failure, a killed pip (cancel event) or a failed health check discard the stage; older overlays and the base packages are untouched, so the previous runtime keeps working. `rollback_profile` removes the newest overlay for a profile.
- The host never imports a native runtime when isolation is on: `_validate_feature_runtime("transcribe")` probes the worker, and a warm family worker is retired around installs (`RuntimeSupervisor.restart_family`) so no live interpreter is ever reinstalled into.
- One shared spine (`core/spine/runtime.py`) backs chat tools, MCP tools plus the durable `start_install_runtime_profile` job, the CLI `runtime` group, and the desktop install prompt; every surface reports the same `missing` list and profile ids only (never package names). Analysis paths do not install silently; the UI prompt still asks first.
- `SCENE_RIPPER_APP_SUPPORT_DIR` relocates the managed runtime root so tests and smoke runs cannot touch the user's installation.
- Family migration (second pass, 2026-09-09): every native runtime now has a family (`vision`, `ocr`, `vlm`, `audio`, `alignment` besides `transcription`), a profile (`vision-torch`, `ocr-paddle`, `vlm-local`, `audio-librosa`, `alignment-ctc`) and an isolation seam: `core/runtime_families.isolated(family, call)` decorates the engine functions in `core/analysis/*` (embeddings, boundary embeddings, objects, shots, faces, gaze, ImageNet classification, PaddleOCR, local VLM description/custom query/cinematography, librosa audio, Demucs stems, the CTC alignment engine). When a family is isolated the call runs in that family's worker as an allowlisted `analysis` task (`core/runtime_worker/calls.py`): JSON arguments, paths as strings, provenance callbacks replayed on the host, cancel events forwarded, dependency/model failures keep their exception classes. Workers import `core.analysis` themselves (checkout on the path in source mode; `core`/`models` staged as source beside the worker in frozen builds). Cloud providers and the MLX whisper backend stay in the host by design (workers never receive credentials).
- Cutover is per family and off by default except transcription (`Settings.native_worker_families`, `SCENE_RIPPER_NATIVE_WORKER_FAMILIES`, `scene_ripper runtime isolate`, chat/MCP `update_settings`), so shipping the seam changes nothing until the packaged `native-analysis` smoke (now in all three release workflows) proves a family; `main.py` drops the Torch pre-import once `vision`, `vlm`, `alignment` and `audio` are all isolated, and the MLX pre-import once `vlm` is isolated and the MLX whisper backend is not selectable (`host_needs_runtime`). Features flagged `native_install` cannot be staged with `pip --target`; their profiles install in place and are health-checked in a worker afterwards.

**Evidence (2026-09-09):**

- Files: `core/runtime_profiles.py` (staged `install_profile`, `rollback_profile`, `probe_profile_runtime`, `profile_overlays`), `core/runtime_supervisor.py` (`restart_family`, `default_launch(staged_paths=)`), `core/runtime_worker/tasks.py` (`probe`), `core/dependency_manager.py` (`install_packages(target_dir=, cancel_event=)`), `core/feature_registry.py` (`stage_feature_packages`, worker-probed validation), `core/paths.py` (app-support override), `core/spine/runtime.py`, `core/jobs/runtime_install.py`, `core/chat_tools.py`, `scene_ripper_mcp/tools/runtime.py`, `cli/commands/runtime.py`, `ui/widgets/dependency_widgets.py`, docs.
- Scenario 1: `tests/test_runtime_profiles_surfaces.py::test_status_is_identical_across_spine_chat_and_cli` and `::test_ui_install_prompt_uses_the_staged_profile_path_when_isolated`.
- Scenario 2: `tests/test_runtime_supervisor.py::test_failed_health_check_or_cancel_keeps_previous_runtime`, `::test_rollback_removes_only_the_newest_overlay`, and `tests/test_runtime_profiles_surfaces.py::test_mcp_install_job_is_durable_and_cancellable`.
- Scenario 3: `tests/test_runtime_supervisor.py::test_staged_install_promotes_after_health_check_and_never_imports_in_host` (host `sys.path`/`sys.modules` unchanged), `::test_transcribe_runtime_validation_probes_the_worker_when_isolated`, `::test_staged_probe_launches_a_worker_that_sees_the_staged_directory_first`.
- Scenario 4: family locks serialize accelerator access (`RuntimeSupervisor.run`, U13); no second family exists yet, so cross-family admission budgets remain a U14 follow-up with the next family.
- Scenario 5: credential stripping and staging validation from U13 apply unchanged (`::test_launch_uses_explicit_interpreter_and_strips_credentials`).
- Packaged proof: the macOS `native-worker` smoke (see U13 evidence) exercised the real install path from the frozen app and surfaced the two host-side bugs fixed in `baee3fc`.
- Family proof (source mode, macOS, 2026-09-09): `tests/test_runtime_families.py` (allowlist matches the decorated functions, families off by default, real worker round trip with execution replay, exception-class mapping, `analyze_audio` on a synthetic tone inside the audio worker with librosa never imported by the host, `main.py` workaround gating, settings validation); the `native-analysis` smoke target run locally with every family enabled: `vision-torch` probe OK (torch 2.6.0) and a real YOLO detection in the vision worker, `ocr-paddle` probe OK and PaddleOCR 3.4 reading "SCENE RIPPER" from a synthetic frame in the ocr worker (this surfaced and fixed a PaddleOCR 3.x incompatibility in `_get_ocr_engine`: `show_log`/`use_angle_cls` and the `predict` result shape), `vlm-local` probe OK (mlx_vlm 0.3.9), `alignment-ctc` probe OK (ctc_forced_aligner 0.3.0), `audio-librosa` real analysis. The 113 analysis-related test modules (1767 tests) pass with families off, proving the in-process paths are unchanged.
- Packaged proof per family: the `native-analysis` smoke target is wired into `build-macos.yml`, `build-windows.yml` and `linux-build.yml` after `native-worker`; its evidence (and therefore each family's cutover flag) is owed by CI runs the user must trigger.
- Review closure: the CE review of `57d3bee..c0917ea` (run `20260909-061152-5d34037b`, 10 reviewers) found readiness checks restarting the warm worker, cancellation lost while pip is silent, unserialized concurrent installs, a health check that could pass from the live runtime, rollback reporting success after a failed delete, and the desktop dialog dropping cancel/error; `e54517a` applies 20 fixes with regression tests. Deferred: per-family overlay scoping (with the next family), an asynchronous chat install tool.

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

**Decisions (recorded 2026-09-09):**

- Models are Qt `QAbstractListModel`s owned by `ProjectSignalAdapter` (`adapter.clip_model`, `adapter.frame_model`) and updated from project events before the adapter emits its signals; slots always observe a model that already reflects the change.
- Workspaces keep membership (ordered ids), selection, and `FilterState`; they read `Clip`/`Source`/`Frame` objects only through the model. A browser with a shared model never removes rows from it (`clear()` and removals touch membership only). Standalone browsers (tests, dialogs) fall back to a private model so public browser methods keep their signatures.
- Model mutators assert the owning thread and raise `RuntimeError` otherwise; thumbnail results are delivered to `ClipLibraryModel.thumbnail_ready`, which drops results for clips no longer in the library.
- Renderer cutover is deferred: the `ClipBrowser` card widgets and virtualization stay; the recorded budgets in `docs/architecture/library-models.md` gate a later model-backed clip view. `FrameBrowser` already renders from the shared model through its `QListView`, so `FramesTab` no longer resets the model on every refresh.
- Baseline workloads and budgets (10k clips, two workspaces): populate <= 250 ms, update-50 <= 100 ms, filter toggle <= 150 ms, remove-100 <= 250 ms, five scroll positions <= 600 ms, peak RSS <= 400 MB, model operations <= 25 ms.

**Evidence (2026-09-09, macOS source mode):**

- Files: `ui/models/{__init__,clip_model,frame_model}.py`, `ui/project_adapter.py` (models + `frames_added` signal), `ui/clip_browser.py` (`attach_model`, `library_model`, membership ids replace `_source_lookup`/`_virtual_entries` copies), `ui/frame_browser.py` (`set_model`, `FrameBrowserModel` alias), `ui/tabs/frames_tab.py` (`set_frame_model`), `ui/main_window.py` (attaches models, routes thumbnail results through the model), `scripts/library_model_benchmark.py`, `docs/architecture/library-models.md`.
- Scenario 1: `tests/test_library_models.py::TestBrowsersShareTheModel::test_edit_refreshes_cards_in_both_workspaces_without_resetting_selection` (same card widgets before/after, both selections intact) and `TestFrameBrowserSharesTheModel` (frame metadata edit keeps view selection).
- Scenario 2: `::test_filtered_out_and_offscreen_selection_survive_updates`; existing `tests/test_clip_browser_selection.py`, `tests/test_clip_browser_filters.py`, `tests/test_analyze_tab_clip_sync.py` pass unchanged (82 tests).
- Scenario 3: `TestClipLibraryModel::test_thumbnail_for_removed_clip_is_ignored`, `::test_mutations_refused_off_owner_thread`, `TestFrameLibraryModel::test_mutations_refused_off_owner_thread`.
- Scenario 4: `scripts/library_model_benchmark.py` recorded 1k/10k runs (table in `docs/architecture/library-models.md`); all measurements are within budget, and model-layer cost at 10k is under 5 ms per operation. `TestModelScale` bounds the model layer in CI.
- Validation: the 26 UI suites that touch browsers, adapters, history and thumbnail delivery pass (346 tests) plus `tests/test_library_models.py` (16 tests); ruff clean for changed files.
- Review closure: the CE review of `a172738..72048ce` (run `20260909-043301-f7adb43e`, 10 reviewers) found stale rows after re-detection (`Project.replace_source_clips` now emits `clips_removed`), adapter signal suppression when a model mutation fails, an O(n) frame refresh, a frame-model reset on source undo, double refreshes and stale cached sources in the browser, and agent-context gaps (`frames_tab_frame_ids`, virtual-mode `total_clips`); `068a304` applies 19 fixes with regression tests (`tests/test_library_models.py` now 28 tests). Two P3s stay documented as residual (blanket `set_sources` invalidation, post-undo row order).

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

**Decisions (recorded 2026-09-09):**

- The comparison panel is a view over spine functions and mutates nothing itself; every button is a signal the tab routes through `duplicate_sequence`, `get_sequence_recipe`, `prepare_regeneration` + `publish_recipe`, or `switch_to_sequence`. `compare_sequences` is a new read-only spine function exposed as a chat tool, an MCP tool, and `scene_ripper sequence compare`, so agents see exactly what the panel shows.
- Regeneration in the desktop splits `regenerate_sequence` into `prepare_regeneration` (GUI thread, validates inputs/version/parameters and snapshots candidates) and a `VariationWorker` that runs `run_algorithm` with a cancel event; publishing happens on the GUI thread only when the worker's project/session still match. A cancelled or failed run publishes nothing. Progress and per-sequence busy state flow into the panel; the regenerate dialog shows the same cost/missing-dependency summary as the confirm step.
- Switching A/B keeps the elapsed playhead time and clamps to the arriving sequence's end (`comparable_seconds` in the comparison). The rendered-proxy probe lives in MainWindow; a missing preview offers rendering for that side without hiding the recipe differences.
- Deleting a compared sequence clears only its selector (selection is kept by sequence id across refreshes). Agent- or MCP-created variations appear through the existing `sequences_changed` sync. Read-only projects keep Show/Recipe and disable Duplicate/Regenerate.
- Keyboard: `A`/Left and `B`/Right switch while the panel (or one of its children) has focus; `Ctrl+Shift+B` toggles it (`Ctrl+Shift+C` already toggles the chat panel).

**Evidence (2026-09-09, macOS source mode):**

- Files: `ui/widgets/sequence_comparison.py`, `ui/dialogs/recipe_dialogs.py` (`RecipeInspectDialog`, `RegenerateDialog`), `ui/workers/variation_worker.py`, `ui/tabs/sequence_tab.py` (Compare A/B toggle, `switch_to_sequence`, `inspect_recipe`, `duplicate_sequence`, `regenerate_sequence`/`start_variation`/`cancel_variation`), `ui/main_window.py` (preview probe and per-sequence render), `core/spine/sequences.py` (`compare_sequences`, `prepare_regeneration`, `RegenerationPlan`), `core/chat_tools.py` + `scene_ripper_mcp/tools/sequence.py` + `cli/commands/sequence.py` (`compare`), docs (`sequencers.md`, `agent-tools.md`, `headless-mcp.md`, `surface-compatibility.md`), screenshot `docs/user-guide/images/compare-ab.png` from `scripts/capture_comparison_panel.py`.
- Scenario 1: `tests/test_sequence_comparison.py::TestSequenceTabVariations::test_variation_runs_off_thread_publishes_b_and_leaves_a_untouched` and `::test_duplicate_through_tab_uses_spine_and_fills_slot_b` (A's dict unchanged, B has the changed parameter and parent recipe id, one undo entry).
- Scenario 2: `TestComparisonPanel::test_missing_preview_offers_render_without_blocking_differences`.
- Scenario 3: `::test_cancelled_variation_publishes_nothing` (worker cancelled mid-run, no sequence added, controls re-enabled); missing-dependency warnings surface in the regenerate dialog through the shared cost estimator.
- Scenario 4: `::test_keyboard_switches_and_actions_emit_ids`, `::test_empty_state_until_two_sequences_exist`, `::test_read_only_projects_keep_inspection_but_not_mutation`, `::test_deleting_a_compared_sequence_clears_only_its_slot`, `::test_agent_created_variation_shows_up_in_panel`, `::test_switch_keeps_elapsed_time_and_clamps_to_shorter_sequence`.
- Walkthrough (automation): `::test_import_to_variation_to_export_walkthrough_over_mcp_and_cli` generates A over MCP, regenerates B, compares over MCP and the CLI, and exports both EDLs. UI walkthrough: the panel screenshot above; the same flow was exercised through `SequenceTab` in the tab tests (offscreen).
- Fixed on the way: `SequenceTab._load_active_sequence` fed `SequenceClip` entries to the preview strip (no `thumbnail_path`), which broke switching to any populated sequence.
- Validation: `tests/test_sequence_comparison.py` (18), variation/history/chat/MCP sequence suites (285) pass; ruff clean for changed files.
- Review closure: the CE review of `fff5938..a585f45` (run `20260909-051058-76402b0f`, 9 reviewers) found the Ctrl+Shift+C collision with the chat toggle, missing close/clear handling for the variation worker, a silent sequence switch when a preview render was busy, transform-blind `timelines_identical`, no cancel affordance, A/B keys swallowed by a focused selector, publish racing a pending generation draft, and agent-context gaps; `5c02227` applies 19 fixes with regression tests (`tests/test_sequence_comparison.py` now 26 tests). Deferred: extracting the variation block from `SequenceTab` (U17 cleanup); a per-sequence preview-render agent tool stays a documented gap.

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

**Decisions (recorded 2026-09-09):**

- Dependency contracts: `requirements-engine.txt` is the Qt-free engine contract, `requirements-core.txt` stays the frozen desktop contract (engine + PySide6/mpv), `requirements-optional.txt` the on-demand ML set; pyproject extras `engine`/`desktop`/`mcp`/`ml` resolve the same pins and `tests/test_dependency_contracts.py` fails when they drift (`litellm` is the vendored wheel in the files and the PyPI floor in the extras).
- Gates: `ruff check .` and the scoped mypy gate are required (scope: spine, models, engine/registry, recipe model, native worker/supervisor/profiles, shared item models; widen, never narrow); a `headless-engine` job installs only the engine + MCP contract, asserts PySide6 is absent, and runs the import-boundary tests, `tests/test_headless_engine.py` (synthetic detect → generate → regenerate → compare → EDL → save/load → MCP call in a child interpreter that must not load Qt/ML modules) and the MCP suite; the macOS/Windows test jobs run both test roots. `cli.main`, the MCP server/tools and the runtime modules joined the Qt-free import boundary.
- Wrappers retired: `core.remix.generate_sequence` and `SequenceTab._run_generation` are gone; the legacy keyword dispatcher survives only as `tests/remix_compat.py` for the algorithm suites.
- Still open for final sign-off: the informational macOS/Windows test jobs stay `continue-on-error` until green across a release (docs/releases.md gate-flip rule); Windows/Linux packaged runtime evidence (U13/U14) is owed by CI runs; the remaining native families and the `main.py` Torch/MLX startup workarounds (U14) and the `SequenceTab` variation-block extraction (U16 review) are the outstanding cleanup items.

**Evidence (2026-09-09):**

- Scenario 1: `tests/test_headless_engine.py` (locally: no Qt/ML module loaded; in CI: no PySide6 installed) and the MCP suite passing under a Qt-blocking `sitecustomize` locally.
- Scenario 3: ruff/mypy required in `.github/workflows/quality.yml`; `tests/test_dependency_contracts.py` and `tests/test_spine_imports.py` fail on drift.
- Scenario 4: full desktop suite plus `scene_ripper_mcp/tests` pass after the wrapper removal (old project fixtures, tool payloads, CLI defaults and job history unchanged).
- Scenario 2: release workflow smoke targets unchanged; packaged macOS `native-worker` run recorded under U13.
- Review closure: covered by the `57d3bee..c0917ea` review recorded under U14.

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
