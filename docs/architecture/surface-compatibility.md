# Surface compatibility baseline

Observed on `v1-hardening`, September 6, 2026, before the shared-operation
pilot. This records compatibility obligations and known defects separately.
It is not a claim that every interface already has equivalent capabilities.

## Color analysis

| Surface | Target selection | Default existing-data policy | Application/persistence | Cancellation |
| --- | --- | --- | --- | --- |
| GUI toolbar, pipeline, intentions, chat | Selected clips via `ColorAnalysisWorker` | Skip any non-None palette | GUI signal handler mutates model; project save is separate | Worker event; already submitted pool work may finish |
| GUI Frames | `AnalysisTarget` clips or still images | Skip any non-None palette | GUI calls `Project.update_frame` | Same worker |
| CLI `analyze colors` | All clips, exact IDs or first-eight-character IDs | Skip any non-None palette; `--force` recomputes | Legacy tuple load/save | No color-specific cancellation input |
| Spine / MCP color jobs | All clips or exact IDs | Skip truthy palettes; `skip_existing=False` recomputes | Model update; job runner persists | Event checked between clips |
| Synchronous MCP `analyze_colors` | All project clips | Recompute | Model update and mtime-checked save | No cancellation input |

All use five colors by default. CLI, spine and MCP accept a requested count.
Video inputs use source start/end frame indices. GUI frame targets use an image.
Qt's list signal can convert RGB tuples to lists; persisted palettes load as
tuples. This representation difference is not an editorial difference.

Legacy failure differences to correct explicitly: spine silently drops unknown
IDs and does not account for unprocessed canceled items; GUI silently drops
missing media; CLI counts an empty extraction as success; synchronous MCP aborts
on the first extractor exception. A shared operation must report individual
failures while preserving successful results and each surface's default policy.

## Analysis inventory

The picker registry is `core/analysis_operations.py`. GUI has worker routes for
all twelve registered operations. Spine exposes the corresponding operations
directly or through `analyze_clips`. These are not yet one shared implementation
across GUI and headless adapters.

| Registry operation | Spine function | Dedicated CLI command | Dedicated synchronous MCP tool |
| --- | --- | --- | --- |
| colors | analyze_colors | colors | analyze_colors |
| shots | analyze_shots | shots | analyze_shots |
| classify | classify_content | classify | — |
| detect_objects | detect_objects | objects / people | — |
| extract_text | extract_text | — | — |
| transcribe | transcribe | separate transcribe group | transcribe |
| describe | describe | describe | — |
| cinematography | cinematography | — | — |
| face_embeddings | face_embeddings | — | — |
| gaze | gaze | — | — |
| embeddings | embeddings | — | — |
| custom_query | custom_query | — | — |

MCP jobs additionally expose generic analysis through the spine. Brightness,
volume and boundary embeddings are computed for sequencers outside the picker;
word-level transcription is a distinct sequencing prerequisite. Keep these in
the later analysis/recipe registry migration.

## Sequencer inventory

`ui/algorithm_config.py` contains 23 choices. `core/remix.generate_sequence`
dispatches the ordinary sorting algorithms; dialogs own additional inputs and
workflow steps. GUI chat's `generate_remix` coordinates the GUI workflow.
The CLI exports existing sequences; it does not expose the full GUI recipe
catalog. MCP's `shuffle_sequence` offers random, reverse, by_color and
by_shot_type, not the full catalog below.

| Algorithm key | GUI route / extra workflow |
| --- | --- |
| color | Core sort; chromatic options |
| duration | Core sort |
| brightness | Core sort; brightness computation |
| volume | Core sort; volume computation |
| shuffle | Core shuffle plus options dialog |
| sequential | Original order |
| shot_type | Core sort |
| proximity | Core sort |
| similarity_chain | Core traversal; embeddings |
| match_cut | Core traversal; boundary embeddings |
| exquisite_corpus | Dialog; poem composition |
| storyteller | Dialog; narrative composition |
| free_association | Dialog; iterative language-model proposals |
| cassette_tape | Dialog; phrase matching |
| reference_guided | Dialog; reference sequence and weights |
| signature_style | Dialog; drawing interpretation |
| rose_hobart | Dialog; person selection |
| staccato | Dialog; audio beat slots |
| gaze_sort | Core sort |
| gaze_consistency | Core traversal |
| eyes_without_a_face | Dialog; gaze constraints |
| word_sequencer | Dialog; word mode and parameters |
| word_llm_composer | Dialog; local model and word corpus |

Dialog construction, timeline selection, playback and panel navigation are
currently Qt-owned. Recipe realization must move below these controls without
requiring headless clients to instantiate dialogs. Paid model responses must
be persisted as realized results for later replay.

## Persistence and media-time defects

`tests/fixtures/projects/v1.0.json` through `v1.4.json` exercise the currently
accepted version range with nonzero source offsets and a source/timeline rate
mismatch. These are synthetic compatibility fixtures, not historical files.
The 1.4 fixture additionally includes multiple sequences, a still, audio and
transforms. `test_operation_contracts.py` verifies round trips through `Project`.

Known defects, not desired behavior:

- Legacy tuple load/save callers bypass full-project serialization. The color
  regression test demonstrates that the CLI's old path discarded still frames;
  the full-project path retains both frames and multiple sequences.
- Loading offline video with a callback returning None drops the source, its
  clips and referencing timeline entries. Without a callback it raises
  `MissingSourceError`. The offline characterization test documents the loss;
  U5 replaces this policy with preserved unresolved references.
- `SequenceClip.in_point/out_point` are documented as clip-relative, while the
  exporter consumes source-absolute ranges. MCP insertion starts at zero for a
  clip with a nonzero source start. U8 must resolve this ambiguity explicitly.
- Sequence duration currently counts source frame spans as timeline frames.
  The mixed-rate fixture preserves these numbers; it does not certify correct
  playback duration. Preview/export equivalence needs rendered evidence in U8/U9.

## Verification boundaries

Color compatibility tests mock only extraction and exercise real selection,
serialization and adapter result formatting. The worker test applies its legacy
signal result explicitly; it does not verify GUI-thread delivery or tab refresh.
Existing analysis-pipeline and worker tests cover that separate wiring. The
inventory above is source-inspected; full recipe parity and cross-interface
media-time characterization remain work for the corresponding migration units.

## Shared color operation

The color pilot now routes GUI workers, CLI, spine, synchronous MCP and MCP jobs
through `core/operations/colors.py`. GUI chat, toolbar, pipeline, frame and
intention actions retain their existing worker entry points. Only the numerical
color helper and its runtime smoke test call extraction outside the operation.

The operation snapshots input ranges, paths, file stamps and existing palettes;
computation returns immutable outcomes without editing the project. A one-use
`ColorApplication` rejects mismatched requests and stale inputs, then notifies
the project once for the successful clip batch. The GUI applies that result in
a queued slot on its owning thread. A color-only pipeline no longer repeats the
clip notification at completion. Other analysis operations still notify on
their own completion until migrated.

Failures and canceled work are explicit. Dispatch is bounded by the configured
parallelism; cancellation stops new dispatch and retains in-flight successes.
Historical recompute defaults remain unchanged, including the empty-palette
difference. CLI color saving now uses `Project.save`; synchronous MCP keeps its
mtime check and adds per-target error details while retaining legacy counters.

This is a migration checkpoint, not completion of the architecture plan.
Desktop color execution now uses the shared job runtime through its existing
QThread delivery adapter on all four entry paths (pipeline, chat, Frames, and
intentions). Jobs are explicitly session-only until project data is saved; the
status bar says so. Cancellation retains successful partial results, queued
cancellation skips computation, and result application still precedes completion
on the project owner thread. The QThread shell remains for callers using
`start`, `cancel`, `wait`, and `finished`; its removal condition is migration of
those main-window lifecycle callers to task-aware adapters. MCP saved-project
colors separately use durable result receipts, as described in [shared jobs](shared-jobs.md).

The application guard is process-local and one-use, not durable exactly-once
execution. The first [project-session migration](project-sessions.md) now shares
clip enable/disable and manual sequence insertion, removal, clearing, reorder, timing, track and transform history between desktop and chat. Sequence creation, deletion, renaming, and settings now share that history as well. Clearing removes clips from every track in one undo action. Remaining editorial
commands, locks, durable jobs, media-time correction, recipes, and the remaining
interface migrations are still to be implemented.

Source removal now shares reversible library/sequence-reference changes across
desktop, chat, and MCP. Desktop batch deletion is one edit; Undo restores objects
and refreshes library views without importing media again. The desktop retains its
live-sequence deletion guard. MCP response keys and load/save behavior are preserved.

Generated desktop and agent sequences now publish completed drafts as one undo
action, including replacement and empty-sequence reuse. Redo reuses realized
output; it does not repeat inference or pre-rendering. Remaining headless algorithm
registry and recipe parity belongs to U11-U12, rather than being claimed here.

Checkpoint validation:

- `python -m pytest tests/ scene_ripper_mcp/tests/ -q`: 2,832 passed, 2 skipped.
- `python -m ruff check .`: passed.
- `python -m mypy core/operations core/spine models --follow-imports=silent`:
  passed for 34 source files.
- Actual Qt queued delivery is exercised in `test_color_worker_delivery.py`.
  The import-boundary test now runs in a subprocess: the previous in-process
  native-module purge reproduced a Qt crash on the unchanged baseline.

Review covered correctness, project standards, tests, maintainability, agent
interfaces, API compatibility and failure handling. Per repository instructions,
these were sequential inline passes, not independent reviewers. A reproduced
media-probe permission error was fixed so it fails one target instead of the
whole batch. No known color-pilot findings remain; this does not certify the
unimplemented portions of the plan.

## Transcription batch cutover

GUI clip transcription, spine (including MCP jobs), CLI transcription, and
synchronous MCP transcription now share `core/operations/transcription.py` for
per-clip execution and bounded scheduling. Task ranges, paths, model, language,
backend, segmentation, and parallelism are captured before execution. MLX batches
run serially. Cancellation and critical dependency failures stop further submission;
accepted successes survive and remaining tasks are reported as unprocessed.

Silent clips are successful empty transcripts on every surface. Empty transcripts
now survive project save/load and satisfy skip-existing checks. Synchronous MCP
therefore counts silent clips as transcribed rather than skipped. Existing selection,
model/language defaults, and force/recompute policies remain adapter inputs.
CLI and synchronous MCP no longer require faster-whisper before backend selection;
the shared transcription backend handles its own dependency errors.

GUI model preloading and disk-space status remain adapter responsibilities.
`TranscriptionApplication` now guards both GUI and headless result publication by
project session, clip/source object identity, frame range/rate, existing transcript,
and media device/inode/size/mtime. Each target can be applied once. Application
uses the project session boundary for owner-thread, writable-project, revision,
and reentrancy checks. Headless batches notify once; GUI results apply incrementally.
The GUI relay also scopes progress, status, errors, and results to their worker,
request, and pipeline run, rejecting expired or replaced work before UI delivery.

Alignment, audio-only transcription, and durable recovery on the remaining
surfaces remain later U7 work. This cutover does not mark U7 complete.

### Dedicated MCP transcription job recovery

`start_transcribe` now captures an immutable operation specification and checks
project revision, target snapshots, and media metadata before queued work starts.
Its saved-project runner uses existing result rows and project-first receipts:
computed transcripts survive save failures, and saved transcripts reconcile after
checkpoint failures without duplicate application. Source fingerprints include a
SHA-256 digest, cached only while device/inode/size/mtime/ctime remain unchanged.
Edited managed outputs are conflicts. Silent results are retained and reused.

CLI transcription and synchronous MCP also use the durable runner. CLI retains
its selection and skip-existing rules. CLI `--force` and synchronous MCP request
a refresh whose identity includes the previous transcript and count of committed
transcription receipts for that target. Failed saves leave both unchanged and
reuse pending computations; successful saves advance the count so another
explicit refresh recomputes even silent output. The initial adapter model is
never saved over the runner's updated model.

Generic MCP multi-analysis now shares ordered execution, progress mapping, and
cancellation with spine analysis through `core/operations/analysis_plan.py`.
Its saved-project adapter freezes submitted arguments, checks the queued project
revision, and revalidates transcription media/targets before that step starts.
Transcription uses the durable runner; other steps save their spine results
individually. Every following step reloads the saved model, preserving receipts
and completed work when a later step fails. This is not an atomic whole-plan
transaction, and other analysis families still need their durable migrations.

GUI transcription now submits an immutable operation specification to the shared
job runtime through its existing Qt worker. Preflight and computation run inside
the job; the worker observes progress/results and retains a task ID, terminal
status, and typed outcomes. Cancellation shares one event across the worker and
runtime, and cleanup completes before the worker emits its terminal signal.
Job-start notices use the same current-worker/request/session guard as transcripts.

These GUI jobs are explicitly session-only, including when the editor has a
saved project path: results become unsaved model changes until the user saves.
Their temporary job store closes after delivery; they do not yet provide durable
receipt recovery. Durable GUI publication remains outstanding U7 work.

### Word alignment computation and application

Word alignment now runs through `core/operations/alignment.py`: input snapshots
contain detached transcript text/language and source timing, results are typed,
and serial execution cleans temporary audio on both failure and cancellation.
Cancellation during inference suppresses its late result and marks remaining
targets unprocessed. Empty word lists continue to mean alignment completed.

The GUI worker retains its existing feature preflight but no longer reads live
clip models in its execution loop. Owner-thread delivery distributes words onto
a detached transcript and uses the shared project application guard; edited
text, timing/media changes, cleared tabs, replaced projects, and duplicate
signals cannot overwrite current transcripts. The tab refreshes accepted model
updates only.

`scene_ripper analyze align` and the MCP `start_align_words` job use
`core/jobs/alignment.py` with the shared alignment operation. Both accept exact clip IDs, skip completed word data
by default, and support explicit forced alignment. The shared application guard
publishes results; the adapters save under the project writer lease. Missing
dependencies produce per-target `dependency_missing` results and CLI exit code
4 when no clip succeeds; headless calls never install them implicitly. The MCP
job uses standard progress/result/cancellation tools. Computed results and project
receipts support retries after failed saves or checkpoints. Media content hashes
and transcript inputs guard reuse; forced refreshes advance only after a saved
receipt. Matching identical refresh receipts are all reconciled on retry.
GUI alignment and direct spine calls do not yet use durable receipts. Durable GUI
publication and the explicit capability-install workflow remain outstanding.

The GUI alignment worker now submits immutable alignment metadata to the shared
session job runtime, records a task ID and terminal status, and closes the runtime
before emitting completion. Dependency preparation runs inside the job; cancellation
before or during preparation prevents inference. The Qt adapter drains progress
and detached word results before completion, while `AlignmentDelivery` still
publishes only on the project owner thread for the current run. Like GUI
transcription, these jobs are `session_only` even for saved projects; saving the
project remains required to persist their results.
