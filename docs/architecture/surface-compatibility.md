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

## Description computation cutover

GUI clip/frame descriptions, spine descriptions, CLI descriptions, and MCP
jobs share `core/operations/description.py`. Immutable tasks retain target
IDs, image paths, source paths, and clip ranges. Provider computation remains in
`core/analysis/description.py` and loads lazily.

Description options snapshot the selected model, tier, prompt, input mode, and
parallelism before GUI dispatch or MCP submission (at entry for direct spine calls). Provider calls
receive the selected model explicitly, including video calls and frame fallback.
The local model cache is keyed by requested model and available backend, so a
model change cannot silently reuse another model's weights. API credentials are
resolved at call time and are not part of the operation options.

The shared runner admits at most the configured number of tasks (capped at five)
and serializes local inference. Headless descriptions retain serial scheduling.
Both surfaces use the existing GUI transient-error classification and up to three
retries at 2, 5, and 10 seconds; authentication errors are not retried. Cancellation
interrupts retry waits, stops admission, and suppresses in-flight results while
waiting for active provider calls to return. Empty or `Error...` provider replies
are failures on both surfaces. Unexpected per-task failures preserve other results.

GUI startup still owns local-model preloading and completion signals. GUI input
filtering preserves its existing skip behavior; headless responses retain explicit
skipped/missing-thumbnail entries.

`DescriptionApplication` guards clip/frame publication by project session,
target and source object identity, prior description metadata, clip range, image
path, source path/FPS, and media stat identity (including ctime). Accepted results
use owner-thread project update methods; duplicate delivery does not apply again.
All four GUI entry points bind results to their launching worker, with pipeline
and agent-request checks where applicable. Headless calls report rejected results
as `stale_result`; GUI delivery reports a discarded-result error.

CLI descriptions, dedicated MCP `start_describe`, and description steps in MCP
analysis plans now use `core/jobs/description.py`. Result identity includes the
project/target, source and image fingerprints, resolved options, and local
backend/fallback provenance. Computation is recorded before publication; project
data and receipts save together before cache checkpoints. Retries after save or
checkpoint failure reuse matching results. Existing user-edited descriptions stay
intact by default; CLI `--force` and MCP `force=true` request a new refresh generation.
Missing committed cache payloads fail without recomputing.

CLI retains eight-character clip prefixes, 640x360 thumbnail preparation,
frame-only inference, and its result counters. Generated thumbnails are included
in the saved project when results are committed. CLI/MCP reuse requires matching
options and media plus the same cache directory. The old CLI inference loop is
removed. Direct spine calls remain non-durable; custom queries and cinematography
remain pending U7 work.

Saved GUI projects use `gui_describe` history and a computation journal for both
clip and frame targets. Results are recorded before queued delivery, and matching
results from an interrupted run are reused without model preparation or inference.
Mixed batches compute only missing targets. Identity includes prior description
metadata, image/source fingerprints, resolved options, and runtime provenance.
Delivery verifies the recorded payload, original save location, and owner-thread
application guards before recording a project receipt. Unsaved projects use
session-only history. All four GUI entry points use this path.

GUI computation never saves the project automatically. An explicit successful save
acknowledges only receipts whose output matches the exact saved clip/frame snapshot.
Save As and edited descriptions do not acknowledge the original result; failed
checkpoints can be retried by saving again without inference. GUI and headless
receipts retain separate identities because their publication policies differ.

## Custom-query computation cutover

GUI clip/frame queries and spine/MCP queries share `core/operations/custom_query.py`.
Tasks retain target identity and trimmed query text; options snapshot the provider
model before execution. Cloud admission is bounded to the configured concurrency
(at most five), while local inference stays serial on the calling worker thread.
Both paths use the same transient retry classification and cancellation-aware
waits. Cancellation stops admission and suppresses in-flight results. GUI workers
emit completion even if local model preparation fails.

The existing custom-query legacy tier mapping is preserved: `cpu` means local and
`gpu` means cloud. Existing append/skip result behavior remains, with whitespace
normalized before skip checks.

`CustomQueryApplication` appends a result once, on the owner thread, after checking
project/session identity, clip/source object identity, clip range, source path/FPS,
image/source media identity, prior query results, and the submitted query text.
Spine calls report rejected results as `stale_result`. GUI delivery also checks the
launching worker, pipeline run, cancellation, and agent request before publication;
the main-window callback only refreshes views. Frame tasks cannot resolve through
a colliding clip ID: the Frame model has no custom-query storage yet.

GUI clip queries use the shared job runtime, with model preparation and local
inference on the same runtime thread. Saved projects retain `gui_custom_query`
history and computation receipts; unsaved projects use session-only history.
The GUI journal reuses matching results across restarts and computes only missing
targets. Queued delivery verifies the saved payload and original project location
before owner-thread publication. An explicit project save acknowledges matching
appends, including several queries accumulated before saving. Edited prefixes and
Save As do not acknowledge the original results. Failed checkpoints retry on the
next save without inference. All GUI journals reject corrupted committed result
identities before consulting cached history.

Dedicated MCP custom-query jobs and custom-query steps in analysis plans now use
`core/jobs/custom_query.py`. Submission captures the query, model, media identity,
and backend provenance. Computation is recorded before one project save publishes
the request's successful appends and receipts together. A failed save reuses those
results; a failed checkpoint reconciles the exact saved append without inference
or duplication. After successful completion, a later request appends a fresh
result as before. Manual edits become inputs to the next append. Missing committed
payloads fail without repeating paid computation. Later analysis steps reload the
saved model so they cannot overwrite custom-query receipts with older state.

## Cinematography computation cutover

GUI clip/frame analysis and spine/MCP cinematography share
`core/operations/cinematography.py`. Tasks retain target identity, image/video paths,
and frame ranges; options snapshot tier, input mode, cloud model, and local model
before dispatch. The local provider now loads the configured cinematography model
explicitly. Local inference is serial on the calling worker thread; cloud admission
is bounded to the configured concurrency, capped at five.

Transient failures use the shared retry classification and cancellable waits.
Cancellation stops admission, suppresses in-flight results, drains active calls,
and still emits GUI completion. Outcomes serialize analysis data so provider,
incremental GUI signals, and final results do not share mutable analysis objects.
Frame targets force frame mode. Existing video extraction fallback and headless
shot-type projection remain. Guarded publication and durable cinematography
recovery are subsequent U7 steps.

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

Unsaved-project GUI jobs remain session-only. Saved-project transcription jobs
retain history as `gui_transcribe`; their outcomes remain unsaved model changes
until the user saves. Computed outcomes are retained in the shared journal
described below independently of project publication.

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
Direct spine calls do not yet use durable receipts. GUI alignment records its
detached word outcomes separately from headless publication, as described below.
The explicit capability-install workflow remains outstanding.

The GUI alignment worker now submits immutable alignment metadata to the shared
session job runtime, records a task ID and terminal status, and closes the runtime
before emitting completion. Dependency preparation runs inside the job; cancellation
before or during preparation prevents inference. The Qt adapter drains progress
and detached word results before completion, while `AlignmentDelivery` still
publishes only on the project owner thread for the current run. Like GUI
transcription, job history is `session_only` for unsaved projects. Saved-project
alignment jobs retain history as `gui_align_words`; saving the project remains
required to persist its edits.

For an already saved GUI project, `core/jobs/gui_alignment.py` records successful
word outcomes in the shared computed-result store before Qt delivery. Input
identity includes transcript data, media hashes, project/source identity, and a
receipt generation for explicit refreshes. Matching restarts reuse computation
without dependency preparation. The owner-thread delivery checks the output
against its immutable recorded payload, rejects changed save destinations and
stale media, and attaches a receipt only after guarded application. Ordinary
project saves include words and receipts together; inference never saves unrelated
edits automatically. Unsaved projects remain memory-only. GUI alignment uses a
separate result identity from headless alignment because it records word outcomes
before segment publication; it does not reuse headless segment-result entries.
Saved-project transcription and alignment job history survives runtime cleanup.

Saved GUI transcription now uses the same `core/jobs/gui_results.py` journal.
Its adapter includes the resolved transcription options and prior transcript in
the input identity, publishes matching cache hits without model preparation, and
sends only cache misses through the existing bounded parallel runner. Silent
transcripts are recorded too. Receipts are attached only after the owner-thread
guard accepts an unchanged queued payload and save destination; normal saves
persist the transcript and receipt together. Changed media is rejected before
preparation, after preparation, and before recording or delivering results.
GUI transcription uses its own outcome-cache identity; it does not reuse
headless transcript-result entries. Neither GUI adapter saves unrelated edits
automatically, and projects without a save location remain memory-only.

The common atomic save path now acknowledges matching GUI result receipts after
the project file is written, under the same writer lease. This covers synchronous
and background saves. `core/jobs/gui_checkpoints.py` validates the serialized
snapshot's project/path/target identity and transcript against the recorded
outcome, including word distribution for alignment. Newer live edits are never
used to acknowledge an older background-save snapshot. Changed or historical
nonmatching outputs stay pending; identical saved refreshes are all reconciled.
Pending rows are read in bounded queries through one database connection.

A checkpoint error cannot turn an already-written project into a failed save.
The error is logged and a later ordinary save retries acknowledgement without
inference or result application. Missing caches are not recreated by saving;
unknown receipts and headless result kinds are left to their owning recovery
paths.

Saved GUI transcription/alignment jobs use `OperationSpec.publication=owner_thread`.
Their runtime retains canonical project association and owner leases for crash
recovery, while the editor keeps its project writer and applies accepted results
on its owner thread. The default worker-publication path still acquires its own
writer. Older operation JSON and IDs omit the new default field and stay stable.
GUI job kinds are distinct from headless jobs: `completed` describes finished
computation, and `publication: explicit_project_save` describes the publication
policy, not the current saved state. New GUI runtimes recover abandoned owners
as crashed without restarting inference or disturbing live jobs. Other GUI
operation families still need their history migrations.

Short durable job-store connection scopes are serialized within the process to
avoid the SQLite connection open/close deadlock observed on macOS. Inference and
project-file writes stay outside these scopes and retain their concurrency.

### Content classification cutover

All classification surfaces use `core/operations/classification.py` for immutable
tasks and outcomes, validated labels, and serial MobileNet inference. Cancellation
while waiting for the model or during inference suppresses late results. The
vocabulary comes from the selected weights. Empty labels mean completed analysis.

| Surface | Existing-data policy | Publication and recovery |
| --- | --- | --- |
| GUI pipeline and chat | Skip non-None labels | Shared job runtime; guarded owner-thread publication; saved-project journal |
| GUI Frames | Same policy, explicit frame identity | Same journal; frame labels remain separate from object detections |
| CLI `analyze classify` | Skip existing; `--force` refreshes | Durable runner saves project before checkpoint; retains 320x180 analysis images without replacing display thumbnails |
| Dedicated and generic MCP jobs | Skip existing | Same durable runner; frozen submitted options and input revision |
| Direct spine | Skip existing by default | Shared guarded application; caller owns saving |

GUI computation is recorded before delivery but does not save unrelated project
edits. Explicit saves acknowledge exact labels and receipts; failed checkpoints
can be acknowledged by a later save. Reopening after a failed save reuses matching
computation. Unsaved GUI projects retain session-only history. Save As, modified
queued payloads, edited labels, and target/media changes cannot acknowledge or
apply stale output. Unrelated notes preserve reuse.

The existing QThread API remains a compatibility shell for start/cancel/wait and
completion callers. Its removal condition is migrating those callers to the
task-aware adapter. Object detection and the remaining U7 families still await
their own cutovers; this does not mark U7 complete.

### Object detection computation

GUI workers, CLI `analyze objects` / `analyze people`, and spine detection
(including MCP jobs) now use `core/operations/object_detection.py`. Tasks retain
clip/frame identity; results contain immutable detections and person counts.
Legacy signals and model adapters receive fresh dictionaries and bounding-box
lists. Malformed responses become per-item failures; empty detections are valid.

Inference is serial across shared callers because the YOLO model is a singleton.
The GUI parallelism argument remains accepted. Cancellation before model access
or during inference suppresses late replies and prevents later work. Model-load
failure stops the batch without repeated downloads; untouched targets are
reported as unprocessed. Existing data-selection and publication policies remain
at the adapters. Guarded publication and durable object-detection recovery are
the next U7 migration steps.

### Guarded object-detection publication

Object-detection results now publish through `ObjectDetectionApplication` on the
project owner's thread. GUI pipeline, agent, and Frames launchers use a queued
delivery adapter bound to the worker, project session, target type, and active
pipeline/reply context. CLI and spine use the same application guard. Changed
media, target replacement, changed source/range, edits to output fields, and
duplicate replies cannot overwrite newer state. Unrelated notes remain intact.

People-only analysis checks `person_count` when skipping existing results and
updates only that field. Full object analysis updates both detections and count.
Frames persist person counts, including zero; both clips and frames preserve a
valid empty detections list across save/load. Missing analysis remains `None`.
Durable object-detection result recovery remains the next U7 step.

### Durable headless object detection

CLI object/people commands, the dedicated MCP job, and object detection within
generic analysis plans now use `core/jobs/object_detection.py`. Saved result
identity includes source/image fingerprints, source range, display and analysis
image paths, confidence, detection mode, and the named YOLO weights release and
runtime package versions. Force requests create a new result generation.

Computation is recorded before project save; the project is saved before result
checkpointing. Explicit retries reuse recorded computation after failed saves
and reconcile completed project saves after checkpoint failures. CLI retries
retain their 320x180 analysis-image identity without replacing display images.
People-only output validation checks count without overwriting object detections.
Legacy populated results retain the existing explicit-force reuse policy.
GUI object-detection job history and recovery remain the next migration step.

### GUI object-detection recovery

Pipeline, agent, and Frames object-detection workers now use the shared job
runtime. Saved projects keep job history and record computed detections before
queued owner-thread publication. Retries reuse matching records; unsaved projects
remain session-only. Completion settles after runtime cleanup.

The GUI journal captures target type, prior affected outputs, source/frame
identity, image/source fingerprints, options, and runtime identity. People-only
records exclude object metadata from affected outputs. Delivery verifies the
recorded payload and save location before applying it. A project save checkpoints
only receipts whose exact affected outputs are present in the saved snapshot,
including zero people and empty detections. Changed outputs or Save As cannot
acknowledge an unrelated result. Other U7 analysis families still await migration.

### Shared face computation and guarded publication

GUI face analysis and spine/MCP face-embedding analysis now use
`core/operations/faces.py`. Workers receive source/range snapshots and return
immutable bounding boxes and embeddings. The GUI worker no longer mutates clips;
launch-bound queued delivery applies accepted results on the project owner thread.
Spine publication uses the same source, target, and prior-output guards.

Cancellation suppresses late replies, model-load failures stop further inference,
and completion is emitted after cleanup even on cancellation or failure. Shared
face jobs serialize model use and unloading. Invalid provider data and unreadable
video samples are failures; valid empty face results remain distinct from missing
analysis and survive save/load. Existing embedding rounding in saved projects is
unchanged. Durable face job recovery remains the next U7 step.

### Durable headless face analysis

Dedicated MCP face jobs, face steps in generic analysis plans, and CLI
`analyze faces <project> [--sample-interval <seconds>] [--force]` now use
`core/jobs/faces.py`. Source/range fingerprints, sampling options, runtime package
versions, and model identity bind recorded results to their inputs. Forced
refreshes create new generations; changed saved face data is preserved by default.

The journal retains full-precision embeddings. Output reconciliation accepts
either the recorded value or exactly its existing five-decimal project
serialization, allowing checkpoint retries after save without another inference
call. One lazily acquired model session spans a job's result commits and is
released on success or failure; cache-only retries do not load the model.
GUI face job history and recovery remain the next U7 step.

### GUI face recovery

The desktop face-analysis launcher now uses the shared job runtime. Saved-project
jobs journal successful results before queued GUI delivery; explicit retry can
reuse these computations after reopening without loading the face model. Unsaved
projects remain session-only. Workers never save unrelated project edits.

Owner-thread delivery validates the launch context, target inputs, save location,
and recorded payload before applying results and attaching receipts. Explicit
project saves acknowledge only matching saved faces, using the existing five-digit
embedding precision while computation and queued delivery retain full precision.
Valid empty results are preserved, missing sources fail per item, and existing
faces can be skipped even when their source is offline. Pending inference shares
one model session; cache-only recovery does not initialize it.

Gaze, embeddings, OCR, remaining analysis routes, and workflow orchestration are
still pending in U7. This face cutover does not complete U7.

### Shared gaze computation and guarded publication

GUI and spine/MCP gaze analysis use `core/operations/gaze.py` with immutable source
and clip snapshots. Workers no longer mutate clips. Owner-thread delivery rejects
changed targets, source media, sessions, pipeline runs, and agent replies, and
deduplicates queued results before refreshing the existing gaze UI.

Shared gaze batches serialize model use and unloading, avoid loading for cancelled
or skipped work, and discard inference returned after cancellation. Per-item
failures remain visible without losing successful siblings. Invalid angles and
unreadable video samples are failures. The legacy `no_gaze_detected` outcome and
GUI signal signature remain compatible; GUI progress now counts every requested
item, including skips. Pipeline summaries retain gaze errors.

Durable gaze job recovery remains the next U7 step. This cutover does not complete
U7 or add native-process isolation.

### Durable headless gaze analysis

Dedicated MCP gaze jobs, gaze steps in combined analysis plans, and CLI
`analyze gaze <project> [--sample-interval <seconds>] [--force]` journal completed
observations through `core/jobs/gaze.py`. Source and range fingerprints, sampling
options, algorithm/model identity, and runtime package versions bind each result.
Successful computations survive failed saves, and reconciliation recognizes the
project format's two-decimal angle precision without rounding the journal payload.

A completed no-gaze observation has an explicit receipt even though all three
gaze fields are empty. The existing public `no_gaze_detected` response remains;
retries reuse its receipt instead of repeating inference. Decode/model failures
have no successful observation receipt. Forced refresh can replace old angles
with a completed empty observation. Ordinary retries preserve edited angles.

One model session spans pending items and their commits. Cache-only retries never
initialize the model. GUI gaze history and recovery remain the next U7 step.

### Durable GUI gaze observations

Saved-project GUI gaze workers now use shared job history and journal each
successful observation before queued publication. Retrying after interruption
reuses matching computations without loading the model. Unsaved projects remain
session-only; analysis never saves unrelated desktop edits automatically.

The owner-thread delivery validates launch context, target inputs, save location,
and the recorded result before applying it once. Explicit project saves checkpoint
matching receipts using the existing two-decimal gaze serialization. Both the
typed observation signal and legacy populated-gaze signal can arrive without
duplicating application or UI refresh.

A completed empty observation clears old gaze fields and has a durable receipt.
Normal retries skip saved empty observations only when media, range, options,
runtime, and current empty fields match; an explicit refresh recomputes them.
Missing analysis and failed decoding remain distinct from completed empty results.
The headless public `no_gaze_detected` response remains compatible.

Embeddings, OCR, remaining audio/frame analysis, and workflow orchestration remain
U7 work. This gaze cutover does not complete U7 or isolate native inference.

### Shared embedding computation and guarded publication

GUI and spine/MCP clip-thumbnail embeddings now use
`core/operations/embeddings.py`. Bounded batches return immutable vectors;
only the project owner applies results after validating clip, source, thumbnail,
and prior embedding state. Queued results from cancelled jobs, replaced workers,
or changed project sessions are rejected, and duplicate delivery applies once.

The shared operation rejects zero, non-finite, and wrong-sized vectors. A batch
with the wrong number of vectors fails explicitly instead of silently truncating
its results. Missing thumbnails fail per item, successful earlier batches remain
available after a later batch failure, and cancellation rejects late results.
Migrated embedding jobs serialize model use and cleanup; cancelled waiters do
not unload another job's model. Progress includes skipped and failed targets.

The legacy `embedding_ready(clip_id)` signal remains a computation notification;
the main window uses the typed outcome signal for guarded publication. Pipeline
summaries now retain embedding errors, and result refresh does not dirty the
project a second time. Existing headless result fields remain, with an additive
`unprocessed` list for cancellation and aborted later batches.

Embedding job history/recovery, boundary and sequencer-specific embedding routes,
OCR, remaining audio/frame analysis, and workflow orchestration remain U7 work.
Native-process isolation remains a later milestone.

### Durable headless thumbnail embeddings

Dedicated MCP `start_generate_embeddings`, embedding steps in combined analysis
plans, and CLI `analyze embeddings <project> [--chunk-size <n>] [--force]` use
`core/jobs/embeddings.py`. Submitted jobs capture target ranges, thumbnail/source
identity, options, and runtime metadata. Receipt validation rejects corrupt or
changed inputs before reuse; existing manually edited vectors are preserved
unless explicitly refreshed.

Inference remains batched. Each computed batch is recorded before project
publication, including vectors beyond a project-save batch boundary. Failed
saves reuse those vectors, and failed checkpoints reconcile saved receipts.
Thumbnail vectors retain their full saved precision. Only missing work loads
the model; ownership spans computation batches and a fatal model/batch error
stops later dispatch. Cancellation preserves accepted results and leaves
recorded but unapplied neighbors available for an explicit retry.

Public MCP job names remain unchanged. GUI embedding job history and recovery,
remaining embedding routes, OCR, and the rest of U7 remain outstanding.

### Durable GUI thumbnail embeddings

Saved-project GUI embedding workers now adapt the shared job runtime and use
`core/jobs/gui_embeddings.py` to journal complete computed batches before queued
publication. Matching retries reuse recorded vectors without loading or unloading
the model. Unsaved projects remain session-only, and analysis does not save
unrelated desktop edits.

The journal binds source, thumbnail, range, previous vector/model, options, and
runtime metadata. Owner-thread delivery additionally verifies the exact recorded
payload and current save location before attaching its receipt. Explicit saves
checkpoint only matching saved vectors/model identifiers at full precision.
Changed output, Save As, invalid vectors, and corrupt cache records cannot be
acknowledged as matching results. Both legacy and typed computation signals
remain available; only typed delivery applies project changes.

Boundary/sequencer embedding routes, OCR, remaining audio/frame analysis, and
workflow orchestration remain U7 work. Native-process isolation remains later.

The MCP analysis adapter translates shared operation kinds to their existing
public job names at submission. Gaze, faces, objects, classification, and
cinematography are covered by tests that submit to the real job runtime and
verify completed jobs and saved receipts, with provider inference mocked.
### Sequencer thumbnail embedding prerequisites

Similarity-chain and Staccato thumbnail prerequisites delegate to the shared
embedding operation, including bounded batches, vector validation, cancellation,
and model ownership. Sequencing uses detached clip snapshots: prerequisite
embeddings are inputs to the proposed sequence and do not silently update the
project's analysis fields. Use explicit embedding analysis to persist those fields.
Staccato requires an embedding for every clip, including when no thumbnail can be
computed; similarity-chain retains its existing fallback for missing embeddings.

This U7 migration does not yet provide durable sequencer prerequisite jobs.
Boundary embeddings, the full sequencing operation/recipe migration, and remaining
analysis workflow migrations are still outstanding.
