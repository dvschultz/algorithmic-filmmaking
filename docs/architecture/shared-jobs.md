# Shared jobs lifecycle

`core/jobs` owns the SQLite store, executor, cancellation events, and per-project
mutex. It imports neither Qt nor MCP. Historical `scene_ripper_mcp.jobs` imports
re-export the same classes and constants. The initial SQL schema is unchanged;
it is embedded in a Python module so source and frozen runtimes load it without
an unpackaged SQL resource. Existing rows and task IDs remain readable.

Handles are registered before workers can start. A rejected executor submission
records failure instead of leaving a queued orphan. A cancelled queued job skips
its runner. Conditional SQL updates preserve the first terminal decision and
prevent progress from reversing a cancellation request. Explicit failed work
results retain their payload and are recorded as failed.

`ui/workers/job_adapter.py` polls shared state on its Qt owner thread. Signals
carry task IDs and emit a terminal outcome once, including when completed
history was purged before polling. Existing GUI workers remain in place until
their workflows migrate; the adapter itself does not mutate project models.

The adapter's `result_ready(task_id, output)` signal delivers any recorded
terminal payload before the completed/cancelled/failed notification, exactly once
per task. Consumers that need partial outcomes should apply through this signal
and use outcome signals for status; do not apply again through `completed`'s
legacy payload. MCP `get_job_result` similarly includes available output on error
responses without changing their `success: false` or terminal error code. Neither
surface treats a partial payload as overall job success.

## Saved-result recovery

`core/jobs/spec.py` defines immutable `OperationSpec` metadata: operation kind and
version, canonical arguments and input snapshot, originating session/revision,
and cancellation/persistence capabilities. JSON encoding detaches nested values
and rejects non-string keys, nonfinite numbers, and arbitrary Python objects.
The runtime rejects mismatched kind/arguments or store persistence before
inserting work. Noncancellable operations reject caller cancellation; they cannot
borrow an external cancellation event.

Both color adapters supply specs. Desktop specs include normalized color policy,
parallelism, originating session and mutation generation. MCP specs include the
accepted project content revision and target snapshot; the runner verifies them
after acquiring the writer lease and before extraction. Changed queued inputs
fail instead of silently using a newer snapshot. Parameters are read from the
detached spec, so caller list mutation cannot alter queued work.

The nullable `operation_json` job column preserves old rows. Status projections
expose only the operation ID, kind/version and capabilities, never arguments or
input details. `job_history` describes the store, not a guarantee that project
changes were committed. Legacy submissions remain supported during migration;
explicit idempotency keys retain their existing first-submission behavior and
return the original job and its original metadata. Metadata does not serialize
or dynamically resolve executable callables. Per-item `ResultSpec` identities
remain separate from batch job identity; color uses one version constant for both.

`core/jobs/commits.py` identifies an operation by canonical project path, operation
kind/version, target, normalized arguments, and input snapshot. The serialized
identity is detached from mutable caller data and hashed into a result ID.

Publication has three steps under the project writer lease:

1. Store each computed JSON payload and digest immediately in SQLite with
   synchronous FULL, before computing the next item.
2. Stage accepted output and receipts on one private project model. Revalidate
   every staged input, payload digest, output and receipt before publishing a
   group of up to 16 results in one atomic project save.
3. Mark the group's results committed in one SQLite transaction only after the
   project save succeeds. A missing result or digest mismatch rolls back the
   entire checkpoint transaction.

A retry after a failed project save reuses the stored computation in a fresh
model. A retry after a checkpoint failure verifies the project receipt and output
and acknowledges it without applying twice. Missing or corrupt results, changed
inputs, removed receipts, and edited committed outputs fail closed.

Sequence-generation retries with an explicit idempotency key restore the failed
job's persisted `OperationSpec`, including its original seed and result identity.
They revalidate the stored asset parameters and input snapshot before dispatch.
A fresh unkeyed invocation remains an independent generation; direct engine
retries must retain their original spec rather than construct a new request.
Sequence publication checks cancellation both before application and when the
batch is flushed. An interrupted publication discards the detached edit and the
runtime reports cancellation, while the computed recipe remains available for
an explicit retry.

`result_batch()` owns the private model and writer lease. Normal exit flushes a
partial group; exceptional exit discards unsaved changes. Failed application or
publication makes the batch unusable, even if a caller catches the exception;
retry requires a fresh model. Cancellation flushes already-computed successes.
`commit_result()` retains its single-result API through a one-item batch. Color
serialization runs in the MCP job executor; GUI color application remains
session-only and uses the existing separate project-save flow.

A local diagnostic with 256 targets, mocked extraction and a 93,850-byte project
compared the same engine with one-item and 16-item groups. Project saves fell
from 256 to 16; time inside project save fell from 0.901s to 0.058s, and total run
time from 1.602s to 0.527s. These measurements isolate grouping cost; they do not
predict media extraction latency.

Project schema 1.5 adds `job_results` receipts. Earlier files migrate with the
existing exact-byte backup policy. Schema-aware 1.4 clients open 1.5 read-only,
preventing them from silently dropping receipts. Ordinary save and Save As retain
receipts. SQLite adds a separate result table without altering old job IDs or rows.
Deleting or purging job history retains computed results for recovery; automatic
result-cache cleanup is not implemented yet.

`core/jobs/colors.py` is the first production caller, through MCP
`start_analyze_colors`. Each clip retains its own result ID within a commit group.
Media path, frame range,
mtime/size, project/source IDs, and color count determine reuse. Legacy palettes
without managed receipts keep their skip-existing behavior. Managed palettes are
verified on retry; changed media gets a new result identity. Media fingerprints
use mtime/size, not full content hashes. Missing cached receipt data blocks this
pilot because it cannot safely establish which palettes it owns.

## Unsaved project sessions

`JobRuntime.for_session()` creates an isolated in-memory SQLite store for one
unsaved project. It uses the same executor, terminal transitions, cancellation,
and idempotency behavior. Submission and status projections explicitly include
`persistence: session_only`. Completed means computation completed; it does not
mean the project was saved. Existing durable MCP response shapes are unchanged.

The Qt adapter emits `started(task_id, persistence)` on its owner thread before
terminal delivery, so migrated workflows can label unsaved work. It remains the
workflow's responsibility to apply computed data on the project owner thread and
save it through the project model. Saving does not retroactively turn session
history into durable receipts. New disk-backed operations use a regular runtime.

`shutdown()` stops workers but retains history for final UI polling.
`close_session()` waits for workers and discards the in-memory store; call it off
the UI thread after final polling. Closing is idempotent. No job or cache history
is available to a newly created session, even with the same idempotency key.
Session runtimes reject saved project paths, and the durable result commit path
rejects an in-memory store.

Desktop color analysis now uses this runtime across pipeline, chat, Frames, and
intention entry points. `ColorAnalysisWorker` keeps its QThread API for existing
start/cancel/wait/finished callers, but computation and task state run through a
session runtime. Its thread relays progress from a queue and delivers the final
immutable batch before analysis completion. Application stays in the GUI's
owner-thread slot. The status bar explains that results remain unsaved until a
project save, including when the project already has a file path.

Cancellation uses one event shared by the Qt adapter and runtime. The runtime
retains a returned partial payload on cancelled jobs, so completed palettes are
still applied while undispatched targets remain unprocessed. Pre-start
cancellation skips computation. The adapter closes its session store after
reading the terminal result. Other desktop workflows still await migration.
The QThread compatibility shell can be removed once main-window callers adopt
task-aware start, wait and cleanup instead of the legacy worker API.

This is a partial U6 implementation. Recovery covers recorded results; a crash
after a provider response but before recording it cannot guarantee avoiding a
repeat call. Paid workflows, other analysis runners, and GUI workflow migration
remain outstanding. The color pilot now batches project saves while preserving
per-item recovery identities.

## Runtime ownership and restart recovery

Each disk-backed runtime acquires a unique OS-backed owner lease before accepting
work. Its owner ID is written atomically with each job row. The lease uses the
existing managed OS-lock primitive keyed by owner UUID; no PID or heartbeat
timeout establishes liveness. Runtime leases may close on the last worker's
thread; project-writer leases retain their owner-thread checks. The OS releases
the lease after process death. Lock records remain
permanent so deleting a record cannot split its lock identity.

Boot recovery probes these leases. Live owners are skipped; abandoned owners'
queued, running and cancelling jobs become crashed. Completed and other terminal
states stay unchanged, and saved result receipts remain available for retry.
Nonblocking shutdown retains the lease until the last worker settles. Submission
and shutdown are serialized so no job can be inserted after its lease is released;
closed runtimes reject submissions before writing a row.

The additive `owner_id` column is installed under an SQLite write transaction,
preserving historical job IDs and public projections. Legacy rows without owners
retain the earlier running/cancelling recovery policy; their queued rows remain
untouched. Concurrent old binaries that perform the unconditional legacy sweep
are not supported. Session-only stores need neither leases nor restart recovery.
This does not add cross-process cancellation or a distributed queue: a runtime
can cancel its own handles, and project writer leases still guard file mutation.

## Cinematography publication

GUI clip and frame analysis and headless clip analysis share detached computation
in `core/operations/cinematography.py`. `CinematographyApplication` captures the
originating project session, target identity, media stamps, clip range and source
FPS, and existing cinematography and shot type. It applies a successful result
once, on the project owner thread, only while those inputs remain unchanged.
Unrelated edits such as notes do not invalidate the result.

The GUI delivery object also checks the current worker, request, pipeline run,
and cancellation state before publication. Frame targets remain explicit even
when a frame and clip share an ID. Model notifications update frame views; the
clip callback only refreshes views after the shared application publishes.
Headless callers receive `stale_result` for a rejected successful computation.

Saved-project cinematography jobs now use `core/jobs/cinematography.py` from both
the dedicated MCP tool and multi-step analysis jobs. Receipt identity includes
project/source identity, media fingerprints, range/FPS, resolved cloud/local
models, input mode, and runtime availability. Computation is recorded before
model publication; project saves precede cache checkpoints. Retrying a failed
save reuses recorded inference, and a failed checkpoint reconciles the exact
saved output, including its derived shot type. Existing or manually edited
analysis remains intact. Missing or corrupt committed receipts reject replay.

GUI cinematography workers now use the shared runtime for saved and unsaved
projects. Saved projects record clip/frame results through
`GuiCinematographyCache` before queued delivery. A restart or failed save can
reuse matching inference without model preparation. Mixed batches publish cached
hits and compute only misses. Unsaved projects keep session-only history.

GUI receipt identity includes the explicit target type, prior analysis and shot
type, frame association, task/media identity, resolved options, and runtime.
Delivery verifies the recorded payload and current project location before
applying it. Only an explicit project save acknowledges matching analysis and
derived shot type; Save As or edited output does not acknowledge the old receipt.
An explicit refresh after saving starts a new generation. Other U7 workflow
families remain outstanding.

## Content classification computation

GUI content classification, CLI `analyze classify`, and headless `classify_content` share immutable tasks,
options, and outcomes in `core/operations/classification.py`. The runner admits
one inference at a time across shared classification jobs because they use one
MobileNet singleton. Waiting for that model is cancellable. Cancelled inference
does not publish its result or start later targets.

The GUI worker retains its signals and parallelism argument for caller
compatibility, but inference runs serially. Completion is emitted even after
pre-start cancellation or errors. Empty label lists remain valid successes;
malformed labels or confidence values become per-item failures.

`ClassificationApplication` applies results once on the project owner thread,
checking the originating session, target/source objects, media, clip range/FPS,
frame association, and prior labels. CLI-generated analysis thumbnails may differ
from the stored display thumbnail; applying labels preserves the display path.
GUI delivery also checks cancellation and the current worker, request, and
pipeline. Model notifications refresh views after publication.

Frames persist ImageNet labels in `object_labels`, separate from object detection.
An empty list means classification completed without matching labels; `None`
means no classification. Older frame files retain their existing detection data
and load with no classification. Clip serialization also preserves empty labels.

Saved-project CLI classification and both dedicated and multi-step MCP jobs now
record results through `core/jobs/classification.py`. Receipts include source and
image fingerprints, range/FPS, options, display-image identity, and the selected
MobileNet weights and installed runtime versions. Inference uses the vocabulary
from those weights rather than a separately downloaded label file.

Failed saves reuse recorded computation. Failed checkpoints reconcile exact saved
labels, including CLI results made with a separate analysis image. Existing user
labels remain intact. CLI `--force` starts a new result generation after a
successful checkpoint; an interrupted refresh reuses its recorded results.
Missing or corrupt committed receipts reject replay.

GUI classification now runs through the shared runtime for clip and frame
targets. Saved projects record immutable outcomes before queued delivery; matching
results can be reused after reopening without preparing the model. Unsaved
projects retain session-only job history. Cache identity includes target type,
prior labels, frame association or clip range, source identity/FPS, media
fingerprints, and model runtime. Unrelated notes do not invalidate reuse.

Delivery verifies the recorded payload and original save location before applying
labels and recording the receipt on the project owner thread. An explicit save
acknowledges only the matching labels, including an empty successful result.
Save As and manually changed labels do not acknowledge old results. A refresh
after save starts a new generation. The other remaining workflow families are
still outstanding in U7.

## Ordered frame analysis

`core/operations/frame_analysis.py` tracks the requested steps and each frame's
outcomes without importing Qt. `ui/workers/frame_analysis.py` coordinates the
existing analysis workers, starting the next operation only after the previous
worker's native thread has finished. The main window gates capabilities and
displays progress; it no longer counts completion signals or saves implicitly.

Provider options are captured at submission. Each step constructs fresh tasks
and publication adapters after preceding results have updated the project.
Changing settings during a run therefore cannot change later provider choices.
Existing metadata is preserved, including valid empty results. A frame is marked
analyzed only when every requested operation succeeds or already has a result.
Failures leave successful metadata available for inspection and explicit saving.

Cancellation prevents subsequent steps. Replaced runs retain their workers until
native completion but discard late results. Publication verifies project/session,
save location, frame object and media identity, request ownership, and durable
receipts. Dependency dialogs also cannot redirect an existing request into a
replacement project or frame. Closing the window waits for active workers to
settle through the existing close preflight.

This completes the frame coordinator slice, not U7: intention-workflow planning
and the remaining analysis/import route audit are still required.

## Intention phase plan

`core/operations/intention.py` defines the ordered download, detection, thumbnail,
analysis, and building steps without Qt. The existing intention coordinator
projects this plan through its signals and retains partial source results.
Phase completion is consumed once; out-of-order signals and cancellation from
transition observers cannot start later steps. Download notifications are counted
once per requested URL, and duplicate successful detection callbacks do not
advance the source queue.

Color, shot-type, and Storyteller workflows check their required metadata before
building. Missing dependencies produce an error instead of a successful analysis
completion. Failed workflows remain inspectable and do not schedule sequence
finalization. Successful finalization captures its workflow, plan, project, and
session so a delayed callback cannot finalize a replacement run.

This is the phase-plan portion of the intention migration. Worker dispatch,
native-thread advancement, and run ownership across the remaining intention
adapters still need to leave `MainWindow`. U7 remains incomplete.

The route audit identified an off-thread mutation and duplicated loop in the
former main-window thumbnail worker. The thumbnail cutover below resolves that
portion of worker extraction.

## Shared thumbnail computation

`core/operations/thumbnails.py` owns detached tasks, per-clip outcomes, generation,
and guarded owner-thread publication. Headless backfill and detection use it
through `core/spine/thumbnails.py`. All five desktop thumbnail entry points use
`ui/workers/thumbnail_worker.py` and the shared session job runtime. The main
window no longer defines a thumbnail worker or assigns returned paths to models.

The GUI preserves its 160x90 default; headless calls preserve their 320x180
default and explicit dimensions. Existing thumbnails are skipped unless forced.
Generated cache identities include media identity, range, FPS, and dimensions.
Forced refresh creates a new file instead of overwriting a referenced artifact.
Generation writes a temporary file and publishes it atomically only after
checking cancellation and source identity.

Delivery validates the project/session/save location, clip/source objects,
range/FPS, source media, prior thumbnail, and request ownership. Replaced and
cancelled workers remain retained until native thread completion. Reset cancels
all retained thumbnail workers; closing waits for them through the existing
close preflight. Model changes use `Project.update_clips()` on the owner thread;
saving remains explicit. Intention thumbnails also check their originating plan.

Remaining intention analysis worker ownership and the broader
U7 route audit are still required.

## Intention detection ownership

`ui/workers/intention_detection.py` owns serial detection for one intention plan.
It snapshots detection settings, buffers results, and advances to the next source
or phase only after the current worker's native `finished` signal. The main
window no longer implements the detection loop or waits for old intention
workers during replacement.

`DetectionApplication` in `core/operations/detection.py` publishes on the project
owner thread after validating session, save location, media, source identity,
and existing clip objects. Imported source IDs are preserved. Failed or stale
sources produce per-source errors while later sources can still succeed.
Publication does not save the project.

Cancellation, restart, reset, and close retain active workers until they exit.
Callbacks also check the originating coordinator and plan; an old standalone
detection completion cannot clear a replacement intention worker. The controller
checks ownership again after model and UI callbacks before advancing the plan.

U7 remains partial: intention analysis ownership plus the remaining
route audit still need migration.

## Intention download ownership

`ui/workers/intention_download.py` binds the existing recoverable URL download
worker to one intention run. Detection and download adapters share
`IntentionRun` identity checks for the project, session, save path, coordinator,
plan, and requesting agent. Dependency and directory dialogs recheck that
identity before dispatch; cancelled or failed gates terminate the current run.
Directory selection uses the existing validated-path dialog helper.

The worker captures immutable `DownloadItem` records when each file completes.
`DownloadApplication` admits them on the owner thread, validating media identity
and retaining existing source IDs and editorial metadata. Successful sources can
appear in Collect while other downloads continue. Cancellation preserves sources
already admitted and rejects later delivery; it does not implicitly save.

Only native worker completion advances the plan. Before that transition, the
adapter rechecks source and media identity so removed or replaced sources do not
enter detection. The coordinator reconciles terminal summaries, counts failures
and missing outcomes once, and keeps successful downloads and local files when
other URLs fail. Distinct Qt slots preserve sender identity for progress and
native completion while reusing the shared download channel's retention.

Reset, replacement, and close cancel workers while retaining them until native
completion. A worker that cannot start is released immediately. The remaining
intention analysis adapters and the broader U7 route audit are still open.
