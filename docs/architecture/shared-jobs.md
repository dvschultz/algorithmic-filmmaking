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

## Saved-result recovery

`core/jobs/commits.py` identifies an operation by canonical project path, operation
kind/version, target, normalized arguments, and input snapshot. The serialized
identity is detached from mutable caller data and hashed into a result ID.

Publication has three steps under the project writer lease:

1. Store the computed JSON payload and digest in SQLite with synchronous FULL.
2. Validate inputs and disk revision, apply to a freshly loaded project, and save
   both the output and its result-ID/digest receipt in one atomic project file.
3. Mark the stored result committed only after the project save succeeds.

A retry after a failed project save reuses the stored computation in a fresh
model. A retry after a checkpoint failure verifies the project receipt and output
and acknowledges it without applying twice. Missing or corrupt results, changed
inputs, removed receipts, and edited committed outputs fail closed.

Project schema 1.5 adds `job_results` receipts. Earlier files migrate with the
existing exact-byte backup policy. Schema-aware 1.4 clients open 1.5 read-only,
preventing them from silently dropping receipts. Ordinary save and Save As retain
receipts. SQLite adds a separate result table without altering old job IDs or rows.
Deleting or purging job history retains computed results for recovery; automatic
result-cache cleanup is not implemented yet.

`core/jobs/colors.py` is the first production caller, through MCP
`start_analyze_colors`. Each clip is committed separately. Media path, frame range,
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
remain outstanding. Per-clip project saves favor recovery over throughput.

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
