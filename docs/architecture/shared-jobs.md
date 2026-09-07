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

This is a partial U6 implementation. Recovery covers recorded results; a crash
after a provider response but before recording it cannot guarantee avoiding a
repeat call. Paid workflows, other analysis runners, GUI workflow migration, and
explicit session-only jobs remain outstanding. Per-clip project saves favor
recovery over throughput. The legacy boot sweep/idempotency behavior is preserved;
sharing a jobs database across independent live processes requires further
ownership work.
