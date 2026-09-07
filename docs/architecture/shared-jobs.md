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

This is the lifecycle extraction portion of U6. Immutable normalized operation
inputs, durable result IDs, project-before-checkpoint commit reconciliation,
charged-result reuse, and explicit session-only jobs remain outstanding. The
legacy boot sweep/idempotency behavior is preserved; sharing a jobs database
across independent live processes requires further ownership work.
