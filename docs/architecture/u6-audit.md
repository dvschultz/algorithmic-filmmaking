# U6 implementation audit

Scope: U6 in `docs/plans/2026-09-06-1218-refactor-shared-editing-engine-plan.md`.
This records evidence and remaining work; it does not mark the larger plan done.

| Requirement | Evidence | State |
| --- | --- | --- |
| Shared lifecycle and adapter terminal semantics | `core/jobs/runtime.py`, `ui/workers/job_adapter.py`, `tests/test_job_lifecycle.py`; desktop color uses the runtime through `ColorAnalysisWorker` | Implemented for the pilot; other workflows migrate in U7 |
| Partial output survives cancellation and reaches callers | Runtime retains returned output; Qt emits it before terminal status; MCP includes it in terminal error responses. Tests cover completed, cancelled and failed Qt delivery and failed/cancelled MCP output | Implemented |
| Project commit before checkpoint, without duplicate application | `core/jobs/commits.py`, `tests/test_job_recovery.py` inject failures before project save and after save/before checkpoint | Implemented for the saved color pilot |
| Recorded computation reused without repeating the call | Recovery tests count computation calls across retries and reject changed output, corrupt payloads and missing cache records | Shared mechanism verified; paid workflow adoption belongs to U7 |
| Unsaved work visibly session-only | In-memory store and runtime; desktop color status message; `tests/test_session_jobs.py` and `tests/test_color_worker_delivery.py` | Implemented |
| Old history/task IDs remain readable | Additive store migration; `test_legacy_migration_rows_and_import_aliases_survive`; MCP store tests | Implemented |
| Safe runtime restart ownership | `tests/test_job_ownership.py` covers live processes, killed owner with queued/running work, nonblocking shutdown and insertion/shutdown race | Implemented for owner-aware runtimes; legacy limitations documented |
| Operation identity and immutable normalized arguments | `OperationSpec` stores canonical detached JSON and a deterministic ID; runtime verifies kind/arguments against the spec. Both color adapters supply normalized parameters and input snapshots | Implemented for the pilot; legacy submissions remain during U7 migration |
| Session/input revisions and execution capabilities | Specs record session/input revision; runtime enforces persistence and cancellation capabilities. MCP color verifies queued project revision and target snapshot before extraction; desktop application retains its owner/session/target guards | Implemented for the pilot |
| Batched result commits and serialization cost | Saved color pilot records and saves one target at a time. It preserves commit order but does not batch project serialization | Not implemented |

Next, batch saved-result publication while retaining failure-injection coverage
at both commit boundaries. U6 remains open until this approach requirement is
implemented and checked. Adopting the same operation metadata for remaining
analysis families belongs to their U7 migrations.
