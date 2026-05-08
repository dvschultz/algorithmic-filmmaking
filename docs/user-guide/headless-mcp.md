# Headless MCP — driving Scene Ripper from Claude Code

Scene Ripper exposes a Model Context Protocol (MCP) server at
`scene-ripper-mcp` that an external agent (Claude Code, Hermes, etc.) can
drive without the GUI running. Long-running operations are split into a
**polling-jobs** pattern so a calling agent never blocks on hours-long
multi-film analysis.

This guide is the caller-facing reference: the lifecycle, the structured
errors, idempotency, and the recommended polling cadence. For a
walkthrough with a concrete example, see
[`headless-mcp-tutorial.md`](./headless-mcp-tutorial.md).

## Quick start

1. Install the MCP server: `pip install -e .[mcp]`
2. Configure your agent to launch `scene-ripper-mcp` over stdio (or
   `scene-ripper-mcp --transport http --port 8765` for HTTP).
3. Start a job — for example, scene detection over an existing project:

   ```
   start_detect_scenes_bulk(
       project_path="/Users/me/films/myfilm.sceneripper",
       source_ids=["src-1", "src-2"],
       sensitivity=3.0,
   )
   → {"task_id": "abc-123", "status": "queued", "poll_interval": 5}
   ```

4. Poll status:

   ```
   get_job_status(task_id="abc-123")
   → {"status": "running", "progress": 0.42, "status_message": "...",
      "queue_position": null, "blocking_job_id": null, ...}
   ```

5. Once `status` reaches a terminal value (`completed` / `failed` /
   `cancelled` / `crashed`), fetch the result:

   ```
   get_job_result(task_id="abc-123")
   → {"success": true, "status": "completed",
      "result": {"succeeded": [...], "failed": [...]}}
   ```

## Tool surface

### Long-running ops (start / poll / fetch / cancel)

| `start_*` tool | What it does |
|---|---|
| `start_detect_scenes_bulk` | Scene-detect every source in a project |
| `start_detect_scenes_new_project` | Create a fresh project from a video file and detect scenes |
| `start_generate_thumbnails` | Generate/backfill clip thumbnails for downstream visual analysis |
| `start_analyze_colors` | Extract dominant colors per clip |
| `start_analyze_shots` | Classify shot type (wide / medium / close-up / xclose) |
| `start_transcribe` | Whisper transcription per clip |
| `start_download_videos` | Bulk video downloads (YouTube / Vimeo / Internet Archive) |

Every `start_*` returns immediately with `{task_id, status, poll_interval}`.
The job runs in a background thread; the response payload is **not**
included — you must poll `get_job_status` until terminal, then call
`get_job_result`.

### Generic job management

| Tool | What it does |
|---|---|
| `get_job_status(task_id)` | Status, progress, queue position. **No payload** (R28). |
| `get_job_result(task_id)` | Final result on completed; sanitized error on failed/cancelled/crashed. Errors `not_terminal` while still running. |
| `cancel_job(task_id)` | Signals cancellation. Job transitions running → cancelling → cancelled. |
| `list_jobs(status_filter, kind_filter, project_filter)` | Safe-projection list. Use to discover in-flight work at session start. |
| `purge_old_jobs(days=30)` | Delete terminal-status rows older than `days`. Running and queued rows are never purged. |

### Synchronous tools (unchanged from v0)

The original synchronous catalog (project / clip / sequence / export
queries and mutations, `download_video`, `download_videos`, `analyze_*`,
`detect_scenes`) is still exposed for backward compatibility (R5). For
multi-film batches use the `start_*` job variants instead.

## Job lifecycle

```
                             ┌──────────────┐
            start_*  ───────▶│   queued     │
                             └───────┬──────┘
                                     │ worker pulls + acquires per-project lock
                                     ▼
                             ┌──────────────┐
                             │   running    │◀──── progress updates
                             └───┬─────┬────┘
                                 │     │
              cancel_job ────────┘     │
                                 ▼     │
                         ┌───────────┐ │
                         │cancelling │ │
                         └─────┬─────┘ │
                               │       │
            spine fn returns   │       │
                  ▼            ▼       ▼
           ┌──────────┐  ┌────────────┐
           │completed │  │ cancelled  │
           └──────────┘  └────────────┘
                                       │
                                  spine raised
                                       ▼
                               ┌──────────────┐
                               │   failed     │
                               └──────────────┘

server boot finds rows still in running/cancelling? → marked crashed.
```

## Calling-agent UX contract

- **Discover in-flight work at session start.** When the agent begins a
  session, call `list_jobs(status_filter=["queued", "running"])` to see
  what's already running. Background work survives MCP server restarts
  (rows in `running` / `cancelling` are flipped to `crashed` on the next
  boot, so the caller can decide whether to retry).
- **Use the `poll_interval` returned by `start_*` and `get_job_status`.**
  It's the recommended cadence; ignoring it just costs more SQLite reads.
  Default is 5 seconds.
- **`get_job_result` is a separate call.** Status reports do not include
  the result payload — fetch it explicitly once the job is terminal.
- **Concurrent same-project jobs serialise.** Two `start_*` calls against
  the same project queue serially through a per-project mutex; the second
  job's `get_job_status` will report `queue_position` and
  `blocking_job_id` so you can reason about the wait.

## Idempotency

`start_*` tools accept an optional `idempotency_key` (max 255 chars). The
composite scope is `(kind, project_path, idempotency_key)`:

- A key matching an already-`completed` row returns the **same `task_id`**
  without spawning a new worker. Safe to retry the same call after a flaky
  network or restart.
- A key matching a `failed` / `cancelled` / `crashed` row deletes the
  prior row and **spawns a fresh job** — terminal-error states do not
  poison the key.
- The same key against a different `project_path` is a different scope —
  it spawns a new job.

There is **no automatic TTL** on terminal rows. Call `purge_old_jobs()`
explicitly when you want to prune history.

## Project-modification guard

Every `start_*` that touches a project file captures the file's mtime at
submit time. If the file mtime drifts between submit and save (someone
else opened and saved it in the GUI, for example), the worker aborts
the save with a structured error:

```json
{
  "success": false,
  "error": {
    "code": "project_modified_externally",
    "path": "/path/to/project.sceneripper",
    "expected_mtime": 1234567890.0,
    "current_mtime": 1234567899.0
  },
  "result": { /* per-item progress preserved up to abort */ }
}
```

This is **best-effort, not a guarantee** — last-writer-wins is still
possible under undetected races. The v1 scope assumes the GUI is closed
when MCP is driving the project. Per-item results that landed before the
abort (e.g. clips with `dominant_colors` set in memory) are returned in
`result` so the caller can decide how to recover.

## Structured errors

| `error.code` | Meaning |
|---|---|
| `job_not_found` | The supplied `task_id` does not exist. |
| `not_terminal` | `get_job_result` called while the job was still running or queued. |
| `already_terminal` | `cancel_job` called on a row already in a terminal state. |
| `invalid_idempotency_key` | Key exceeds 255 chars. |
| `source_files_missing` | `Project.load()` could not resolve every source file. |
| `project_modified_externally` | mtime guard tripped — the project file changed between submit and save. |
| `feature_unavailable` | An optional ML dependency is not installed and the op cannot run. Install via the GUI / CLI; MCP does not auto-install. |
| `cancelled` | `asyncio.CancelledError` was caught and translated (FastMCP SDK defence). |
| `invalid_url` | URL failed scheme/host validation before reaching yt-dlp. |
| `source_file_missing` | A source's video file is no longer on disk. |

`get_job_status` and `list_jobs` **do not** include `args_json`,
`result_json`, or `error` — that payload is gated behind `get_job_result`
to avoid leaking sensitive arguments or absolute paths into the agent's
context (R28).

Tracebacks stored in failed-job rows are sanitised: type + message + the
last 10 frames, capped at 4 KB, with absolute source paths stripped.

## Crash recovery

If the MCP server process dies mid-job (SIGKILL, OOM, machine reboot),
the in-memory worker state is gone but the job row in `<cache>/jobs.db`
survives. On the next boot, every row found in `running` or `cancelling`
is unconditionally flipped to `crashed`:

```
status = "crashed"
error  = "server restarted while in flight"
```

Per-item results that the spine fn already wrote to the project file
**before the crash** are preserved (most analysis ops commit per-clip).
The caller is expected to re-issue the same `start_*` with the same
`idempotency_key` — terminal-error rows don't block resubmission, and
skip-existing semantics on each spine fn make the retry resume from
where it left off.

## Local-only

The MCP server is local-only — no authentication, rate limiting, or
remote network exposure. The job database (`<cache>/jobs.db`) is created
with mode `0o600` so other users on the machine cannot read job history
(R29). Run it under your user account, behind whatever transport your
agent uses (stdio or local HTTP).
