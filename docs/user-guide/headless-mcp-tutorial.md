# Tutorial — Driving Scene Ripper from Claude Code

This is a hands-on walkthrough for the headless MCP feature. By the end you will
have a multi-film analysis running unattended in Claude Code while you do
something else.

If you want the reference (every tool, every error code, the lifecycle
diagrams), read [`headless-mcp.md`](./headless-mcp.md) instead. This doc is
the "show me the happy path" version.

---

## What you'll build

A workflow where you tell Claude Code:

> "Detect scenes on every video in `~/Videos/Film/`, then run color and
> shot-type analysis on every clip. Don't block — let me know when it's done."

Claude Code will kick off the work, you can close the chat, and the work
keeps running in the background. When you come back, you ask "is it done?"
and Claude Code polls and shows you the result.

---

## Prerequisites

- Scene Ripper installed and working from source: `pip install -e ".[mcp]"`
- Claude Code 1.0+ (`claude --version`)
- A directory with one or more video files
- A scratch directory you can write to (e.g. `~/scene-ripper-projects/`)

> **GUI vs MCP.** The v1 design assumes the desktop GUI is **closed** when
> MCP is driving a project. The mtime guard catches most cross-process
> collisions but isn't a hard lock. If you want to use both at once,
> work in different projects.

---

## Step 1 — Verify the server runs

```bash
scene-ripper-mcp --transport stdio
```

The process should start, print initialization logs to stderr, and wait on
stdin. Press `Ctrl-C` to stop. If it errors, check that `pip install -e ".[mcp]"`
completed and `which scene-ripper-mcp` resolves to your venv.

---

## Step 2 — Wire it into Claude Code

```bash
claude mcp add scene-ripper -- /full/path/to/scene-ripper-mcp
```

(Use `which scene-ripper-mcp` to get the full path.)

Then in any Claude Code session:

```
/mcp
```

You should see `scene-ripper` listed. Expand it and you'll see the tool
catalog: `start_detect_scenes_bulk`, `start_analyze_colors`, `get_job_status`,
`list_jobs`, etc.

---

## Step 3 — Single-video walkthrough

A good first test: scan one video into a fresh project.

In Claude Code:

> Use scene-ripper-mcp. Detect scenes on `~/Videos/example.mp4` and write
> the project to `~/scene-ripper-projects/example.sceneripper`. Poll until
> done and show me the clip count.

What Claude Code does under the hood:

1. Calls `start_detect_scenes_new_project(video_path, output_project_path)`.
   Returns immediately with `{task_id, status: "queued", poll_interval: 5}`.
2. Calls `get_job_status(task_id)` every 5 seconds. You'll see status flip
   `queued → running` with a `progress` value climbing from 0 to 1.
3. When `status` is `completed`, calls `get_job_result(task_id)` to fetch
   the per-source breakdown.
4. Optionally calls `list_clips(project_path)` to summarise the clips.

Expected total time: 10-60 seconds for a 5-minute video on a modern Mac.

If something goes wrong:

- `source_file_missing` — check the video path; it must be absolute.
- `feature_unavailable` — the underlying analysis dependency isn't
  installed. MCP doesn't auto-install; do it from the GUI's Settings >
  Dependencies, or `pip install -r requirements-optional.txt`.

---

## Step 4 — Multi-film analysis (the real use case)

This is what the whole feature is for. Suppose you have a project with 12
sources and you want every clip analyzed.

In Claude Code:

> Open `~/scene-ripper-projects/big-project.sceneripper`. List the sources,
> then start a color-analysis job over all clips. Don't wait — return
> immediately and tell me the task_id.

Claude Code's flow:

```
list_sources(project_path)
→ 12 sources, 583 clips total

start_analyze_colors(
    project_path="...big-project.sceneripper",
    idempotency_key="initial-color-pass-2026-05",
)
→ {"task_id": "abc-123", "status": "queued", "poll_interval": 5}
```

Now close the chat. The job runs in the MCP server's background thread pool.

A while later, in a new session:

> Check on scene-ripper-mcp job `abc-123`.

```
get_job_status(task_id="abc-123")
→ {"status": "running", "progress": 0.42,
   "status_message": "Color analysis (245/583): clip-2a4f", ...}
```

Or, when finished:

```
get_job_status → {"status": "completed", "progress": 1.0, ...}
get_job_result  → {"result": {"succeeded": [...583 entries...], "failed": []}}
```

### Stacking ops

You can chain analyses naturally:

> When the color job is done, also run shot-type classification.

Claude Code will poll `get_job_status` until the colors job is `completed`,
then call `start_analyze_shots(...)`. If you submit the shots job before the
colors one finishes against the same project, the second job goes to
`queued` with `queue_position: 1` and `blocking_job_id` pointing at the
colors task — the per-project mutex serialises them.

---

## Step 5 — Cancellation

> Cancel job `abc-123`.

```
cancel_job(task_id="abc-123")
→ {"success": true, "ok": true}
```

The worker observes the cancel event between clips, finishes the in-flight
clip cleanly, saves what's been done so far, and the row goes
`running → cancelling → cancelled`. Per-clip results that landed before
cancel are preserved on disk in the project file — re-issue the same
`start_analyze_colors` call (with the same `idempotency_key`) to resume:
skip-existing semantics mean only the unprocessed clips are revisited.

---

## Step 6 — Crash recovery

This is the bit that makes hours-long jobs safe. If the MCP server dies
mid-job (machine sleeps, you reboot, OOM), the in-memory worker state is
gone but the job row in `<cache>/jobs.db` survives.

Next time you boot the server (which Claude Code does automatically when
you make a tool call), it runs an unconditional sweep:

```sql
UPDATE jobs SET status='crashed',
                error='server restarted while in flight',
                finished_at=NOW
WHERE status IN ('running', 'cancelling');
```

So in Claude Code:

> List any crashed jobs.

```
list_jobs(status_filter=["crashed"])
→ [{"task_id": "abc-123", "kind": "analyze_colors",
    "project_path": "...", "finished_at": ...}]
```

Then re-issue with the same idempotency key:

> Re-start the same color analysis with idempotency key
> `initial-color-pass-2026-05`.

Because the prior row is in a terminal-error state (`crashed`), the runtime
deletes it and spawns a fresh job. Skip-existing in the spine means the
clips that already had `dominant_colors` populated are not re-analyzed —
the run picks up where it crashed.

---

## Step 7 — Discovering in-flight work at session start

When you start a new Claude Code session, ask:

> What scene-ripper jobs are running?

```
list_jobs(status_filter=["queued", "running"])
```

This is the recommended first call in any session — background work from
prior sessions will be there waiting, and the LLM context doesn't carry
over.

---

## Step 8 — Cleaning up old job history

Job rows accumulate forever by default — there is **no automatic TTL**. To
prune:

> Purge job history older than 30 days.

```
purge_old_jobs(days=30)
→ {"deleted_count": 42}
```

Running and queued rows are never deleted, regardless of age.

---

## A complete real-world example

Here is a session transcript (lightly edited) showing the full F1 flow.

```
You: Use scene-ripper-mcp. Open ~/projects/film-school.sceneripper.
     Run color, shot-type, and transcription on every clip. Don't wait —
     start each as a separate job and tell me the three task IDs.

Claude: I'll start three jobs in parallel.
        - start_analyze_colors → task_id: a1b2-...
        - start_analyze_shots  → task_id: c3d4-...
        - start_transcribe     → task_id: e5f6-...

        These will serialise behind the per-project mutex (same project),
        but they'll run unattended. Check back later with `get_job_status`.

[an hour later]

You: How are the analyses doing?

Claude: list_jobs(project_filter="...film-school.sceneripper")

        a1b2-...: completed (583/583 succeeded)
        c3d4-...: running   (310/583, status_message "...clip-310")
        e5f6-...: queued    (queue_position: 1, blocking_job_id: c3d4-...)

You: Show me the colors result.

Claude: get_job_result(task_id="a1b2-...")

        583 clips analyzed, 0 failures.
        Most common dominant color (avg): #4a3a2c (warm earth-tone).
```

That's the whole thing. The point of the design is that the human is in
control of timing without ever blocking on the work.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `start_*` returns immediately, but `get_job_status` says `queued` and never advances | Per-project mutex held by another in-flight job | Check `blocking_job_id`; cancel the holder if stuck. |
| `get_job_result` returns `not_terminal` | Job is still running | Poll `get_job_status` first, only call `get_job_result` once status is `completed`/`failed`/`cancelled`/`crashed`. |
| `project_modified_externally` error | The `.sceneripper` file changed between submit and save (GUI was open) | Close the GUI; re-issue with same idempotency_key. |
| `feature_unavailable` for color/shots/transcribe | Optional ML deps missing | Install via GUI Settings > Dependencies, or `pip install -r requirements-optional.txt`. |
| Many `crashed` rows after a reboot | Boot sweep correctly flipped in-flight rows | Re-issue with the same idempotency_keys; skip-existing handles resume. |
| `list_jobs` returns 0 even though I started a job in another session | Default DB at `<settings.cache_dir>/jobs.db` is per-user; check `cache_dir` is the same in both sessions | `python -c "from core.settings import load_settings; print(load_settings().cache_dir)"` |

---

## Reference

For the complete tool surface, structured error codes, the lifecycle state
diagram, and idempotency semantics, see
[`headless-mcp.md`](./headless-mcp.md).
