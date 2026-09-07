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

## CLI Setup And Examples

The MCP server and CLI are installed from the same source checkout:

```bash
python -m pip install -e .[mcp]
scene_ripper --help
scene-ripper-mcp --transport stdio
```

Use the CLI when you want a direct shell workflow and MCP when an agent
needs to keep jobs running, poll status, and fetch results later.

Common CLI commands:

```bash
# Create a project by detecting scenes in a video
scene_ripper detect /path/to/video.mp4

# Transcribe clips in an existing project
scene_ripper transcribe /path/to/video.sceneripper

# Run visual analysis
scene_ripper analyze describe /path/to/video.sceneripper

# Export a sequence
scene_ripper export sequence /path/to/video.sceneripper -o ./out.mp4

# Test YouTube API credentials
scene_ripper test-youtube-key
```

Common MCP recipes:

```
# Create a new project from a source video and detect scenes
start_detect_scenes_new_project(
    video_path="/path/to/video.mp4",
    project_path="/path/to/video.sceneripper",
    sensitivity=3.0,
    idempotency_key="video-detect-v1",
)

# Transcribe existing project clips
start_transcribe(
    project_path="/path/to/video.sceneripper",
    model_name="base",
    language="en",
    idempotency_key="video-transcribe-v1",
)

# Download several videos for a batch project
start_download_videos(
    urls=[
        "https://www.youtube.com/watch?v=...",
        "https://vimeo.com/...",
    ],
    download_dir="/path/to/downloads",
    idempotency_key="batch-download-v1",
)
```

Headless configuration is environment-first. Set cloud provider keys in
the shell that launches the CLI or MCP server:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export YOUTUBE_API_KEY=...
```

For local LLM workflows, run Ollama before launching the agent and select
an Ollama model in the calling tool or project settings:

```bash
ollama serve
ollama pull llama3.1
```

Ollama does not need an API key. Cloud keys are still required for cloud
VLM/LLM operations such as Gemini descriptions or OpenAI-backed text
workflows.

For local HTTP MCP testing:

```bash
scene-ripper-mcp --transport http --port 8765
```

## Tool surface

### Long-running ops (start / poll / fetch / cancel)

| `start_*` tool | What it does |
|---|---|
| `start_detect_scenes_bulk` | Scene-detect every source in a project |
| `start_detect_scenes_new_project` | Create a fresh project from a video file and detect scenes |
| `start_generate_thumbnails` | Generate/backfill clip thumbnails for downstream visual analysis |
| `start_analyze_clips` | Canonical multi-operation analysis job using UI operation keys |
| `start_analyze_colors` | Extract dominant colors per clip |
| `start_analyze_shots` | Classify shot type (wide / medium / close-up / xclose) |
| `start_analyze_classify` | Classify thumbnail content with ImageNet labels |
| `start_detect_objects` | Detect objects and person counts |
| `start_extract_text` | Extract visible text with resumable OCR/VLM results; optional `force` refresh |
| `start_transcribe` | Whisper transcription per clip |
| `start_transcribe_audio` | Standalone imported audio transcription by exact audio-source ID; preserves existing results unless `force=true` |
| `start_import_audio` | Import an audio file and save; repeated canonical paths return the existing audio ID |
| `start_import_images` | Import a batch of still images and save; copy by default or reference originals with `copy_files=false` |
| `start_extract_frames` | Extract a video source into saved frame items; supports `interval`, `all`, and `smart` modes and an optional `clip_id` |
| `start_align_words` | Add word timestamps to existing transcripts; requires installed alignment runtime |
| `start_describe` | Generate VLM descriptions |
| `start_analyze_cinematography` | Generate rich film-language analysis |
| `start_detect_faces` | Extract face embeddings |
| `start_analyze_gaze` | Estimate gaze direction |
| `start_generate_embeddings` | Extract DINOv2 visual embeddings |
| `start_generate_boundary_embeddings` | Extract resumable first/last-frame DINOv2 pairs for Match Cut; optional `force` refresh |
| `start_custom_query` | Run a yes/no visual query against clips |
| `start_download_videos` | Bulk video downloads (YouTube / Vimeo / Internet Archive) |

Import audio with `scene_ripper import-audio project.sceneripper voice.wav`, or
submit `start_import_audio` and poll `get_job_status` / `get_job_result`.
Relative media paths use the project directory. MCP paths must satisfy the shared
safe-root and traversal checks. Results include `audio_source_id`, `filename`,
`duration`, and `status`. Repeated imports preserve the existing ID without probing
again; failed saves reuse recorded metadata and IDs, and interrupted checkpoints
acknowledge the saved import on retry. An MCP `idempotency_key` also deduplicates
job submission. GUI and headless result journals remain separate.

Import still images with
`scene_ripper import-images project.sceneripper first.png second.jpg`.
Use `--reference` to retain references to originals instead of copying them into
the project. MCP offers the same policy through `start_import_images` and
`copy_files`. Relative input paths use the project directory; MCP applies the
shared safe-root and traversal checks to every item. A batch saves valid images
and reports rejected items in `errors`; no valid images means the job fails.
Results include `frame_ids`, `imported_count`, `errors`, and `status`.

Interrupted saves reuse the original frame IDs, copied images and thumbnails.
An interrupted checkpoint acknowledges the saved batch on retry. A fresh
invocation intentionally appends new frames; use an MCP `idempotency_key` when
retrying the same submission. Changed inputs, order, copy policy, runtime or
generated artifacts prevent stale recovery. GUI and headless journals are separate.

Frame extraction is also available as
`scene_ripper extract-frames project.sceneripper SOURCE_ID --interval 10`.
Add `--clip-id CLIP_ID` to limit the range, or `--mode all` / `--mode smart`.
Each new invocation appends a new batch. MCP callers can retry the same
submission with an `idempotency_key` to avoid creating another job. Poll
`get_job_status`, then `get_job_result` for `frame_ids` and `frame_count`.
Interrupted saves reuse the original generated files; an interrupted checkpoint
acknowledges the saved batch before a later invocation starts another batch.
Changed source media, options, runtime, or generated artifacts prevent stale
recovery. GUI and headless journals are separate; desktop saves remain explicit.

OCR jobs retain successful inference for retries after a failed project save.
This applies to `start_extract_text` and the `extract_text` operation in
`start_analyze_clips`. An empty text list means analysis completed without finding
text and is preserved across save/load. Existing results are skipped; pass
`force=true` to `start_extract_text` to replace them. Failed inference is not
stored as an empty observation. The submitted VLM model is captured when queued.

The CLI exposes the same recovery through
`scene_ripper analyze extract-text project.json --method hybrid`. Use repeated
`--clip-id` options for exact clip IDs, `--model` to choose a VLM, and `--force`
to replace existing observations. Recovery requires the same configured cache
directory and unchanged media, extraction options, and runtime identity.

Download entry points save a verified file receipt after each successful item.
Retries reuse unchanged local files and download missing files again. Modified
files appear in `failed` with code `download_output_changed`; use another output
directory or intentionally remove the modified file before requesting a fresh
download. Earlier receipts survive a later failure or cancellation. This checks
local file integrity, not whether the remote video has changed. Desktop, CLI,
and headless downloads share receipts when their output directory and download
policies match and they use the same configured cache directory.

`start_analyze_clips` accepts the same operation keys used by the GUI
Analyze tab: `colors`, `shots`, `classify`, `detect_objects`,
`extract_text`, `transcribe`, `describe`, `cinematography`,
`face_embeddings`, `gaze`, `embeddings`, `boundary_embeddings`, and `custom_query`. When using
`custom_query`, pass the query text in the `query` argument.

Custom-query jobs save computed results before publishing them to the project.
Retrying after a failed project save reuses matching results; retrying a failed
cache checkpoint does not append them twice. A new request after successful
completion still appends a fresh query result. Query text, provider model, and
media identity are captured when the job is submitted. This also applies to
custom-query steps in `start_analyze_clips`.

`start_describe` and `describe` steps in `start_analyze_clips` retain computed
results before saving them to the project. Retrying after a failed save reuses
matching results rather than calling the provider again. Model, prompt, input mode,
image/source media, and local backend changes prevent reuse. Existing descriptions
are preserved by default; pass `force=true` to `start_describe` to refresh them.
CLI `analyze describe` shares these receipts when options, media, and the configured
cache directory match. Missing committed cache payloads produce an error rather
than an automatic paid recomputation.

`start_analyze_classify` and `classify` steps in `start_analyze_clips` retain
recorded classification results across failed saves and checkpoint failures.
Retries with matching media, model runtime, and options reuse computation and
preserve existing or manually edited labels. CLI `analyze classify` uses the same
recovery path; `--force` requests a fresh generation after a completed run.
Desktop classification also retains clip and frame computation for saved
projects. Results remain unsaved until the project is saved; unsaved projects
keep only session history.

`start_analyze_cinematography` and cinematography steps in `start_analyze_clips`
also retain completed inference before saving the project. Retry after a failed
save or cache checkpoint reuses matching results. Submission captures the cloud
and local models, input mode, media identity, and local backend availability.
Existing cinematography and manually edited shot types are preserved on retry.
Missing or corrupt committed receipts produce an error rather than an automatic
provider call. Desktop cinematography also retains computation for saved projects;
its receipts are separate and are acknowledged only when you save the project.

Every `start_*` returns immediately with `{task_id, status, poll_interval}`.
The job runs in a background thread; the response payload is **not**
included — you must poll `get_job_status` until terminal, then call
`get_job_result`.

### Generic job management

| Tool | What it does |
|---|---|
| `get_job_status(task_id)` | Status, progress, queue position. **No payload** (R28). |
| `get_job_result(task_id)` | Final result on completed; sanitized error plus any available `result` on failed/cancelled/crashed. Errors `not_terminal` while still running. |
| `cancel_job(task_id)` | Signals cancellation. Job transitions running → cancelling → cancelled. |
| `list_jobs(status_filter, kind_filter, project_filter)` | Safe-projection list. Use to discover in-flight work at session start. |
| `purge_old_jobs(days=30)` | Delete terminal-status rows older than `days`. Running and queued rows are never purged. |

A terminal `success: false` describes the overall job. If the response also
contains `result`, inspect its per-item outcomes: some items may have succeeded
before cancellation or failure. An absent `result` means no output was recorded.
Status and list endpoints continue to omit payloads, and nonterminal jobs do not
expose their unfinished output through `get_job_result`.

Migrated color and bulk-detection jobs expose an `operation` summary in status/list responses:
its ID, kind, version, `cancellable`, and persistence mode. Input details remain
private. These queued jobs fail if their project revision or media snapshot
changes before execution. Submit it again against the current inputs to retry.
An explicit idempotency key still refers to the original job; use a new key when
requesting different work.

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

server boot finds jobs whose runtime owner has exited? -> marked crashed.
```

## Calling-agent UX contract

- **Discover in-flight work at session start.** When the agent begins a
  session, call `list_jobs(status_filter=["queued", "running"])` to see
  what's already running. Job history survives MCP server restarts.
  Abandoned queued/running/cancelling jobs become `crashed` on the next
  boot, so the caller can decide whether to retry. Jobs owned by another
  live runtime keep running.
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
survives. Each runtime holds an OS-backed lease, and its ID is recorded with
its jobs. On the next boot, queued/running/cancelling rows whose owner lease
is no longer held become `crashed`. Another live server's jobs are preserved:

```
status = "crashed"
error  = "job runtime exited"
```

Older job rows without owner IDs retain their prior running/cancelling recovery
policy and error message; legacy queued rows remain untouched. Do not share the
database with older binaries that still run an unconditional boot sweep.

Results written to the project file **before the crash** are preserved. Re-issue
the same `start_*` with the same `idempotency_key`: terminal-error rows do not
block resubmission. Color analysis uses recorded results and project receipts to
reconcile retries; other operations retain their existing save and skip-existing
behavior. Work that was never saved or recorded may need to run again.

## Local-only

The MCP server is local-only — no authentication, rate limiting, or
remote network exposure. The job database (`<cache>/jobs.db`) is created
with mode `0o600` so other users on the machine cannot read job history
(R29). Run it under your user account, behind whatever transport your
agent uses (stdio or local HTTP).

## Retained editing sessions

Use `open_project_session(project_path)` to keep sequence edits and undo history
across MCP calls. It returns a `session_id`. Reopening the same file reuses its
session; `get_project_session(session_id)` reports sequences and history state.

The initial retained editing tools are:

- `create_session_sequence(session_id, name, fps=30.0)`
- `rename_session_sequence(session_id, sequence_id, name)`
- `delete_session_sequence(session_id, sequence_id)`
- `undo_project_session(session_id)` and `redo_project_session(session_id)`
- `close_project_session(session_id)`

Each successful edit saves the project before returning. Undo and redo also
save, and never rerun analysis, downloads, or generation. A failed save discards
the unpublished in-memory edit; the next call reloads the on-disk project.
Closing the session or stopping the server discards history, not saved edits.

Unmigrated path-based tools remain available. They do not add to retained undo
history. If they or another process change the saved file, the retained session
reloads and reports `history_reset: true`. An undo request after such a change
finds no old history to replay. The next edit uses the reloaded project.

The server serializes retained session calls on an owner thread and acquires
writer ownership during each operation. A desktop-owned file returns
`project_busy`. Ownership is released between calls. Newer-schema projects can
be inspected but cannot be edited. Cancelling a request that is already running
does not roll back its save; inspect the session before retrying that edit.

If an open project path is retargeted through a symlink, close and reopen the
session to accept the new location. Existing session handles reject that change.

### Retained timeline edits

`get_session_timeline(session_id, sequence_id)` returns tracks and stable timeline
clip IDs. The following tools target that sequence even when it is inactive:

- `insert_session_clips` inserts enabled library clip IDs into an existing track.
  Omit `start_frame` to append after the track's latest end. Explicit placement
  does not ripple existing clips. Unknown or disabled IDs reject the whole batch.
- `remove_session_clips` removes timeline IDs; `ripple` defaults to false.
- `reorder_session_clips` packs the selected track in the requested timeline-ID
  order, followed by omitted clips in their existing order.
- `edit_session_timeline_clip` accepts a `changes` object for timing, track, or
  transforms. In/out points are absolute source frames, with an exclusive out.
- `clear_session_timeline` clears all tracks while preserving their structure.

Each call is one saved undoable edit. The existing path-based `reorder_sequence`,
`remove_from_sequence`, `clear_sequence`, and `shuffle_sequence` tools now join
the same retained session when called through the running server. Shuffle redo
uses its saved order, without running randomness or analysis again. `add_to_sequence`, tag/note edits, and source removal also join retained history.
Remaining unmigrated operations, such as media analysis and imports, still save
independently and invalidate previous retained history.

### Library edits and legacy insertion

`add_to_sequence` now joins retained history and uses absolute source-frame
ranges. Missing tracks are created in the same undo action as insertion; undo
removes those new tracks. Track creation accepts indices 0–255. Unknown or
disabled library clips are skipped, and a batch with no insertable clips does
not create tracks.

`add_clip_tags`, `remove_clip_tags`, `add_clip_note`, and `remove_source` preserve
their existing response fields and now share the retained session. Undoing source
removal restores its model references, including clips used in sequences.

`set_session_clips_disabled(session_id, clip_ids, disabled)` edits enabled state.
`update_session_clip(session_id, clip_id, fields)` edits metadata. Transcript
values use a list of objects with `start_time`, `end_time`, and `text`; optional
word objects use `start`, `end`, and `text`. Times must be finite, nonnegative,
and ordered. Use an empty list to clear a transcript. JSON conversion happens
before the edit, so malformed transcript input cannot partially change a clip.


## Shared job lifecycle

The MCP job API and existing task IDs are unchanged. Runtime, store, and project
mutex implementations now live under `core/jobs`; old MCP imports remain
compatible. Terminal job decisions are immutable: late progress or cancellation
cannot reopen a completed job. Queued cancellation skips the runner, and a
returned `success: false` is recorded as a failed job with its result preserved.

The desktop has a Qt adapter over this runtime. Existing desktop workflows are
still being migrated.

Color-analysis jobs record each computed palette immediately, then save outputs
and receipts to the project in groups of up to 16. Cancellation saves the final
partial group. Start another color job after a failure to reuse its
recorded results: a failed project save retries application, while a failed job
checkpoint verifies the saved palette without applying it twice. Edited palettes
are preserved and reported as a conflict.

`start_transcribe` also records each successful transcript, including silent
clips, and saves project receipts in groups of up to 16. Resubmitting after a
failed save or checkpoint reuses verified computed results. Edited managed
transcripts are reported as conflicts. Pending jobs reject changed project or
media metadata; execution fingerprints source contents and revalidates them
before publication. Cancellation preserves completed targets for retry.
The CLI transcription command and synchronous MCP `transcribe` tool also save
these receipts in the shared job cache. CLI `--force` and synchronous MCP retain
their explicit refresh behavior: a failed save can reuse its pending computation,
while a successfully saved refresh allows the next request to compute again,
including for silent clips. CLI runs without `--force` still skip existing
transcripts. `start_analyze_clips` also uses durable transcription when its
operation list includes `transcribe`, preserving its skip-existing behavior even
for transcripts created with another model. It saves between operations and reloads
the project for following steps, so a later failure does not lose a completed
transcript or its receipts. Other operations in that list retain their existing
computation behavior; the full list is not an atomic transaction. GUI
transcription now caches computations for projects that already have a save
location and includes receipts in normal project saves. GUI and headless
transcription currently use separate computation identities.

`start_align_words` and CLI `analyze align` also cache computed word timestamps
before saving project receipts. Matching retries recover failed saves and
reconcile failed checkpoints, including repeated forced runs with identical
output. Transcript text, timing, language, and media fingerprints guard reuse.
Completed word data is preserved by default; `force=True` requests a refresh.
A failed refresh save can reuse its pending computation, while a saved refresh
allows the next forced request to compute again. Cached recovery does not require
loading the alignment runtime. Saved GUI projects separately cache word outcomes
before delivery and include their receipts in normal project saves; those entries
are not reused by headless alignment.

Ordinary project saves also acknowledge matching GUI transcript and alignment
receipts after writing the file. A failed acknowledgement leaves the saved file
intact and can be retried by saving again. Headless result receipts retain their
existing job-specific checkpoint paths.

Saved GUI transcription/alignment jobs also appear in job history as
`gui_transcribe` and `gui_align_words`, associated with their project path.
Their `completed` status means computation finished. The result marker
`publication: explicit_project_save` describes their save policy; it does not
claim the current editor state was saved. Existing job status/result tools can
inspect these records. Abandoned GUI jobs become crashed when a runtime performs
recovery; inference is never automatically restarted.

`start_detect_scenes_bulk` also records computed scenes, saving one source and
its receipt at a time. Retrying identical inputs reuses the recorded clip IDs;
a failed save retries publication, and a failed checkpoint reconciles the saved
result. Edited sources or clips are reported as conflicts instead of overwritten.
Queued jobs reject changed project or media inputs. Successful sources remain
saved if another source fails or the job is cancelled. This recovery path applies
to bulk detection on existing projects, not new-project detection.

Projects save in schema 1.5 to retain these receipts; schema-aware 1.4 clients can
inspect them read-only. Job-history cleanup retains the computed-result cache.
Keep that cache when moving projects between installations if you need to retry
managed color analysis, transcription, alignment, or bulk detection; missing cached results are reported instead of silently
recomputing them.
