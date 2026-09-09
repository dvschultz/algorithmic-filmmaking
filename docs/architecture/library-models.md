# Shared library item models (KTD13 / U15)

`ui/models/clip_model.py` and `ui/models/frame_model.py` hold the one
in-process copy of clip, source and frame data that the workspaces render.
`ProjectSignalAdapter` owns both models and updates them from project events
*before* the matching Qt signal fires, so every slot sees a model that already
reflects the change.

## Ownership

| Layer | Owns | Never owns |
|---|---|---|
| `Project` | truth: clips, sources, frames, session history | Qt state |
| `ClipLibraryModel` / `FrameLibraryModel` | ordered rows, `Clip`/`Source`/`Frame` objects, thumbnail landing | membership, selection, filters |
| `ClipBrowser` (Cut, Analyze) | membership (`_virtual_ids`), `selected_clips`, `FilterState`, realized cards | copies of clips or sources |
| `FrameBrowser` | `QListView` selection and zoom | frame data |

`ClipBrowser.attach_model(model)` switches a browser from its private model
(standalone use, tests, dialogs) to the adapter's shared one. A browser with a
shared model never removes rows from it: `clear()` / `remove_clips_by_ids()`
only touch membership. `get_source_for_clip()` is a library lookup, so it
answers for any clip in the project, not only clips the workspace shows.

`FrameBrowser.set_model(model)` renders the view straight from the shared
model; `set_frames` becomes a no-op and `clear` only drops the selection, so
`FramesTab.update_frame_browser()` effectively just toggles the empty state (a
model reset would drop the view's selection).

Undo/redo of source removal arrives as one `sources_changed` event; both
models reconcile incrementally (`sync_project`) so surviving rows keep their
identity and selection. Restored rows are appended, so model row order can
differ from project order after an undo; browsers order their own membership.
Re-detection (`Project.replace_source_clips`) emits `clips_removed` for the old
clips before `clips_added`, so stale rows never linger in the model.

If a model mutation fails inside the adapter (for example an off-thread
project mutation hitting the owner-thread assertion), the adapter logs the
traceback and still emits its Qt signal so the views are not silently starved.

## Threading

All model mutators assert they run on the model's owning thread and raise
`RuntimeError` otherwise. Workers deliver results through Qt signals to the
GUI thread as before; `MainWindow._on_thumbnail_ready` calls
`clip_model.thumbnail_ready(clip_id, path)`, which returns `False` and drops
the result when the clip has already left the project.

## Recorded baseline and budgets

Measured with `QT_QPA_PLATFORM=offscreen python scripts/library_model_benchmark.py`
on 2026-09-09 (macOS 25.5, Apple Silicon, Python 3.13, PySide6 6.x, 20 sources,
two attached workspaces, 1400x900 viewport). Pre-U15 numbers came from the same
operations on the previous browser-owned bookkeeping (single workspace).

| metric | 1,000 clips | 10,000 clips | pre-U15 (10k, one workspace) | budget (10k) |
|---|---|---|---|---|
| model populate (upsert all) | 0.6 ms | 6.8 ms | n/a | 25 ms |
| browser populate, virtual (first workspace) | 47 ms | 94 ms | 65 ms | 250 ms |
| second workspace populate | 42 ms | 70 ms | n/a | 250 ms |
| update 50 clips, both workspaces, layout preserved | 7 ms | 30 ms | 37 ms | 100 ms |
| scroll to five positions (virtual re-realization) | 262 ms | 276 ms | 259 ms | 600 ms |
| filter toggle on/off | 24 ms | 54 ms | 47 ms | 150 ms |
| remove 100 clips (project + model + both workspaces) | 36 ms | 101 ms | 38 ms (browser only) | 250 ms |
| 1,000 stale thumbnail results ignored | 0.3 ms | 0.4 ms | n/a | 5 ms |
| realized cards | 55 | 60 | 60 | <= 2 x viewport rows |
| peak RSS | 236 MB | 288 MB | n/a | 400 MB |

Model-layer cost at 10k is under 5 ms per operation (`entries()` 3 ms,
scattered removal of 99 rows 0.5 ms); the remaining time is the existing grid
rebuild (~38 ms per workspace per `processEvents`). Budgets are roughly 2x the
measured values and gate the renderer cutover: replacing card widgets with a
model-backed view must not exceed them. `tests/test_library_models.py`
bounds the model layer only (10k upsert/refresh/entries/remove under 2 s) so
CI stays free of widget timing noise.

## Not done here

- Card rendering still uses `ClipThumbnail` widgets realized from membership;
  a `QListView`-style renderer for clips is the follow-up gated on the budgets
  above.
- Agent context (`core/gui_state.py`) still reads selection through the tabs;
  it does not keep a separate project copy, so no change was needed.
