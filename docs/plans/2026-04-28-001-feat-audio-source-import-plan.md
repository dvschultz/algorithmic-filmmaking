---
title: 'feat: Audio source import (Staccato + transcription)'
type: feat
status: completed
date: 2026-04-28
---

# feat: Audio source import (Staccato + transcription)

## Summary

Add a new `AudioSource` type that lets users import audio files (MP3/WAV/M4A/FLAC/OGG) into a project alongside videos. Audio sources are not cut into clips and never appear in the sequencer output — they exist to feed audio-consuming tools. v1 wires them into the Collect tab (import + library), the Staccato dialog (replacing today's per-session `QFileDialog`), and transcription (so podcasts/interviews can be transcribed without a video carrier). Stem separation, sequence music selection, and YouTube audio-only download are explicitly deferred.

---

## Problem Frame

Today, audio in Scene Ripper is ephemeral. The Staccato sequencer's music track is picked via raw `QFileDialog.getOpenFileName` (`ui/dialogs/staccato_dialog.py:610`) and forgotten when the dialog closes — every session re-picks the same file. Transcription accepts audio paths in `core/transcription.py:659`, but no UI surfaces let a user run it on an imported audio file. Sequences already mux audio onto export via `Sequence.music_path` (`core/sequence_export.py:38`), but that path is also raw, not project-managed.

The user-facing pain is "I have to find this music file again every time I open the dialog." The deeper architectural pain is that audio-only assets have no first-class home in the project model, blocking any future tool that would want to operate on them.

---

## Requirements

- R1. Users can import local audio files (MP3, WAV, M4A, FLAC, OGG) into a project.
- R2. Imported audio appears in the Collect tab in a section visually distinct from videos.
- R3. Audio sources persist in saved projects (round-trip through save/load).
- R4. The Staccato dialog selects its music track from project audio sources via a picker, with an "Import new…" escape hatch that auto-selects the new import.
- R5. Existing Staccato behavior with a freshly-imported file is unchanged (same waveform analysis, same beat detection, same stem-separation flow).
- R6. Users can transcribe an audio source from the Analyze tab without a video carrier.
- R7. The chat agent and MCP server can list, identify, and import audio sources.
- R8. Audio sources never appear in the sequencer output, scene detection, or video clip pipelines.

---

## Scope Boundaries

- Audio sources are **not** cut into clips, scene-detected, or treated as visual media.
- No audio editing (trimming, fading, volume normalization) — sources are read-only references.
- No waveform thumbnails in v1 — the audio library uses a generic icon. Waveform rendering exists per-file inside the Staccato dialog already; replicating it as a small library thumbnail is deferred.

### Deferred to Follow-Up Work

- **Stem separation as a standalone tool**: Demucs is currently invoked inside Staccato. A standalone "extract stems" entry point on an `AudioSource` would be useful but is outside v1 scope.
- **`Sequence.music_path` rewiring**: Sequences still use raw paths for export muxing. Migrating to `audio_source_id` is a follow-up that requires a project-format migration.
- **YouTube audio-only download**: yt-dlp supports `bestaudio`, but the UX (audio-only search filter, format picker) is its own surface area.
- **Audio waveform thumbnails in the library grid**: skipped for v1 to keep dependencies minimal.

---

## Context & Research

### Relevant Code and Patterns

- `models/clip.py:63-178` — `Source` dataclass with `to_dict(base_path)` / `from_dict(data, base_path)` serialization, including path-traversal protection and `_absolute_path` fallback. `AudioSource` follows the same serialization shape.
- `core/project.py:611-820` — `Project` manages `_sources`, `_clips`, `_sequences` with cached lookup dicts, observer pattern (`add_observer`, `_notify_observers("sources_changed", …)`), and explicit `add_source` / `remove_source` mutators. Audio sources mirror this exactly.
- `core/project.py:169-216, 438-587` — Project save/load functions iterate `data.get("sources", [])`. Audio sources are added as a parallel `data.get("audio_sources", [])` list, defaulting to empty for backward compatibility.
- `ui/tabs/collect_tab.py:22-191` — `CollectTab` exposes `add_source`, `remove_source`, `update_source_*`, `_on_files_dropped`. Audio analogues mirror these methods.
- `ui/dialogs/staccato_dialog.py:608-623` — `_on_select_file` is the single entry point for music-file selection. Replacing this with a project-audio-source picker is the only user-visible change in Staccato.
- `core/chat_tools.py:464` — `list_sources(project)` is the pattern; `list_audio_sources` mirrors it.
- `core/transcription.py:659-700` — `transcribe_audio` already accepts an arbitrary `audio_path`. No core changes; only a new UI/agent surface to invoke it on an `AudioSource`.

### Institutional Learnings

- `pyside6-project-state-mutation` skill — when adding new project-state collections, mutator methods must invalidate cached lookup dicts and call `_notify_observers`, otherwise GUI doesn't refresh.
- `feature-registry-reload-filter-scope` — `librosa` is in the `audio_analysis` feature registry; do not depend on it eagerly during import (Staccato already gates it correctly).

### External References

None for this work — patterns are well-established in the repo.

---

## Key Technical Decisions

- **New `AudioSource` model in its own file (`models/audio_source.py`)**: avoids polluting `Source` with nullable video-specific fields (`fps`, `width`, `height`, `cut`, `color_profile`). Type system enforces "you can't run scene detection on this" without runtime branches. Counter-argument was that a discriminator scales better for future media types, but no such media type is imminent.
- **Audio lives as a side-by-side section in the Collect tab, not a new tab**: keeps audio discoverable alongside video import. A new tab would add nav weight and create the "where does audio live?" question for users.
- **No waveform thumbnail in the library**: generic audio icon for v1. Waveform rendering exists per-file in Staccato; library-grid thumbnails would require offline waveform precomputation and add no proven user value.
- **Staccato keeps its `music_path` attribute internally; the picker resolves to a path**: minimizes blast radius. The dialog's worker, waveform widget, and stem cache all work in terms of `Path`. Only the *selection mechanism* changes.
- **Transcription bonus is one shallow unit**: `core/transcription.py` already takes `audio_path`, so the work is exposing "transcribe this audio source" as an Analyze-tab action and an agent tool. No new transcription core code.
- **Agent and MCP exposure are combined into one unit**: both consume the same `AudioSource` shape and `Project.audio_sources` list; splitting them would duplicate test setup with no benefit.

---

## Open Questions

### Resolved During Planning

- **Should audio sources be copied into the project directory or referenced in place?**: Reference in place, like videos. Honors existing `to_dict(base_path)` serialization pattern with relative-path-plus-absolute-fallback.
- **What audio formats?**: Match the formats already accepted by `_AUDIO_FORMATS` in Staccato dialog. Verify the constant during U3 and reuse it.
- **Where does the import worker live?**: `ui/workers/audio_import_worker.py`, mirroring the pattern of other workers. Inherits from `CancellableWorker` (`ui/workers/base.py`) for consistency, even though import is fast.

### Deferred to Implementation

- **Exact UI layout for the Collect tab audio section**: vertical split vs. tabbed sub-sections vs. side-by-side. Try the simplest (vertical stack, audio below videos) and adjust if it feels cramped during implementation.
- **Should transcribing an audio source store the transcript on the `AudioSource` or in a separate model?**: store directly on `AudioSource.transcript: Optional[str]`, mirroring how `Clip.transcript` works. Confirm during U6.

---

## Implementation Units

- U1. **`AudioSource` model**

**Goal:** Define the `AudioSource` dataclass with serialization that mirrors `Source`'s pattern, but with audio-only fields.

**Requirements:** R1, R3, R8

**Dependencies:** None

**Files:**
- Create: `models/audio_source.py`
- Test: `tests/test_audio_source_model.py`

**Approach:**
- Dataclass fields: `id` (UUID), `file_path: Path`, `duration_seconds: float`, `sample_rate: int = 0`, `channels: int = 0`, `transcript: Optional[str] = None`, `transcript_segments: Optional[list] = None` (parallel to `Clip.transcript`/`transcript_segments`).
- Implement `to_dict(base_path: Optional[Path] = None) -> dict` and `from_dict(data: dict, base_path: Optional[Path] = None)` with the same path-traversal protection and `_absolute_path` fallback as `Source`.
- Properties: `filename` (file_path.name), `duration_str` (formatted MM:SS).

**Patterns to follow:**
- `models/clip.py:63-178` (`Source` class)

**Test scenarios:**
- Happy path: round-trip `AudioSource` → `to_dict` → `from_dict` preserves all fields.
- Happy path: `to_dict(base_path=...)` produces relative `file_path` and `_absolute_path` fallback.
- Edge case: `from_dict` resolves a relative path against `base_path` when the absolute fallback is missing.
- Error path: `from_dict` raises `ValueError` when a relative path traverses outside `base_path` (mirror `Source.from_dict` security check).
- Edge case: `transcript` and `transcript_segments` default to `None` and round-trip cleanly when populated.

**Verification:**
- The new test file passes in isolation.
- `Source` tests are unaffected.

---

- U2. **Project state: `audio_sources` collection**

**Goal:** Wire `AudioSource` into `Project` with the same observer / cache / mutator discipline as `Source`. Persist through save/load.

**Requirements:** R3, R8

**Dependencies:** U1

**Files:**
- Modify: `core/project.py`
- Test: `tests/test_project_audio_sources.py`

**Approach:**
- Add `_audio_sources: list[AudioSource]` to `Project.__init__`.
- Mirror the `Source` API: `audio_sources` property, `audio_sources_by_id` cached dict, `add_audio_source`, `remove_audio_source(id)`, `get_audio_source(id)`.
- Each mutator invalidates the cache and calls `_notify_observers("audio_sources_changed", self._audio_sources)`.
- In the project save function (currently around `core/project.py:210-216`), serialize `_audio_sources` under `data["audio_sources"]`.
- In the project load function (around `core/project.py:489-516`), deserialize from `data.get("audio_sources", [])` — defaulting to empty preserves backward compatibility with existing project files.
- Validate during load: log+skip malformed entries, like `Source` validation does.

**Patterns to follow:**
- `core/project.py:611-820` (`Project._sources` lifecycle)
- `core/project.py:169-216, 438-587` (save/load source iteration)

**Test scenarios:**
- Happy path: `add_audio_source(audio)` appends to list, invalidates cache, fires `audio_sources_changed` observer.
- Happy path: `remove_audio_source(id)` removes by id and fires observer.
- Happy path: `audio_sources_by_id` returns the cached dict and refreshes after add/remove.
- Happy path: project save/load round-trips `audio_sources` field.
- Edge case: loading a legacy project file without `audio_sources` key initializes `_audio_sources` to `[]` without error.
- Edge case: malformed audio source entry in JSON is logged and skipped (mirror `Source` validation).
- Integration: adding an audio source does not affect `_sources` or `sources_changed` observers.

**Verification:**
- New test file passes.
- Existing project save/load tests are unaffected.

---

- U3. **Audio import worker**

**Goal:** Background worker that reads an audio file, extracts duration / sample rate / channel count via ffprobe, and produces an `AudioSource` ready to add to the project.

**Requirements:** R1

**Dependencies:** U1

**Files:**
- Create: `ui/workers/audio_import_worker.py`
- Test: `tests/test_audio_import_worker.py`

**Approach:**
- Inherit `CancellableWorker` (`ui/workers/base.py`) — consistent with other workers even though import is fast.
- Signals: `progress(int, int)`, `audio_ready(AudioSource)`, `error(str)`, `finished()`.
- Use existing FFmpeg helpers (look for `ffprobe` calls in `core/ffmpeg.py`) to extract duration, sample rate, channel count. Validate file is parseable as audio (reject if duration ≤ 0 or no audio streams).
- Build and emit an `AudioSource` instance; do not mutate the project — that's the caller's responsibility, mirroring how video import works.

**Patterns to follow:**
- `ui/workers/base.py` (`CancellableWorker`)
- `core/ffmpeg.py` ffprobe helpers (verify exact API during implementation)
- Existing video-import worker for argument shape (locate during implementation; likely in `ui/workers/`)

**Test scenarios:**
- Happy path: importing a known WAV fixture emits `audio_ready` with non-zero duration, sample rate, channels.
- Error path: importing a file with no audio stream emits `error` and does not emit `audio_ready`.
- Error path: importing a missing path emits `error`.
- Edge case: cancellation mid-import does not emit `audio_ready`.

**Verification:**
- New test file passes; existing worker tests unaffected.

---

- U4. **Collect tab — audio section**

**Goal:** Add an audio import affordance and a separate audio library widget below the video grid in the Collect tab.

**Requirements:** R1, R2

**Dependencies:** U2, U3

**Files:**
- Modify: `ui/tabs/collect_tab.py`
- Possibly create: `ui/widgets/audio_library_list.py` (simple list view, not a card grid)
- Test: `tests/test_collect_tab_audio.py`

**Approach:**
- Toolbar gains a "+ Audio" button alongside existing "+ Local" / "+ YouTube" buttons. Uses `QFileDialog.getOpenFileName` with the audio formats filter (verify and reuse the `_AUDIO_FORMATS` constant from `staccato_dialog.py:611` — extract to a shared module if not already there).
- Drag-drop in `_on_files_dropped` routes audio extensions (.mp3/.wav/.m4a/.flac/.ogg) to the audio import worker; videos continue to flow to the video import path. Detection is by extension only.
- Below the video grid, render a simple list/table of audio sources: filename, duration, file size, remove button. Generic audio icon (no waveform thumbnail in v1). Section has a small header label so it's visually distinct.
- New `CollectTab` methods: `add_audio_source`, `remove_audio_source`, `get_audio_source`. Same shape as the video equivalents.
- Observer wiring: subscribe to `audio_sources_changed` and re-render the list when the project state changes.

**Patterns to follow:**
- `ui/tabs/collect_tab.py:22-191` (existing `CollectTab` shape and `add_source` / `remove_source` API)
- `ui/theme.py` UISizes constants for button heights / list row heights

**Test scenarios:**
- Happy path: clicking "+ Audio" → picking a file → triggers import worker → after `audio_ready`, project gains the audio source and the list shows it.
- Happy path: dropping a `.wav` file routes to audio import; dropping a `.mp4` routes to video import.
- Happy path: removing an audio source from the list calls `project.remove_audio_source` and the list refreshes.
- Edge case: dropping a file with an unknown extension is rejected without crashing.
- Integration: `audio_sources_changed` observer fires the list refresh — change project state programmatically and assert the list updates.

**Verification:**
- Test file passes.
- Manual: import a sample audio file, confirm it persists across save/load.

---

- U5. **Staccato dialog: audio source picker**

**Goal:** Replace `QFileDialog`-based music selection with a picker populated from `project.audio_sources`. Keep an "Import new…" escape hatch.

**Requirements:** R4, R5

**Dependencies:** U2, U3

**Files:**
- Modify: `ui/dialogs/staccato_dialog.py`
- Test: `tests/test_staccato_dialog_audio_picker.py`

**Approach:**
- Replace the "Select File" button + `_on_select_file` (`ui/dialogs/staccato_dialog.py:608-623`) with a `QComboBox` listing `project.audio_sources` (display: filename + duration). The combo also has an "Import new…" item at the bottom.
- Selecting an existing audio source sets `self._music_path = source.file_path` and triggers `_analyze_audio()` — same code path as today.
- Selecting "Import new…" opens `QFileDialog`, runs the import worker, adds the resulting `AudioSource` to the project, then auto-selects it in the combo.
- If `project.audio_sources` is empty when the dialog opens, the combo shows a placeholder ("No audio sources — import one") and only the "Import new…" item is enabled.
- `_music_path` attribute and downstream logic (waveform analysis, stem cache key, generation flow) are unchanged.

**Patterns to follow:**
- Existing combo-box dialogs in `ui/dialogs/` (e.g., `free_association_dialog.py` for clip selection patterns)
- UI consistency: `COMBO_BOX_MIN_HEIGHT` / `COMBO_BOX_MIN_WIDTH` from `ui/theme.py:UISizes`

**Test scenarios:**
- Happy path: opening the dialog with N audio sources populates the combo with N items.
- Happy path: selecting an existing source sets `music_path` and triggers audio analysis.
- Happy path: selecting "Import new…" → file dialog → worker → audio source added to project → combo auto-selects the new source.
- Edge case: opening the dialog with zero audio sources shows the placeholder and disables generation until import.
- Integration: re-opening the dialog after a previous session preserves the previously-imported audio source (proves persistence end-to-end).
- Edge case: a deleted/missing file path on a previously-imported audio source surfaces a clear error rather than crashing the analysis worker.

**Verification:**
- Test file passes.
- Manual: open the Staccato dialog after a fresh import; pick the audio source; generate a sequence; confirm the same waveform/beats/output as the pre-change implementation.

---

- U6. **Transcribe an audio source (Analyze tab + agent tool)**

**Goal:** Let users transcribe an audio source without a video carrier. Surface as an Analyze-tab action and an agent tool. Result lives on `AudioSource.transcript` / `transcript_segments`.

**Requirements:** R6, R7

**Dependencies:** U1, U2

**Files:**
- Modify: `ui/tabs/analyze_tab.py` (add an audio-source transcription entry point — likely a button/menu item that's enabled when an audio source is selected)
- Modify: `core/chat_tools.py` (new `transcribe_audio_source(project, audio_source_id)` tool)
- Possibly create: `ui/workers/audio_transcribe_worker.py` (or reuse existing transcription worker if its shape allows)
- Test: `tests/test_audio_transcription_flow.py`

**Approach:**
- Reuse `core/transcription.py:transcribe_audio` (line 659) — it already accepts an arbitrary `audio_path`. No core changes.
- Worker emits transcript text + timestamped segments on completion; main thread sets them on the `AudioSource` and notifies `audio_sources_changed`.
- Agent tool: `transcribe_audio_source(audio_source_id: str)` — mirrors the existing video transcription tool's shape and error reporting.
- UI surface: in the Analyze tab, when an audio source is selected (vs. a video source), show a "Transcribe" action. Other analyze actions (object detection, OCR, color, etc.) are not enabled for audio.

**Patterns to follow:**
- `core/chat_tools.py` — existing transcription tool for shape and error contract (locate during implementation; search for `transcribe` in chat_tools)
- `core/gui_state.py` — pattern for tracking selected source so the agent tool knows what's selected
- Existing analyze-tab action gating (action button enabled/disabled based on selected source type)

**Test scenarios:**
- Happy path: transcribing a known short audio fixture populates `audio_source.transcript` and `transcript_segments`.
- Happy path: agent tool `transcribe_audio_source` returns a structured success result with segment count.
- Error path: transcribing a missing audio source id returns `{"success": False, "error": ...}` with a helpful message ("Use list_audio_sources …" pattern).
- Edge case: cancelling mid-transcription leaves `audio_source.transcript` unset.
- Integration: re-running transcription on an already-transcribed audio source replaces the previous transcript and fires the observer.

**Verification:**
- Test file passes.
- Manual: import an audio file, transcribe it, confirm the transcript persists across save/load (round-trips through U1's serialization).

---

- U7. **Agent tools + MCP exposure**

**Goal:** Make audio sources first-class to the chat agent and the MCP server.

**Requirements:** R7

**Dependencies:** U2, U3

**Files:**
- Modify: `core/chat_tools.py` (new tools)
- Modify: `core/gui_state.py` (track selected audio source if relevant)
- Modify: `ui/chat_worker.py` (register new tools — verify exact registration site during implementation)
- Modify: `scene_ripper_mcp/server.py` (expose audio source ops)
- Test: `tests/test_chat_tools_audio_sources.py`, `tests/test_mcp_audio_sources.py`

**Approach:**
- Tools to add in `core/chat_tools.py`, mirroring existing source patterns:
  - `list_audio_sources(project) -> dict` — returns id, filename, duration, transcript-presence flag.
  - `get_audio_source(project, audio_source_id) -> dict` — full detail for one.
  - `import_audio_source(project, file_path) -> dict` — invokes the import worker synchronously (or signals back via `_wait_for_worker` if needed; check `pyside6-agent-worker-double-start` learning).
- MCP server: surface `list_audio_sources` and `get_audio_source` read-only operations. Defer write operations (import) until there's a clear external use case.
- Agent error messages reference `list_audio_sources` for discovery, matching the existing `list_sources` pattern in `core/chat_tools.py:2240`.

**Patterns to follow:**
- `core/chat_tools.py:464` (`list_sources`)
- `core/chat_tools.py:2240, 2274, 2328` (error message conventions)
- `pyside6-agent-async-worker-pattern` skill for worker-backed agent tools
- `tool-executor-result-format` skill for return shape

**Test scenarios:**
- Happy path: `list_audio_sources` returns all imported audio sources with id/filename/duration.
- Happy path: `get_audio_source(id)` returns full detail including transcript when present.
- Happy path: `import_audio_source(path)` adds an audio source and returns its id.
- Error path: `get_audio_source` with an unknown id returns `{"success": False, "error": ...}` referencing `list_audio_sources`.
- Error path: `import_audio_source` with a missing file path returns a structured error.
- Integration (MCP): MCP `list_audio_sources` returns the same shape as the chat-tool variant.

**Verification:**
- Test files pass.
- Manual: ask the chat agent to import and then transcribe an audio file end-to-end.

---

- U8. **User documentation**

**Goal:** Document audio import, the Collect-tab audio section, the Staccato picker, and audio transcription so users can discover the feature.

**Requirements:** R1, R2, R4, R6

**Dependencies:** U4, U5, U6, U7

**Files:**
- Modify: `docs/user-guide/index.md` or relevant per-tab doc (locate during implementation; the doc structure is `docs/user-guide/`)
- Possibly create: `docs/user-guide/audio-sources.md`

**Approach:**
- Short section: "Importing audio files" — supported formats, drag-drop, "+ Audio" button.
- Update Staccato user-guide section to reference the new picker.
- Short section on transcribing audio (point at the existing transcription docs and note that audio sources work the same way).

**Test expectation:** none — documentation only.

**Verification:**
- Markdown renders cleanly.
- Cross-references resolve.

---

## System-Wide Impact

- **Interaction graph:** New `audio_sources_changed` observer event on `Project`. Subscribers: `CollectTab` audio list (U4), `StaccatoDialog` combo box (U5), Analyze-tab action gating (U6). Forgetting to subscribe at any of these surfaces is the primary "GUI doesn't update" failure mode (mitigated by the `pyside6-project-state-mutation` learning).
- **Error propagation:** Import worker emits `error(str)` on bad files; UI shows the error in a status label, not a blocking dialog. Agent tools return `{"success": False, "error": ...}` matching existing convention.
- **State lifecycle risks:** A project save mid-import would persist a `_audio_sources` list that's missing the in-flight import. Same risk exists today with video import — accept the risk; document if surfaced in QA.
- **API surface parity:** `list_audio_sources` mirrors `list_sources` so any agent that knew how to find a source can find an audio source.
- **Integration coverage:** End-to-end test that imports → persists → re-loads → uses-in-Staccato proves the data layer integrates with the UI layer.
- **Unchanged invariants:** `Source` shape, `Project.sources` API, `Sequence.music_path` (still raw path; rewiring deferred), all existing video-clip pipelines, scene detection, sequencer output. None of these touch `AudioSource`.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Drag-drop classifier mistakes a video for audio (or vice versa) — extension-only detection | Limit accepted audio extensions to `.mp3/.wav/.m4a/.flac/.ogg` and let everything else fall through to existing video logic. Reject unknown extensions explicitly. |
| Project file format change breaks legacy projects | `data.get("audio_sources", [])` defaults to empty; old projects load cleanly. New projects with audio sources written by the new code are forward-compatible (older code would silently ignore the new field). |
| Agent worker double-start when "Import new…" is invoked while a previous import is in flight | Apply the `pyside6-agent-worker-double-start` learning — gate the import button while a worker is running. |
| Stale audio source pointing at a moved/deleted file | `from_dict` already handles absolute fallback; surface a clear error in the dialog when the file is missing rather than crashing analysis. |
| Transcription on an audio source overlaps with video-clip transcription somehow | Audio transcription writes only to `AudioSource.transcript` / `transcript_segments`. No path crosses into `Clip.transcript`. |

---

## Documentation / Operational Notes

- No migration script needed — `audio_sources` defaults to empty when absent.
- No new dependencies. Audio formats are read via existing FFmpeg/ffprobe; transcription uses existing whisper paths; Staccato's audio analysis was already gated on `librosa` via `audio_analysis` feature registry.
- No changes to release/build pipelines.

---

## Sources & References

- Existing `Source` model: `models/clip.py:63-178`
- Project state pattern: `core/project.py:611-820`
- Staccato file selection (current): `ui/dialogs/staccato_dialog.py:608-623`
- Transcription core (already accepts audio_path): `core/transcription.py:659-700`
- Sequence music export (deferred from this plan): `core/sequence_export.py:38`
- Agent tool pattern: `core/chat_tools.py:464` (`list_sources`)
