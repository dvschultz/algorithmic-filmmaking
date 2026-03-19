---
title: "feat: Staccato Music Track Export and Timeline Waveform"
type: feat
status: completed
date: 2026-03-19
---

# feat: Staccato Music Track Export and Timeline Waveform

## Overview

Add end-to-end music track support for staccato sequences: persist the music file path on the Sequence model, mux the music audio onto the exported video, and show an audio waveform track in the timeline. The staccato sequence is cut to the music's beats, so audio and video are inherently time-aligned.

## Problem Statement / Motivation

Staccato generates sequences synced to a music track's beats and onsets, but the music file is discarded after generation. Users must manually combine the exported video with their music in a separate tool. This breaks the "one-click export" workflow and defeats the purpose of beat-driven editing.

## Proposed Solution

Three phases, each independently shippable:

### Phase 1: Persist `music_path` on Sequence

- Add `music_path: Optional[str] = None` to the `Sequence` dataclass in `models/sequence.py`
- Follow the `reference_source_id` pattern: conditionally serialize in `to_dict()`, read with `.get()` in `from_dict()`
- Store as a relative path (same pattern as `SequenceClip.prerendered_path`)
- Pass `music_path` from `StaccatoDialog` to `sequence_tab` via the existing `sequence_metadata` dict pattern used by Reference Guide

**Files:**

| File | Change |
|------|--------|
| `models/sequence.py` | Add `music_path` field, update `to_dict()`/`from_dict()` with relative-path handling |
| `ui/dialogs/staccato_dialog.py` | Change `sequence_ready` signal to `Signal(list, object)` to pass music_path, or expose as property |
| `ui/tabs/sequence_tab.py` | Update `_apply_staccato_sequence` to pass `sequence_metadata={"music_path": str(music_path)}` |

### Phase 2: Mux Music Audio onto Exported Video

- Add `music_path: Optional[Path] = None` to `ExportConfig` in `core/sequence_export.py`
- Add `_mux_audio()` method to `SequenceExporter` that runs after `_concat_segments()`
- FFmpeg command: two inputs, map video from concat output + audio from music file
- Use `-shortest` to trim music if longer than video
- If music is shorter than video, pad with silence or let video continue silently
- Apply FFmpeg path escaping rules per `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`

**FFmpeg command pattern:**

```bash
ffmpeg -i concat_output.mp4 -i music.mp3 \
  -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 192k \
  -shortest \
  -y output.mp4
```

**Files:**

| File | Change |
|------|--------|
| `core/sequence_export.py` | Add `music_path` to `ExportConfig`. Add `_mux_audio()` method. Call after concat in `export()`. Update `export_sequence()` convenience function |
| `ui/main_window.py` | In `_on_sequence_export_click()`, pass `sequence.music_path` into ExportConfig |

### Phase 3: Timeline Audio Waveform Track

- Create `ui/timeline/audio_track_item.py` — a `QGraphicsItem` that draws a waveform
- Reuse the peak-calculation logic from `WaveformWidget.paintEvent()` but adapted for QGraphicsScene coordinates and zoom-aware rendering
- Add the audio track item below video tracks in `TimelineScene.rebuild()`
- Load waveform samples when `TimelineWidget.load_sequence()` detects a `music_path`
- Waveform computation runs in a background thread (reuse the librosa loading pattern from `StaccatoAnalyzeWorker`)
- Track must respond to zoom changes (`pixels_per_second`) to stay aligned with video clips

**Layout:**
```
[Ruler]
[Track 1 - Video clips]
[Track 2 - Video clips (if multi-track)]
[Audio Track - Waveform] ← new, non-interactive, fixed height
```

**Files:**

| File | Change |
|------|--------|
| New: `ui/timeline/audio_track_item.py` | QGraphicsRectItem subclass with waveform paint |
| `ui/timeline/timeline_scene.py` | Add audio track in `rebuild()`, update `_update_scene_rect()` height |
| `ui/timeline/timeline_widget.py` | Add `set_audio_waveform()` method, update playhead height, trigger waveform load on `load_sequence()` |

## Technical Considerations

### Path Persistence
- Music path stored relative to project directory (same as `prerendered_path`)
- On reload, resolve relative → absolute via `(base_path / music_path).resolve()`
- If the file doesn't exist at load time, log a warning and set `music_path = None`

### FFmpeg Audio Muxing
- Use `-c:v copy` to avoid re-encoding video (fast)
- Use `-c:a aac -b:a 192k` for audio encoding (or match ExportConfig's audio_codec)
- Apply path escaping from `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`
- Wrap subprocess in `try/finally` for cleanup per `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`
- If music_path doesn't exist at export time, skip muxing and log warning

### Timeline Single-Sequence Invariant
- Per `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`, the scene owns the sequence. Audio waveform data should live on the scene (or be a child item of the scene), not shadowed in the widget.
- Per `docs/solutions/runtime-errors/qgraphicsscene-missing-items-20260124.md`, `rebuild()` must create the audio track item. If the item is missing from `_track_items`, operations will crash.

### Waveform Performance
- Only render the visible portion of the waveform (clip to viewport)
- Downsample to pixel width (already done in WaveformWidget)
- Cache peaks at a reasonable resolution; recompute on zoom only if needed
- Background thread for initial librosa load (apply guard-flag pattern per `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`)

## Acceptance Criteria

### Phase 1: Model
- [x] `Sequence.music_path` persists through save/load cycle
- [x] Staccato dialog passes music_path to sequence via `sequence_metadata`
- [x] Old projects without `music_path` load without errors (backward compatible)
- [x] music_path stored as relative path in project JSON

### Phase 2: Export
- [x] Exporting a staccato sequence with music_path produces a video with the music audio track
- [x] Video stream is not re-encoded (`-c:v copy`)
- [x] Music longer than video is trimmed
- [x] Music shorter than video: video continues (or silence fills)
- [x] Missing music file at export time: export succeeds without audio, logs warning
- [x] Export without music (non-staccato sequences) works unchanged

### Phase 3: Timeline
- [x] Staccato sequences show a waveform track below video tracks
- [x] Non-staccato sequences show no audio track
- [x] Waveform stays time-aligned with video clips across zoom levels
- [x] Playhead extends through the audio track
- [x] Missing music file: no audio track shown, no crash
- [x] Waveform loads asynchronously (doesn't block UI)

### Tests
- [x] Unit test: Sequence.to_dict/from_dict roundtrip with music_path
- [x] Unit test: ExportConfig with music_path builds correct FFmpeg args
- [x] Unit test: _mux_audio skips gracefully when music file missing

## Dependencies & Risks

- **librosa** is already a dependency (used for audio analysis)
- **FFmpeg** `-c:v copy` with audio mux is well-established; no risk of quality loss
- **Risk**: MPS/CUDA device detection in waveform loading — use CPU only for librosa (it doesn't use GPU)
- **Risk**: Very long music files could produce large peak arrays — mitigate with downsampling

## Sources & References

- Sequence model pattern: `models/sequence.py:172-269` (reference_source_id)
- Export pipeline: `core/sequence_export.py:47-443`
- Timeline architecture: `ui/timeline/timeline_scene.py`, `ui/timeline/timeline_widget.py`
- Waveform rendering: `ui/widgets/waveform_widget.py:57-142`
- Staccato dialog: `ui/dialogs/staccato_dialog.py:224-622`
- Sequence metadata passing: `ui/tabs/sequence_tab.py:693-756` (_apply_dialog_sequence)
- FFmpeg path escaping: `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`
- Timeline state invariant: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- Scene rebuild requirement: `docs/solutions/runtime-errors/qgraphicsscene-missing-items-20260124.md`
- Subprocess cleanup: `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`
- Worker guard pattern: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
