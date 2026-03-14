---
title: "refactor: Unify sequence playback controls"
type: refactor
status: completed
date: 2026-03-03
---

# Unify Sequence Playback Controls

## Overview

Remove the duplicate play/stop buttons from the timeline toolbar and wire the VideoPlayer's existing controls to drive full sequence playback. Currently the Sequence tab has two independent sets of transport controls that behave differently — the timeline toolbar drives the sequence engine while the VideoPlayer controls only play the current loaded clip.

## Problem Statement

The Sequence tab has two play buttons:

1. **Timeline toolbar** (`ui/timeline/timeline_widget.py:81-92`) — `play_btn` + `stop_btn`. These emit `playback_requested`/`stop_requested` signals that MainWindow's sequence engine handles for clip-by-clip playback.
2. **VideoPlayer controls** (`ui/video_player.py:288-302`) — `play_btn` + `stop_btn`. These directly toggle MPV pause on the currently loaded clip. They do NOT advance through the sequence.

This creates confusion: pressing play on the VideoPlayer only plays the current clip in isolation. Pressing play on the timeline toolbar plays the full sequence. There's also a bug where pressing the VideoPlayer's play button while the sequence engine is active (`_is_playing == True`) triggers clip-skip instead of pause.

## Proposed Solution

1. **Remove** `play_btn` and `stop_btn` from `TimelineWidget` toolbar
2. **Add a `sequence_mode` flag** to `VideoPlayer` that changes play/stop behavior
3. **In sequence mode**, VideoPlayer's play/stop emit signals instead of controlling MPV directly — MainWindow's sequence engine handles them
4. **Update MainWindow** to connect VideoPlayer's new signals and update the VideoPlayer's play icon during sequence playback

## Acceptance Criteria

- [x] Timeline toolbar no longer has play/stop buttons
- [x] VideoPlayer play button starts full sequence playback from current playhead position
- [x] VideoPlayer play button pauses sequence playback when playing
- [x] VideoPlayer stop button stops sequence playback and resets playhead
- [x] VideoPlayer play/pause icon stays in sync with sequence playback state
- [x] Frame stepping (back/forward) still works during paused sequence
- [x] Position slider still reflects playback position within current clip
- [x] Speed control still works (disabled during playback, enabled when paused/stopped)
- [x] A/B loop controls unaffected
- [x] Non-sequence VideoPlayer instances (clip details sidebar) unaffected — they keep direct MPV control

## Technical Approach

### File Changes

#### `ui/video_player.py`

Add sequence mode with new signals:

```python
# New signals
play_requested = Signal()      # Emitted in sequence mode when play is clicked
stop_requested = Signal()      # Emitted in sequence mode when stop is clicked

# New flag
self._sequence_mode = False
```

Modify `_toggle_playback()`:
```python
def _toggle_playback(self):
    if self._sequence_mode:
        self.play_requested.emit()
        return
    # existing MPV toggle logic
```

Modify `stop()` (or add sequence-mode path):
```python
# When stop_btn is clicked in sequence mode, emit signal
# instead of directly controlling MPV
```

Add `set_sequence_mode(enabled: bool)` and `set_playing(playing: bool)` methods:
```python
def set_sequence_mode(self, enabled: bool):
    """Enable sequence mode — play/stop emit signals instead of controlling MPV."""
    self._sequence_mode = enabled

def set_playing(self, playing: bool):
    """Update play/pause icon state (called by sequence engine)."""
    if playing:
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
    else:
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
```

#### `ui/timeline/timeline_widget.py`

Remove from toolbar (lines 80-92):
- `self.play_btn` and `self.stop_btn`
- Their signal connections in `_connect_signals()` (lines 146-147)
- `_on_play_clicked()` and `_on_stop_clicked()` methods (lines 162-169)

Update `set_playing()` method (lines 171-178):
- Remove play_btn icon toggling (VideoPlayer now handles this)
- Keep the method as a no-op or remove if nothing else uses it

Keep signals `playback_requested` and `stop_requested` on the class — they're still emitted, just from a different source now.

#### `ui/tabs/sequence_tab.py`

Wire VideoPlayer signals instead of timeline signals:
```python
# In _create_timeline_view():
self.video_player.set_sequence_mode(True)
self.video_player.play_requested.connect(self._on_playback_requested)
self.video_player.stop_requested.connect(self._on_stop_requested)

# Remove or keep timeline signal connections (timeline no longer emits these)
```

Update `_on_playback_requested` to get playhead frame from timeline:
```python
def _on_playback_requested(self):
    frame = self.timeline.get_playhead_frame()
    self.playback_requested.emit(frame)
```

#### `ui/main_window.py`

Update sequence playback methods to use VideoPlayer's `set_playing()` instead of `timeline.set_playing()`:

- `_on_playback_requested()` (line 4411): `sequence_tab.video_player.set_playing(True)`
- `_pause_playback()` (line 4621): `sequence_tab.video_player.set_playing(False)`
- `_stop_playback()` (line 4643): `sequence_tab.video_player.set_playing(False)`

The existing `_on_video_state_changed` handler needs adjustment — since we're now going through the sequence engine for play/stop, the state change signal from MPV should NOT trigger clip advancement when the user presses stop. This is already handled by the `_is_playing` flag check at line 4602.

### What Stays the Same

- `TimelineWidget` toolbar keeps: Clear, + Track, Fit, 100%, Export Sequence
- `TimelineWidget` signals `playback_requested`/`stop_requested` can be removed from the class
- VideoPlayer frame stepping, position slider, speed combo, A/B loop — all unchanged
- Sidebar VideoPlayer — unaffected (not in sequence mode)
- MainWindow sequence engine logic (`_play_clip_at_frame`, `_on_playback_tick`, etc.) — unchanged
- The `_on_video_state_changed` handler — works as-is since it checks `_is_playing`

### Edge Cases

- **Empty sequence**: Play button should be a no-op when sequence has no clips (MainWindow already checks `sequence.duration_frames == 0`)
- **Clicking play during source switch**: The `_pending_sequence_playback_*` fields already handle mid-switch state
- **Frame stepping during pause**: Works because frame stepping calls MPV directly, not through the sequence engine

## Sources & References

- Completed timeline playback plan: `docs/plans/archive/2026-01-24-feat-timeline-playback-plan.md`
- MPV integration plan: `docs/plans/2026-02-27-feat-mpv-video-player-integration-plan.md`
- Sequence tab redesign: `docs/plans/archive/2026-01-28-feat-sequence-tab-ui-redesign-plan.md`
- Current playback engine: `ui/main_window.py:4391-4648`
- Timeline toolbar: `ui/timeline/timeline_widget.py:70-147`
- VideoPlayer controls: `ui/video_player.py:283-348`
