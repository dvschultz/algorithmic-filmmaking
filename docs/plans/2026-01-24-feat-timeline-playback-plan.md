# Plan: Timeline Playback

## Overview

Add playback functionality to the timeline so users can preview their sequence in real-time. This completes the core timeline functionality by connecting the playhead to video playback with seamless clip transitions.

## Current State

**Timeline** (`ui/timeline/timeline_widget.py`):
- Has playhead that can be positioned via click/drag
- Emits `playhead_changed(time_seconds)` signal
- Has `get_clip_at_playhead()` returning (SequenceClip, Clip, Source)
- No play/pause controls

**VideoPlayer** (`ui/video_player.py`):
- Has `load_video(path)`, `seek_to(seconds)`, `play_range(start, end)`
- Plays single source video
- Has its own play/pause button (for clip preview)

**MainWindow Connection**:
- `playhead_changed` → `video_player.seek_to(time_seconds)`
- Only seeks within current source - doesn't switch sources

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TimelineWidget                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Toolbar: [▶ Play] [■ Stop]  Remix: [...] Generate   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Timeline View + Playhead                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │ playback_requested(start_frame)
         │ playhead_changed(time_seconds)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      MainWindow                              │
│  - Coordinates playback between timeline and player         │
│  - Manages source switching when crossing clip boundaries   │
│  - Uses QTimer for playback loop                            │
└─────────────────────────────────────────────────────────────┘
         │ load_video() / seek_to() / play()
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      VideoPlayer                             │
│  - Plays video from current source                          │
│  - Reports position for playhead sync                       │
└─────────────────────────────────────────────────────────────┘
```

## Files to Modify

| File | Change |
|------|--------|
| `ui/timeline/timeline_widget.py` | Add play/stop buttons, playback signals |
| `ui/main_window.py` | Add playback coordination, source switching |
| `ui/video_player.py` | Add signal for position sync during playback |

## Implementation Steps

### Step 1: Add Play/Stop Buttons to Timeline Toolbar

**File:** `ui/timeline/timeline_widget.py`

Add play button to toolbar after "Timeline" label:
```python
# In _create_toolbar():
# Play button (before remix controls)
self.play_btn = QPushButton()
self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
self.play_btn.setToolTip("Play sequence")
self.play_btn.setFixedSize(32, 28)
toolbar.addWidget(self.play_btn)

# Stop button
self.stop_btn = QPushButton()
self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
self.stop_btn.setToolTip("Stop playback")
self.stop_btn.setFixedSize(32, 28)
toolbar.addWidget(self.stop_btn)

toolbar.addSpacing(16)  # Separator before remix controls
```

Add signals:
```python
# Add to class signals
playback_requested = Signal(int)  # start_frame
stop_requested = Signal()
```

Connect buttons:
```python
# In _connect_signals():
self.play_btn.clicked.connect(self._on_play_clicked)
self.stop_btn.clicked.connect(self._on_stop_clicked)

def _on_play_clicked(self):
    """Request playback from current playhead position."""
    frame = self._playhead.get_frame() if self._playhead else 0
    self.playback_requested.emit(frame)

def _on_stop_clicked(self):
    """Request playback stop."""
    self.stop_requested.emit()
```

### Step 2: Add Playback State Management

**File:** `ui/timeline/timeline_widget.py`

Add methods to update UI during playback:
```python
def set_playing(self, playing: bool):
    """Update UI to reflect playback state."""
    if playing:
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.play_btn.setToolTip("Pause sequence")
    else:
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setToolTip("Play sequence")
```

### Step 3: Add Position Reporting to VideoPlayer

**File:** `ui/video_player.py`

Add signal for external position sync:
```python
# Add signal
position_updated = Signal(int)  # position in milliseconds

# In _on_position_changed():
def _on_position_changed(self, position: int):
    """Handle position change."""
    self.position_slider.setValue(position)
    self.position_updated.emit(position)  # Add this line
    # ... rest of method
```

### Step 4: Add Playback Coordination to MainWindow

**File:** `ui/main_window.py`

Add imports:
```python
from PySide6.QtCore import QTimer
```

Add instance variables in `__init__`:
```python
# Playback state
self._is_playing = False
self._playback_start_frame = 0
self._current_playback_clip = None  # Currently playing SequenceClip
self._playback_timer = QTimer()
self._playback_timer.setInterval(33)  # ~30fps update rate
self._playback_timer.timeout.connect(self._on_playback_tick)
```

Connect timeline signals:
```python
# In _setup_timeline():
self.timeline.playback_requested.connect(self._on_playback_requested)
self.timeline.stop_requested.connect(self._on_stop_requested)

# Connect video player position for playhead sync
self.video_player.position_updated.connect(self._on_video_position_updated)
```

### Step 5: Implement Playback Logic

**File:** `ui/main_window.py`

```python
def _on_playback_requested(self, start_frame: int):
    """Start sequence playback from given frame."""
    if self._is_playing:
        # Toggle pause
        self._pause_playback()
        return

    sequence = self.timeline.get_sequence()
    if sequence.duration_frames == 0:
        return  # Nothing to play

    self._is_playing = True
    self._playback_start_frame = start_frame
    self.timeline.set_playing(True)

    # Load and play the clip at current position
    self._play_clip_at_frame(start_frame)

def _play_clip_at_frame(self, frame: int):
    """Load and play the clip at given timeline frame."""
    seq_clip, clip, source = self.timeline.get_clip_at_playhead()

    if not seq_clip:
        # No clip at this position - stop or skip to next
        self._stop_playback()
        return

    self._current_playback_clip = seq_clip

    # Calculate source position
    # frame_in_clip = where we are relative to clip start on timeline
    frame_in_clip = frame - seq_clip.start_frame
    # source_frame = in_point + offset into clip
    source_frame = seq_clip.in_point + frame_in_clip
    source_seconds = source_frame / source.fps

    # Calculate end of this clip in source time
    end_seconds = seq_clip.out_point / source.fps

    # Load source and play range
    self.video_player.load_video(source.file_path)
    self.video_player.play_range(source_seconds, end_seconds)

    # Start timer to monitor for clip transitions
    self._playback_timer.start()

def _on_playback_tick(self):
    """Called during playback to check for clip transitions."""
    if not self._is_playing:
        self._playback_timer.stop()
        return

    # Check if we've reached end of current clip
    if not self._current_playback_clip:
        return

    # Get current timeline position from playhead
    current_time = self.timeline.get_playhead_time()
    current_frame = int(current_time * self.timeline.sequence.fps)

    # Check if we've moved past current clip
    if current_frame >= self._current_playback_clip.end_frame():
        # Move to next clip
        next_frame = self._current_playback_clip.end_frame()
        self.timeline.set_playhead_time(next_frame / self.timeline.sequence.fps)
        self._play_clip_at_frame(next_frame)

def _on_video_position_updated(self, position_ms: int):
    """Sync timeline playhead to video position during playback."""
    if not self._is_playing or not self._current_playback_clip:
        return

    seq_clip = self._current_playback_clip
    source_data = self.timeline._clip_lookup.get(seq_clip.source_clip_id)
    if not source_data:
        return

    _, source = source_data

    # Convert video position to source frame
    source_seconds = position_ms / 1000.0
    source_frame = int(source_seconds * source.fps)

    # Calculate timeline frame
    # timeline_frame = start_frame + (source_frame - in_point)
    frame_offset = source_frame - seq_clip.in_point
    timeline_frame = seq_clip.start_frame + frame_offset
    timeline_seconds = timeline_frame / self.timeline.sequence.fps

    # Update playhead (don't re-emit to avoid loops)
    if self.timeline._playhead:
        self.timeline._playhead.set_time(timeline_seconds)

def _pause_playback(self):
    """Pause playback."""
    self._is_playing = False
    self._playback_timer.stop()
    self.video_player.player.pause()
    self.timeline.set_playing(False)

def _on_stop_requested(self):
    """Stop playback."""
    self._stop_playback()

def _stop_playback(self):
    """Stop playback and reset state."""
    self._is_playing = False
    self._playback_timer.stop()
    self._current_playback_clip = None
    self.video_player.player.stop()
    self.timeline.set_playing(False)
```

### Step 6: Handle End of Sequence

**File:** `ui/main_window.py`

Update `_play_clip_at_frame` to handle end of sequence:
```python
def _play_clip_at_frame(self, frame: int):
    """Load and play the clip at given timeline frame."""
    # Check if we're past the end of sequence
    sequence = self.timeline.get_sequence()
    if frame >= sequence.duration_frames:
        self._stop_playback()
        # Reset playhead to beginning
        self.timeline.set_playhead_time(0)
        return

    # ... rest of existing implementation
```

### Step 7: Handle Video Player State Changes

**File:** `ui/main_window.py`

Connect to video player state:
```python
# In _setup_connections():
self.video_player.player.playbackStateChanged.connect(self._on_video_state_changed)

def _on_video_state_changed(self, state):
    """Handle video player state changes."""
    from PySide6.QtMultimedia import QMediaPlayer

    if not self._is_playing:
        return

    if state == QMediaPlayer.StoppedState:
        # Clip ended - check if we should continue to next
        if self._current_playback_clip:
            next_frame = self._current_playback_clip.end_frame()
            self.timeline.set_playhead_time(next_frame / self.timeline.sequence.fps)
            self._play_clip_at_frame(next_frame)
```

## Data Flow During Playback

```
1. User clicks Play button
2. TimelineWidget emits playback_requested(current_frame)
3. MainWindow._on_playback_requested():
   - Gets clip at current playhead position
   - Calculates source position from timeline frame
   - Loads correct source video
   - Calls video_player.play_range()
   - Starts playback timer
4. During playback:
   - VideoPlayer emits position_updated(ms)
   - MainWindow converts to timeline frame
   - Updates playhead position
5. When clip ends:
   - Timer detects frame >= clip.end_frame()
   - Loads next clip's source
   - Continues playback seamlessly
6. At end of sequence:
   - Stop playback
   - Reset playhead to beginning
```

## Verification

1. **Manual testing:**
   - Import video, run scene detection
   - Add clips to timeline (or generate remix)
   - Click Play - should start playing from playhead
   - Playhead should move during playback
   - Should switch sources at clip boundaries
   - Should stop at end of sequence
   - Click timeline during playback to seek
   - Pause/resume functionality works

2. **Edge cases:**
   - Play with empty timeline (should do nothing)
   - Play when playhead is past all clips
   - Play timeline with gaps between clips
   - Multiple tracks (future - play topmost clip)

## Checklist

- [x] Add play/stop buttons to timeline toolbar
- [x] Add `playback_requested` and `stop_requested` signals
- [x] Add `set_playing()` method to update button icon
- [x] Add `position_updated` signal to VideoPlayer
- [x] Add playback state variables to MainWindow
- [x] Add QTimer for playback loop
- [x] Implement `_on_playback_requested()`
- [x] Implement `_play_clip_at_frame()` with source switching
- [x] Implement `_on_video_position_updated()` for playhead sync
- [x] Implement `_on_playback_tick()` for clip transitions
- [x] Implement pause/stop functionality
- [x] Handle end of sequence
- [x] Handle gaps between clips (show black)
- [x] Test with multi-clip sequence
