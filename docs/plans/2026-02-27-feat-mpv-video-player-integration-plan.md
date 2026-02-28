---
title: "feat: MPV-Based Video Player Integration"
type: feat
status: completed
date: 2026-02-27
origin: docs/brainstorms/2026-02-27-mpv-video-player-brainstorm.md
---

# feat: MPV-Based Video Player Integration

## Overview

Replace the current PySide6 QMediaPlayer-based video player with an MPV-powered player using python-mpv (libmpv). This is a clean swap with no fallback — QMediaPlayer is removed entirely. Beyond the engine swap, add frame-accurate seeking, frame stepping, playback speed control, and A/B looping. The thumbnail scrubber is deferred to a follow-up enhancement.

The migration is structured to minimize risk: first fix the leaky abstraction (consumers accessing internal `.player`), then swap the engine, then add new features.

## Problem Statement / Motivation

The current Qt Multimedia player (`QMediaPlayer` + `QVideoWidget`) has four pain points:
1. **Sluggish seeking/scrubbing** — keyframe-based seeking, no frame accuracy
2. **Poor codec support** — relies on OS-level codecs; Linux requires GStreamer plugins that often aren't installed
3. **No editing-grade features** — no frame stepping, no speed control, no A/B looping
4. **Basic feel** — minimal controls, no professional video editing UX

MPV solves all four: frame-accurate seeking is built-in, it bundles FFmpeg for near-universal codec support, and it exposes rich playback properties (speed, A/B loop, frame stepping) through a clean API.

(See brainstorm: `docs/brainstorms/2026-02-27-mpv-video-player-brainstorm.md`)

## Proposed Solution

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      VideoPlayer                          │
│  (public API — unchanged interface + new methods)         │
│                                                           │
│  ┌─────────────────────┐  ┌──────────────────────────┐   │
│  │   MpvContainer      │  │   MpvSignalBridge        │   │
│  │   (QWidget + wid)   │  │   (property observers    │   │
│  │   or                │  │    → Qt signals)          │   │
│  │   MpvGLWidget       │  │                           │   │
│  │   (QOpenGLWidget)   │  │   position_changed(float) │   │
│  │                     │  │   duration_changed(float)  │   │
│  │   mpv.MPV instance  │  │   pause_changed(bool)      │   │
│  │                     │  │   media_loaded()            │   │
│  └─────────────────────┘  └──────────────────────────┘   │
│                                                           │
│  Controls: play/pause, stop, slider, time, speed,        │
│            frame step, (A/B loop in sequence only)        │
└──────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
  ClipDetailsSidebar   SequenceTab/MainWindow   Agent Tools
  (public API only)    (public API only)        (public API only)
```

### Key Design Decisions

All decisions carried forward from brainstorm:

1. **Clean swap, no fallback** — Remove QMediaPlayer entirely, require libmpv
2. **Enhanced player** — Frame stepping, speed control, A/B loop (thumbnail scrubber deferred)
3. **Full cross-platform** — macOS + Windows + Linux
4. **Dual distribution** — pip install (documented prerequisite) + PyInstaller bundle
5. **Preserve + expand public API** — Existing methods kept, new methods added, internal `.player` access eliminated from all consumers

### New Decision: Embedding Strategy

**Start with `wid` (window ID) embedding on all platforms.** This is the simpler approach and works reliably on Windows and Linux. On macOS, it has documented stability concerns but works for most use cases. If macOS issues arise during Phase 1 testing, upgrade to `QOpenGLWidget` + `MpvRenderContext` for macOS only.

Rationale: The OpenGL approach adds significant complexity (render context lifecycle, GL proc address resolution, frame buffer management). Starting simple lets us validate the full integration before adding that complexity. The `wid` approach is a contained change in `MpvContainer` that can be swapped to `MpvGLWidget` without affecting the rest of VideoPlayer.

### New Decision: A/B Loop vs Clip Range

Map the existing clip range system to MPV's `ab-loop-a`/`ab-loop-b` properties internally. The user-facing "A/B loop" feature (manual markers) is separate and only available when clip range is NOT active. When `set_clip_range()` is called, it sets `ab-loop-a` and `ab-loop-b`. When the user sets manual A/B loop markers, it uses the same properties. They cannot both be active simultaneously.

### New Decision: Thumbnail Scrubber Deferred

The thumbnail-on-hover scrubber requires a separate data pipeline (pre-generating thumbnail strips per source) that is independent of the MPV integration. Defer to a follow-up plan to keep this plan focused.

### New Decision: Speed Control Scope

Playback speed is available in both ClipDetailsSidebar and SequenceTab for direct video preview. During automated sequence playback (the timer-based clip-transition system in MainWindow), speed is fixed at 1x to avoid timeline synchronization complexity. The speed control widget is disabled during sequence playback.

## Technical Approach

### Complete Public API (Post-Migration)

```python
class VideoPlayer(QWidget):
    # --- Signals ---
    position_updated = Signal(int)         # milliseconds (preserved)
    duration_changed = Signal(int)         # milliseconds (new)
    media_loaded = Signal()                # fires when file is ready (new)
    playback_state_changed = Signal(bool)  # True=playing, False=paused/stopped (new)

    # --- Existing Methods (preserved) ---
    def load_video(self, path: Path) -> None
    def seek_to(self, seconds: float) -> None
    def set_clip_range(self, start_seconds: float, end_seconds: float) -> None
    def clear_clip_range(self) -> None
    def play_range(self, start_seconds: float, end_seconds: float) -> None

    # --- New Methods ---
    def play(self) -> None
    def pause(self) -> None
    def stop(self) -> None
    def shutdown(self) -> None            # Clean MPV termination

    # --- New Properties ---
    @property
    def is_playing(self) -> bool
    @property
    def duration_ms(self) -> int
    @property
    def playback_speed(self) -> float
    @playback_speed.setter
    def playback_speed(self, speed: float) -> None

    # --- New Feature Methods ---
    def frame_step_forward(self) -> None
    def frame_step_backward(self) -> None
    def set_ab_loop(self, a_seconds: float, b_seconds: float) -> None
    def clear_ab_loop(self) -> None
```

### Implementation Phases

#### Phase 1: API Cleanup (Pre-Migration)

Refactor all consumers to use public API only, while still on QMediaPlayer. This decouples the engine swap from the API change.

**1a. Expand VideoPlayer public API on current QMediaPlayer engine**

Add `play()`, `pause()`, `stop()`, `is_playing`, `duration_ms`, `media_loaded` signal, `playback_state_changed` signal, and `shutdown()` (no-op for QMediaPlayer). These are thin wrappers around the internal `self.player`.

Files: `ui/video_player.py`

**1b. Refactor ClipDetailsSidebar**

Replace all `self.video_player.player.*` accesses:
- `self.video_player.player.mediaStatusChanged` → connect to `self.video_player.media_loaded`
- `self.video_player.player.pause()` → `self.video_player.pause()`
- `self.video_player.player.stop()` → `self.video_player.stop()`
- Remove `_pending_clip_range` pattern — `media_loaded` signal replaces `mediaStatusChanged`

Files: `ui/clip_details_sidebar.py`

**1c. Refactor MainWindow**

Replace all `self.sequence_tab.video_player.player.*` accesses:
- `.player.playbackStateChanged` → `video_player.playback_state_changed`
- `.player.playbackState() != QMediaPlayer.PlayingState` → `not video_player.is_playing`
- `.player.stop()` → `video_player.stop()`
- `.player.pause()` → `video_player.pause()`
- `.player.duration()` → `video_player.duration_ms`

Files: `ui/main_window.py`

**1d. Refactor agent tools**

Replace all `player.player.*` accesses in `chat_tools.py`:
- `player.player.play()` → `player.play()`
- `player.player.pause()` → `player.pause()`
- `player.player.duration()` → `player.duration_ms`

Files: `core/chat_tools.py`

**1e. Verify**

Run full test suite. Manually test all three consumer flows (sidebar clip preview, sequence playback, agent commands). Everything should work identically — this phase changes no behavior.

#### Phase 2: MPV Engine Swap

Replace the QMediaPlayer internals with MPV. External behavior is identical.

**2a. Add dependencies**

- Add `python-mpv>=1.0.8` to `requirements.txt` and `requirements-core.txt`
- Add locale fix to `main.py`: `locale.setlocale(locale.LC_NUMERIC, 'C')` after PySide6 imports, before QApplication creation
- Add startup check: if `import mpv` fails, show a dialog with platform-specific install instructions

Files: `requirements.txt`, `requirements-core.txt`, `main.py`

**2b. Create MpvSignalBridge**

A QObject that registers MPV property observers and emits Qt signals. Handles thread safety (MPV callbacks → Qt main thread).

```python
class MpvSignalBridge(QObject):
    position_changed = Signal(float)   # seconds
    duration_changed = Signal(float)   # seconds
    pause_changed = Signal(bool)
    media_loaded = Signal()
    eof_reached = Signal()
```

Key implementation details:
- Store all observer function references in `self._observers` list to prevent GC
- MPV's `Signal.emit()` from callbacks is safe — Qt auto-queues cross-thread signals
- Guard all callbacks with `if value is not None` (MPV sends None when no file loaded)

Files: New file `ui/mpv_bridge.py` or inline in `ui/video_player.py`

**2c. Rewrite VideoPlayer internals**

Replace QMediaPlayer/QVideoWidget/QAudioOutput with:
- `MpvContainer(QWidget)` — sets `WA_DontCreateNativeAncestors` + `WA_NativeWindow`, creates `mpv.MPV(wid=str(int(self.winId())), vo='gpu', keep_open='yes', idle='yes', hwdec='auto')`
- `MpvSignalBridge` — bridges property observers to Qt signals
- Map all public methods to MPV API:
  - `load_video(path)` → `self._mpv.play(str(path))`
  - `seek_to(seconds)` → `self._mpv.seek(seconds, 'absolute', 'exact')`
  - `play()` → `self._mpv.pause = False`
  - `pause()` → `self._mpv.pause = True`
  - `stop()` → `self._mpv.pause = True; self._mpv.seek(0, 'absolute')`
  - `set_clip_range(start, end)` → set `ab-loop-a`, `ab-loop-b`, seek to start
  - `clear_clip_range()` → set `ab-loop-a='no'`, `ab-loop-b='no'`
  - `shutdown()` → `self._mpv.terminate()`
- Set `hr-seek=yes` for exact A/B loop boundaries
- Convert position from seconds (float) to milliseconds (int) for `position_updated` signal

Remove all QMediaPlayer/QVideoWidget/QAudioOutput imports.

Files: `ui/video_player.py`

**2d. Handle multiple instances**

Two VideoPlayer instances (sidebar + sequence) create two independent MPV instances. To prevent audio overlap:
- When sequence playback starts, mute the sidebar player: `sidebar_player._mpv.mute = True`
- When sidebar loads a new clip, it already pauses (existing behavior)
- No other coordination needed — MPV instances are fully independent

Files: `ui/main_window.py` (add mute coordination)

**2e. Handle shutdown**

Add `shutdown()` calls to `MainWindow.closeEvent()`:
```python
def closeEvent(self, event):
    self.sequence_tab.video_player.shutdown()
    self.clip_details_sidebar.video_player.shutdown()
    # ... existing worker cleanup ...
    super().closeEvent(event)
```

Files: `ui/main_window.py`

**2f. Verify**

Run full test suite. Manually test all three consumer flows. Test on macOS specifically for embedding stability.

#### Phase 3: Enhanced Controls

Add new player features incrementally.

**3a. Frame stepping**

Add frame-step forward/backward buttons to the controls bar:
- `frame_step_forward()` → `self._mpv.command('frame-step')`
- `frame_step_backward()` → `self._mpv.command('frame-back-step')`
- In clip range mode, clamp at boundaries (don't step past clip start/end)
- Add to both sidebar and sequence tab players

UI: Two small buttons (|< and >|) between stop and slider.

Files: `ui/video_player.py`

**3b. Playback speed control**

Add a speed selector dropdown or slider:
- Options: 0.25x, 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x, 4x
- `self._mpv.speed = speed_value`
- Disabled during automated sequence playback (MainWindow timer-based mode)
- Available in both sidebar and sequence tab

UI: Compact dropdown to the right of the time label.

Files: `ui/video_player.py`, `ui/main_window.py` (disable during sequence playback)

**3c. A/B loop markers (Sequence tab only)**

Add UI for setting manual A/B loop points during direct video preview:
- "Set A" / "Set B" / "Clear" buttons or keyboard shortcuts
- Only available when clip range is NOT active
- When set, uses MPV's native `ab-loop-a`/`ab-loop-b`
- Visual markers on the slider

UI: Small A/B/Clear buttons in an expandable section below controls.

Files: `ui/video_player.py`

**3d. Improved controls styling**

Update control bar to match the app's professional theme:
- Custom-styled buttons using theme colors
- Hover states
- Active state indicators (playing = accent color, A/B loop active = different color)
- Consistent sizing per `UISizes` constants

Files: `ui/video_player.py`, `ui/theme.py` (add any new style rules)

#### Phase 4: Packaging & Distribution

Update all packaging configurations for MPV.

**4a. Requirements documentation**

Add installation instructions to README or project docs:
- macOS: `brew install mpv`
- Linux: `sudo apt install libmpv-dev` / `sudo dnf install mpv-libs-devel`
- Windows: Download from mpv.io or `choco install mpv`

Files: Project docs

**4b. PyInstaller spec — macOS**

- Add `libmpv.dylib` and its transitive dependencies to `binaries`
- Collect dependencies using `otool -L` and mpv's `dylib_unhell.py` script
- Rewrite library paths from `@rpath` to `@loader_path`
- Remove `PySide6.QtMultimedia` and `PySide6.QtMultimediaWidgets` from hiddenimports (move to excludes)
- Disable UPX for dylibs (breaks code signing on Apple Silicon)

Files: `packaging/macos/scene_ripper.spec`

**4c. PyInstaller spec — Windows**

- Add `mpv-2.dll` to `binaries`
- Download pre-built DLL from mpv.io builds (64-bit)
- Place alongside the frozen executable
- Remove QtMultimedia from hiddenimports

Files: `packaging/windows/scene_ripper.spec`

**4d. Linux AppImage**

- Remove GStreamer plugin entries from `AppImageBuilder.yml`
- Add `libmpv.so` and FFmpeg dependencies
- Test that video playback works without system GStreamer

Files: `packaging/linux/AppImageBuilder.yml`

**4e. CI workflows**

- macOS workflow: add `brew install mpv` step
- Windows workflow: add mpv DLL download step
- Linux workflow: add `sudo apt install libmpv-dev` step
- Add smoke test: launch app, load a test video, verify playback (if feasible in CI)

Files: `.github/workflows/build-macos.yml`, `.github/workflows/build-windows.yml`

**4f. Startup error handling**

If `import mpv` fails at runtime:
- Show a user-friendly QMessageBox with platform-specific install instructions
- Include a "Copy to clipboard" button for the install command
- Exit gracefully (don't show a Python traceback)

Files: `main.py` or `ui/main_window.py`

## System-Wide Impact

### Interaction Graph

```
User clicks Play → VideoPlayer.play()
    → mpv.pause = False
    → mpv property observer fires 'time-pos' on mpv thread
    → MpvSignalBridge.position_changed emits (Qt auto-queues to main thread)
    → VideoPlayer._on_position_changed slot
    → Updates slider, time label
    → Emits position_updated(int ms)
    → MainWindow._on_video_position_updated
    → TimelineWidget.set_playhead_time()
```

### Error Propagation

- MPV file load errors → `media_loaded` signal never fires; log error via MPV log handler
- MPV seek errors → silently ignored (MPV clamps to valid range)
- MPV codec errors → logged via MPV log handler; video shows black frame
- libmpv missing → caught at import time, dialog shown, app exits

### State Lifecycle Risks

- **Partial shutdown**: If `terminate()` is called from an MPV callback, it deadlocks. Mitigated by always calling from Qt main thread (closeEvent).
- **Stale observers**: If observer functions are garbage collected, position updates silently stop. Mitigated by storing references in `self._observers` list.
- **Locale corruption**: PySide6 overrides `LC_NUMERIC` on import. If another PySide6 import happens after our locale fix, it could re-corrupt. Mitigated by asserting locale in VideoPlayer.__init__.

### API Surface Parity

- **VideoPlayer public API**: Expanded (new methods/signals), no removed methods
- **Agent tools**: Updated to use new public API, same functionality
- **ClipDetailsSidebar**: Updated to use new signals, same behavior
- **MainWindow**: Updated to use new public API, same playback orchestration

## Acceptance Criteria

### Functional Requirements

- [x]Video loads and plays in ClipDetailsSidebar (clip preview with range looping)
- [x]Video loads and plays in SequenceTab (timeline sequence playback)
- [x]Agent tools (play, pause, seek) work via chat interface
- [x]Frame-accurate seeking — scrubbing the slider shows exact frames, not nearest keyframes
- [x]Frame stepping forward and backward works in both sidebar and sequence tab
- [x]Playback speed control (0.25x–4x) works
- [x]A/B loop markers work in sequence tab when clip range is not active
- [x]Clip range mode constrains playback to clip boundaries (same as current behavior)
- [x]Position slider shows clip-relative time in clip range mode
- [x]Looping works at clip boundaries (same as current behavior)
- [x]Timeline playhead syncs with video position during playback
- [x]Two VideoPlayer instances work simultaneously without crashes
- [x]Audio from only one player at a time (sidebar muted during sequence playback)

### Codec Support

- [x]H.264/MP4 plays correctly
- [x]H.265/HEVC plays correctly
- [x]ProRes plays correctly (macOS-sourced footage)
- [x]VP9/WebM plays correctly (YouTube downloads)
- [x]AV1 plays correctly (newer YouTube downloads)

### Cross-Platform

- [x]Works on macOS (Apple Silicon and Intel)
- [x]Works on Windows 10/11
- [x]Works on Ubuntu 22.04+
- [x]No GStreamer dependency on Linux (MPV brings its own codecs)

### Packaging

- [x]PyInstaller macOS bundle includes libmpv and all dependencies
- [x]PyInstaller Windows bundle includes mpv-2.dll
- [x]Linux AppImage includes libmpv
- [x]CI workflows install libmpv for builds
- [x]Startup shows clear error if libmpv is missing (pip install scenario)

### Non-Functional

- [x]Seeking latency ≤ 200ms for frame-accurate seeks
- [x]No memory leaks from property observer lifecycle
- [x]Clean shutdown — no hanging processes after app close
- [x]No locale-related numeric parsing errors on non-English systems

### Quality Gates

- [x]All existing tests pass (no regressions)
- [x]New tests for VideoPlayer public API
- [x]Manual testing of all three consumer flows on macOS
- [x]No direct `.player` access remains anywhere in codebase (grep verification)

## Alternative Approaches Considered

### OpenVideo (Web-Based) — Rejected

TypeScript library for browsers. Would require QWebEngineView (+200MB Chromium), Python↔JS bridge via QWebChannel, dual state management, AGPL license concerns. Too complex for a native desktop app. (See brainstorm for full analysis.)

### QMediaPlayer with improvements — Rejected

Could improve the existing player with workarounds (thumbnail scrubbing via FFmpeg subprocess, approximate frame seeking). But fundamental limitations remain: codec support depends on OS, no true frame accuracy, no speed/loop APIs. Throwing effort at Qt Multimedia's limitations is less productive than adopting MPV.

### OpenGL Render API from the start — Deferred

The MPV render API (`MpvRenderContext` + `QOpenGLWidget`) provides the most robust cross-platform rendering, especially on macOS. But it adds significant complexity (GL context management, frame buffer objects, proc address resolution). Starting with `wid` embedding is simpler and allows us to validate the full integration first. If macOS stability issues arise, the MpvContainer widget can be swapped to MpvGLWidget without changing any consumer code.

## Dependencies & Prerequisites

### System Dependencies

| Dependency | Version | Platform | Purpose |
|------------|---------|----------|---------|
| libmpv | >= 0.36 | All | Video playback engine |
| python-mpv | >= 1.0.8 | All (pip) | Python bindings for libmpv |

### Existing Dependencies Removed

| Dependency | Notes |
|------------|-------|
| PySide6.QtMultimedia | No longer used for playback (still used if other parts need it) |
| PySide6.QtMultimediaWidgets | QVideoWidget removed |
| GStreamer (Linux) | No longer needed — MPV bundles FFmpeg |

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| macOS `wid` embedding instability | Medium | High | Phase 1 spike test on macOS. Fallback: swap MpvContainer → MpvGLWidget (contained change) |
| libmpv not installed by users | High | Medium | Startup dialog with platform-specific install instructions |
| Property observer GC causing silent signal drops | Medium | High | Store all observer references in `self._observers` list; add integration test |
| Locale corruption causing numeric parsing errors | Low | High | Set `LC_NUMERIC='C'` in main.py; assert in VideoPlayer.__init__ |
| PyInstaller bundling complexity (macOS dylib chain) | Medium | Medium | Use mpv's `dylib_unhell.py` tool; test bundle in CI |
| Audio overlap from two MPV instances | Low | Low | Mute sidebar during sequence playback |
| Frame stepping past clip boundary | Low | Low | Clamp at boundaries in clip range mode |

## Testing Strategy

### Unit Tests (New)

- `tests/test_video_player_api.py` — Test the public API contract with a mock MPV backend
  - `test_load_video_emits_media_loaded`
  - `test_set_clip_range_sets_ab_loop`
  - `test_clear_clip_range_clears_ab_loop`
  - `test_position_updated_emits_milliseconds`
  - `test_play_pause_stop_state`
  - `test_frame_step_clamps_at_clip_boundary`
  - `test_speed_property`
  - `test_shutdown_terminates_mpv`

### Integration Tests

- `tests/test_video_player_integration.py` — Test with real MPV instance (requires libmpv)
  - `test_load_and_seek_real_video`
  - `test_clip_range_loop`
  - `test_two_instances_independent`

### Manual Test Protocol

Before merge, verify on macOS:
1. Open app, load a source video → plays in sidebar
2. Run scene detection → clip preview works with range looping
3. Switch to Sequence tab → timeline playback works
4. Seek via slider → frame-accurate (visible improvement over QMediaPlayer)
5. Frame step forward/backward → shows individual frames
6. Change speed to 2x → playback doubles, audio pitch shifts
7. Close app → exits cleanly, no hanging process
8. Repeat on Windows and Linux

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-02-27-mpv-video-player-brainstorm.md](docs/brainstorms/2026-02-27-mpv-video-player-brainstorm.md) — Key decisions: MPV over OpenVideo, clean swap no fallback, enhanced player scope, cross-platform, preserve public API

### Internal References

- Current video player: `ui/video_player.py`
- Clip details sidebar (consumer): `ui/clip_details_sidebar.py`
- Sequence tab (consumer): `ui/tabs/sequence_tab.py`
- Agent tools (consumer): `core/chat_tools.py`
- Main window (orchestrator): `ui/main_window.py`
- Styled slider: `ui/widgets/styled_slider.py`
- Theme system: `ui/theme.py`
- PyInstaller macOS spec: `packaging/macos/scene_ripper.spec`
- PyInstaller Windows spec: `packaging/windows/scene_ripper.spec`
- Linux AppImage config: `packaging/linux/AppImageBuilder.yml`
- Linux Qt Multimedia / GStreamer learnings: `docs/solutions/deployment/linux-pyside6-distribution-packaging.md`

### External References

- [python-mpv GitHub](https://github.com/jaseg/python-mpv) — Python bindings
- [python-mpv PyPI v1.0.8](https://pypi.org/project/mpv/) — Package page
- [mpv manual](https://mpv.io/manual/stable/) — Property and command reference
- [mpv-examples Qt embedding (C++)](https://github.com/mpv-player/mpv-examples/blob/master/libmpv/qt/qtexample.cpp) — Reference implementation
- [mpv-examples Qt OpenGL (C++)](https://github.com/mpv-player/mpv-examples/blob/master/libmpv/qt_opengl/mpvwidget.cpp) — Render API reference
- [PySide6 keybinding issue #200](https://github.com/jaseg/python-mpv/issues/200) — Keyboard events in PySide6
- [mpv AB-Loop seeking issue #9169](https://github.com/mpv-player/mpv/issues/9169) — Need `hr-seek=yes` for exact loops
- [mpv macOS stability (mpv-examples README)](https://github.com/mpv-player/mpv-examples/blob/master/libmpv/README.md) — macOS notes
