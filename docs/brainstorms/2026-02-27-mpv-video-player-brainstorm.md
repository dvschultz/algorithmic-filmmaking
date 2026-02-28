# MPV-Based Video Player Integration

**Date**: 2026-02-27
**Status**: Brainstorm
**Trigger**: Current Qt Multimedia player is slow, limited codec support, no frame-accurate seeking, feels basic

## What We're Building

Replace the current QMediaPlayer-based video player with an MPV-powered player using python-mpv (libmpv). Beyond a drop-in engine swap, enhance the player with frame-accurate scrubbing, thumbnail-on-hover timeline, A/B looping, and playback speed controls.

### Current State

- `ui/video_player.py` uses PySide6 `QMediaPlayer` + `QVideoWidget`
- Player is embedded in 3 places: clip details sidebar, sequence tab, and agent tools
- Public API: `load_video()`, `seek_to()`, `play_range()`, `set_clip_range()`, `clear_clip_range()`
- Signals: `position_updated` (emits ms position to MainWindow/timeline/agents)
- Pain points: sluggish seeking, poor codec support, no frame-accurate preview, basic look and feel

### Target State

- MPV-backed player with the same integration points
- Frame-accurate seeking and scrubbing
- Near-universal codec/container support (H.264, H.265, ProRes, VP9, AV1, etc.)
- Hardware-accelerated decoding
- Enhanced controls: speed adjustment, frame stepping, A/B loop markers
- Thumbnail-on-hover timeline scrubber
- Professional look and feel matching the rest of the app's theme

## Why This Approach

### Why MPV over OpenVideo (web-based)

OpenVideo (https://github.com/openvideodev/openvideo) was considered but rejected because:
- It's a TypeScript/WebCodecs library designed for browsers
- Would require embedding Chromium via QWebEngineView (+200MB dependency)
- Python-JS bridge via QWebChannel adds latency and complexity
- Two separate state systems to keep in sync
- AGPL license implications
- Debugging spans two runtimes

### Why MPV over QMediaPlayer

- **Codec support**: MPV uses FFmpeg internally, plays virtually everything
- **Seeking**: Frame-accurate seeking is a core MPV feature
- **Hardware acceleration**: Native support for VA-API, VDPAU, VideoToolbox, D3D11
- **Maturity**: Powers professional tools, battle-tested at scale
- **Customization**: Extensive property/option system for fine-tuning playback
- **Size**: ~5MB dependency vs 200MB for a web engine

## Key Decisions

1. **Clean swap, no fallback** - Remove QMediaPlayer entirely. Require libmpv as a dependency. One code path to maintain.

2. **Enhanced player scope** - Not just an engine swap. Add frame-accurate scrubbing, thumbnail timeline, A/B looping, playback speed controls, and improved controls UI.

3. **Full cross-platform** - macOS + Windows + Linux from the start.

4. **Dual distribution** - Support both pip install (libmpv as documented prerequisite) and PyInstaller bundle (libmpv bundled).

5. **Preserve public API** - Keep `load_video()`, `seek_to()`, `play_range()`, `set_clip_range()`, `position_updated` signal so all 3 integration points (sidebar, sequence tab, agents) work without changes to calling code.

## Technical Considerations

### MPV Embedding in Qt

python-mpv embeds into Qt widgets using the window ID approach:
```python
container = QWidget()
container.setAttribute(Qt.WA_DontCreateNativeAncestors)
container.setAttribute(Qt.WA_NativeWindow)
mpv_player = mpv.MPV(wid=str(int(container.winId())), vo='gpu')
```

On macOS, the video output backend should be `gpu` or `libmpv` for best results. On Linux, `x11` or `gpu`. On Windows, `gpu` or `d3d11`.

### Frame Rate Consideration

When MPV renders into the default framebuffer, it syncs video frames with audio, which can cap the UI frame rate at the video's FPS (24/30). Mitigation: use `vo='libmpv'` with OpenGL rendering for decoupled frame rates, or accept the cap since this is a video-focused app.

### Signal/Slot Bridge

MPV uses property observers (callbacks) instead of Qt signals. Bridge pattern:
- MPV `observe_property('time-pos', callback)` -> emit Qt `position_updated` signal
- MPV `observe_property('duration', callback)` -> emit duration signal
- Thread safety: MPV callbacks fire on MPV's thread; use `QMetaObject.invokeMethod` or `Signal.emit` (thread-safe in Qt) to cross to the GUI thread

### Platform-Specific libmpv Installation

| Platform | Dev Install | Bundle |
|----------|-------------|--------|
| macOS | `brew install mpv` | Bundle `libmpv.dylib` via PyInstaller hook |
| Linux | `apt install libmpv-dev` / `dnf install mpv-libs-devel` | AppImage with bundled lib |
| Windows | `choco install mpv` or download from mpv.io | Bundle `mpv-2.dll` via PyInstaller hook |

### New Features to Add

| Feature | MPV Property/Command | Notes |
|---------|---------------------|-------|
| Frame-accurate seeking | `seek <pos> absolute exact` | Built-in, just use `exact` flag |
| Frame stepping | `frame-step` / `frame-back-step` | Forward and backward |
| Playback speed | `speed` property (0.25x - 4x) | Real-time adjustment |
| A/B looping | `ab-loop-a`, `ab-loop-b` | Native loop points |
| Thumbnail timeline | `mpv.screenshot_raw()` or pre-generated | May need separate thumbnail pipeline |

### Files That Will Change

| File | Change |
|------|--------|
| `ui/video_player.py` | Major rewrite - MPV engine, new controls |
| `ui/clip_details_sidebar.py` | Minor - adapt to new player API (should be minimal if API preserved) |
| `ui/tabs/sequence_tab.py` | Minor - same as above |
| `core/chat_tools.py` | Minor - agent tools should work via same API |
| `ui/main_window.py` | Minor - signal connections |
| `requirements.txt` | Add `python-mpv` |
| `ui/widgets/` | New widgets for enhanced controls (speed slider, frame step buttons, thumbnail scrubber) |
| PyInstaller specs | Add libmpv bundling hooks |

## Open Questions

_None - all major questions resolved during brainstorming._

## Resolved Questions

1. **Web vs native?** - Native (MPV). Web embed adds too much complexity for a desktop app.
2. **Scope?** - Enhanced player, not just engine swap. Add frame-accurate controls, speed, A/B loop.
3. **Fallback?** - No. Clean swap, remove QMediaPlayer entirely.
4. **Platforms?** - All three: macOS, Windows, Linux.
5. **Distribution?** - Both pip install and PyInstaller bundle.
