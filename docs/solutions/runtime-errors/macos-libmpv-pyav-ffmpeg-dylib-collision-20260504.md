---
module: Scene Ripper
date: 2026-05-04
problem_type: runtime_error
component: macos_runtime
symptoms:
  - "Class AVFFrameReceiver is implemented in both av/.dylibs/libavdevice and Homebrew ffmpeg/libavdevice"
  - "Class AVFAudioReceiver is implemented in both av/.dylibs/libavdevice and Homebrew ffmpeg/libavdevice"
  - "This may cause spurious casting failures and mysterious crashes"
root_cause: duplicate_native_ffmpeg_runtime
resolution_type: code_fix
severity: high
tags: [macos, libmpv, pyav, ffmpeg, dyld, startup-imports]
---

# Troubleshooting: macOS libmpv/PyAV FFmpeg Dylib Collision

## Problem

Scene Ripper could emit Objective-C duplicate-class warnings on macOS when both Homebrew FFmpeg libraries and PyAV's bundled FFmpeg libraries were loaded into the same Python process.

The warning appeared as:

```text
objc: Class AVFFrameReceiver is implemented in both .../site-packages/av/.dylibs/libavdevice... and /opt/homebrew/Cellar/ffmpeg/.../libavdevice...
objc: Class AVFAudioReceiver is implemented in both .../site-packages/av/.dylibs/libavdevice... and /opt/homebrew/Cellar/ffmpeg/.../libavdevice...
```

## Root Cause

`main.py` prepended `/opt/homebrew/lib` or `/usr/local/lib` to `DYLD_LIBRARY_PATH` so `python-mpv` could find `libmpv.dylib`. That made every Homebrew FFmpeg dylib visible to the process.

At the same time, startup package imports were too eager:

- `ui/__init__.py` imported `ui.main_window`.
- `core/__init__.py` imported `core.scene_detect`.
- `scenedetect.backends` imported its PyAV backend, which imported `av`.

That meant ordinary submodule imports could load PyAV's bundled FFmpeg, and `python-mpv` could load Homebrew FFmpeg. Loading both native runtimes defines the same AVFoundation Objective-C classes twice.

## Solution

Keep native media runtimes out of startup imports and avoid broad dynamic-library path mutation:

1. Remove the global macOS `DYLD_LIBRARY_PATH` mutation from `main.py`.
2. Make `ui.video_player` import `python-mpv` lazily when playback initializes.
3. Make `ui.video_player` point `ctypes.util.find_library("mpv")` at one exact `libmpv.dylib` path instead of exposing all of Homebrew's library directory.
4. Convert `ui/__init__.py` and `core/__init__.py` package exports to lazy `__getattr__` lookups.
5. Prevent PySceneDetect from importing its PyAV backend during `core.scene_detect` import, because Scene Ripper uses the OpenCV backend explicitly.
6. Change `is_faster_whisper_available()` to use `importlib.util.find_spec()` so availability checks do not import `faster_whisper` or PyAV.

## Verification

Use import-boundary tests to ensure startup does not load `mpv`, `av`, or `scenedetect` unnecessarily:

```bash
python -m pytest tests/test_video_player_runtime.py tests/test_transcription_runtime_imports.py -v
```

Useful manual check:

```bash
python -c "import sys; import ui.video_player; import core.scene_detect; print('mpv' in sys.modules, 'av' in sys.modules)"
```

Expected output is `False False`.
