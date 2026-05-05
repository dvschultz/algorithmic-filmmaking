---
title: Cut tab hides newly detected clips when thumbnail generation fails
date: 2026-05-05
category: ui-bugs
module: ui
problem_type: runtime_error
component: cut_tab
severity: high
symptoms:
  - "Project has clips from multiple sources, but Cut tab only shows existing source clips"
  - "Invalid duration for option ss: 1514513/9000"
  - "ThumbnailWorker logs clip source_id different from fallback source id"
root_cause: thumbnail_ready_required_for_clip_card
resolution_type: ui_state_sync
tags:
  - cut-tab
  - thumbnails
  - ffmpeg
  - pyscenedetect
  - source-id
---

# Cut Tab Hides Newly Detected Clips When Thumbnail Generation Fails

## Problem

After cutting a new video in a project that already contains clips, the project model can contain the new clips, but the Cut tab still only shows the previous footage. Logs show all clips in the project, followed by FFmpeg thumbnail failures such as:

```text
Invalid duration for option ss: 1514513/9000
```

## Root Cause

The Cut tab browser only received new clip cards from `_on_thumbnail_ready`. If thumbnail generation failed before emitting that signal, valid detected clips remained in the project model but never appeared in the browser.

The FFmpeg failure happened because fractional/rational timestamps reached the `-ss` argument as Python fraction strings. FFmpeg expects decimal seconds, not values like `1514513/9000`.

## Solution

- Add detected clip cards to the Cut tab immediately after detection completes, before thumbnail generation starts.
- Keep thumbnail generation as an update path: when a thumbnail is ready, update the existing card instead of relying on thumbnail readiness to create it.
- Normalize thumbnail timestamps to finite decimal seconds before building FFmpeg commands.
- Normalize PySceneDetect frame rates to plain floats when creating `Source` metadata.

## Verification

Run:

```bash
python -m pytest tests/test_color_profile.py::TestSceneDetectorGrayscaleIntegration::test_progress_frame_callback_accepts_frame_timecode tests/test_color_profile.py::TestSceneDetectorGrayscaleIntegration::test_scene_detect_fps_normalized_to_float tests/test_thumbnail_generation.py tests/test_cut_tab_detection_visibility.py tests/test_clip_browser_selection.py::test_virtual_clip_browser_realizes_only_visible_window -v
ruff check core/thumbnail.py core/scene_detect.py ui/main_window.py ui/tabs/cut_tab.py tests/test_thumbnail_generation.py tests/test_cut_tab_detection_visibility.py tests/test_color_profile.py
```
