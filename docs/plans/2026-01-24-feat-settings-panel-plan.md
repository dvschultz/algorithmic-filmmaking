---
title: "feat: Add Settings Panel"
type: feat
date: 2026-01-24
---

# feat: Add Settings Panel

## Overview

Add a persistent settings panel to Scene Ripper accessible via File > Settings (or Preferences on macOS). The panel allows users to configure cache/storage paths, detection defaults, and export settings. Settings persist between sessions using Qt's `QSettings` for platform-native storage.

## Problem Statement / Motivation

Currently, Scene Ripper has hardcoded values for:
- Thumbnail cache location (`~/.cache/scene-ripper/thumbnails`)
- Download directory (`~/Movies/Scene Ripper Downloads`)
- Detection sensitivity (only adjustable per-session via slider)
- Export parameters (fixed CRF, preset, codec)

Users cannot:
- Redirect cache to a faster drive or external storage
- Set default detection sensitivity
- Configure export quality presets
- Disable automatic analysis features to speed up processing

## Proposed Solution

Create a `SettingsDialog` (modal QDialog) with three tabs:
1. **Paths & Storage** - Directory locations
2. **Detection** - Scene detection defaults
3. **Export** - Video export quality and format

Use `QSettings` for persistence and a `Settings` dataclass as the in-memory representation.

## Technical Approach

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MainWindow    │────▶│  SettingsDialog  │────▶│    QSettings    │
│                 │     │                  │     │  (persistence)  │
│  File > Settings│     │  - Tabs UI       │     └─────────────────┘
└─────────────────┘     │  - Validation    │              │
                        │  - Apply/Cancel  │              ▼
                        └──────────────────┘     ~/.config/Algorithmic
                                 │                Filmmaking/Scene Ripper.conf
                                 ▼
                        ┌──────────────────┐
                        │ Settings(dataclass)│
                        │                  │
                        │ - paths          │
                        │ - detection      │
                        │ - export         │
                        └──────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
    ThumbnailGenerator    VideoDownloader    SequenceExporter
      (cache_dir)         (download_dir)     (ExportConfig)
```

### Files to Create

| File | Purpose |
|------|---------|
| `core/settings.py` | `Settings` dataclass, `load_settings()`, `save_settings()` |
| `ui/settings_dialog.py` | `SettingsDialog` QDialog with tabs |

### Files to Modify

| File | Changes |
|------|---------|
| `ui/main_window.py` | Add File > Settings menu, load settings on startup, pass to workers |
| `core/thumbnail.py` | Accept `cache_dir` from settings |
| `core/downloader.py` | Accept `download_dir` from settings |
| `core/sequence_export.py` | Use settings for default ExportConfig |
| `core/scene_detect.py` | Use settings for default DetectionConfig |
| `main.py` | Initialize settings on app start |

### Settings Model

```python
# core/settings.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QSettings

@dataclass
class Settings:
    # Paths
    thumbnail_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "scene-ripper" / "thumbnails")
    download_dir: Path = field(default_factory=lambda: Path.home() / "Movies" / "Scene Ripper Downloads")
    export_dir: Path = field(default_factory=lambda: Path.home() / "Movies")

    # Detection
    default_sensitivity: float = 3.0
    min_scene_length_seconds: float = 0.5
    auto_analyze_colors: bool = True
    auto_classify_shots: bool = True

    # Export
    export_quality: str = "medium"  # high, medium, low
    export_resolution: str = "original"  # original, 1080p, 720p, 480p
    export_fps: str = "original"  # original, 24, 30, 60
```

### Quality Presets

| Preset | CRF | FFmpeg Preset | Approx Bitrate |
|--------|-----|---------------|----------------|
| High | 18 | slow | ~10 Mbps |
| Medium | 23 | medium | ~5 Mbps |
| Low | 28 | fast | ~2 Mbps |

### Resolution Options

| Option | Max Width | Max Height | Behavior |
|--------|-----------|------------|----------|
| Original | source | source | No scaling |
| 1080p | 1920 | 1080 | Scale to fit, preserve aspect ratio |
| 720p | 1280 | 720 | Scale to fit, preserve aspect ratio |
| 480p | 854 | 480 | Scale to fit, preserve aspect ratio |

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Settings                                              [x]   │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────┬────────────┬──────────┐                        │
│ │  Paths  │  Detection │  Export  │                        │
│ └─────────┴────────────┴──────────┘                        │
│                                                             │
│ ┌─ Paths & Storage ──────────────────────────────────────┐ │
│ │                                                         │ │
│ │ Thumbnail Cache:  [~/.cache/scene-ripper/thumb...][...] │ │
│ │                                                         │ │
│ │ Download Folder:  [~/Movies/Scene Ripper Downl...][...] │ │
│ │                                                         │ │
│ │ Export Folder:    [~/Movies                      ][...] │ │
│ │                                                         │ │
│ │ Cache size: 245 MB        [Clear Cache]                 │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│         [Restore Defaults]            [Cancel]  [OK]        │
└─────────────────────────────────────────────────────────────┘
```

### Detection Tab

```
┌─ Detection Defaults ─────────────────────────────────────┐
│                                                          │
│ Default Sensitivity:  ──●──────── 3.0                    │
│                       (1.0 = more scenes, 10.0 = fewer)  │
│                                                          │
│ Min Scene Length:     [0.5    ] seconds                  │
│                                                          │
│ ☑ Auto-analyze colors after detection                    │
│ ☑ Auto-classify shot types after detection               │
│                                                          │
│ Note: Changes apply to future detections only            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Export Tab

```
┌─ Export Defaults ────────────────────────────────────────┐
│                                                          │
│ Quality:      [▼ Medium (balanced)    ]                  │
│               • High - Best quality, larger files        │
│               • Medium - Balanced (recommended)          │
│               • Low - Smaller files, faster encoding     │
│                                                          │
│ Resolution:   [▼ Original             ]                  │
│               • Original, 1080p, 720p, 480p              │
│                                                          │
│ Frame Rate:   [▼ Original             ]                  │
│               • Original, 24 fps, 30 fps, 60 fps         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Acceptance Criteria

### Functional

- [x] File > Settings opens modal settings dialog (Cmd+, on macOS)
- [x] Settings persist to disk and restore on app relaunch
- [x] Path settings support Browse button with folder picker
- [x] Invalid paths show validation error on Apply (red border, message)
- [x] Non-existent directories are auto-created on Apply
- [x] "Restore Defaults" resets all fields to defaults (requires OK to persist)
- [x] "Clear Cache" shows confirmation, deletes thumbnails, updates size display
- [x] Detection settings update the main window slider default
- [x] Export settings apply to Export Selected, Export All, and Export Sequence
- [x] Changes only apply on OK/Apply, Cancel discards

### Edge Cases

- [ ] Opening settings while background operation running: affected settings disabled
- [ ] Path changed to network/cloud folder: accept with tooltip warning
- [x] Settings file corrupted: load defaults, warn user
- [x] Fresh install (no settings file): use defaults seamlessly

## Dependencies & Risks

### Dependencies
- No new packages required - uses PySide6 QSettings (already available)
- QSettings already configured via `setApplicationName`/`setOrganizationName` in main.py

### Risks
| Risk | Mitigation |
|------|------------|
| Cache path change orphans existing thumbnails | Show warning, offer to move/delete old cache |
| Settings change during operation causes issues | Disable affected settings while workers running |
| Platform differences in QSettings storage | Document config file locations for troubleshooting |

## Implementation Phases

### Phase 1: Core Settings Infrastructure
- Create `core/settings.py` with dataclass and load/save functions
- Implement `QSettings` integration
- Add unit tests for settings serialization

### Phase 2: Settings Dialog UI
- Create `ui/settings_dialog.py` with tabs
- Implement validation and error display
- Add Browse buttons for paths
- Wire OK/Cancel/Restore Defaults

### Phase 3: Integration
- Add File > Settings menu to `main_window.py`
- Modify `ThumbnailGenerator`, `VideoDownloader` to accept paths
- Update workers to use settings defaults
- Apply export settings to all export operations

### Phase 4: Polish
- Add "Clear Cache" with size calculation
- Add keyboard shortcut (Cmd+, / Ctrl+,)
- Disable settings for active operations
- Add tooltips to all controls

## Success Metrics

- Settings persist across app restarts
- All three directory settings can be changed and used
- Export quality presets produce expected file sizes
- No crashes or data loss when changing settings

## References & Research

### Internal References
- Existing config pattern: `core/scene_detect.py:17-28` (DetectionConfig)
- Export config pattern: `core/sequence_export.py:20-32` (ExportConfig)
- Thumbnail cache: `core/thumbnail.py:19`
- Download dir: `core/downloader.py:43`
- QSettings setup: `main.py:8-9` (setApplicationName/setOrganizationName)
- Dialog patterns: `ui/main_window.py:447-454` (QFileDialog usage)
- Combo box pattern: `ui/clip_browser.py:268-273`

### Documented Learnings
- `docs/solutions/qthread-destroyed-duplicate-signal-delivery-20260124.md` - Use guards for signal handlers
- `docs/solutions/ffmpeg-path-escaping-20260124.md` - Sanitize paths derived from user input

### Platform Config File Locations
- **macOS**: `~/Library/Preferences/com.Algorithmic Filmmaking.Scene Ripper.plist`
- **Linux**: `~/.config/Algorithmic Filmmaking/Scene Ripper.conf`
- **Windows**: Registry `HKEY_CURRENT_USER\Software\Algorithmic Filmmaking\Scene Ripper`
