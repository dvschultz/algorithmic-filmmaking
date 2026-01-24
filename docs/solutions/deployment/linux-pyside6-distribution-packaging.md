---
title: "Linux Distribution Packaging for PySide6/Qt Applications"
category: deployment
tags:
  - pyside6
  - qt
  - linux
  - appimage
  - cross-platform
  - github-actions
  - xdg-compliance
  - gstreamer
  - packaging
module: core/settings.py, packaging/linux/
symptoms:
  - "Application fails to find default directories on Linux"
  - "Video playback not working on Linux (Qt Multimedia backend missing)"
  - "No single-file distribution for Linux users"
  - "~/Movies default path incorrect for Linux (should be ~/Videos)"
  - "Missing GStreamer plugins cause silent audio/video failures"
solved_date: 2026-01-24
---

# Linux Distribution Packaging for PySide6/Qt Applications

## Problem

Making a Python/PySide6 desktop application distributable on Linux with proper platform conventions, multimedia support, and portable packaging.

## Root Cause

1. **macOS-centric defaults** - Paths like `~/Movies` don't exist on Linux
2. **Qt Multimedia backend** - Linux requires GStreamer (not bundled with PySide6)
3. **Distribution fragmentation** - Different package names across distros
4. **No standard Python app format** - Unlike macOS `.app` or Windows `.exe`

## Solution

### 1. XDG-Compliant Path Handling

Detect platform and use appropriate directory conventions:

```python
# core/settings.py
import os
import sys
from pathlib import Path

def _get_videos_dir() -> Path:
    """Get platform-appropriate videos directory."""
    if sys.platform == "linux":
        xdg_videos = os.environ.get("XDG_VIDEOS_DIR")
        if xdg_videos:
            return Path(xdg_videos)
        return Path.home() / "Videos"
    elif sys.platform == "darwin":
        return Path.home() / "Movies"
    else:
        return Path.home() / "Videos"
```

### 2. Document GStreamer Dependencies

Qt Multimedia on Linux requires GStreamer. Document per-distro installation:

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

**Fedora:**
```bash
sudo dnf install ffmpeg gstreamer1-plugins-good gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly-free gstreamer1-libav
```

**Arch:**
```bash
sudo pacman -S ffmpeg gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

### 3. AppImage Packaging

Create portable single-file distribution:

```yaml
# packaging/linux/AppImageBuilder.yml
version: 1
AppDir:
  path: ./AppDir
  app_info:
    id: com.example.scene-ripper
    name: Scene Ripper
    icon: scene-ripper
    version: 1.0.0
    exec: usr/bin/python3
    exec_args: "$APPDIR/usr/src/main.py $@"

  apt:
    arch: amd64
    sources:
      - sourceline: deb http://archive.ubuntu.com/ubuntu jammy main universe
    include:
      - python3
      - gstreamer1.0-plugins-good
      - gstreamer1.0-plugins-bad
      - gstreamer1.0-libav

  runtime:
    env:
      PYTHONPATH: "$APPDIR/usr/lib/python3/site-packages"

AppImage:
  arch: x86_64
```

### 4. GitHub Actions CI

Test on Linux automatically:

```yaml
# .github/workflows/linux-build.yml
name: Linux Build

on: [push, pull_request]

jobs:
  test-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt update
          sudo apt install -y ffmpeg gstreamer1.0-plugins-good \
            gstreamer1.0-libav libxcb-cursor0 libegl1

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install -r requirements.txt

      - name: Test imports
        run: python -c "from ui.main_window import MainWindow; print('OK')"
        env:
          QT_QPA_PLATFORM: offscreen
```

## Key Insights

1. **XDG compliance is mandatory** - Linux users expect standard paths
2. **GStreamer has no fallback** - Qt Multimedia silently fails without it
3. **Package names vary by distro** - Same plugin, different names
4. **CI needs `QT_QPA_PLATFORM=offscreen`** - No display server in GitHub Actions
5. **`libxcb-cursor0` often missing** - Ubuntu 22.04+ moved it to a separate package

## Prevention Checklist

- [ ] Use `sys.platform` checks for platform-specific paths
- [ ] Respect XDG environment variables on Linux
- [ ] Document GStreamer requirements prominently
- [ ] Test on multiple distros (Ubuntu, Fedora, Arch at minimum)
- [ ] Add Linux CI workflow from the start
- [ ] Consider AppImage for portable distribution

## Files Modified

- `core/settings.py` - Platform-aware path detection
- `README.md` - Linux installation documentation
- `packaging/linux/AppImageBuilder.yml` - AppImage recipe
- `packaging/linux/scene-ripper.desktop` - Desktop entry
- `packaging/linux/build-appimage.sh` - Build script
- `.github/workflows/linux-build.yml` - CI workflow

## Related Documentation

- [docs/plans/2026-01-24-feat-linux-app-distribution-plan.md](../plans/2026-01-24-feat-linux-app-distribution-plan.md) - Full implementation plan
- [Qt for Linux Deployment](https://doc.qt.io/qt-6/linux-deployment.html)
- [AppImage Documentation](https://docs.appimage.org/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/)
