---
title: "feat: Linux App Distribution"
type: feat
date: 2026-01-24
priority: medium
---

# feat: Linux App Distribution

## Overview

Create a distributable Linux version of Scene Ripper. The core Python/Qt application is already cross-platform compatible; this plan focuses on packaging, testing, and distribution.

## Current State Analysis

### Already Cross-Platform

| Component | Implementation | Linux Status |
|-----------|---------------|--------------|
| UI Framework | PySide6 (Qt 6) | ✅ Native support |
| Scene Detection | PySceneDetect + OpenCV | ✅ Works |
| Video Processing | FFmpeg subprocess | ✅ Uses `shutil.which()` |
| Path Handling | `pathlib.Path` | ✅ Portable |
| Settings Storage | QSettings | ✅ Auto-uses `~/.config/` |
| ML Models | torch + transformers | ✅ Linux wheels available |

### Needs Work

| Component | Issue | Effort |
|-----------|-------|--------|
| Qt Multimedia | Requires GStreamer backend | Small |
| Default Paths | `~/Movies` doesn't exist on Linux | Small |
| Packaging | No distribution setup | Medium |
| CI/CD | No Linux build pipeline | Medium |
| FFmpeg | Bundle vs. require system install | Decision |

## Proposed Solution

### Phase 1: Linux Compatibility Fixes

Address minor code issues for Linux:

#### 1.1 Fix Default Paths

```python
# core/settings.py - Current
download_dir: Path = field(
    default_factory=lambda: Path.home() / "Movies" / "Scene Ripper Downloads"
)

# Proposed - XDG-compliant
def _get_default_download_dir() -> Path:
    """Get platform-appropriate download directory."""
    import sys
    if sys.platform == "linux":
        # Use XDG_VIDEOS_DIR or fallback to ~/Videos
        xdg_videos = os.environ.get("XDG_VIDEOS_DIR")
        if xdg_videos:
            return Path(xdg_videos) / "Scene Ripper Downloads"
        return Path.home() / "Videos" / "Scene Ripper Downloads"
    return Path.home() / "Movies" / "Scene Ripper Downloads"
```

#### 1.2 Document GStreamer Dependencies

Qt Multimedia on Linux requires GStreamer. Add to README:

```markdown
### Linux Dependencies

```bash
# Ubuntu/Debian
sudo apt install ffmpeg gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Fedora
sudo dnf install ffmpeg gstreamer1-plugins-good gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly-free gstreamer1-libav

# Arch
sudo pacman -S ffmpeg gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```
```

#### 1.3 Verify Video Playback

Test QMediaPlayer on Linux. If issues persist, consider:
- VLC backend via python-vlc (more codec support)
- MPV backend via mpv Python bindings
- Fallback to FFplay for playback

### Phase 2: Choose Distribution Method

#### Option Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **AppImage** | Single file, no install, portable | Large file (~500MB+), no auto-update | Quick distribution, testing |
| **Flatpak** | Sandboxed, auto-updates, Flathub | Complex manifest, sandbox permissions | Production distribution |
| **pip install** | Simple, familiar to devs | User installs FFmpeg, no desktop icon | Developers, CI |
| **.deb/.rpm** | Native feel, apt/dnf install | Distro-specific, maintenance burden | Enterprise/specific distros |

#### Recommended: AppImage for MVP, Flatpak for Production

**Phase 2a: AppImage (Quick Win)**
- Use `appimage-builder` or `linuxdeploy`
- Bundle Python + dependencies
- Ship without FFmpeg (require system install initially)

**Phase 2b: Flatpak (Later)**
- Create Flathub manifest
- Bundle FFmpeg
- Handle sandbox permissions for file access

### Phase 3: AppImage Packaging

#### 3.1 Create AppImage Build Script

```yaml
# AppImageBuilder.yml
version: 1
script:
  - pip install --target=AppDir/usr/lib/python3.11/site-packages -r requirements.txt

AppDir:
  path: ./AppDir
  app_info:
    id: com.algorithmic-filmmaking.scene-ripper
    name: Scene Ripper
    icon: scene-ripper
    version: 0.1.0
    exec: usr/bin/python3
    exec_args: "$APPDIR/usr/src/main.py $@"

  apt:
    arch: amd64
    sources:
      - sourceline: deb http://archive.ubuntu.com/ubuntu jammy main universe
    include:
      - python3
      - python3-pip
      - libgl1
      - libegl1
      # GStreamer for Qt Multimedia
      - gstreamer1.0-plugins-good
      - gstreamer1.0-plugins-bad
      - gstreamer1.0-libav

  files:
    include:
      - usr/src/

  runtime:
    env:
      PYTHONPATH: "$APPDIR/usr/lib/python3.11/site-packages"

AppImage:
  arch: x86_64
  update-information: guess
```

#### 3.2 Create Desktop Entry

```ini
# scene-ripper.desktop
[Desktop Entry]
Type=Application
Name=Scene Ripper
Comment=Automatic scene detection for video collage
Exec=scene-ripper %F
Icon=scene-ripper
Categories=AudioVideo;Video;
MimeType=video/mp4;video/x-matroska;video/webm;video/quicktime;
Terminal=false
```

#### 3.3 Build Script

```bash
#!/bin/bash
# build-appimage.sh

set -e

# Clean previous build
rm -rf AppDir *.AppImage

# Create AppDir structure
mkdir -p AppDir/usr/src
mkdir -p AppDir/usr/share/applications
mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps

# Copy source
cp -r *.py core/ models/ ui/ AppDir/usr/src/

# Copy desktop entry and icon
cp scene-ripper.desktop AppDir/usr/share/applications/
cp assets/icon.png AppDir/usr/share/icons/hicolor/256x256/apps/scene-ripper.png

# Build AppImage
appimage-builder --recipe AppImageBuilder.yml
```

### Phase 4: Testing Infrastructure

#### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/linux-build.yml
name: Linux Build

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Install system dependencies
        run: |
          sudo apt update
          sudo apt install -y ffmpeg \
            gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad \
            gstreamer1.0-libav \
            libxcb-xinerama0 \
            libegl1

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run import test
        run: python -c "from ui.main_window import MainWindow; print('Import OK')"

      - name: Run tests
        run: python -m pytest tests/ -v
        if: hashFiles('tests/') != ''

  build-appimage:
    runs-on: ubuntu-22.04
    needs: test-linux
    steps:
      - uses: actions/checkout@v4

      - name: Build AppImage
        run: |
          wget -O appimage-builder https://github.com/AppImageCrafters/appimage-builder/releases/download/v1.1.0/appimage-builder-1.1.0-x86_64.AppImage
          chmod +x appimage-builder
          ./appimage-builder --recipe AppImageBuilder.yml

      - name: Upload AppImage
        uses: actions/upload-artifact@v4
        with:
          name: Scene-Ripper-x86_64.AppImage
          path: "*.AppImage"
```

#### 4.2 Test Matrix

| Distro | Version | Test Type |
|--------|---------|-----------|
| Ubuntu | 22.04 LTS | Full (CI) |
| Ubuntu | 24.04 LTS | Full (CI) |
| Fedora | 39+ | Manual |
| Arch | Rolling | Manual |

### Phase 5: Documentation

Update README with Linux installation:

```markdown
## Linux Installation

### Option 1: AppImage (Recommended)

1. Download `Scene-Ripper-x86_64.AppImage` from Releases
2. Make executable: `chmod +x Scene-Ripper-x86_64.AppImage`
3. Run: `./Scene-Ripper-x86_64.AppImage`

### Option 2: From Source

```bash
# Install system dependencies
sudo apt install ffmpeg gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-libav python3-pip

# Clone and install
git clone https://github.com/your-repo/scene-ripper
cd scene-ripper
pip install -r requirements.txt
python main.py
```
```

## Technical Considerations

### FFmpeg Bundling Decision

| Approach | Size Impact | Maintenance | User Experience |
|----------|-------------|-------------|-----------------|
| Require system FFmpeg | Smaller (~200MB) | Easy | User must install |
| Bundle FFmpeg | Larger (~400MB) | Harder updates | Just works |

**Recommendation**: Start with system FFmpeg requirement. Bundle later if user friction is high.

### Qt Platform Plugins

Linux Qt apps need platform plugins. Ensure these are bundled:
- `libqxcb.so` - X11 support
- `libqwayland*.so` - Wayland support (optional but recommended)

### Wayland vs X11

Scene Ripper should work on both. Test:
- `QT_QPA_PLATFORM=xcb ./scene-ripper` - Force X11
- `QT_QPA_PLATFORM=wayland ./scene-ripper` - Force Wayland

## Acceptance Criteria

### Phase 1: Compatibility
- [x] App launches on Ubuntu 22.04 from source
- [ ] Video playback works with GStreamer
- [x] Default paths use XDG directories on Linux
- [x] Settings persist in `~/.config/`

### Phase 2: Packaging
- [x] AppImage builds successfully (config created)
- [ ] AppImage runs on Ubuntu 22.04 (needs Linux testing)
- [ ] AppImage runs on Fedora 39 (needs Linux testing)
- [x] Desktop file shows in app menu

### Phase 3: CI/CD
- [x] GitHub Actions builds AppImage on push
- [x] AppImage artifact available for download
- [x] Import test passes on CI

### Phase 4: Documentation
- [x] README has Linux installation section
- [x] Dependencies documented per distro
- [ ] Troubleshooting section for common issues

## Dependencies & Risks

### Dependencies

- `appimage-builder` or `linuxdeploy` for packaging
- Ubuntu 22.04+ for CI (GStreamer compatibility)
- GitHub Actions (free for open source)

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GStreamer codec issues | Medium | High | Test with various video formats, document requirements |
| AppImage size too large | Medium | Low | Optimize dependencies, consider Flatpak later |
| Wayland compatibility | Low | Medium | Test on both, default to X11 if issues |
| GPU acceleration issues | Medium | Medium | Ensure Mesa/GPU drivers documented |

## Implementation Order

```
Phase 1 (Small fixes)     → 1-2 hours
├── Fix default paths
├── Document GStreamer deps
└── Test video playback

Phase 2 (Packaging setup) → 4-6 hours
├── Create AppImageBuilder.yml
├── Create desktop entry
├── Create build script
└── Test AppImage locally

Phase 3 (CI/CD)           → 2-3 hours
├── Create GitHub workflow
├── Upload artifacts
└── Test on fresh Ubuntu

Phase 4 (Documentation)   → 1-2 hours
├── Update README
├── Add troubleshooting
└── Document dependencies
```

## References

### Internal References
- FFmpeg handling: `core/ffmpeg.py`
- Settings/paths: `core/settings.py`
- Video player: `ui/video_player.py`

### External References
- [AppImage Documentation](https://docs.appimage.org/)
- [appimage-builder Guide](https://appimage-builder.readthedocs.io/)
- [Qt for Linux Deployment](https://doc.qt.io/qt-6/linux-deployment.html)
- [GStreamer Qt Integration](https://gstreamer.freedesktop.org/documentation/qt6d3d11/)
- [Flatpak Python Guide](https://docs.flatpak.org/en/latest/python.html)

---

*Generated: 2026-01-24*
