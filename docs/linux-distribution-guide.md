# Linux Distribution Guide for PySide6/Qt Applications

## Prevention Checklist for Cross-Platform Projects

### 1. Project Setup Phase

#### XDG Path Compliance
- [ ] Use `XDG_CONFIG_HOME` for configuration (`~/.config/appname`)
- [ ] Use `XDG_DATA_HOME` for application data (`~/.local/share/appname`)
- [ ] Use `XDG_CACHE_HOME` for caches (`~/.cache/appname`)
- [ ] Check `XDG_VIDEOS_DIR`, `XDG_DOCUMENTS_DIR` for media/document defaults
- [ ] Never hardcode `~/.appname` - use platform-appropriate paths

```python
# Pattern: Platform-aware path resolution
import os
import sys
from pathlib import Path

def get_config_dir() -> Path:
    if sys.platform == "linux":
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "myapp"
        return Path.home() / ".config" / "myapp"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MyApp"
    else:
        return Path(os.environ.get("APPDATA", "")) / "MyApp"
```

#### Qt/PySide6 Configuration
- [ ] Set `QApplication.setOrganizationName()` and `setApplicationName()` before QSettings use
- [ ] QSettings automatically respects XDG on Linux when organization/app names are set
- [ ] Test QSettings storage location: `~/.config/OrgName/AppName.conf`

```python
app = QApplication(sys.argv)
app.setApplicationName("Scene Ripper")
app.setOrganizationName("Algorithmic Filmmaking")
# QSettings will now use: ~/.config/Algorithmic Filmmaking/Scene Ripper.conf
```

### 2. Multimedia Dependencies

#### GStreamer Requirements (Critical for Qt Multimedia on Linux)
Qt Multimedia uses GStreamer as its backend on Linux. Missing plugins cause silent failures.

- [ ] Document required GStreamer packages:
  - `gstreamer1.0-plugins-good` - Essential codecs
  - `gstreamer1.0-plugins-bad` - Additional codecs
  - `gstreamer1.0-plugins-ugly` - Patent-encumbered codecs (MP3, etc.)
  - `gstreamer1.0-libav` - FFmpeg-based decoders
  - `gstreamer1.0-gl` - OpenGL video output

- [ ] Add runtime checks for GStreamer:

```python
def check_gstreamer():
    """Verify GStreamer is available for Qt Multimedia."""
    import subprocess
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "--version"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
```

#### OpenGL/EGL Dependencies
- [ ] Include `libgl1`, `libegl1` for Qt rendering
- [ ] Include `libxcb-xinerama0` for X11 multi-monitor support

### 3. CI/CD Setup

#### GitHub Actions Linux Testing

```yaml
# Minimum Ubuntu version for PySide6 6.6+
runs-on: ubuntu-22.04

steps:
  - name: Install system dependencies
    run: |
      sudo apt-get update
      sudo apt-get install -y \
        ffmpeg \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libxcb-xinerama0 \
        libegl1 \
        libgl1
```

#### Import Testing
Always test imports in CI to catch missing dependencies early:

```yaml
- name: Run import test
  run: |
    python -c "from PySide6.QtWidgets import QApplication"
    python -c "from PySide6.QtMultimedia import QMediaPlayer"
    python -c "from ui.main_window import MainWindow"
```

### 4. AppImage Packaging

#### Directory Structure

```
AppDir/
├── AppRun                      # Launcher script
├── usr/
│   ├── bin/python3             # Python interpreter
│   ├── lib/
│   │   ├── python3/site-packages/  # Python packages
│   │   └── x86_64-linux-gnu/
│   │       ├── qt6/plugins/    # Qt plugins
│   │       └── gstreamer-1.0/  # GStreamer plugins
│   ├── share/
│   │   ├── applications/app.desktop
│   │   └── icons/hicolor/...
│   └── src/                    # Application source
└── *.desktop                   # Desktop entry symlink
```

#### Critical Environment Variables

```bash
# In AppRun script
export PYTHONPATH="${APPDIR}/usr/lib/python3/site-packages"
export QT_PLUGIN_PATH="${APPDIR}/usr/lib/x86_64-linux-gnu/qt6/plugins"
export GST_PLUGIN_PATH="${APPDIR}/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
export LD_LIBRARY_PATH="${APPDIR}/usr/lib:${APPDIR}/usr/lib/x86_64-linux-gnu"
```

#### Desktop Entry Requirements

```ini
[Desktop Entry]
Type=Application
Name=My App
Exec=myapp %F
Icon=myapp
Terminal=false
Categories=AudioVideo;Video;
MimeType=video/mp4;video/x-matroska;
StartupNotify=true
```

---

## Common Pitfalls to Avoid

### 1. Path Handling

| Pitfall | Prevention |
|---------|------------|
| Hardcoding `~/` paths | Use `Path.home()` and XDG variables |
| Assuming case-insensitive filesystem | Linux is case-sensitive - match exactly |
| Using backslashes in paths | Use `pathlib.Path` for cross-platform paths |
| Hardcoding `/tmp` | Use `tempfile.gettempdir()` |

### 2. Qt/PySide6 Specific

| Pitfall | Prevention |
|---------|------------|
| Missing Qt platform plugins | Bundle `libxcb*` and set `QT_PLUGIN_PATH` |
| Silent multimedia failures | GStreamer plugins must be bundled |
| Wayland compatibility issues | Test on both X11 and Wayland |
| Font rendering issues | Bundle fallback fonts or use system fonts |

### 3. Packaging

| Pitfall | Prevention |
|---------|------------|
| Missing shared libraries | Use `ldd` to verify all dependencies |
| Wrong architecture | Explicitly specify `x86_64` or `aarch64` |
| Broken symlinks in AppImage | Use `readlink -f` to resolve paths |
| Icon not showing | Use absolute path in desktop entry |

### 4. Runtime

| Pitfall | Prevention |
|---------|------------|
| Assuming FFmpeg is in PATH | Check at startup, show helpful error |
| Root user file permissions | Test as non-root user |
| SELinux/AppArmor blocking | Test on Fedora (SELinux) and Ubuntu (AppArmor) |
| DBus session unavailable | Handle gracefully, not required for core function |

---

## Testing Recommendations

### 1. Distribution Matrix

Test on at least these distributions:

| Distribution | Why | Notes |
|--------------|-----|-------|
| Ubuntu 22.04 LTS | Most common | Base for many derivatives |
| Ubuntu 24.04 LTS | Latest LTS | Check newer lib versions |
| Fedora 39+ | RHEL ecosystem | Different package manager, SELinux |
| Debian 12 | Stability-focused | Older packages |
| Arch Linux | Rolling release | Latest everything |
| Linux Mint | Desktop-focused | Popular with end users |

### 2. Virtual Machine Test Script

```bash
#!/bin/bash
# test-appimage.sh - Run in VM

set -e

APPIMAGE="$1"

echo "=== System Info ==="
cat /etc/os-release | head -5
echo ""

echo "=== Testing AppImage ==="
chmod +x "$APPIMAGE"

# Test execution
echo "Starting application..."
timeout 10 "$APPIMAGE" --help || echo "No --help flag"

# Test GUI (requires display)
if [ -n "$DISPLAY" ]; then
    timeout 30 "$APPIMAGE" &
    PID=$!
    sleep 5
    if ps -p $PID > /dev/null; then
        echo "Application started successfully"
        kill $PID
    else
        echo "Application crashed"
        exit 1
    fi
fi

echo "=== Test Complete ==="
```

### 3. Automated CI Testing

```yaml
# .github/workflows/linux-test-matrix.yml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-22.04, ubuntu-24.04]
        python: ['3.11', '3.12']
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y xvfb gstreamer1.0-plugins-good
      - name: Test with Xvfb
        run: xvfb-run python -c "from PySide6.QtWidgets import QApplication; app = QApplication([])"
```

### 4. Manual Testing Checklist

Before release, manually verify:

- [ ] Application launches without errors
- [ ] Video playback works (audio and video)
- [ ] File dialogs open correctly
- [ ] Settings persist across restarts
- [ ] Drag-and-drop works
- [ ] Copy/paste works
- [ ] Keyboard shortcuts work
- [ ] Window resizing works
- [ ] Multi-monitor works
- [ ] System tray integration (if applicable)

---

## Best Practices for Qt/PySide6 on Linux

### 1. Application Initialization

```python
#!/usr/bin/env python3
import sys
import os
import logging

# Set up logging before any Qt imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Platform-specific setup before QApplication
if sys.platform == "linux":
    # Ensure XDG variables have sensible defaults
    if "XDG_RUNTIME_DIR" not in os.environ:
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

def main():
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("My App")
    app.setOrganizationName("My Organization")

    # Application code here

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

### 2. Error Handling for Missing Dependencies

```python
def check_dependencies():
    """Check for required external dependencies."""
    errors = []

    # Check FFmpeg
    import shutil
    if not shutil.which("ffmpeg"):
        errors.append(
            "FFmpeg not found. Install with: sudo apt install ffmpeg"
        )

    # Check GStreamer
    if not shutil.which("gst-inspect-1.0"):
        errors.append(
            "GStreamer not found. Install with: "
            "sudo apt install gstreamer1.0-plugins-good"
        )

    if errors:
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Missing Dependencies")
        msg.setText("Required dependencies are missing:")
        msg.setDetailedText("\n".join(errors))
        msg.exec()
        return False

    return True
```

### 3. Settings with Fallbacks

```python
from PySide6.QtCore import QSettings
from pathlib import Path
import sys
import os

class Settings:
    def __init__(self):
        self.qsettings = QSettings()

    def get_download_dir(self) -> Path:
        """Get download directory with platform-aware default."""
        stored = self.qsettings.value("paths/download_dir")
        if stored and Path(stored).exists():
            return Path(stored)

        # Platform-aware default
        if sys.platform == "linux":
            xdg = os.environ.get("XDG_VIDEOS_DIR")
            if xdg:
                return Path(xdg) / "Downloads"
            return Path.home() / "Videos" / "Downloads"
        elif sys.platform == "darwin":
            return Path.home() / "Movies" / "Downloads"
        else:
            return Path.home() / "Videos" / "Downloads"
```

### 4. Multimedia with Graceful Degradation

```python
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtCore import QUrl
import logging

logger = logging.getLogger(__name__)

class VideoPlayer:
    def __init__(self):
        self.player = QMediaPlayer()
        self.player.errorOccurred.connect(self._on_error)

    def _on_error(self, error, error_string):
        logger.error(f"Media player error: {error} - {error_string}")

        # Provide actionable error message
        if "gstreamer" in error_string.lower():
            self._show_gstreamer_help()

    def _show_gstreamer_help(self):
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Video Playback Error")
        msg.setText("Video playback failed due to missing codecs.")
        msg.setDetailedText(
            "Install GStreamer plugins:\n\n"
            "Ubuntu/Debian:\n"
            "  sudo apt install gstreamer1.0-plugins-good \\\n"
            "    gstreamer1.0-plugins-bad gstreamer1.0-libav\n\n"
            "Fedora:\n"
            "  sudo dnf install gstreamer1-plugins-good \\\n"
            "    gstreamer1-plugins-bad-free gstreamer1-libav"
        )
        msg.exec()
```

### 5. Desktop Integration

```python
import subprocess
import sys

def open_file_manager(path):
    """Open file manager at path, cross-platform."""
    if sys.platform == "linux":
        subprocess.run(["xdg-open", str(path)])
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["explorer", str(path)])

def open_url(url):
    """Open URL in default browser."""
    if sys.platform == "linux":
        subprocess.run(["xdg-open", url])
    elif sys.platform == "darwin":
        subprocess.run(["open", url])
    else:
        subprocess.run(["start", url], shell=True)
```

### 6. Thread Safety

```python
from PySide6.QtCore import QThread, Signal, QObject

class Worker(QObject):
    """Example of proper Qt threading pattern."""
    finished = Signal()
    error = Signal(str)
    progress = Signal(int)
    result = Signal(object)

    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_fn(*self.args, **self.kwargs)
            self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

def run_in_thread(task_fn, *args, on_finished=None, on_error=None, **kwargs):
    """Run a function in a background thread."""
    thread = QThread()
    worker = Worker(task_fn, *args, **kwargs)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    if on_finished:
        worker.result.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)

    thread.start()
    return thread, worker
```

---

## Quick Reference: Package Names

### Ubuntu/Debian

```bash
# Qt/PySide6 runtime
sudo apt install libxcb-xinerama0 libegl1 libgl1

# GStreamer for Qt Multimedia
sudo apt install \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-gl

# Build tools for AppImage
sudo apt install patchelf desktop-file-utils fakeroot
```

### Fedora/RHEL

```bash
# Qt/PySide6 runtime
sudo dnf install libxcb mesa-libEGL mesa-libGL

# GStreamer for Qt Multimedia
sudo dnf install \
    gstreamer1-plugins-good \
    gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly-free \
    gstreamer1-libav

# RPM Fusion for more codecs
sudo dnf install \
    https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install gstreamer1-plugins-bad-freeworld
```

### Arch Linux

```bash
# Qt/PySide6 runtime
sudo pacman -S libxcb mesa

# GStreamer for Qt Multimedia
sudo pacman -S \
    gst-plugins-good \
    gst-plugins-bad \
    gst-plugins-ugly \
    gst-libav
```

---

## Summary

1. **Plan for Linux from Day 1** - Don't treat it as an afterthought
2. **XDG compliance is mandatory** - Users expect standard paths
3. **GStreamer is Qt Multimedia's backend** - Bundle or document requirements
4. **Test on multiple distributions** - Ubuntu alone is not enough
5. **CI/CD catches issues early** - Automate Linux testing
6. **Provide helpful error messages** - Guide users to solutions
7. **AppImage is the most portable format** - But requires careful bundling
