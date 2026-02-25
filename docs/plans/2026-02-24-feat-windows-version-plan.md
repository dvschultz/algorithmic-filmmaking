---
title: "feat: Windows Version with Full Feature Parity"
type: feat
status: completed
date: 2026-02-24
origin: docs/brainstorms/2026-02-24-windows-version-brainstorm.md
---

# feat: Windows Version with Full Feature Parity

## Overview

Port Scene Ripper to Windows with full feature parity for general (non-technical) users. Deliver a PyInstaller `.exe` wrapped in an Inno Setup installer, with FFmpeg and yt-dlp auto-downloaded on first launch. Apple Silicon ML features replaced with cross-platform equivalents (faster-whisper CPU, LiteLLM cloud APIs).

The codebase is partially Windows-ready (PySide6, pathlib, argument-array subprocess calls, platform-gated Apple deps). There are ~24 specific issues to fix across platform paths, binary resolution, subprocess handling, dependency management, safe path validation, and packaging.

## Problem Statement / Motivation

Users have requested a Windows version. The current app only builds and distributes for macOS (.app/.dmg) and Linux (AppImage). Windows represents the largest desktop OS market share. The developer does not have a Windows machine, so the build/test strategy must rely on CI and cloud VMs.

## Proposed Solution

Fix platform-specific code in 6 phases, then package and distribute via GitHub Actions CI. Each phase is independently testable and deployable.

(see brainstorm: docs/brainstorms/2026-02-24-windows-version-brainstorm.md for key decisions)

## Technical Approach

### Architecture

No architectural changes. All fixes are additive platform branches in existing code. New files are limited to packaging configs and CI workflow.

### Implementation Phases

---

#### Phase 1: Platform Foundation

Fix the core platform abstractions that everything else depends on.

##### 1a. `core/paths.py` — Add Windows branches

**Files**: `core/paths.py` (lines 30-38, 64-72)

Add `sys.platform == "win32"` branch to `get_app_support_dir()` and `get_log_dir()`:

```python
# core/paths.py
def get_app_support_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Scene Ripper"
    elif sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA",
                    Path.home() / "AppData" / "Local")) / "Scene Ripper"
    return Path.home() / ".local" / "share" / "scene-ripper"

def get_log_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "Scene Ripper"
    return get_app_support_dir() / "logs"
```

Use `LOCALAPPDATA` (not `APPDATA`) to match the pattern in `core/settings.py` `_get_cache_dir()` — managed binaries are large data, not roaming config.

##### 1b. `core/binary_resolver.py` — Windows binary lookup

**Files**: `core/binary_resolver.py` (lines 21-31, 50-51, 56-63)

Three changes:

1. Append `.exe` on Windows when checking managed path:
```python
def find_binary(name: str) -> str | None:
    managed_dir = get_managed_bin_dir()
    if managed_dir:
        suffixes = [".exe", ""] if sys.platform == "win32" else [""]
        for suffix in suffixes:
            managed_path = managed_dir / (name + suffix)
            if managed_path.is_file():
                return str(managed_path)
    # ... rest unchanged
```

2. Replace `os.access(path, os.X_OK)` with `path.is_file()` in extra path search (line 58) — `X_OK` is unreliable on Windows.

3. Add Windows extra search paths:
```python
if sys.platform == "win32":
    _EXTRA_SEARCH_PATHS.extend([
        r"C:\Program Files\FFmpeg\bin",
        r"C:\ffmpeg\bin",
        str(Path.home() / "scoop" / "shims"),
    ])
```

##### 1c. `core/binary_resolver.py` — Subprocess helper

**Files**: `core/binary_resolver.py` (new function)

Create a central helper to avoid repeating platform checks in ~20 subprocess call sites:

```python
import subprocess

def get_subprocess_kwargs() -> dict:
    """Return platform-appropriate kwargs for subprocess calls."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
```

**Success criteria**:
- [ ] `get_app_support_dir()` returns `%LOCALAPPDATA%\Scene Ripper` on Windows
- [ ] `get_log_dir()` returns `%LOCALAPPDATA%\Scene Ripper\logs` on Windows
- [ ] `find_binary("ffmpeg")` finds `ffmpeg.exe` in managed dir on Windows
- [ ] `find_binary("ffmpeg")` falls through to `shutil.which()` if not in managed dir
- [ ] `get_subprocess_kwargs()` returns `CREATE_NO_WINDOW` flag on Windows
- [ ] Unit tests for all three behaviors (parameterized for win32/darwin/linux)

**Estimated effort**: Small

---

#### Phase 2: FFmpeg & Subprocess Fixes

Fix all subprocess calls that would break or misbehave on Windows.

##### 2a. Replace bare `"ffmpeg"` / `"ffprobe"` calls

**Files**:
- `core/transcription.py` (lines 333, 395, 524) — 3 bare `"ffmpeg"` calls
- `core/analysis/audio.py` (lines 146, 220, 370) — 2 bare `"ffmpeg"`, 1 bare `"ffprobe"`
- `core/analysis/embeddings.py` (line 207) — 1 bare `"ffmpeg"`

Replace each bare string with `find_binary("ffmpeg")` or `find_binary("ffprobe")` from `core/binary_resolver.py`. Add import and resolve at function entry:

```python
from core.binary_resolver import find_binary

ffmpeg = find_binary("ffmpeg") or "ffmpeg"  # fallback to PATH lookup
```

##### 2b. Fix FFmpeg null device

**Files**: `core/analysis/audio.py` (lines 376-377)

```python
# Before
"-f", "null",
"-",

# After
"-f", "null",
"NUL" if sys.platform == "win32" else "-",
```

##### 2c. Apply `CREATE_NO_WINDOW` to all subprocess calls

**Files** (all subprocess call sites):
- `core/ffmpeg.py` (~5 calls)
- `core/thumbnail.py` (~1 call)
- `core/sequence_export.py` (~3 calls)
- `core/downloader.py` (~3 calls)
- `core/transcription.py` (~4 calls)
- `core/analysis/audio.py` (~3 calls)
- `core/analysis/embeddings.py` (~1 call)
- `core/dependency_manager.py` (~2 calls)

Add `**get_subprocess_kwargs()` to each `subprocess.run()` and `subprocess.Popen()` call.

##### 2d. Guard `signal.SIGTERM`

**Files**: `cli/utils/signals.py` (line 196)

```python
signal.signal(signal.SIGINT, _signal_handler)
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, _signal_handler)
```

**Success criteria**:
- [ ] All 7 bare ffmpeg/ffprobe calls replaced with `find_binary()` + fallback
- [ ] FFmpeg null device uses `"NUL"` on Windows
- [ ] No console window flashes during subprocess operations on Windows
- [ ] CLI does not crash on `import` on Windows (SIGTERM guard)
- [ ] All existing tests still pass on macOS/Linux (no regressions)

**Estimated effort**: Small-Medium (many files but mechanical changes)

---

#### Phase 3: Windows Dependency Manager

Extend `core/dependency_manager.py` to download Windows binaries on first launch.

##### 3a. Platform dispatch for download URLs

**Files**: `core/dependency_manager.py` (lines 29-40, 195-270)

Replace hardcoded macOS ARM64 URLs with platform dispatch:

```python
def _get_ffmpeg_url() -> str:
    if sys.platform == "win32":
        return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    elif platform.machine() == "arm64" and sys.platform == "darwin":
        return "https://www.osxexperts.net/ffmpeg7arm.zip"
    raise RuntimeError(f"No FFmpeg download available for {sys.platform}/{platform.machine()}")

def _get_ytdlp_url() -> str:
    if sys.platform == "win32":
        return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
    elif sys.platform == "darwin":
        return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
    raise RuntimeError(f"No yt-dlp download available for {sys.platform}")
```

##### 3b. Binary naming with `.exe` suffix

Update `ensure_ffmpeg()`, `ensure_ffprobe()`, `ensure_yt_dlp()` to use `.exe` suffix on Windows:

```python
ext = ".exe" if sys.platform == "win32" else ""
ffmpeg_path = bin_dir / f"ffmpeg{ext}"
```

##### 3c. Remove ARM64-only guard

Replace `if platform.machine() != "arm64": raise RuntimeError(...)` with platform dispatch that supports both macOS ARM64 and Windows x64.

##### 3d. Skip `ensure_python()` on frozen Windows builds

```python
def ensure_python(...):
    if getattr(sys, 'frozen', False) and sys.platform == "win32":
        return None  # Python is bundled by PyInstaller
    # ... existing logic
```

##### 3e. BtbN archive extraction

BtbN Windows FFmpeg zips have binaries in `ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe`. The existing `_extract_zip_binary()` already searches for the filename anywhere in the archive (line 162-167), so this should work. Verify with a test.

**Success criteria**:
- [ ] `ensure_ffmpeg()` downloads Windows x64 FFmpeg on `sys.platform == "win32"`
- [ ] `ensure_ffprobe()` downloads Windows x64 FFprobe
- [ ] `ensure_yt_dlp()` downloads `yt-dlp.exe` on Windows
- [ ] `ensure_python()` is a no-op on frozen Windows builds
- [ ] All `ensure_*` functions still work on macOS ARM64 (no regressions)
- [ ] Downloaded binaries are found by `find_binary()` (Phase 1b integration)

**Estimated effort**: Medium

---

#### Phase 4: Safe Roots, Cache Paths & Misc Fixes

##### 4a. Windows safe path roots

**Files**: `core/chat_tools.py` (lines 82-98), `scene_ripper_mcp/security.py` (lines 8-24)

Add Windows-specific safe roots:

```python
if sys.platform == "win32":
    # Add all existing drive roots (C:\, D:\, etc.)
    import string
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:\\")
        if drive.exists():
            safe_roots.append(drive)
    # Add temp directory (may be outside user profile)
    safe_roots.append(Path(tempfile.gettempdir()).resolve())
```

Note: Adding all existing drive roots is necessary because Windows users commonly store video files on secondary drives (`D:\Videos\`).

##### 4b. Windows cache fallback paths

**Files**:
- `core/thumbnail.py` (line 29)
- `core/analysis/classification.py` (line 33)
- `core/analysis/detection.py` (line 58)

Replace `~/.cache/` fallback with platform-aware cache:

```python
from core.settings import load_settings
# Or use the settings cache_dir which already handles Windows
```

Alternatively, use `core/paths.py` to add a `get_cache_dir()` function that mirrors the settings logic.

##### 4c. Default download directory

**Files**: `core/downloader.py` (line 52)

Replace `Path.home() / "Movies" / "Scene Ripper Downloads"` with the platform-aware `_get_videos_dir()` from settings:

```python
from core.settings import _get_videos_dir
default_dir = _get_videos_dir() / "Scene Ripper Downloads"
```

##### 4d. Update Deno error messages

**Files**: `core/downloader.py` (lines 369-380)

Replace `brew install deno` with platform-specific instructions:

```python
if sys.platform == "win32":
    deno_hint = "Install Deno: winget install DenoLand.Deno"
elif sys.platform == "darwin":
    deno_hint = "Install Deno: brew install deno"
else:
    deno_hint = "Install Deno: curl -fsSL https://deno.land/install.sh | sh"
```

**Success criteria**:
- [ ] Agent tools accept file paths on any Windows drive (D:\, E:\, etc.)
- [ ] Agent tools accept paths in `%TEMP%` on Windows
- [ ] Cache directories use `%LOCALAPPDATA%` on Windows (no dot-directories)
- [ ] Default download dir is `~/Videos/Scene Ripper Downloads` on Windows
- [ ] Deno error messages show Windows install instructions on Windows

**Estimated effort**: Small

---

#### Phase 5: Windows CI

Add a GitHub Actions workflow to run tests on Windows and build the installer.

##### 5a. Test workflow

**Files**: `.github/workflows/windows-ci.yml` (new)

```yaml
name: Windows CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install FFmpeg
        run: choco install ffmpeg -y
      - name: Install dependencies
        run: pip install -r requirements-core.txt pytest pytest-asyncio
      - name: Run tests
        env:
          QT_QPA_PLATFORM: offscreen
        run: pytest tests/ -v --tb=short
```

Run on every push/PR to main (matching Linux CI pattern). Install FFmpeg via Chocolatey for tests that need it. Use `QT_QPA_PLATFORM=offscreen` for headless Qt.

##### 5b. Platform-specific test markers

**Files**: `tests/conftest.py` (additions)

Add fixtures and markers for platform-specific tests:

```python
import sys
import pytest

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
unix_only = pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
```

##### 5c. Windows-specific unit tests

**Files**: `tests/test_windows_compat.py` (new)

Test the platform-specific code paths:
- `get_app_support_dir()` returns correct Windows path
- `find_binary()` appends `.exe` on Windows
- `get_subprocess_kwargs()` returns `CREATE_NO_WINDOW` on Windows
- `_get_ffmpeg_url()` returns BtbN URL on Windows
- Safe path roots include drive letters on Windows
- FFmpeg null device is `"NUL"` on Windows

These tests mock `sys.platform` so they run on all CI platforms.

**Success criteria**:
- [ ] Full pytest suite passes on `windows-latest` GitHub Actions runner
- [ ] New Windows-specific tests pass on all platforms (via mocking)
- [ ] CI runs on every push/PR to main
- [ ] Existing macOS and Linux CI still pass

**Estimated effort**: Small-Medium

---

#### Phase 6: Windows Packaging

Create the PyInstaller spec, Inno Setup script, and build workflow.

##### 6a. Windows PyInstaller spec

**Files**: `packaging/windows/scene_ripper.spec` (new)

Based on `packaging/macos/scene_ripper.spec`. Key differences:
- Remove `BUNDLE` section (macOS-only)
- Replace `keyring.backends.macOS` with `keyring.backends.Windows` in hiddenimports
- Add `icon='..\\..\\assets\\icon.ico'` to EXE
- Keep `console=False`
- Keep same `excludes` list (torch, transformers, etc. — on-demand)
- Bundle `faster-whisper` and `ctranslate2` (cannot use managed Python on Windows)

##### 6b. Windows icon

**Files**: `assets/icon.ico` (new)

Convert existing `assets/icon.icns` to `.ico` format. Include sizes: 16x16, 32x32, 48x48, 64x64, 128x128, 256x256.

##### 6c. Inno Setup script

**Files**: `packaging/windows/scene_ripper.iss` (new)

Per-user install (no admin required) to `{localappdata}\Programs\Scene Ripper`:
- Start Menu shortcut
- Optional desktop shortcut
- Uninstaller (preserves `%LOCALAPPDATA%\Scene Ripper` user data)
- License file
- App icon

```ini
[Setup]
AppName=Scene Ripper
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\Scene Ripper
DefaultGroupName=Scene Ripper
PrivilegesRequired=lowest
OutputBaseFilename=SceneRipper-Setup-{#MyAppVersion}
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\scene_ripper.exe

[Files]
Source: "..\..\dist\scene_ripper\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Scene Ripper"; Filename: "{app}\scene_ripper.exe"
Name: "{autodesktop}\Scene Ripper"; Filename: "{app}\scene_ripper.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
```

##### 6d. Build workflow

**Files**: `.github/workflows/build-windows.yml` (new)

Tag-triggered (matching macOS pattern) + manual dispatch:

```yaml
name: Build Windows
on:
  push:
    tags: ['v*']
  workflow_dispatch:
    inputs:
      version:
        description: 'Version number'
        required: true

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements-core.txt pyinstaller
      - name: Build with PyInstaller
        run: pyinstaller packaging/windows/scene_ripper.spec --distpath dist --workpath build --noconfirm
      - name: Build Inno Setup installer
        uses: jrsoftware/iscc-action@v1
        with:
          script: packaging/windows/scene_ripper.iss
      - name: Upload installer
        uses: actions/upload-artifact@v4
        with:
          name: SceneRipper-Windows-Installer
          path: packaging/windows/Output/*.exe
      - name: Upload to release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: packaging/windows/Output/*.exe
```

##### 6e. Build script (local)

**Files**: `packaging/windows/build.ps1` (new)

PowerShell script for local builds (when testing on a Windows VM):

```powershell
pip install -r requirements-core.txt pyinstaller
pyinstaller packaging/windows/scene_ripper.spec --distpath dist --workpath build --noconfirm
```

**Success criteria**:
- [ ] `pyinstaller packaging/windows/scene_ripper.spec` produces a working `scene_ripper.exe`
- [ ] Inno Setup produces a `SceneRipper-Setup-X.Y.Z.exe` installer
- [ ] Installer installs to `%LOCALAPPDATA%\Programs\Scene Ripper` without admin
- [ ] Start menu shortcut launches the app
- [ ] Uninstaller removes the app but preserves user data
- [ ] GitHub Actions builds and uploads the installer on tag push
- [ ] App icon displays correctly in Explorer, taskbar, and Start menu

**Estimated effort**: Medium-Large

---

## System-Wide Impact

### Interaction Graph

Platform changes touch these interaction chains:
1. `dependency_manager.ensure_*()` → `paths.get_managed_bin_dir()` → `paths.get_app_support_dir()` — must agree on Windows path
2. `find_binary()` → managed dir lookup → `.exe` suffix — must match what dependency_manager downloads
3. Every subprocess call → `get_subprocess_kwargs()` → `CREATE_NO_WINDOW` flag — affects all operations
4. `chat_tools._validate_path()` → safe roots → drive letter roots — affects all agent tool calls

### Error & Failure Propagation

- **Download failure**: `ensure_*()` → `RuntimeError` → shown in first-launch dialog. User retries or manually installs.
- **Binary not found**: `find_binary()` → `None` → `"ffmpeg"` fallback → `FileNotFoundError` from subprocess → caught by worker, shown as error in UI.
- **File locking**: Windows may block file deletion during processing. `PermissionError` in temp cleanup should be caught and logged (not raised).
- **Antivirus quarantine**: Downloaded `.exe` disappears. `find_binary()` returns `None`. Same flow as binary-not-found, but error message should suggest whitelisting.

### State Lifecycle Risks

- **Partial FFmpeg download**: Atomic temp-file-then-rename pattern (already used in `dependency_manager.py`) prevents partial downloads from appearing as valid binaries.
- **Interrupted install**: Inno Setup handles rollback automatically.
- **process.terminate() on Windows**: Calls `TerminateProcess()` (hard kill, no cleanup). May leave partial output files. Existing behavior — not introducing new risk.

### API Surface Parity

All changes are internal. No public API changes. Agent tools continue to work identically. CLI commands continue to work identically.

## Acceptance Criteria

### Functional Requirements

- [ ] App launches on Windows 10+ and displays the main window
- [ ] First-launch auto-downloads FFmpeg and yt-dlp
- [ ] Scene detection works (import video → cut → thumbnails appear)
- [ ] Transcription works via faster-whisper (CPU)
- [ ] VLM analysis works via LiteLLM cloud API
- [ ] YouTube search and download works
- [ ] Sequence editing and rendering works
- [ ] Agent chat works with file path validation on all drives
- [ ] Settings persist to `%LOCALAPPDATA%` / `%APPDATA%`
- [ ] CLI commands work (scene-ripper detect, analyze, etc.)
- [ ] Inno Setup installer installs and uninstalls cleanly

### Non-Functional Requirements

- [ ] No console window flashes during any operation
- [ ] No SIGTERM crash in CLI
- [ ] Installer does not require admin privileges
- [ ] Full pytest suite passes on Windows CI

### Quality Gates

- [ ] All existing tests pass on macOS, Linux, and Windows CI
- [ ] New Windows-specific unit tests cover path, binary, and subprocess logic
- [ ] Manual smoke test on a real Windows environment before first release

## Dependencies & Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyInstaller antivirus false positives | High | Medium | Document SmartScreen "Run anyway" instructions. Add code signing later. |
| PySide6 DPI scaling issues | Medium | Medium | Test at 125%/150% on Windows VM. Qt 6 handles DPI better than Qt 5. Fix specific widgets if needed. |
| BtbN FFmpeg URL changes | Low | High | Pin to a specific release tag rather than `latest`. Add fallback URL. |
| Windows file locking blocks temp cleanup | Medium | Low | Catch `PermissionError` in cleanup, log and continue. |
| Faster-whisper ctranslate2 DLLs not bundled by PyInstaller | Medium | High | Add ctranslate2 data files to spec `datas`. Test transcription in built .exe. |
| 260-char path limit | Low | Medium | Skip for v1. Document if reported. |

## Alternative Approaches Considered

1. **MSIX / Windows Store** — Cleaner install UX but more complex tooling, less community knowledge for PySide6. Rejected for v1. (see brainstorm)
2. **Portable zip** — Simplest packaging but no Start menu / uninstaller. Rejected in favor of proper installer. (see brainstorm)
3. **Local GPU inference (CUDA)** — Better transcription speed but adds ~500MB CUDA libs and NVIDIA driver requirement. Rejected for v1 — CPU-only. (see brainstorm)
4. **CLI-only first** — Fastest path but doesn't serve general users who need the GUI. Rejected. (see brainstorm)

## Sources & References

### Origin

- **Brainstorm document**: [docs/brainstorms/2026-02-24-windows-version-brainstorm.md](docs/brainstorms/2026-02-24-windows-version-brainstorm.md) — Key decisions carried forward: PyInstaller + Inno Setup packaging, faster-whisper CPU + LiteLLM cloud APIs, first-launch dependency download, Windows 10+ minimum, no code signing v1.

### Internal References

- macOS PyInstaller spec: `packaging/macos/scene_ripper.spec`
- macOS build CI: `.github/workflows/build-macos.yml`
- Linux build CI: `.github/workflows/linux-build.yml`
- Core requirements (for bundling): `requirements-core.txt`
- Dependency manager: `core/dependency_manager.py`
- Binary resolver: `core/binary_resolver.py`
- Platform paths: `core/paths.py`
- Settings (Windows-aware): `core/settings.py` (lines 279-339)

### External References

- BtbN FFmpeg Windows builds: https://github.com/BtbN/FFmpeg-Builds/releases
- yt-dlp releases: https://github.com/yt-dlp/yt-dlp/releases
- Inno Setup documentation: https://jrsoftware.org/ishelp/
- PyInstaller Windows notes: https://pyinstaller.org/en/stable/usage.html#windows
- PySide6 Windows deployment: https://doc.qt.io/qtforpython-6/deployment/index.html
