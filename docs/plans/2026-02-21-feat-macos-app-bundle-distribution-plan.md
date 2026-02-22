---
title: "feat: macOS .app Bundle Distribution"
type: feat
status: active
date: 2026-02-21
origin: docs/brainstorms/2026-02-21-macos-app-bundle-brainstorm.md
---

# macOS .app Bundle Distribution

## Overview

Package Scene Ripper as a signed, notarized macOS .app bundle so users can download a DMG from GitHub Releases and run the app without a terminal, Python environment, or manual dependency installation. The core video editing workflow (collect, cut, sequence, render) works immediately; ML-powered features download their dependencies on first use.

## Problem Statement

The app currently requires `pip install`, Python 3.11+, FFmpeg, and numerous system dependencies. This is a barrier for non-technical users and even developers who don't want to manage a Python environment for a GUI tool. The goal is a one-download, drag-to-Applications install experience.

## Proposed Solution

Use PyInstaller to create a macOS .app bundle targeting Apple Silicon (arm64) only. Bundle core dependencies (PySide6, OpenCV, LiteLLM, etc.) in the app. External binaries (FFmpeg, yt-dlp) and heavy ML packages (PyTorch, transformers, etc.) are downloaded on first use to `~/Library/Application Support/Scene Ripper/`. Distribute as a branded, code-signed, notarized DMG via GitHub Releases with automated CI builds on tagged releases.

(see brainstorm: `docs/brainstorms/2026-02-21-macos-app-bundle-brainstorm.md`)

## Technical Approach

### Architecture

The .app bundle contains the frozen Python runtime, core dependencies, and app code. Everything else lives outside the bundle in the user's Application Support directory:

```
Scene Ripper.app/                    # ~1.3-1.5 GB (signed + notarized)
  Contents/
    MacOS/Scene Ripper               # Frozen executable
    Frameworks/                      # sys._MEIPASS — bundled Python + deps
    Resources/icon.icns              # App icon
    Info.plist                       # Bundle metadata

~/Library/Application Support/Scene Ripper/
  bin/                               # Downloaded binaries
    ffmpeg
    ffprobe
    yt-dlp
    deno                             # If needed for yt-dlp
  python/                            # Standalone Python 3.11 framework
    bin/python3
    lib/python3.11/
  packages/                          # On-demand Python packages
    torch/
    transformers/
    ultralytics/
    ...
  packages.json                      # Installed package manifest
  config.json                        # App settings (migrated from XDG on first frozen launch)
  cache/                             # Thumbnails, model cache
```

### Key Technical Decision: On-Demand Python Packages

**Problem (identified by SpecFlow analysis):** The brainstorm chose `pip install --target`, but `sys.executable` in a PyInstaller-frozen app points to the frozen binary, not a Python interpreter. Running `sys.executable -m pip` re-launches the app instead of invoking pip.

**Solution:** Bundle a standalone Python 3.11 framework on first-run setup.

1. On first launch (or first time an ML feature is requested), download the official Python 3.11 macOS arm64 framework from python.org (~30 MB compressed).
2. Extract to `~/Library/Application Support/Scene Ripper/python/`.
3. Use that Python's pip for all `--target` installations into `packages/`.
4. At app startup, add `packages/` to `sys.path` so the frozen app can import on-demand packages.
5. Ship a `package_manifest.json` embedded in the .app that specifies exact versions, platform wheels, and size estimates for each on-demand package.

**Why not alternatives:**
- *Direct wheel download + extract*: Doesn't resolve dependency trees or handle compiled extensions reliably.
- *System Python fallback*: User may not have Python, or may have 3.12 while the app needs 3.11 ABI compatibility.

### Implementation Phases

#### Phase 1: Foundation — Frozen App Infrastructure

Build the core infrastructure needed for the .app to launch and work for basic (non-ML) features.

**1.1 Create `core/paths.py` — Resource and binary resolution module**

```python
# core/paths.py
"""Centralized path resolution for frozen and source modes."""

import os
import sys
from pathlib import Path

def is_frozen() -> bool:
    return getattr(sys, 'frozen', False)

def get_base_path() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent

def get_app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "Scene Ripper"

def get_managed_bin_dir() -> Path:
    return get_app_support_dir() / "bin"

def get_managed_packages_dir() -> Path:
    return get_app_support_dir() / "packages"

def get_managed_python_dir() -> Path:
    return get_app_support_dir() / "python"
```

- [x] Create `core/paths.py` with `is_frozen()`, `get_base_path()`, `get_app_support_dir()`, `get_managed_bin_dir()`, `get_managed_packages_dir()`, `get_managed_python_dir()`
- [x] Create `core/binary_resolver.py` with `find_binary(name: str) -> Optional[str]` that checks: managed bin dir → Homebrew paths → `shutil.which()`
- [x] Refactor all 11+ `shutil.which()` call sites to use `find_binary()`:
  - `core/ffmpeg.py` lines 19, 28, 168, 228, 290
  - `core/thumbnail.py` line 13
  - `core/sequence_export.py` line 41
  - `core/downloader.py` line 87
  - `core/analysis/description.py` line 53
  - `ui/dialogs/intention_import_dialog.py` line 53
- [x] Generalize `downloader.py:_get_subprocess_env()` (lines 18-45) to include the managed bin dir when frozen
- [x] Add `sys.path` injection at app startup: insert `get_managed_packages_dir()` into `sys.path` if it exists

**1.2 Make `litellm` imports lazy**

Currently `core/analysis/description.py` line 22 and `core/analysis/cinematography.py` line 26 have top-level `import litellm`. These need to become lazy imports inside the functions that use them, to prevent the full litellm dependency tree from being required at import time.

- [x] Move `import litellm` in `core/analysis/description.py` from line 22 to inside functions that use it
- [x] Move `import litellm` in `core/analysis/cinematography.py` from line 26 to inside functions that use it
- [x] Verify no other top-level imports of on-demand packages exist

**1.3 Separate dependency lists**

- [x] Create `requirements-core.txt` (bundled in .app):
  - PySide6, OpenCV, scenedetect, numpy, Pillow, scikit-learn
  - litellm, httpx, tenacity (LLM integration — needed for chat agent)
  - keyring, google-api-python-client
  - click (CLI, if exposed)
- [x] Create `requirements-optional.txt` (on-demand):
  - torch, torchvision, transformers, einops
  - ultralytics
  - paddleocr (+ paddlepaddle)
  - faster-whisper
  - lightning-whisper-mlx, mlx-vlm, mlx
  - librosa
  - yt-dlp (downloaded as standalone binary, not pip)
- [x] Keep `requirements.txt` as the union of both (for development)

**1.4 Settings path migration**

When running frozen on macOS, use `~/Library/Application Support/Scene Ripper/` instead of `~/.config/scene-ripper/`.

- [x] In `core/settings.py`, when `is_frozen() and sys.platform == 'darwin'`, use macOS-native paths:
  - Config: `~/Library/Application Support/Scene Ripper/config.json`
  - Cache: `~/Library/Caches/com.scene-ripper.app/`
- [x] Add one-time migration: on first frozen launch, if `~/.config/scene-ripper/config.json` exists, copy it to the new location
- [ ] Ensure keyring service name uses the bundle identifier consistently: `com.algorithmic-filmmaking.scene-ripper`

#### Phase 2: On-Demand Dependency Manager

**2.1 Binary downloader**

- [x] Create `core/dependency_manager.py` with:
  - `ensure_ffmpeg() -> Path` — downloads static arm64 FFmpeg from osxexperts.net if not present
  - `ensure_ffprobe() -> Path` — same for FFprobe
  - `ensure_yt_dlp() -> Path` — downloads latest standalone yt-dlp binary
  - `update_yt_dlp()` — re-downloads yt-dlp to pick up new versions
  - All downloads go to `~/Library/Application Support/Scene Ripper/bin/`
  - Progress callback support for UI integration
  - Download verification (file size check, executable permissions)
  - Partial download cleanup on failure
  - Disk space pre-flight check before large downloads

**2.2 Python package installer**

- [x] `ensure_python() -> Path` — downloads python-build-standalone 3.11 macOS arm64 if not present
- [x] `install_package(specifier: str, progress_callback) -> bool` — runs `<managed_python> -m pip install --target <packages_dir> <specifier>`
- [x] `is_package_available(module_name: str) -> bool` — attempts import from packages dir
- [x] Create `package_manifest.json` embedded in the app:
  ```json
  {
    "python_version": "3.11.11",
    "python_url": "https://www.python.org/ftp/python/3.11.11/python-3.11.11-macos11.pkg",
    "packages": {
      "torch": {
        "pip_specifier": "torch>=2.4,<2.6",
        "size_mb": 350,
        "features": ["shot_classification", "embeddings", "object_detection", "descriptions_local"]
      },
      "transformers": {
        "pip_specifier": "transformers>=4.36,<5.0",
        "size_mb": 100,
        "depends_on": ["torch"],
        "features": ["shot_classification", "embeddings"]
      }
    }
  }
  ```
- [x] ABI compatibility check: store `compat_version` marker (Python version + app version). If mismatch after app update, prompt user to re-download packages.

**2.3 Feature availability registry**

- [x] Create `core/feature_registry.py` that maps features to their dependency requirements:
  ```python
  FEATURE_DEPS = {
      "scene_detection": {"binaries": ["ffmpeg"]},
      "thumbnails": {"binaries": ["ffmpeg"]},
      "video_download": {"binaries": ["yt-dlp"]},
      "transcribe": {"packages": ["faster-whisper"], "binaries": ["ffmpeg"]},
      "transcribe_mlx": {"packages": ["lightning-whisper-mlx"]},
      "describe_local": {"packages": ["torch", "transformers"]},
      "shot_classify": {"packages": ["torch", "transformers"]},
      "object_detect": {"packages": ["ultralytics", "torch"]},
      "ocr": {"packages": ["paddleocr"]},
      "audio_analysis": {"packages": ["librosa"]},
  }
  ```
- [x] `check_feature(name: str) -> tuple[bool, list[str]]` — returns (available, missing_deps)
- [x] `install_for_feature(name: str, progress_callback)` — installs all deps for a feature

#### Phase 3: UI Integration

**3.1 First-run experience**

When FFmpeg is missing on first launch, the app should still open normally but show a setup prompt.

- [x] Intercept `RuntimeError("FFmpeg not found")` in `FFmpegProcessor.__init__()` and `ThumbnailGenerator.__init__()` — when frozen, don't raise; instead set a `ffmpeg_available = False` flag
- [x] Show a non-modal "Setup Required" banner at the top of the main window:
  > "Scene Ripper needs FFmpeg to process videos. [Download FFmpeg] (~150 MB)"
- [ ] Gray out features requiring FFmpeg with tooltip: "FFmpeg required — click to download"
- [x] Download progress shown in a modal dialog with cancel button
- [x] On success, dismiss the banner and enable features without restart

**3.2 On-demand download prompts**

When a user triggers a feature requiring an uninstalled package:

- [x] Show dialog: "This feature requires [package_name] (~[size]MB). Download now?"
- [x] Download/install progress with cancel support
- [ ] On completion, retry the original action automatically
- [x] Handle download failure: "Download failed. [Retry] [Cancel]" — clean up partial files

**3.3 Settings page — dependency management**

- [x] Add a "Dependencies" section in the Settings dialog showing:
  - FFmpeg: installed/missing + version + [Update] button
  - yt-dlp: installed/missing + version + [Update] button
  - Each ML feature group: installed/missing + size + [Install]/[Remove] button
  - Total disk usage of on-demand downloads
  - [Reset All Dependencies] button to wipe packages/ and bin/
- [x] Show managed Python status: installed/missing + version

**3.4 Version update checker**

- [x] On app launch (background thread), query GitHub Releases API: `GET /repos/{owner}/{repo}/releases/latest`
- [x] Throttle: check at most once per 24 hours (store last check timestamp in settings)
- [x] If newer version found, show dismissible banner: "Scene Ripper [version] is available. [Download]"
- [x] "Download" opens the GitHub Release page in the browser
- [x] Add "Check for updates" toggle in Settings (opt-out, default on)

#### Phase 4: PyInstaller Build Configuration

**4.1 Create the `.spec` file**

- [x] Create `packaging/macos/scene_ripper.spec` with:
  - `Analysis(['main.py'])` entry point
  - `hiddenimports` for PySide6 modules used (QtMultimedia, QtMultimediaWidgets, QtNetwork) and shiboken6
  - Aggressive `excludes` list: all 30+ unused PySide6 modules (QtWebEngine, Qt3D, QtQuick, QtQml, QtBluetooth, QtCharts, QtSensors, QtSerialPort, QtTest, QtPdf, etc.)
  - Exclude on-demand packages: torch, transformers, ultralytics, paddleocr, faster_whisper, lightning_whisper_mlx, mlx_vlm, mlx, librosa, scipy
  - Exclude unused stdlib: tkinter, unittest, test
  - `console=False` for windowed mode
  - `strip=False` and `upx=False` (both break macOS code signing)
  - `target_arch=None` (inherits runner arch = arm64)
  - `bundle_identifier='com.algorithmic-filmmaking.scene-ripper'`
  - `info_plist` with `NSHighResolutionCapable`, `LSMinimumSystemVersion: '13.0'`, `CFBundleShortVersionString` from env var
  - `icon='assets/icon.icns'`

**4.2 Create entitlements file**

- [x] Create `packaging/macos/entitlements.plist`:
  - `com.apple.security.cs.allow-unsigned-executable-memory` (required for Python/NumPy)
  - `com.apple.security.cs.disable-library-validation` (required for loading on-demand `.so`/`.dylib` from Application Support)
  - `com.apple.security.cs.allow-jit` (required for NumPy/PyTorch JIT)
  - `com.apple.security.network.client` (for downloads, API calls)
  - `com.apple.security.files.user-selected.read-write` (for video file access)

**4.3 Create app icon**

- [x] Create `assets/icon.icns` (macOS icon set: 16x16 through 1024x1024)
- [x] Create `assets/icon.png` for Linux (reuse from the iconset)
- [ ] Set the window icon in `main.py` via `app.setWindowIcon(QIcon(icon_path))`

**4.4 Create DMG background**

- [x] Create `assets/dmg-background.png` (660x400) with drag-to-Applications visual

**4.5 Local build script**

- [x] Create `packaging/macos/build.sh`:
  - Install core deps from `requirements-core.txt`
  - Run PyInstaller with the spec file
  - Post-build: verify the .app launches
  - Optional: code sign locally for testing
  - Create DMG using `create-dmg`

**4.6 Add logging to file when frozen**

- [x] In `main.py`, when `is_frozen()`, configure logging to write to `~/Library/Logs/Scene Ripper/scene-ripper.log` in addition to stderr
- [ ] Add log rotation (max 5 MB, keep 3 backups)
- [ ] Add "Open Log File" option in Help menu for diagnostics

#### Phase 5: CI/CD Pipeline

**5.1 GitHub Actions workflow**

- [x] Create `.github/workflows/build-macos.yml`:
  - Trigger: push tags matching `v*` + `workflow_dispatch` with version input
  - Runner: `macos-14` (Apple Silicon arm64)
  - Steps:
    1. Checkout code
    2. Setup Python 3.11
    3. Install `requirements-core.txt` + PyInstaller
    4. Import signing certificate from secrets (base64-decoded .p12 → temp keychain)
    5. Store notarization credentials (`xcrun notarytool store-credentials`)
    6. Run PyInstaller with spec file
    7. Code sign the .app (`codesign --force --deep --options runtime --entitlements ... --sign ...`)
    8. Verify signature (`codesign --verify --deep --strict`)
    9. Create branded DMG (`create-dmg` with background, icon positioning, Applications alias)
    10. Notarize DMG (`xcrun notarytool submit --wait`)
    11. Staple ticket (`xcrun stapler staple`)
    12. Upload DMG to GitHub Release

**5.2 Required GitHub Secrets**

| Secret | Value |
|--------|-------|
| `MACOS_CERTIFICATE` | Base64-encoded Developer ID Application .p12 |
| `MACOS_CERTIFICATE_PWD` | Password for the .p12 |
| `KEYCHAIN_PASSWORD` | Random password for temp CI keychain |
| `CODESIGN_IDENTITY` | `"Developer ID Application: Name (TEAMID)"` |
| `APPLE_ID` | Apple ID email |
| `APPLE_TEAM_ID` | Developer Team ID |
| `APPLE_APP_PASSWORD` | App-specific password from appleid.apple.com |

**5.3 Release workflow**

- [ ] Tag-based release: `git tag v0.2.0 && git push --tags` triggers build
- [ ] DMG artifact attached to the GitHub Release automatically
- [ ] Release notes template with download size noted

#### Phase 6: Testing & Validation

- [ ] Local PyInstaller build produces a launchable .app
- [ ] .app launches on a clean macOS 13+ system without Python installed
- [ ] Core workflow works without FFmpeg (UI degrades gracefully, shows download prompt)
- [ ] FFmpeg download + install works, enables cut/render features
- [ ] yt-dlp download + install works, enables YouTube downloads
- [ ] On-demand Python package install works for at least one ML feature (e.g., shot classification)
- [ ] Code signing passes: `codesign --verify --deep --strict`
- [ ] Gatekeeper passes: `spctl --assess --type exec --verbose`
- [ ] Notarization succeeds: `xcrun stapler validate`
- [ ] DMG opens with branded background and drag-to-Applications alias
- [ ] Version update checker finds newer release and shows banner
- [ ] App update (replacing .app in /Applications) preserves: settings, downloaded deps, API keys in Keychain
- [ ] Keychain API keys survive app update (same bundle ID, same Team ID signing)
- [ ] Settings migration from XDG paths works on first frozen launch

## Alternative Approaches Considered

| Approach | Why Rejected |
|----------|-------------|
| **Briefcase (BeeWare)** | Less battle-tested with complex ML dependency stacks, smaller community |
| **Nuitka** | Very long compile times, harder to debug, less mature macOS .app support |
| **py2app** | Less actively maintained than PyInstaller, fewer PySide6 success stories |
| **Bundle everything (~3-4 GB)** | Too large for initial download; most users won't use all ML features |
| **Minimal + all on-demand (~500 MB)** | Core workflow (cut/render) needs PySide6+OpenCV which are already 1.2 GB |
| **Universal binary (arm64+x86_64)** | Doubles bundle size, Intel Mac market is shrinking, MLX only works on arm64 |
| **pip install --target from frozen app directly** | `sys.executable` points to frozen binary, not a Python interpreter — pip cannot run |
| **System Python fallback for pip** | Unreliable — user may not have Python, or may have wrong version (ABI mismatch) |

(see brainstorm for full approach discussion)

## System-Wide Impact

### Interaction Graph

- **`main.py` startup** → checks `is_frozen()` → injects managed packages into `sys.path` → sets up logging to file → creates QApplication → applies theme → shows MainWindow
- **`FFmpegProcessor.__init__`** → calls `find_binary("ffmpeg")` → if missing + frozen → sets `ffmpeg_available=False` → UI shows download prompt instead of raising
- **On-demand download triggered** → `DependencyManager` runs in QThread worker → emits progress signals → UI shows progress → on completion emits `finished` signal → feature retries
- **yt-dlp update** → `DependencyManager.update_yt_dlp()` → downloads to bin/ → replaces old binary → no app restart needed
- **Settings save** → if frozen on macOS → writes to `~/Library/Application Support/` → keyring stores via bundle ID

### Error & Failure Propagation

- **Download fails midway**: Worker catches exception → emits `error(message)` signal → UI shows retry dialog → partial file cleaned up
- **pip install fails**: Worker captures stderr → emits error with pip output → UI shows "Installation failed" with details + retry
- **ABI mismatch after update**: App detects `compat_version` mismatch on startup → shows "Dependencies need updating" banner → user re-downloads
- **Code signing breaks from manual modification**: No code inside the .app is modified at runtime — all mutable data is in Application Support

### State Lifecycle Risks

- **Partial package install**: If pip install is interrupted, the `packages/` directory may have incomplete files. Mitigation: install to a temp dir first, move to `packages/` on success (atomic operation).
- **Keychain ACL after update**: macOS Keychain ties access to code signatures. If the signing identity changes (different certificate, different Team ID), API keys become inaccessible. Mitigation: always sign with the same Team ID; use `kSecAttrAccessGroup` tied to Team ID.
- **Stale yt-dlp**: YouTube breaks old versions regularly. Mitigation: auto-update check on first use each session, with manual update button in Settings.

### API Surface Parity

- All `shutil.which()` call sites (11+) must be migrated to `find_binary()`
- `core/settings.py` path resolution must branch on `is_frozen()` for macOS
- `main.py` must add startup hooks for `sys.path` injection and logging setup

## Acceptance Criteria

### Functional Requirements

- [ ] User can download a DMG from GitHub Releases
- [ ] User can drag .app to /Applications and launch it without Python installed
- [ ] Gatekeeper does not block the app (signed + notarized)
- [ ] Core workflow (import video, cut scenes, browse clips, arrange sequence, render) works after FFmpeg download
- [ ] FFmpeg download prompt appears on first action requiring it
- [ ] yt-dlp download prompt appears on first YouTube download attempt
- [ ] At least one ML feature (shot classification) works after on-demand package install
- [ ] Version update banner appears when a new release is available
- [ ] Settings, API keys, and downloaded dependencies survive app updates
- [ ] Settings are migrated from XDG paths on first frozen launch

### Non-Functional Requirements

- [ ] Initial DMG download: < 1 GB compressed
- [ ] App launch time: < 5 seconds to main window visible
- [ ] FFmpeg download: < 2 minutes on broadband
- [ ] On-demand install progress is visible and cancellable
- [ ] Crash logs available at `~/Library/Logs/Scene Ripper/`

### Quality Gates

- [ ] `codesign --verify --deep --strict` passes
- [ ] `spctl --assess --type exec` passes
- [ ] `xcrun stapler validate` passes
- [ ] All existing unit tests pass with `is_frozen() == False` (no regression)
- [ ] Manual smoke test on clean macOS 13+ without dev tools

## Dependencies & Prerequisites

- Apple Developer account with Developer ID Application certificate
- GitHub Actions secrets configured for signing + notarization
- App icon (.icns) and DMG background (.png) assets created
- `create-dmg` available via Homebrew on CI runner
- macOS 14 arm64 GitHub Actions runner available (`macos-14`)

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Qt plugin discovery fails in bundle | Medium | App won't launch | PyInstaller handles this; add `QT_PLUGIN_PATH` fallback in runtime hook |
| On-demand pip install fails (ABI issues) | Medium | ML features broken | Pin exact package versions in manifest; test full install matrix in CI |
| Notarization rejected | Low | Can't distribute | Test entitlements locally first; check notarytool log for specifics |
| PySide6 size doesn't shrink with excludes | Low | Bundle too large | Measure actual savings early in Phase 4; accept up to 2 GB if needed |
| Python 3.11 arm64 download URL changes | Low | First-run install breaks | Fallback URLs; version pinned in manifest; alert on failure |
| yt-dlp goes stale within weeks | High | YouTube downloads fail | Auto-update on each session; standalone binary in Application Support |
| Keychain prompts on every update | Medium | Bad UX | Sign with consistent Team ID; use access group |
| GitHub Actions macOS runner unavailable | Low | Can't build in CI | Use `workflow_dispatch` for manual builds on local machine as fallback |

## Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: Foundation | Medium — centralized path/binary resolution, lazy imports, dep separation |
| Phase 2: Dependency Manager | Large — download logic, pip orchestration, manifest system, progress UI |
| Phase 3: UI Integration | Medium — first-run UX, download prompts, settings page, update checker |
| Phase 4: PyInstaller Config | Medium — spec file, entitlements, icon, DMG, build script |
| Phase 5: CI/CD | Medium — GitHub Actions workflow, secrets, signing, notarization |
| Phase 6: Testing | Medium — smoke tests on clean systems, signing verification |

Recommended order: Phase 1 → Phase 4 (get a basic .app working) → Phase 2 → Phase 3 → Phase 5 → Phase 6

## Future Considerations

- **Linux AppImage refresh**: Apply the same dependency manager pattern to the Linux AppImage build
- **Windows .exe**: PyInstaller supports Windows; the `core/paths.py` and `core/binary_resolver.py` modules are cross-platform
- **Auto-update mechanism**: Replace "open browser" with in-app delta updates (e.g., Sparkle framework)
- **App Store distribution**: Would require full sandboxing, which conflicts with subprocess calls (FFmpeg, yt-dlp)
- **Homebrew cask**: `brew install --cask scene-ripper` pointing to GitHub Releases

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-02-21-macos-app-bundle-brainstorm.md](../brainstorms/2026-02-21-macos-app-bundle-brainstorm.md) — Key decisions carried forward: PyInstaller, Apple Silicon only, core+on-demand bundle strategy, GitHub Releases distribution, branded DMG, in-app update checker.

### Internal References

- Existing Linux packaging: `packaging/linux/AppImageBuilder.yml`, `packaging/linux/build-appimage.sh`
- Binary discovery pattern: `core/downloader.py:18-45` (`_get_subprocess_env()`)
- All `shutil.which()` call sites: `core/ffmpeg.py`, `core/thumbnail.py`, `core/sequence_export.py`, `core/downloader.py`, `core/analysis/description.py`, `ui/dialogs/intention_import_dialog.py`
- Top-level litellm imports: `core/analysis/description.py:22`, `core/analysis/cinematography.py:26`
- Settings paths: `core/settings.py:290-325`
- App entry point: `main.py:13-43`
- Keyring usage: `core/settings.py:61-146`

### External References

- PyInstaller macOS bundling: https://pyinstaller.org/en/stable/
- PyInstaller code signing wiki: https://github.com/pyinstaller/pyinstaller/wiki/Recipe-OSX-Code-Signing
- Apple notarization: https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution
- `create-dmg` (shell): https://github.com/create-dmg/create-dmg
- Static FFmpeg for macOS arm64: https://osxexperts.net/
- GitHub Actions macOS runners: https://github.com/actions/runner-images
