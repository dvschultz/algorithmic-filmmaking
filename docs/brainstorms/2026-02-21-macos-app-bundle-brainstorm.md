# Brainstorm: macOS .app Bundle Distribution

**Date:** 2026-02-21
**Status:** Ready for planning

## What We're Building

A downloadable macOS .app bundle for Scene Ripper so users can install and run the app without a terminal, Python environment, or manual dependency installation. Distributed as a DMG via GitHub Releases with code signing and notarization.

---

## Why This Approach

The app currently requires `pip install`, a working Python 3.11+ environment, FFmpeg, and various system dependencies. This is a barrier for non-technical users and even technical users who don't want to manage Python environments for a GUI tool.

---

## Key Decisions

### 1. Platform Target: macOS Apple Silicon only (arm64)

- Intel Mac users excluded (simplifies build, MLX features work natively)
- No universal binary needed — avoids doubling bundle size
- MLX acceleration (Whisper, VLMs) included by default

### 2. Packaging Tool: PyInstaller

- Most battle-tested Python-to-app bundler
- Strong PySide6/Qt6 support (handles Qt plugins, platform plugins)
- Built-in macOS code signing and notarization support
- Large community with documented solutions for edge cases
- Produces a `.app` bundle from a `.spec` configuration file

### 3. Bundle Strategy: Core + On-Demand Downloads

**Bundled in .app (~1.5-2 GB):**
- Python 3.11 runtime
- PySide6 (~1.1 GB — the dominant cost)
- OpenCV (~100 MB)
- PySceneDetect
- NumPy, Pillow, scikit-learn
- LiteLLM, httpx, tenacity
- Core app code

**Downloaded on first use (not bundled):**
- FFmpeg + FFprobe (static binaries, ~80-150 MB)
- yt-dlp (with self-update mechanism since YouTube breaks older versions)
- PyTorch (~350 MB) — only needed for ML features
- transformers (~100 MB) — only for HuggingFace models
- PaddleOCR + PaddlePaddle (~200-400 MB) — only for OCR
- ultralytics (~30-50 MB) — only for object detection
- librosa + scipy (~80 MB) — only for audio analysis
- ML model weights (SigLIP 2, DINOv2, Moondream, Whisper) — 1-5 GB total
- MLX stack (mlx, mlx-vlm, lightning-whisper-mlx)

**Rationale:** Core video editing workflow (collect, cut, sequence, render) works immediately. ML-powered features (describe, classify, transcribe, detect objects) trigger downloads when first used.

### 4. External Binary Management

- **FFmpeg/FFprobe**: Download static arm64 builds on first launch. Store in `~/Library/Application Support/Scene Ripper/bin/`. App checks this path before `shutil.which()`.
- **yt-dlp**: Download on first use. Include an update mechanism (yt-dlp has `--update` support, or we pip-install into a managed venv).
- **Deno**: Download if needed for yt-dlp YouTube challenge solving.

### 5. Code Signing & Notarization

- Apple Developer account available ($99/year)
- Sign .app bundle with Developer ID certificate
- Notarize with Apple for Gatekeeper approval
- Users get a clean install experience (no right-click > Open workaround)

### 6. Distribution: GitHub Releases

- CI builds DMG automatically on tagged release (e.g., `v0.2.0`)
- GitHub Actions workflow on macOS runner (arm64)
- DMG uploaded as release asset
- Users download from Releases page

### 7. CI Pipeline

- Triggered by git tag push (`v*`)
- macOS arm64 GitHub Actions runner
- Steps: install deps → PyInstaller build → code sign → create DMG → notarize → upload to release
- Secrets needed: Apple Developer certificate, notarization credentials, GitHub token

---

## Architecture: On-Demand Dependency Manager

The app needs a dependency manager that:

1. **Checks** if a dependency is available before features that need it
2. **Downloads** missing dependencies with a progress UI
3. **Stores** them in `~/Library/Application Support/Scene Ripper/`
4. **Updates** the app's PATH/PYTHONPATH to find them
5. **Gracefully disables** features when dependencies are missing (show "Install X to use this feature" instead of crashing)

### Dependency Categories

| Category | Examples | Trigger |
|----------|----------|---------|
| **Binaries** | FFmpeg, FFprobe, yt-dlp, Deno | First use of cut/download features |
| **Python packages** | PyTorch, transformers, ultralytics, paddleocr | First use of ML features |
| **ML models** | SigLIP 2, DINOv2, Whisper weights | First use of specific analysis |

### UI Considerations

- First-launch setup wizard (optional) to pre-download common dependencies
- Per-feature "Download required" buttons in the UI
- Download progress in status bar or modal dialog
- Settings page showing installed/missing optional dependencies

---

## Estimated Bundle Sizes

| Component | Size |
|-----------|------|
| Python runtime | ~30 MB |
| PySide6 + Qt6 | ~1,100 MB |
| OpenCV | ~100 MB |
| App code + small deps | ~100 MB |
| **Initial .app** | **~1.3-1.5 GB** |
| **DMG (compressed)** | **~500-800 MB** |

After on-demand downloads (all features):
| Additional downloads | ~1-2 GB |
| ML model weights | ~1-5 GB |
| **Total on disk** | **~4-8 GB** |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PySide6 Qt plugin discovery fails in bundle | App won't launch | PyInstaller handles this, but needs testing. Set `QT_PLUGIN_PATH` explicitly. |
| On-demand pip install has no isolation | Dependency conflicts possible | Use `pip install --target` to a dedicated directory in Application Support, not inside the signed .app bundle |
| yt-dlp goes stale for YouTube | YouTube downloads stop working | Self-update mechanism, or download latest on each use |
| GitHub Actions doesn't have arm64 macOS runners | Can't build | Apple Silicon runners are available (macos-14+). Verify availability. |
| Code signing certificate expires | Users get Gatekeeper warnings | Set calendar reminder, renew annually |
| Large download size deters users | Low adoption | Compress DMG aggressively, document size upfront |
| PyTorch/CUDA not needed on Apple Silicon | Wasted space if bundled | Don't bundle — download CPU/MPS-only torch on demand |

---

## Resolved Questions

1. **PySide6 tree-shaking**: Yes — strip aggressively. Exclude unused Qt modules (QtWebEngine, Qt3D, QtQuick, QtBluetooth, etc.) via PyInstaller `--exclude-module`. Could save 200-400 MB.

2. **On-demand Python package installation**: Use `pip install --target` to `~/Library/Application Support/Scene Ripper/packages/`. Add that directory to `sys.path` at startup. Simpler than a managed venv, works with PyInstaller's frozen Python, and doesn't modify the signed .app bundle.

3. **Auto-update for the app itself**: Yes — check GitHub Releases API on launch, show a banner if a new version is available with a download link.

4. **DMG styling**: Yes — branded DMG with custom background image, app icon, and Applications alias for drag-to-install.
