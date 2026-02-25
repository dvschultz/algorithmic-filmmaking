# Windows Version Brainstorm

**Date**: 2026-02-24
**Status**: Complete

## What We're Building

A Windows version of Scene Ripper with full feature parity, packaged as a PyInstaller `.exe` installer for general (non-technical) users. FFmpeg and yt-dlp download automatically on first launch. Apple Silicon-only ML features (MLX Whisper, MLX VLM) are replaced with cross-platform equivalents (faster-whisper for transcription, LiteLLM cloud APIs for VLM analysis).

## Why This Approach

- **PyInstaller .exe**: Matches the existing macOS packaging approach (`.spec` file already exists). Well-documented for PySide6 apps. GitHub Actions can build it without a Windows machine.
- **First-launch dependency download**: Keeps installer small (~50MB vs ~150MB+). Mirrors the macOS auto-download pattern already in `core/dependency_manager.py`.
- **faster-whisper + LiteLLM API**: Avoids GPU driver complexity (CUDA toolkit, NVIDIA deps). faster-whisper runs on CPU. VLM analysis goes through cloud APIs the user already configures. No new local ML dependencies.
- **CI-based build + cloud VM for debugging**: Cheapest path. GitHub Actions `windows-latest` is free. Cloud free-tier VM available for manual debugging when needed.

## Key Decisions

1. **Packaging**: PyInstaller `.exe` wrapped in Inno Setup installer, built via GitHub Actions CI
2. **ML backends**: faster-whisper (CPU-only) for transcription, LiteLLM APIs for VLM — no local GPU inference on Windows v1
3. **Binary dependencies**: FFmpeg + yt-dlp auto-downloaded on first launch (Windows builds from BtbN/GitHub releases)
4. **Testing**: GitHub Actions `windows-latest` runners as primary test platform; free-tier cloud VM for manual GUI debugging
5. **Feature scope**: Full parity with macOS (minus Apple-only hardware acceleration)
6. **Min Windows version**: Windows 10+
7. **Code signing**: Skip for v1 (users click "Run anyway")
8. **Timeline**: ASAP

## Current State Analysis

The codebase is partially Windows-ready. Here's what works and what doesn't.

### Already Windows-Compatible

- **PySide6 UI**: Inherently cross-platform, no platform-specific Qt code
- **Path handling**: Uses `pathlib.Path` consistently, no string concatenation with `/`
- **Config/cache dirs**: `core/settings.py` already has `os.name == "nt"` branches
- **Filename sanitization**: `core/downloader.py` already checks Windows reserved names (CON, PRN, AUX, etc.)
- **Open folder**: `ui/main_window.py` has `sys.platform == "win32"` branch using `explorer`
- **Apple-only deps**: Already gated with `sys_platform == 'darwin'` in requirements.txt
- **Subprocess calls**: All use argument arrays (never `shell=True`)
- **Clip path serialization**: `models/clip.py` uses `.as_posix()` with cross-drive fallback

### Must Fix (12 Issues)

#### Critical (Would break on Windows)

| # | File | Issue |
|---|------|-------|
| 1 | `core/dependency_manager.py` | All download URLs are macOS ARM64 binaries. All `ensure_*` functions reject non-ARM64. Need Windows download URLs (FFmpeg from gyan.dev, yt-dlp.exe from GitHub). |
| 2 | `core/transcription.py` (lines 333, 395, 524) | Bare `"ffmpeg"` subprocess calls bypass `find_binary()`. Would fail if FFmpeg is not on system PATH. |
| 3 | `core/analysis/audio.py` (lines 146, 220, 370) | Same bare `"ffprobe"` and `"ffmpeg"` calls bypassing `find_binary()`. |
| 4 | `core/analysis/embeddings.py` (line 207) | Same bare `"ffmpeg"` call. |
| 5 | `core/analysis/audio.py` (line 376-377) | FFmpeg null output uses `"-f", "null", "-"` — must be `"NUL"` on Windows. |
| 6 | `cli/utils/signals.py` | `signal.SIGTERM` cannot be caught on Windows. May cause runtime error. |

#### High (Non-standard / broken behavior)

| # | File | Issue |
|---|------|-------|
| 7 | `core/paths.py` | `get_app_support_dir()` and `get_log_dir()` fall to Linux paths on Windows. Need `%APPDATA%` branch. |
| 8 | `core/binary_resolver.py` | Managed binary lookup uses name without `.exe`. `os.access(path, os.X_OK)` unreliable on Windows. |
| 9 | `core/chat_tools.py` + `scene_ripper_mcp/security.py` | Safe path roots include `Path("/tmp")` (doesn't exist on Windows). No Windows-specific safe roots. |
| 10 | `core/downloader.py` | Default download dir `~/Movies/...` is macOS-specific. Should use settings `_get_videos_dir()`. |

#### Medium (Works but non-ideal)

| # | File | Issue |
|---|------|-------|
| 11 | `core/thumbnail.py`, `core/analysis/classification.py`, `core/analysis/detection.py` | Fallback cache paths use `~/.cache/` (Linux convention). Creates dot-directories on Windows. |
| 12 | Subprocess calls generally | Should add `CREATE_NO_WINDOW` flag to prevent console windows flashing behind the GUI. |

## Build & Packaging Plan

### PyInstaller Spec

Create `packaging/windows/scene_ripper.spec` modeled after the existing `packaging/macos/scene_ripper.spec`:
- Bundle PySide6, all Python dependencies
- Include app icon (`.ico` format — need to create from existing macOS `.icns`)
- Set `console=False` to suppress console window
- Use `--onedir` mode (faster startup than `--onefile`)

### GitHub Actions Workflow

Create `.github/workflows/build-windows.yml`:
- Trigger: tag push (matching macOS/Linux pattern)
- Runner: `windows-latest`
- Steps: checkout, install Python 3.11, pip install deps, run tests, PyInstaller build
- Artifact: upload `.exe` installer or `.zip` of the built directory

### Dependency Auto-Download (Windows)

Extend `core/dependency_manager.py` with Windows support:
- FFmpeg: Download from `https://github.com/BtbN/FFmpeg-Builds/releases` (Windows x64 static build)
- yt-dlp: Download `yt-dlp.exe` from GitHub releases
- No need for managed Python (it's bundled by PyInstaller)

### Code Signing

Windows Defender SmartScreen will flag unsigned `.exe` files. Options:
- **No signing initially**: Users click "Run anyway" — acceptable for early adopters
- **Self-signed**: Still triggers SmartScreen
- **Paid code signing cert**: ~$200-400/year. Eliminates SmartScreen warnings. Consider later.

## Testing Strategy

### Automated (CI)

- Run full `pytest` suite on `windows-latest` GitHub Actions runner
- Use `QT_QPA_PLATFORM=offscreen` for headless Qt testing (same as Linux CI)
- Build PyInstaller `.exe` and verify it launches (`--help` flag or smoke test)

### Manual (When Needed)

- **Free cloud VM**: Azure free tier, AWS free tier, or GitHub Codespaces for manual GUI testing
- **UTM on Mac**: Free, open-source VM. Can run Windows 11 ARM. Good for quick local checks but ARM may introduce its own quirks.
- **Key manual tests**: Installer runs, first-launch dependency download works, scene detection + export pipeline completes, video player renders correctly

### What Can't Be Automated

- Visual appearance of PySide6 widgets on Windows (font rendering, DPI scaling, dark mode)
- Antivirus/SmartScreen behavior with the installer
- First-launch UX (dependency download progress, error messages)
- Multi-monitor / high-DPI behavior

## Resolved Questions

1. **Windows 10 vs 11**: Windows 10+ minimum. Covers ~95% of Windows users. Python 3.11 and PySide6 both support it.

2. **Code signing budget**: Skip for now. Users click "Run anyway". Revisit if distribution grows.

3. **Installer format**: Inno Setup wrapping the PyInstaller output. Start menu shortcut, uninstaller, optional desktop icon. Free and well-documented.

4. **GPU transcription**: CPU-only for v1. Keeps things simple, no CUDA dependency. Can add GPU support later.
