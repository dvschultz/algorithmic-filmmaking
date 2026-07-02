# Installation

Scene Ripper packaged builds are distributed from the GitHub Releases page:

https://github.com/dvschultz/algorithmic-filmmaking/releases/latest

## Requirements

- macOS 13 Ventura or newer on Apple Silicon for the downloadable `.dmg`.
- Windows 10/11 64-bit for the Windows installer.
- A recent x86_64 Linux desktop for the AppImage.
- Python 3.11 or newer when running from source.
- At least 5 GB of free disk space. Use 10 GB or more if you plan to run local transcription, face detection, embeddings, or import large videos.

The macOS app bundle declares `LSMinimumSystemVersion=13.0`, so macOS should block launch on older unsupported systems before the app starts.

## macOS

1. Open the latest release.
2. Expand **Assets**.
3. Download the Apple Silicon DMG named like `Scene-Ripper-<version>-arm64.dmg`.
4. Open the DMG and drag `Scene Ripper.app` into `Applications`.
5. Launch from `Applications`.

The downloadable macOS app is Apple Silicon only. On Intel Macs, run from source instead.

## Windows

1. Open the latest release.
2. Expand **Assets**.
3. Download `SceneRipper-Setup-<version>.exe`.
4. Run the installer.

## Linux

1. Open the latest release.
2. Expand **Assets**.
3. Download `Scene_Ripper-<version>-x86_64.AppImage`.
4. Mark it executable and run it:

```bash
chmod +x Scene_Ripper-*-x86_64.AppImage
./Scene_Ripper-*-x86_64.AppImage
```

## Storage

The app downloads and caches runtime tools, thumbnails, logs, and optional ML packages outside the project repository:

- macOS: `~/Library/Application Support/Scene Ripper/`
- Windows: `%LOCALAPPDATA%\Scene Ripper\`
- Linux: `~/.local/share/scene-ripper/`

Large local models and package installs can push this folder above 5 GB. If a model install fails, check available disk space first.

## On-Demand Installs

Some features install their dependencies the first time you use them, after an in-app prompt:

- **Python packages** are installed via pip into the app's managed packages directory inside the app-support folder above (PyTorch, object detection, OCR, audio analysis, and similar ML libraries).
- **Binaries** (FFmpeg/FFprobe, yt-dlp, Deno) and **model weights** (YOLO, MediaPipe, local vision models) are downloaded as needed.

Nothing is installed system-wide, nothing downloads until you use a feature that needs it, and removing the app-support directory removes everything the app installed.
