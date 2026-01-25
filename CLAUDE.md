# Scene Ripper - Project Conventions

## Overview

Pure Python MVP for automatic scene detection in video files. Built with PySide6, PySceneDetect, and FFmpeg.

## Technology Stack

- **UI Framework**: PySide6 (Qt 6 for Python)
- **Scene Detection**: PySceneDetect with AdaptiveDetector
- **Video Processing**: FFmpeg (via subprocess)
- **Python Version**: 3.11+

## Project Structure

```
algorithmic-filmmaking/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── ui/
│   ├── main_window.py      # Main application window
│   ├── clip_browser.py     # Thumbnail grid browser
│   └── video_player.py     # Video preview player
├── core/
│   ├── scene_detect.py     # PySceneDetect wrapper
│   ├── ffmpeg.py           # FFmpeg operations
│   └── thumbnail.py        # Thumbnail generation
└── models/
    └── clip.py             # Data models (Source, Clip)
```

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure FFmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html

# Run the app
python main.py
```

## Key Patterns

### Scene Detection
Use `AdaptiveDetector` for dynamic footage (default). Threshold range: 1.0 (sensitive) to 10.0 (less sensitive).

### Background Processing
Heavy operations (detection, thumbnails) run in `QThread` workers to keep UI responsive.

### FFmpeg Safety
Always use argument arrays, never shell interpolation. Validate paths before processing.

### User Settings
Always use paths and configuration from `core/settings.py` (loaded via `load_settings()`). Never hardcode paths or use defaults when a user-configurable setting exists:
- `settings.download_dir` for video downloads
- `settings.project_dir` for project files
- `settings.cache_dir` for thumbnails and cache

When adding new features that involve file paths, check if a relevant setting exists and use it. If not, consider adding one to the Settings dataclass.

## Testing

```bash
# Run with a test video
python main.py
# Drag-drop a video file or use Import button
```

## Common Commands

```bash
# Scene detection CLI test
python -c "from core.scene_detect import SceneDetector; print(SceneDetector)"

# Check FFmpeg
ffmpeg -version
```
