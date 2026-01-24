# Scene Ripper

Automatic scene detection and algorithmic video remixing for filmmakers and video artists.

## What It Does

Scene Ripper analyzes video files to detect scene boundaries, displays them as browsable thumbnails, and lets you remix clips into new sequences using algorithmic composition.

**Key Features:**
- Automatic scene detection with adjustable sensitivity
- Thumbnail grid browser with preview
- Multi-track timeline for composition
- Algorithmic remix (shuffle with constraints)
- Export individual clips or complete sequences
- YouTube/Vimeo URL import

## Installation

### Linux

**Option 1: AppImage (Recommended)**

Download the latest AppImage from [Releases](https://github.com/dvschultz/algorithmic-filmmaking/releases):

```bash
# Download (replace VERSION with actual version)
wget https://github.com/dvschultz/algorithmic-filmmaking/releases/download/vVERSION/Scene_Ripper-VERSION-x86_64.AppImage

# Make executable and run
chmod +x Scene_Ripper-*-x86_64.AppImage
./Scene_Ripper-*-x86_64.AppImage
```

**Option 2: From Source**

```bash
# Install system dependencies first (see System Dependencies below)
# Ubuntu/Debian:
sudo apt install python3 python3-pip ffmpeg \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Clone and install
git clone https://github.com/dvschultz/algorithmic-filmmaking.git
cd algorithmic-filmmaking
pip install -r requirements.txt

# Run
python main.py
```

### macOS

```bash
# Install FFmpeg
brew install ffmpeg

# Clone and install
git clone https://github.com/dvschultz/algorithmic-filmmaking.git
cd algorithmic-filmmaking
pip install -r requirements.txt

# Run
python main.py
```

### Windows

1. Install Python 3.11+ from [python.org](https://python.org)
2. Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
3. Clone and run:

```bash
git clone https://github.com/dvschultz/algorithmic-filmmaking.git
cd algorithmic-filmmaking
pip install -r requirements.txt
python main.py
```

## Quick Start

**Run:**
```bash
python main.py
```

**Basic Workflow:**
1. Import a video file (drag-drop or Import button)
2. Adjust sensitivity and click "Detect Scenes"
3. Browse detected clips in the thumbnail grid
4. Drag clips to the timeline to build a sequence
5. Click "Export Sequence" to render the final video

## Features

### Scene Detection
- PySceneDetect with AdaptiveDetector
- Sensitivity slider (1.0 sensitive → 10.0 less sensitive)
- Background processing keeps UI responsive
- Progress reporting

### Clip Browser
- Thumbnail grid of detected scenes
- Duration labels on each clip
- Click to preview, double-click for full playback
- Drag-drop to timeline

### Timeline
- Multi-track composition
- Drag to reposition clips
- Playhead synchronization with preview
- Remix algorithms (Shuffle - no same-source consecutive)
- "Generate" button for algorithmic sequencing

### Export
- Individual clip export (frame-accurate)
- Batch export (all or selected)
- Sequence export (timeline → single video)
- Opens output folder on completion

### URL Import
- YouTube and Vimeo support via yt-dlp
- Automatic scene detection after download

## Project Structure

```
algorithmic-filmmaking/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── core/
│   ├── scene_detect.py  # PySceneDetect wrapper
│   ├── ffmpeg.py        # FFmpeg operations
│   ├── thumbnail.py     # Thumbnail generation
│   ├── downloader.py    # yt-dlp wrapper
│   ├── sequence_export.py
│   └── remix/
│       └── shuffle.py   # Constrained shuffle
├── models/
│   ├── clip.py          # Source, Clip dataclasses
│   └── sequence.py      # Sequence, Track, SequenceClip
└── ui/
    ├── main_window.py   # Main application window
    ├── clip_browser.py  # Thumbnail grid
    ├── video_player.py  # Preview player
    └── timeline/        # Timeline components
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| UI Framework | PySide6 (Qt 6) |
| Scene Detection | PySceneDetect |
| Video Processing | FFmpeg |
| Video Download | yt-dlp |
| Computer Vision | OpenCV |

## Requirements

```
PySide6>=6.6
scenedetect[opencv]>=0.6.4
opencv-python>=4.8
numpy>=1.24
yt-dlp>=2024.1
```

### System Dependencies

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
# FFmpeg and Qt multimedia backend (GStreamer)
sudo apt install ffmpeg \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav
```

**Linux (Fedora):**
```bash
sudo dnf install ffmpeg \
    gstreamer1-plugins-good \
    gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly-free \
    gstreamer1-libav
```

**Linux (Arch):**
```bash
sudo pacman -S ffmpeg \
    gst-plugins-good \
    gst-plugins-bad \
    gst-plugins-ugly \
    gst-libav
```

**Windows:**
Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Roadmap

**Complete:**
- [x] Scene detection
- [x] Thumbnail browser
- [x] Video preview
- [x] Individual clip export
- [x] URL import (YouTube/Vimeo)
- [x] Basic timeline
- [x] Sequence export
- [x] Shuffle remix algorithm
- [x] Linux support (AppImage packaging)

**In Progress:**
- [ ] Timeline playback preview
- [ ] Similarity-based sequencing
- [ ] Motion-based ordering

**Future:**
- [ ] Clip tagging (mood, motion, color)
- [ ] FAISS vector similarity search
- [ ] CLI for batch processing
