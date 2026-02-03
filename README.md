# Scene Ripper

Automatic scene detection, AI-powered analysis, and algorithmic video remixing for filmmakers and video artists.

## What It Does

Scene Ripper analyzes video files to detect scene boundaries, enriches clips with AI-generated metadata (descriptions, shot types, transcripts), and lets you remix clips into new sequences using algorithmic composition or AI assistance.

**Key Features:**
- Automatic scene detection with adjustable sensitivity
- Text-based scene detection for karaoke, subtitles, and slides
- Intention-first workflow (one-click sequence creation from URLs)
- AI-powered clip analysis (descriptions, shot types, transcription, colors)
- Film language analysis with scene reports and editing suggestions
- Integrated chat agent for AI-assisted editing
- Thumbnail grid browser with collapsible source headers
- Multiple sequencing modes with shot type filtering and direction control
- Export individual clips or complete sequences
- YouTube/Internet Archive import
- Project save/load with full state persistence

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

**Tab-Based Workflow:**

The app uses a 5-tab workflow, though you can skip or revisit tabs as needed:

| Tab | Purpose |
|-----|---------|
| **Collect** | Import videos (local files, YouTube, Internet Archive) |
| **Cut** | Detect scenes and browse clips |
| **Analyze** | Enrich clips with AI metadata (descriptions, transcripts, etc.) |
| **Sequence** | Arrange clips using various sequencing modes |
| **Render** | Export final video or EDL |

**Basic Flow:**
1. **Collect**: Import a video file (drag-drop or file browser)
2. **Cut**: Adjust sensitivity and click "Detect Scenes"
3. **Analyze**: (Optional) Run analysis to add descriptions, transcripts, shot types
4. **Sequence**: Add clips to timeline, use shuffle or Exquisite Corpus mode
5. **Render**: Export as MP4 or EDL for external editors

## Features

### Scene Detection
- PySceneDetect with AdaptiveDetector
- Sensitivity slider (1.0 sensitive → 10.0 less sensitive)
- **Text-based detection mode** for karaoke, subtitles, and presentation slides—uses OCR to detect text changes as scene boundaries with configurable ROI and similarity threshold
- Background processing keeps UI responsive
- Progress reporting

### AI-Powered Analysis

Enrich clips with metadata using local or cloud AI models:

| Analysis | Description | Models |
|----------|-------------|--------|
| **Describe** | Natural language description of clip content | GPT-4o, Claude, Gemini, Moondream (local) |
| **Classify** | Shot type detection (close-up, medium, wide, etc.) | CLIP (local), VideoMAE (cloud) |
| **Cinematography** | Full film language analysis (shot size, angle, movement, lighting, composition) | Gemini (video or frame) |
| **Transcribe** | Speech-to-text | faster-whisper (local) |
| **Colors** | Dominant color palette extraction | OpenCV |
| **Objects** | Object detection | YOLO |

Supports both local (free, private) and cloud (higher quality) processing tiers.

**Shot Type Classification Tiers:**
- **CPU (local)**: CLIP zero-shot classification with ensemble prompts—runs on thumbnails, free and private
- **Cloud**: VideoMAE model on Replicate—analyzes video segments for significantly better accuracy, requires API key

### Cinematography Analysis

Rich VLM-powered analysis providing professional-grade film language breakdown:

| Category | Properties |
|----------|------------|
| **Shot Size** | Granular classification: ELS, VLS, LS, MLS, MS, MCU, CU, BCU, ECU, Insert |
| **Camera Angle** | low_angle, eye_level, high_angle, dutch_angle, birds_eye, worms_eye + emotional effect |
| **Camera Movement** | static, pan, tilt, track, handheld, crane, arc (video mode only) |
| **Composition** | Subject position, headroom, lead room, visual balance |
| **Lighting** | Style (high/low key), direction, quality (hard/soft), color temperature |
| **Technical** | Dutch tilt detection, lens type estimation, depth of field |
| **Derived** | Emotional intensity, suggested pacing |

**Analysis Modes:**
- **Frame mode**: Fast analysis from thumbnails
- **Video mode**: Analyzes full clip segment for camera movement detection (requires Gemini API)

### Film Language Analysis

Generate scene reports with cinematography analysis:
- **Pacing metrics**: Shot duration distribution, rhythm analysis
- **Visual consistency**: Color palette coherence, lighting continuity
- **Continuity warnings**: Potential jump cuts, axis crosses
- **Editing suggestions**: AI-powered recommendations for improving flow

### Chat Agent

Integrated AI assistant that can help with editing tasks:
- Navigate between tabs
- Select and filter clips by metadata
- Run analysis operations
- Build sequences programmatically
- Answer questions about your project

Configure your preferred LLM provider (OpenAI, Anthropic, Gemini, Ollama) in Settings.

### Sequencing Modes

Multiple ways to arrange clips:

- **Manual**: Drag-drop clips to timeline
- **Shuffle**: Randomize with constraints (no same-source consecutive)
- **Exquisite Corpus**: Generate poetry from extracted text, then sequence clips by matching lines

**Sorting options:**
- **Shot Type**: Filter by Wide Shot, Full Shot, Medium Shot, Close-up, or Extreme Close-up
- **Duration**: Sort by clip length (shortest first or longest first)
- **Color**: Arrange by dominant color (rainbow, warm to cool, cool to warm)

### Intention-First Workflow

Start from a creative intention and let the app handle the rest:
1. Click a sequence card or create a new intention
2. Paste YouTube URLs or select local files in the import dialog
3. The app automatically downloads, detects scenes, generates thumbnails, analyzes clips, and builds your sequence
4. All in one flow—no tab-switching required

### Clip Browser
- Thumbnail grid of detected scenes
- **Collapsible source headers**: Clips grouped by source with expand/collapse, showing clip count and selection state—reduces clutter when working with many sources
- Duration labels on each clip
- Click to preview, double-click for full playback
- Filter by metadata (shot type, has transcript, etc.)

### Export
- Individual clip export (frame-accurate)
- Batch export (all or selected)
- Sequence export (timeline → single video)
- EDL export for external NLE software
- Dataset export for ML training

### Source Import
- Local video files (drag-drop or browser)
- YouTube via yt-dlp
- Internet Archive collections
- Automatic scene detection after download

### Projects
- Save/load project state (sources, clips, sequences, settings)
- JSON-based project format
- Auto-recovery support

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

## Configuration

### API Keys (Optional)

Cloud AI features require API keys. Configure in Settings → API Keys:

| Provider | Features | Get Key |
|----------|----------|---------|
| OpenAI | GPT-4o descriptions, chat | [platform.openai.com](https://platform.openai.com) |
| Anthropic | Claude descriptions, chat | [console.anthropic.com](https://console.anthropic.com) |
| Google | Gemini descriptions (supports video input), chat | [aistudio.google.com](https://aistudio.google.com) |
| Replicate | VideoMAE shot classification (cloud tier) | [replicate.com](https://replicate.com) |
| YouTube | Search and metadata | [console.cloud.google.com](https://console.cloud.google.com) |

Keys are stored securely in your system keyring. Environment variables (e.g., `ANTHROPIC_API_KEY`) take priority.

**Local-only operation**: Scene detection, transcription (Whisper), and local vision (Moondream) work without any API keys.

## Technology Stack

| Component | Technology |
|-----------|------------|
| UI Framework | PySide6 (Qt 6) |
| Scene Detection | PySceneDetect |
| Video Processing | FFmpeg |
| Video Download | yt-dlp |
| Computer Vision | OpenCV |
| Transcription | faster-whisper |
| LLM Integration | LiteLLM (multi-provider) |
| Local Vision | Moondream 2B |

## Requirements

**Core:**
```
PySide6>=6.6
scenedetect[opencv]>=0.6.4
opencv-python>=4.8
numpy>=1.24
yt-dlp>=2024.1
```

**AI Features (optional but recommended):**
```
litellm              # Multi-provider LLM support
faster-whisper       # Local transcription
transformers         # Local vision (Moondream)
```

See `requirements.txt` for the full list.

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

## CLI

Command-line interface for batch processing:

```bash
# Scene detection
python -m cli.main detect video.mp4 --threshold 3.0

# Transcription
python -m cli.main transcribe video.mp4

# YouTube search
python -m cli.main youtube search "query"

# Export project
python -m cli.main export project.json --format edl

# Full help
python -m cli.main --help
```

## Roadmap

**Complete:**
- [x] Scene detection with AdaptiveDetector
- [x] Text-based scene detection (karaoke/subtitle mode)
- [x] Thumbnail browser with filtering
- [x] Collapsible source headers in clip browser
- [x] Video preview player
- [x] Individual and batch clip export
- [x] URL import (YouTube, Internet Archive)
- [x] Multi-track timeline
- [x] Sequence export (MP4, EDL)
- [x] Shuffle remix algorithm
- [x] Linux support (AppImage packaging)
- [x] AI-powered descriptions (GPT-4o, Claude, Gemini, Moondream)
- [x] Shot type classification and filtering
- [x] Cinematography analysis (VLM-powered film language breakdown)
- [x] Transcription (faster-whisper)
- [x] Color analysis with direction control
- [x] Integrated chat agent
- [x] Exquisite Corpus sequencer
- [x] CLI for batch processing
- [x] Project save/load
- [x] Intention-first workflow
- [x] Film language analysis / scene reports

**In Progress:**
- [ ] Timeline playback preview
- [ ] Object detection (YOLO)

**Future:**
- [ ] FAISS vector similarity search
- [ ] Similarity-based sequencing
- [ ] Motion-based ordering
