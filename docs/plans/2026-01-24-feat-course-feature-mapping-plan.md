---
title: "ITP Algorithmic Filmmaking: Course Feature Mapping"
type: feat
date: 2026-01-24
source: docs/research/ITP Algorithmic Filmmaking.md
---

# ITP Algorithmic Filmmaking: Course Feature Mapping

This document maps every tool and technique from the ITP Algorithmic Filmmaking course to Scene Ripper features, identifying what's built, what's needed, and recommended priorities.

## Executive Summary

| Week | Topic | Features Mapped | Built | Remaining |
|------|-------|-----------------|-------|-----------|
| 1 | Video Datasets | 5 | 4 | 1 |
| 2 | Film Analysis | 7 | 2 | 5 |
| 3 | Content Analysis | 11 | 0 | 11 |
| 4 | Dialogue/Text | 3 | 0 | 3 |
| 5 | Generative | 2 | 0 | 2 |
| 6 | Audio/Rhythm | 3 | 0 | 3 |
| **Total** | | **31** | **6** | **25** |

**Recommended Starting Points (High Impact, Lower Complexity):**
1. 🎯 Color extraction + HSV sorting (Week 2)
2. 🎯 Shot type classification (Week 2)
3. 🎯 Whisper transcription (Week 4)
4. 🎯 Beat detection (Week 6)

---

## Week 1: Video Datasets

**Course Focus**: Creating and managing video clip datasets, scene detection, basic sequencing

### Features

| Feature | Course Tool | Status | Priority | Notes |
|---------|-------------|--------|----------|-------|
| Scene detection | PySceneDetect | ✅ Built | - | AdaptiveDetector implemented |
| URL import | yt-dlp | ✅ Built | - | YouTube/Vimeo with validation |
| Random sequencing | Python shuffle | ✅ Built | - | Constrained shuffle algorithm |
| Thumbnail generation | FFmpeg | ✅ Built | - | Mid-clip frame extraction |
| **JSON dataset export** | Custom format | ⬜ Todo | P2 | Interoperable clip metadata format |

### JSON Dataset Export (New Feature)

**What it does**: Export clip metadata in a format that can be shared, imported, or used with external tools.

**Proposed Schema**:
```json
{
  "version": "1.0",
  "exported_at": "2026-01-24T12:00:00Z",
  "source": {
    "file_path": "video.mp4",
    "duration_seconds": 3600,
    "fps": 29.97,
    "resolution": [1920, 1080]
  },
  "clips": [
    {
      "id": "uuid",
      "start_frame": 0,
      "end_frame": 150,
      "start_time": 0.0,
      "end_time": 5.0,
      "duration_seconds": 5.0,
      "thumbnail_path": "thumbnails/clip_001.jpg",
      "metadata": {
        "colors": [...],
        "tags": [...],
        "motion_score": 0.7
      }
    }
  ]
}
```

---

## Week 2: Film Analysis Tools

**Course Focus**: Extracting visual metadata - duration, dimensions, color, shot types

### Features

| Feature | Course Tool | Status | Priority | Notes |
|---------|-------------|--------|----------|-------|
| Clip duration extraction | FFProbe | ✅ Built | - | Via scene detection |
| **Duration visualization** | Matplotlib | ⬜ Todo | P3 | Bar chart of clip lengths |
| **Dimension analysis** | FFProbe | ⬜ Todo | P3 | Detect mixed resolutions |
| **Color extraction (k-means)** | sklearn | ⬜ Todo | P1 | 🎯 High impact |
| **HSV color sorting** | Custom | ⬜ Todo | P1 | 🎯 High impact |
| **Shot type classification** | Pre-trained model | ✅ Built | P1 | CLIP zero-shot |
| Aspect ratio handling | FFmpeg | ⬜ Todo | P2 | Letterbox/pillarbox on export |

### Color Extraction (High Priority)

**Course Notebooks**:
- Color extraction using kmeans
- HSV Color Sorting

**What it does**:
1. Extract dominant colors from each clip (k-means clustering on frames)
2. Store as RGB/HSV values in clip metadata
3. Enable sorting clips by hue, saturation, or brightness
4. Generate "color barcode" visualizations

**Implementation Approach**:
```python
from sklearn.cluster import KMeans
import cv2

def extract_colors(video_path: str, n_colors: int = 5) -> list[tuple]:
    """Extract dominant colors from video using k-means."""
    # Sample frames evenly throughout clip
    frames = sample_frames(video_path, n_samples=10)

    # Flatten all pixels
    pixels = np.vstack([frame.reshape(-1, 3) for frame in frames])

    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)

    # Return colors sorted by frequency
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    sorted_idx = np.argsort(counts)[::-1]

    return [(tuple(colors[i]), counts[i]) for i in sorted_idx]
```

**UI Integration**:
- Color swatches on clip thumbnails
- "Sort by color" option in clip browser
- Color palette filter (e.g., "show warm clips only")

### Shot Type Classification (High Priority)

**Course Notebook**: Shot Type classification

**Categories** (from course):
- Wide shot / Establishing shot
- Medium shot
- Close-up
- Extreme close-up
- Two-shot
- Over-the-shoulder

**Implementation Approach**:
- Use pre-trained model (ResNet or ViT fine-tuned on shot types)
- Run on middle frame of each clip
- Store as metadata tag

**Model Options**:
1. HuggingFace `shot-type-classifier` (if available)
2. Train simple ResNet classifier on labeled dataset
3. Use CLIP zero-shot: "a wide shot", "a close-up", etc.

---

## Week 3: Content Analysis Tools

**Course Focus**: AI-powered content understanding - classification, detection, captioning

### Features

| Feature | Course Tool | Status | Priority | Notes |
|---------|-------------|--------|----------|-------|
| **Frame classification (ImageNet)** | ResNet/EfficientNet | ⬜ Todo | P2 | General object tags |
| **Video classification (X-CLIP)** | X-CLIP | ⬜ Todo | P2 | Zero-shot video labels |
| **Action recognition** | VideoMAE/SlowFast | ⬜ Todo | P2 | "running", "talking", etc. |
| **Object detection (YOLO)** | YOLOv8 | ⬜ Todo | P2 | Count objects per clip |
| **Frame captioning (BLIP2)** | BLIP-2 | ⬜ Todo | P2 | Text description per clip |
| **Video captioning** | mPLUG-Owl | ⬜ Todo | P3 | Narrative description |
| **EDL export** | Custom | ⬜ Todo | P2 | Premiere/Resolve import |
| **Person detection** | YOLOv8/MediaPipe | ⬜ Todo | P2 | Count people per clip |
| **Pose estimation** | MediaPipe/OpenPose | ⬜ Todo | P3 | Body pose matching |
| **Face detection** | MediaPipe | ⬜ Todo | P3 | Face count, direction |
| **Face-based cut matching** | MediaPipe | ⬜ Todo | P3 | Match by face angle |

### Recommended Implementation Order

**Phase A: Basic Classification** (P2)
1. Frame classification with ImageNet labels
2. Object detection with YOLO
3. Person/face detection

**Phase B: Advanced Classification** (P2)
4. X-CLIP zero-shot video classification
5. Action recognition
6. Frame captioning (BLIP-2)

**Phase C: Matching & Export** (P3)
7. Pose estimation for cut matching
8. Face direction matching
9. EDL export for NLE workflows

### Frame Classification (ImageNet)

**Course Notebook**: Frame Classifier with Imagenet

**What it does**: Tag clips with object labels (car, dog, person, tree, etc.)

**Implementation**:
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

def classify_frame(image_path: str, top_k: int = 5) -> list[tuple]:
    """Classify frame using ImageNet model."""
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits.softmax(dim=-1)
    top_probs, top_indices = probs.topk(top_k)

    labels = [model.config.id2label[idx.item()] for idx in top_indices[0]]
    scores = top_probs[0].tolist()

    return list(zip(labels, scores))
```

### X-CLIP Zero-Shot Classification

**Course Notebook**: Video Classifier (X-Clip Zero Shot)

**What it does**: Classify videos using text prompts without training

**Key Advantage**: User-defined categories! ("scary scene", "romantic moment", "action sequence")

**Implementation**:
```python
from transformers import XCLIPProcessor, XCLIPModel

def classify_video_zero_shot(video_frames: list, labels: list[str]) -> dict:
    """Zero-shot video classification with custom labels."""
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")

    inputs = processor(
        text=labels,
        videos=video_frames,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=-1)
    return {label: prob.item() for label, prob in zip(labels, probs[0])}
```

### EDL Export

**Course Notebook**: EDL File Export

**What it does**: Export timeline as Edit Decision List for Premiere/Resolve import

**Format** (CMX 3600):
```
TITLE: Scene Ripper Export
FCM: NON-DROP FRAME

001  001      V     C        00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
* FROM CLIP NAME: clip_001.mp4

002  002      V     C        00:00:05:00 00:00:08:15 00:00:05:00 00:00:08:15
* FROM CLIP NAME: clip_002.mp4
```

---

## Week 4: Dialogue Extraction & Text-Based Editing

**Course Focus**: Speech transcription, text embeddings, script-guided editing

### Features

| Feature | Course Tool | Status | Priority | Notes |
|---------|-------------|--------|----------|-------|
| **Speech transcription** | Whisper | ⬜ Todo | P1 | 🎯 High impact |
| **Sentence similarity** | Sentence Transformers | ⬜ Todo | P2 | Find similar dialogue |
| **Text-guided sequencing** | Custom | ⬜ Todo | P2 | Match clips to script |

### Whisper Transcription (High Priority)

**Course Notebook**: Dialogue Extraction

**What it does**:
1. Extract audio from clips
2. Transcribe with Whisper
3. Store transcript with timestamps
4. Enable text search across clips

**Implementation**:
```python
import whisper

def transcribe_clip(video_path: str, model_size: str = "base") -> list[dict]:
    """Transcribe audio from video clip."""
    model = whisper.load_model(model_size)

    # Whisper can work directly on video files
    result = model.transcribe(str(video_path), word_timestamps=True)

    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": segment.get("words", [])
        })

    return segments
```

**Model Size Tradeoffs**:
| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Very fast | Good | 1GB |
| base | 74M | Fast | Better | 1GB |
| small | 244M | Medium | Good | 2GB |
| medium | 769M | Slow | Very good | 5GB |
| large | 1550M | Very slow | Best | 10GB |

**Recommendation**: Default to `base`, allow user to select larger for accuracy.

### Sentence Similarity

**Course Notebook**: Sentence Similarity

**What it does**:
1. Generate embeddings for each transcribed segment
2. Find clips with similar dialogue
3. Build sequences from a written script

**Implementation**:
```python
from sentence_transformers import SentenceTransformer

def build_embedding_index(transcripts: list[dict]) -> tuple:
    """Build FAISS index for sentence similarity search."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [t["text"] for t in transcripts]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, embeddings

def find_matching_clips(query: str, index, transcripts, k: int = 5):
    """Find clips with dialogue similar to query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)
    return [(transcripts[i], distances[0][j]) for j, i in enumerate(indices[0])]
```

---

## Week 5: Generative Video

**Course Focus**: AI-generated video content (RunwayML, ComfyUI, etc.)

### Features

| Feature | Course Tool | Status | Priority | Notes |
|---------|-------------|--------|----------|-------|
| **External tool integration** | RunwayML API | ⬜ Todo | P3 | Cloud-based generation |
| **Local generation** | ComfyUI/SD | ⬜ Todo | P3 | Requires significant GPU |

### Recommendation

Generative video is rapidly evolving. Rather than build native support, consider:

1. **Export for external tools**: Export clips in formats suitable for RunwayML, ComfyUI, etc.
2. **Import generated content**: Easy import of AI-generated clips back into Scene Ripper
3. **Workflow documentation**: Guides for round-tripping with popular generative tools

---

## Week 6: Rhythmic Sequencing & Audioreactive Editing

**Course Focus**: Music-driven editing, beat detection, rhythm patterns

### Features

| Feature | Course Tool | Status | Priority | Notes |
|---------|-------------|--------|----------|-------|
| **Beat detection** | librosa | ⬜ Todo | P1 | 🎯 High impact |
| **Rhythm patterns** | Custom | ⬜ Todo | P2 | Cut on beats |
| **Audioreactive sequencing** | Custom | ⬜ Todo | P2 | Match clip energy to audio |

### Beat Detection (High Priority)

**Course Notebooks**: Rhythms, Audioreactive Editing

**What it does**:
1. Analyze music track for beat positions
2. Generate cut points aligned to beats
3. Optionally match clip "energy" to musical sections

**Implementation**:
```python
import librosa
import numpy as np

def detect_beats(audio_path: str) -> dict:
    """Detect beats and tempo in audio file."""
    y, sr = librosa.load(audio_path)

    # Tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Downbeats (measure starts) - every 4th beat assuming 4/4
    downbeat_times = beat_times[::4]

    # Energy envelope for audioreactive matching
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)

    return {
        "tempo": float(tempo),
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times.tolist(),
        "energy_envelope": list(zip(rms_times.tolist(), rms.tolist()))
    }

def generate_beat_sequence(clips: list, beat_times: list) -> list:
    """Place clips at beat positions."""
    sequence = []
    clip_idx = 0

    for i, beat_time in enumerate(beat_times):
        if clip_idx >= len(clips):
            clip_idx = 0  # Loop clips

        clip = clips[clip_idx]
        # Trim clip to fit between beats
        if i + 1 < len(beat_times):
            duration = beat_times[i + 1] - beat_time
        else:
            duration = 0.5  # Default for last beat

        sequence.append({
            "clip": clip,
            "start_time": beat_time,
            "duration": duration
        })
        clip_idx += 1

    return sequence
```

### Audioreactive Sequencing

**Advanced Feature**: Match clip visual energy to audio energy

1. Analyze audio energy envelope
2. Score clips by visual "energy" (motion, brightness changes)
3. Place high-energy clips during high-energy audio sections

---

## Implementation Phases

### Phase 3: Visual Analysis (Next Priority)

**Goal**: Color and shot type analysis for intelligent sorting

**Features**:
- [x] Color extraction (k-means, 5 dominant colors)
- [x] HSV color sorting in clip browser
- [ ] Color palette filter
- [x] Shot type classification
- [x] Shot type filter in browser

**Estimated Effort**: 2-3 weeks

### Phase 4: Audio & Speech

**Goal**: Whisper transcription and beat-based editing

**Features**:
- [ ] Whisper integration (local model)
- [ ] Transcript storage and search
- [ ] Beat detection (librosa)
- [ ] Beat-aligned sequencing algorithm
- [ ] Music track import

**Estimated Effort**: 3-4 weeks

### Phase 5: Content Intelligence

**Goal**: Deep content understanding

**Features**:
- [ ] ImageNet frame classification
- [ ] X-CLIP zero-shot video classification
- [ ] Object detection (YOLO)
- [ ] Person/face detection
- [ ] Sentence similarity for dialogue matching

**Estimated Effort**: 4-6 weeks

### Phase 6: Advanced Matching & Export

**Goal**: Professional workflow integration

**Features**:
- [ ] Action classification
- [ ] Pose estimation
- [ ] Face direction matching
- [ ] EDL export
- [ ] FCPXML export (Final Cut Pro)

**Estimated Effort**: 3-4 weeks

---

## Feature Dependency Graph

```
Phase 1 (Done)              Phase 2 (Done)           Phase 3 (Next)
┌──────────────┐           ┌──────────────┐         ┌──────────────┐
│ Scene Detect │──────────>│ Timeline     │────────>│ Color        │
│ Import       │           │ Shuffle      │         │ Analysis     │
│ Export       │           │ Sequence     │         │ Shot Types   │
└──────────────┘           └──────────────┘         └──────────────┘
                                                            │
                                  ┌─────────────────────────┤
                                  ▼                         ▼
                           ┌──────────────┐         ┌──────────────┐
                           │ Phase 4      │         │ Phase 5      │
                           │ Audio/Speech │         │ Content AI   │
                           │ Beat Detect  │         │ Classification│
                           │ Whisper      │         │ Detection    │
                           └──────────────┘         └──────────────┘
                                  │                         │
                                  └─────────┬───────────────┘
                                            ▼
                                     ┌──────────────┐
                                     │ Phase 6      │
                                     │ Advanced     │
                                     │ Matching     │
                                     │ NLE Export   │
                                     └──────────────┘
```

---

## Quick Reference: Course Notebook → Feature

| Notebook | Feature | File Location |
|----------|---------|---------------|
| PySceneDetect | Scene detection | `core/scene_detect.py` ✅ |
| Random Sequencing | Shuffle algorithm | `core/remix/shuffle.py` ✅ |
| JSON file creation | Dataset export | `core/export.py` (todo) |
| Intro to FFMPEG | Clip extraction | `core/ffmpeg.py` ✅ |
| FFProbe | Metadata extraction | `core/ffmpeg.py` (extend) |
| Duration visualization | Duration chart | `ui/analysis/` (todo) |
| Color extraction | K-means colors | `core/analysis/color.py` (todo) |
| HSV Color Sorting | Color sort | `core/analysis/color.py` (todo) |
| Shot Type classification | Shot classifier | `core/analysis/shots.py` ✅ |
| Frame Classifier | ImageNet tags | `core/analysis/classify.py` (todo) |
| Video Classifier (X-CLIP) | Zero-shot labels | `core/analysis/classify.py` (todo) |
| Action Classification | Action tags | `core/analysis/actions.py` (todo) |
| Object Detection (YOLO) | Object counts | `core/analysis/detection.py` (todo) |
| Frame Captioning (BLIP2) | Clip descriptions | `core/analysis/caption.py` (todo) |
| EDL File Export | NLE export | `core/export/edl.py` (todo) |
| Person/Pose Detection | Body detection | `core/analysis/pose.py` (todo) |
| Face Detection | Face detection | `core/analysis/face.py` (todo) |
| Dialogue Extraction | Whisper | `core/analysis/speech.py` (todo) |
| Sentence Similarity | Text embeddings | `core/analysis/embeddings.py` (todo) |
| Rhythms | Beat detection | `core/audio/beats.py` (todo) |
| Audioreactive | Energy matching | `core/audio/reactive.py` (todo) |

---

## Appendix: Model Requirements

### Local Models (CPU-capable)

| Model | Task | Size | Min RAM | Notes |
|-------|------|------|---------|-------|
| ResNet-50 | Frame classification | 98MB | 2GB | Fast, good accuracy |
| Whisper base | Transcription | 74MB | 2GB | Good speed/accuracy |
| YOLOv8n | Object detection | 6MB | 1GB | Fastest YOLO |
| all-MiniLM-L6-v2 | Text embeddings | 90MB | 1GB | Fast sentence embeddings |

### GPU-Recommended Models

| Model | Task | Size | Min VRAM | Notes |
|-------|------|------|----------|-------|
| X-CLIP | Video classification | 600MB | 4GB | Zero-shot capability |
| BLIP-2 | Image captioning | 3GB | 8GB | High quality captions |
| Whisper medium | Transcription | 769MB | 5GB | Better accuracy |
| VideoMAE | Action recognition | 400MB | 4GB | Temporal understanding |

---

*Generated: 2026-01-24*
*Source: ITP Algorithmic Filmmaking course curriculum*
