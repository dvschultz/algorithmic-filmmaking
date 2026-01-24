---
title: "Algorithmic Filmmaking: Video Collage App for Artists"
type: feat
date: 2026-01-24
priority: high
deepened: 2026-01-24
---

# Algorithmic Filmmaking: Video Collage App for Artists

## Enhancement Summary

**Deepened on:** 2026-01-24
**Research agents used:** 8 parallel agents (Tauri, React/Timeline, FAISS, Architecture, Security, Performance, TypeScript, Simplicity)
**Skills applied:** ffmpeg-expert, pyscenedetect-expert, ui-skills, frontend-design, agent-native-architecture

### Key Improvements
1. **Architecture refined**: JSON-RPC 2.0 for Rust-Python IPC, swap Phase 2/3 order (Composition before Intelligence)
2. **Security hardened**: Command injection mitigations, path traversal prevention, domain whitelisting for URLs
3. **Performance optimized**: GPU/CPU detection, memory budgets, thumbnail caching with LRU, model lifecycle management
4. **Simplified MVP path**: Pure Python alternative for 2-4 week prototype vs full 6-12 month build

### Critical Decision Point
**Consider starting with a Pure Python MVP** (PySide6 + PySceneDetect) to validate the concept in 2-4 weeks before investing in the full Tauri architecture. See "Alternative: Pure Python MVP" section.

---

## Overview

A desktop application that enables artists to create collage films by:
1. Importing multiple video sources (local files + YouTube/Vimeo)
2. Automatically detecting and extracting scenes using AI
3. Tagging clips with metadata (mood, motion, content)
4. Recombining clips through algorithmic, timeline-based, or tag-matching approaches
5. Exporting new narrative compositions

**Target Users**: Video artists, experimental filmmakers, VJs, collage artists

## Problem Statement

Artists working with found footage and video collage face tedious manual workflows:
- Manually scrubbing through hours of source material
- Manually marking in/out points for every clip
- No systematic way to find similar or thematically related clips
- Limited tools for algorithmic/generative video editing
- Professional tools (Premiere, Resolve) are designed for linear narrative, not experimental recombination

**Goal**: Reduce the friction between "I have 50 source videos" and "I have a new experimental film" by automating the tedious parts while preserving artistic control.

## Proposed Solution

A Tauri-based desktop app with three core modules:

```
┌─────────────────────────────────────────────────────────────┐
│                    ALGORITHMIC FILMMAKING                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│   INGEST        │   ANALYZE       │   COMPOSE               │
│   ─────────     │   ───────       │   ───────               │
│   • Import      │   • Scene       │   • Timeline editor     │
│     local files │     detection   │   • Algorithmic remix   │
│   • Download    │   • AI tagging  │   • Tag-based matching  │
│     YouTube/    │   • Similarity  │   • Export/render       │
│     Vimeo       │     indexing    │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tauri Frontend (React)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Import   │  │ Library  │  │ Timeline │  │ Preview  │    │
│  │ Panel    │  │ Browser  │  │ Editor   │  │ Player   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↕ IPC (Tauri Commands)
┌─────────────────────────────────────────────────────────────┐
│                    Tauri Rust Backend                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Video    │  │ Project  │  │ Export   │  │ Settings │    │
│  │ Pipeline │  │ Manager  │  │ Engine   │  │ Manager  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↕ JSON-RPC 2.0 over stdin/stdout
┌─────────────────────────────────────────────────────────────┐
│                    Python Services (Sidecar)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PySceneDetect│  │ AI Tagging   │  │ Similarity   │      │
│  │ Service      │  │ Service      │  │ Index        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                    System Dependencies                       │
│     FFmpeg          yt-dlp          Python 3.11+            │
└─────────────────────────────────────────────────────────────┘
```

### Research Insights: IPC Protocol

**Best Practice: JSON-RPC 2.0 over stdin/stdout**

```json
// Request
{"jsonrpc": "2.0", "method": "detectScenes", "params": {"videoPath": "/path/to/video.mp4"}, "id": 1}

// Progress notification (no id = notification)
{"jsonrpc": "2.0", "method": "progress", "params": {"percent": 45, "stage": "analyzing"}}

// Error response
{"jsonrpc": "2.0", "error": {"code": -32000, "message": "Video codec not supported"}, "id": 1}
```

**Implementation Details:**
- Rust backend owns Python process lifecycle (spawn, monitor, gracefully terminate)
- Implement ping/pong heartbeat before dispatching heavy tasks
- If sidecar crashes: detect via process exit, log stderr, notify user, offer restart
- Use `tokio::sync::mpsc` for job queue - process one AI task at a time
- Bundle Python with PyOxidizer or PyInstaller

### Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | React + TypeScript | Large ecosystem, easy to find developers |
| **UI Components** | Radix UI + Tailwind | Accessible, customizable primitives |
| **Desktop Shell** | Tauri 2.x | 10MB bundles, Rust performance, native feel |
| **Video Playback** | HTML5 Video + custom controls | Native browser support, extensible |
| **Timeline** | Remotion or Twick | Frame-accurate, React-native |
| **Virtual Scroll** | TanStack Virtual | Best performance for 1000+ items |
| **State (UI)** | Zustand + Zundo | Minimal boilerplate, native undo/redo |
| **State (Async)** | React Query (TanStack Query) | Server state, caching, background refresh |
| **Backend** | Rust (Tauri) | Performance for file I/O, process management |
| **Scene Detection** | PySceneDetect (Python sidecar) | Best-in-class detection algorithms |
| **AI Tagging** | Hugging Face VideoMAE | Pre-trained, no fine-tuning needed |
| **Similarity** | FAISS (IndexFlatIP) | Sufficient for 10k clips, exact results |
| **Embeddings** | ResNet-18 (512-dim) or ResNet-50 (2048-dim) | Standard, well-supported |
| **Video Processing** | FFmpeg (via subprocess) | Universal, fast, well-documented |
| **Video Download** | yt-dlp (Python) | Feature-rich, actively maintained |
| **Database** | SQLite (via rusqlite/sqlx) | Simple, embedded, portable projects |

### Research Insights: State Management

**Zustand for UI State (document, timeline, selection):**

```typescript
import { create } from 'zustand';
import { temporal } from 'zundo';

interface EditorState {
  tracks: Record<string, Track>;
  clips: Record<string, Clip>;
  selectedClipId: string | null;
  currentFrame: number;
  isPlaying: boolean;

  // Actions
  addClip: (clip: Clip) => void;
  moveClip: (clipId: string, trackId: string, startFrame: number) => void;
  setCurrentFrame: (frame: number) => void;
}

export const useEditorStore = create<EditorState>()(
  temporal((set) => ({
    // ... implementation
  }), { limit: 100 }) // Undo history
);
```

**React Query for Async State (backend calls):**

```typescript
export function useVideoMetadata(filePath: string | null) {
  return useQuery({
    queryKey: ['videoMetadata', filePath],
    queryFn: () => invoke<VideoMetadata>('get_video_metadata', { filePath }),
    enabled: !!filePath,
    staleTime: Infinity, // Metadata doesn't change
  });
}
```

### Data Model

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Project   │──────<│   Source    │──────<│    Clip     │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id          │       │ id          │       │ id          │
│ name        │       │ project_id  │       │ source_id   │
│ created_at  │       │ file_path   │       │ start_frame │
│ updated_at  │       │ file_hash   │       │ end_frame   │
│ fps         │       │ origin_url  │       │ thumbnail   │
│ resolution  │       │ duration    │       │ tags[]      │
└─────────────┘       │ fps         │       │ motion_score│
                      │ metadata    │       │ brightness  │
                      └─────────────┘       └─────────────┘
```

### Research Insights: Frame-Based Timing

**Use frame numbers, not milliseconds, for timeline accuracy:**

```typescript
interface Clip {
  id: string;
  sourceId: string;
  trackId: string;

  // Timeline position (where on the timeline)
  startFrame: number;

  // Source range (what part of the source to use)
  inPoint: number;  // frames
  outPoint: number; // frames

  // Computed
  readonly duration: number; // outPoint - inPoint

  transforms: ClipTransforms;
  effects: Effect[];
  speedMultiplier: number;
}
```

### Research Insights: Embeddings Storage

**Store embeddings separately from SQLite (in FAISS native files):**

```
/project_folder/
  project.db          # SQLite: metadata, relationships
  cache/
    thumbnails/       # Scene thumbnails (WebP, small)
  embeddings/
    scenes.index      # FAISS: scene embeddings
    scenes.mapping    # JSON: FAISS index position -> scene_id
```

**Rationale:**
- FAISS indexes are purpose-built for vector search
- Memory-mapped loading for large indexes
- GPU-ready for future acceleration
- SQLite BLOBs are slower for similarity search

### Implementation Phases

> **IMPORTANT: Phase order changed based on architecture review**
> Composition (formerly Phase 3) now comes before Intelligence (formerly Phase 2) to deliver a usable product sooner.

#### Phase 1: Foundation (Core Pipeline)

**Goal**: Import videos, detect scenes, export clips

**Deliverables**:
- [ ] Tauri project scaffolding with React frontend
- [ ] SQLite database schema and migrations
- [ ] Video import (drag-drop local files)
- [ ] PySceneDetect integration (AdaptiveDetector)
- [ ] Scene list display with thumbnails
- [ ] Basic video preview player
- [ ] Export selected clips to folder

**Technical Tasks**:

```
src-tauri/
├── src/
│   ├── main.rs                 # Tauri entry point
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── import.rs           # Video import commands
│   │   ├── detect.rs           # Scene detection commands
│   │   └── export.rs           # Export commands
│   ├── db/
│   │   ├── mod.rs
│   │   ├── schema.rs           # SQLite schema
│   │   └── migrations/
│   └── services/
│       ├── python_sidecar.rs   # JSON-RPC client
│       └── thumbnail_cache.rs  # LRU cache
├── binaries/                   # Bundled FFmpeg, Python sidecar
└── Cargo.toml
```

```
src-python/
├── __init__.py
├── server.py                   # JSON-RPC server
├── scene_detect.py             # PySceneDetect wrapper
├── requirements.txt
└── build.py                    # PyInstaller build script
```

```
src/ (frontend)
├── App.tsx
├── components/
│   ├── ImportPanel.tsx
│   ├── ClipLibrary.tsx         # Virtual scrolling with TanStack Virtual
│   ├── VideoPreview.tsx
│   └── SceneList.tsx
├── hooks/
│   ├── useTauriCommands.ts
│   └── useTaskProgress.ts      # Event-based progress
└── stores/
    └── projectStore.ts         # Zustand
```

**Research Insights: Drag-Drop Implementation**

```typescript
// React component for drag-drop
import { listen } from "@tauri-apps/api/event";

useEffect(() => {
  const unlistenDrop = listen("tauri://drag-drop", (event) => {
    const paths = event.payload as string[];
    const videoExtensions = [".mp4", ".mkv", ".mov", ".avi", ".webm"];
    const videoFiles = paths.filter((path) =>
      videoExtensions.some((ext) => path.toLowerCase().endsWith(ext))
    );
    if (videoFiles.length > 0) {
      onImport(videoFiles);
    }
  });
  return () => { unlistenDrop.then(fn => fn()); };
}, []);
```

**Research Insights: PySceneDetect Configuration**

```python
from scenedetect import detect, AdaptiveDetector

# AdaptiveDetector is best for handheld/dynamic footage
scenes = detect(
    video_path,
    AdaptiveDetector(
        adaptive_threshold=3.0,  # Lower = more sensitive
        min_content_val=15.0,    # Minimum content change
    )
)
```

| Video Type | Detector | Threshold |
|------------|----------|-----------|
| Standard cuts | ContentDetector | 27 (default) |
| Camera movement | AdaptiveDetector | 3.0 |
| Fades to black | ThresholdDetector | 12 |
| Compression artifacts | HashDetector | 0.395 |
| B&W / grayscale | Any + `--luma-only` | |

**Success Criteria**:
- Import a 10-minute video
- Auto-detect 20+ scenes in under 60 seconds
- Preview any detected scene
- Export scenes as individual files

---

#### Phase 2: Composition (Recombination Engine) - MOVED UP

**Goal**: Enable creative recombination of clips

> **Rationale for moving up**: Users can create collages manually even without AI tagging. This delivers value faster and validates the core UX.

**Deliverables**:
- [ ] Timeline editor component (Remotion or Twick)
- [ ] Drag-drop clip arrangement
- [ ] Algorithmic remix modes:
  - Random shuffle within tag constraints
  - Rhythm-based (match motion intensity curves)
  - Similarity chain (each clip similar to previous)
- [ ] Tag-based auto-assembly ("build a sequence of outdoor scenes")
- [ ] Transition effects (cut, dissolve, fade)
- [ ] Audio handling (mute, original, music track)
- [ ] Real-time preview of sequences

**Technical Tasks**:

```
src/ (frontend)
├── components/
│   ├── Timeline/
│   │   ├── Timeline.tsx        # TanStack Virtual for tracks
│   │   ├── TrackLane.tsx
│   │   ├── ClipBlock.tsx       # @dnd-kit for drag-drop
│   │   ├── Playhead.tsx
│   │   └── TimelineRuler.tsx
│   ├── RemixPanel/
│   │   ├── RemixPanel.tsx
│   │   ├── AlgorithmSelector.tsx
│   │   └── ConstraintBuilder.tsx
│   └── TagMatcher/
│       ├── TagMatcher.tsx
│       └── QueryBuilder.tsx
└── lib/
    ├── remix/
    │   ├── shuffle.ts
    │   ├── rhythm.ts
    │   └── similarity.ts
    └── timeline/
        └── operations.ts
```

**Research Insights: Timeline Component Architecture**

```typescript
import { useVirtualizer } from '@tanstack/react-virtual';
import { useDraggable, useDroppable } from '@dnd-kit/core';

function Timeline({ project }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Virtualize tracks for vertical scrolling
  const trackVirtualizer = useVirtualizer({
    count: project.timeline.trackOrder.length,
    getScrollElement: () => containerRef.current,
    estimateSize: () => 60, // Track height
    overscan: 2,
  });

  return (
    <div ref={containerRef} className="flex-1 overflow-auto">
      {trackVirtualizer.getVirtualItems().map((virtualRow) => (
        <TrackLane key={virtualRow.index} /* ... */ />
      ))}
    </div>
  );
}
```

**Success Criteria**:
- Build a 2-minute sequence by dragging clips
- Generate a 1-minute remix with "shuffle outdoor scenes"
- Preview sequence in real-time
- Export sequence to file

---

#### Phase 3: Intelligence (AI Features) - MOVED DOWN

**Goal**: Add AI tagging, similarity matching, and smart organization

**Deliverables**:
- [ ] Motion intensity analysis (optical flow)
- [ ] Brightness/color analysis per clip
- [ ] AI content tagging (VideoMAE)
- [ ] Clip embedding generation (ResNet-18/50)
- [ ] FAISS similarity index (IndexFlatIP for <10k clips)
- [ ] "Find similar clips" feature
- [ ] Tag-based filtering and search
- [ ] Batch processing queue with progress

**Technical Tasks**:

```
src-python/
├── analysis/
│   ├── motion.py           # Optical flow analysis
│   ├── color.py            # Color/brightness extraction
│   ├── content.py          # VideoMAE classification
│   └── embedding.py        # ResNet feature extraction
├── similarity/
│   ├── index.py            # FAISS index management
│   └── search.py           # Similarity queries
└── requirements.txt        # Add torch, transformers, faiss-cpu
```

**Research Insights: FAISS Index Selection**

| Clips | Index Type | Memory | Search Time | Build Time |
|-------|------------|--------|-------------|------------|
| <1,000 | IndexFlatIP | ~4MB | <10ms | Instant |
| 1,000-10,000 | IndexFlatIP | ~40MB | <10ms | Instant |
| 10,000-100,000 | IndexIVFPQ | ~100MB | <50ms | ~5min |

**For 10k clips, use IndexFlatIP (exact, no training needed):**

```python
import faiss
import numpy as np

# Create index
dimension = 512  # ResNet-18 output
index = faiss.IndexFlatIP(dimension)

# Add vectors (normalized for cosine similarity)
embeddings = np.array(clip_embeddings, dtype='float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Search
query = np.array([query_embedding], dtype='float32')
faiss.normalize_L2(query)
distances, indices = index.search(query, k=5)
```

**Research Insights: GPU/CPU Fallback Strategy**

```python
def get_gpu_capabilities() -> dict:
    caps = {
        "cuda_available": False,
        "recommended_batch_size": 16  # CPU default
    }

    try:
        import torch
        if torch.cuda.is_available():
            caps["cuda_available"] = True
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            if vram_mb >= 8000:
                caps["recommended_batch_size"] = 64
            elif vram_mb >= 4000:
                caps["recommended_batch_size"] = 32
    except ImportError:
        pass

    return caps
```

**Success Criteria**:
- Analyze 100 clips with tags in under 5 minutes (GPU) or 30 minutes (CPU)
- Find 5 similar clips to any selected clip
- Filter library by motion (low/medium/high)
- Filter library by custom tags

---

#### Phase 4: Polish (Production Ready)

**Goal**: YouTube/Vimeo integration, UX polish, performance

**Deliverables**:
- [ ] YouTube/Vimeo URL import (yt-dlp)
- [ ] Download progress and queue management
- [ ] Legal disclaimer and acknowledgment flow
- [ ] Project save/load (SQLite + media references)
- [ ] Export presets (quality, format, resolution)
- [ ] Keyboard shortcuts for common actions
- [ ] Dark/light theme
- [ ] Onboarding tutorial
- [ ] Error handling and recovery
- [ ] macOS/Windows/Linux packaging

**Technical Tasks**:

```
src-python/
├── download/
│   ├── ytdlp_wrapper.py    # yt-dlp integration
│   └── progress.py         # Progress reporting
```

```
src/ (frontend)
├── components/
│   ├── DownloadPanel.tsx
│   ├── ExportDialog.tsx
│   ├── SettingsPanel.tsx
│   └── OnboardingFlow.tsx
```

**Success Criteria**:
- Download a YouTube video and add to library
- Full keyboard-driven workflow possible
- Save project, close app, reopen, resume work
- Export final sequence in H.264/H.265

---

## Security Considerations

### Research Insights: Critical Vulnerabilities and Mitigations

| Category | Severity | Vulnerability | Mitigation |
|----------|----------|---------------|------------|
| Command Injection | CRITICAL | FFmpeg shell interpolation | Use argument arrays, never `sh -c` |
| Command Injection | CRITICAL | yt-dlp URL injection | Strict domain whitelist, `--` separator |
| Path Traversal | HIGH | File import from arbitrary paths | Canonicalization + allowlist roots |
| Deserialization | HIGH | Pickle in IPC | JSON only + Pydantic validation |
| SSRF | HIGH | Internal network access via URLs | DNS resolution + IP blocklist |
| SQL Injection | MEDIUM | Dynamic query construction | Parameterized queries only |
| XSS | MEDIUM | Malicious video metadata | HTML sanitization with ammonia |

### FFmpeg Command Safety

```rust
// SAFE - Use argument array, never shell interpolation
fn execute_ffmpeg(input: &Path, output: &Path) -> Result<()> {
    let validated_input = canonicalize_and_validate(input)?;
    let validated_output = validate_output_path(output)?;

    Command::new("ffmpeg")
        .arg("-i").arg(&validated_input)
        .arg(&validated_output)
        .output()?;
    Ok(())
}
```

### yt-dlp URL Safety

```rust
fn download_video(url: &str, output_dir: &Path) -> Result<PathBuf> {
    let parsed = url::Url::parse(url)?;

    // Whitelist allowed domains
    let allowed_hosts = ["youtube.com", "youtu.be", "vimeo.com"];
    let host = parsed.host_str().ok_or(SecurityError::InvalidUrl)?;

    if !allowed_hosts.iter().any(|h| host.ends_with(h)) {
        return Err(SecurityError::DisallowedDomain);
    }

    Command::new("yt-dlp")
        .arg("--no-exec")           // Disable post-processing exec
        .arg("--no-playlist")       // Prevent playlist expansion
        .arg("--max-filesize").arg("2G")
        .arg("-o").arg(output_template)
        .arg("--")                  // End of options marker
        .arg(url)
        .output()?;

    Ok(output_path)
}
```

### Path Validation

```rust
fn validate_path(user_path: &Path, allowed_roots: &[PathBuf]) -> Result<PathBuf> {
    // Canonicalize to resolve symlinks and ..
    let canonical = user_path.canonicalize()?;

    // Check against allowed roots
    if !allowed_roots.iter().any(|root| canonical.starts_with(root)) {
        return Err(SecurityError::PathTraversal);
    }

    // Verify it's a regular file
    if !canonical.metadata()?.is_file() {
        return Err(SecurityError::NotAFile);
    }

    Ok(canonical)
}
```

---

## Performance Considerations

### Research Insights: Memory Budget (<2GB for 1000 clips)

| Component | Budget | Strategy |
|-----------|--------|----------|
| SQLite + Metadata | 50MB | Indexed queries, no full table loads |
| Thumbnail Cache (memory) | 200MB | LRU cache, 200 thumbnails max |
| FAISS Index | 400MB | Memory-mapped, IndexFlatIP |
| UI State | 100MB | Virtual scrolling, lazy loading |
| Python Sidecar | 500MB | One model loaded at a time |
| Headroom | 750MB | Buffer for spikes |

### Research Insights: Model Lifecycle Management

```python
class ModelManager:
    """Unload ML models after idle timeout to free memory."""

    def __init__(self, idle_timeout_seconds: int = 60):
        self._models: dict[str, any] = {}
        self._idle_timeout = idle_timeout_seconds
        self._unload_timer = None

    def get_model(self, name: str):
        self._cancel_unload_timer()

        if name not in self._models:
            self._unload_all()  # Free memory first
            self._models[name] = self._load_model(name)

        self._schedule_unload()
        return self._models[name]

    def _unload_all(self):
        for model in self._models.values():
            del model
        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Research Insights: Thumbnail Caching

```rust
pub struct ThumbnailCache {
    memory_cache: LruCache<String, Vec<u8>>,  // 200 items max
    cache_dir: PathBuf,
}

// Cache key includes file path + timestamp + last modified
fn cache_key(video_path: &str, timestamp_ms: u64, modified: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}:{}:{}", video_path, timestamp_ms, modified));
    format!("{:x}", hasher.finalize())
}
```

| Resolution | Format | Size | Use Case |
|------------|--------|------|----------|
| 160x90 | WebP | ~5KB | Grid view |
| 80x45 | WebP | ~2KB | Timeline |
| 320x180 | WebP | ~15KB | Preview hover |

### Benchmark Targets

| Operation | Target (GPU) | Target (CPU) |
|-----------|--------------|--------------|
| Scene detection | <2s/min | <10s/min |
| AI tagging (100 clips) | <5min | <30min |
| Preview start | <500ms | <500ms |
| Thumbnail cache hit (memory) | <5ms | <5ms |
| Thumbnail cache hit (disk) | <50ms | <50ms |
| FAISS search (1K clips) | <10ms | <10ms |
| Memory (1K clips) | <2GB | <2GB |
| App startup | <3s | <3s |

---

## Alternative: Pure Python MVP

### Research Insights: Simplicity Review

**Consider starting with a pure Python MVP to validate the concept faster:**

| Approach | Time to MVP | Complexity | When to Use |
|----------|-------------|------------|-------------|
| Full Tauri + React + Python | 6-12 months | HIGH | Production product |
| **Pure Python (PySide6)** | **2-4 weeks** | LOW | **Concept validation** |
| Electron + Python | 4-6 weeks | MEDIUM | Not recommended |

### Pure Python Architecture

```
algorithmic-filmmaking/
├── main.py                 # Entry point
├── ui/
│   ├── main_window.py      # PySide6 main window
│   ├── video_player.py     # QMediaPlayer wrapper
│   └── clip_browser.py     # Grid of thumbnails
├── core/
│   ├── scene_detect.py     # PySceneDetect wrapper
│   ├── ffmpeg.py           # FFmpeg subprocess calls
│   └── thumbnail.py        # Thumbnail generation
├── models/
│   └── clip.py             # Simple dataclass
└── requirements.txt
```

**Dependencies:**
```
PySide6>=6.6
scenedetect>=0.6
opencv-python>=4.8
```

### MVP Feature Set

**Keep (IMPLEMENTED):**
- [x] Drag-drop import
- [x] Scene detection (PySceneDetect)
- [x] Thumbnail display
- [x] Video preview
- [x] Export clips
- [x] Sensitivity slider

**Cut for MVP:**
- YouTube/Vimeo download (use yt-dlp CLI separately)
- AI tagging (VideoMAE)
- Similarity search (FAISS)
- Timeline editor
- Algorithmic remix
- Save/load projects
- Themes, onboarding

### Recommendation

**If you're uncertain about product-market fit:**
1. Build the Python MVP in 2-4 weeks
2. Test with 5-10 target artists
3. If validated, invest in full Tauri architecture
4. If not, iterate on the concept quickly

---

## Alternative Approaches Considered

### 1. Electron Instead of Tauri

**Pros**: Consistent rendering across platforms, larger ecosystem
**Cons**: 100MB+ bundle size, higher memory usage, slower startup
**Decision**: Tauri's performance benefits outweigh Electron's consistency. Video apps need efficiency.

### 2. Pure Python with PyQt

**Pros**: Single language, simpler architecture
**Cons**: PyQt licensing complexity, less modern UI, harder to find frontend devs
**Decision**: Recommended for MVP validation. Not for production.

### 3. Web App with Server-Side Processing

**Pros**: No installation, collaborative features easier
**Cons**: Latency for video operations, bandwidth costs, storage limits
**Decision**: Desktop-first for performance. Could add cloud sync later.

### 4. MoviePy for All Video Processing

**Pros**: Pure Python, high-level API
**Cons**: 10-100x slower than FFmpeg for batch operations
**Decision**: Use FFmpeg directly for performance. MoviePy only for complex compositing if needed.

---

## Acceptance Criteria

### Functional Requirements

- [x] Import local video files (MP4, MOV, AVI, MKV, WebM)
- [ ] Import videos from YouTube/Vimeo URLs
- [x] Automatically detect scenes with configurable sensitivity
- [x] Display scene thumbnails in a browsable library
- [x] Preview any clip with frame-accurate playback
- [ ] Tag clips with AI-detected content labels
- [ ] Manually add/edit/remove tags on clips
- [ ] Search clips by tags, motion level, color
- [ ] Find similar clips based on visual content
- [ ] Arrange clips in a timeline sequence
- [ ] Generate sequences algorithmically (shuffle, rhythm, chain)
- [ ] Export sequences as video files
- [ ] Save and load projects

### Non-Functional Requirements

- [ ] Scene detection: <2 seconds per minute of video (GPU) or <10 seconds (CPU)
- [ ] Clip preview: Start playback within 500ms
- [ ] UI responsiveness: No frame drops during preview
- [ ] Memory: <2GB RAM for 1000-clip library
- [ ] Storage: Project files <10MB (excluding media)
- [ ] Startup: App launches in <3 seconds

### Quality Gates

- [ ] Unit tests for core Rust commands
- [ ] Integration tests for Python sidecar
- [ ] E2E tests for critical user flows (import → detect → export)
- [ ] Cross-platform build verification (macOS, Windows, Linux)
- [ ] Accessibility audit (keyboard navigation, screen reader labels)

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Scene detection accuracy | >90% cut detection | Manual review of 10 test videos |
| Processing speed | <2 min for 10-min video | Benchmark suite |
| User workflow time | 50% reduction vs manual | User testing comparison |
| App stability | <1 crash per 10 hours | Crash reporting |
| Export quality | Visually lossless | A/B comparison with source |

---

## Dependencies & Prerequisites

### Required System Dependencies

- **FFmpeg**: Must be installed and in PATH (or bundled)
- **Python 3.11+**: For ML services sidecar
- **Rust 1.75+**: For building Tauri backend
- **Node.js 20+**: For building React frontend

### Python Dependencies (sidecar)

```
scenedetect>=0.6
opencv-python>=4.8
torch>=2.0
transformers>=4.30
faiss-cpu>=1.7
numpy>=1.24
yt-dlp>=2024.0
pydantic>=2.0
```

### NPM Dependencies

```
react, react-dom
@tanstack/react-query
@tanstack/react-virtual
zustand
zundo
@dnd-kit/core
@radix-ui/react-*
tailwindcss
@tauri-apps/api
```

---

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FFmpeg installation issues | Medium | High | Bundle FFmpeg binaries per platform |
| Python sidecar crashes | Medium | Medium | JSON-RPC error handling, process restart, heartbeat |
| GPU compatibility for ML | Medium | Medium | CPU fallback for all ML operations |
| Video format compatibility | Low | Medium | Rely on FFmpeg's broad format support |
| yt-dlp blocks/rate limits | Medium | Low | Graceful degradation, user notification |
| Large library performance | Low | Medium | Virtual scrolling, background indexing |
| Cross-platform UI bugs | Medium | Low | Test on all platforms, use system webview carefully |
| Security vulnerabilities | Medium | High | Input validation, domain whitelist, path sanitization |
| Python bundling fails cross-platform | Medium | High | Prototype packaging for all 3 OSes in week 1 |

---

## Immediate Next Steps

**Week 1 (De-risk):**
1. Prototype Tauri + bundled Python sidecar + bundled FFmpeg on all 3 platforms
2. Define JSON-RPC contract for core operations (import, detect, export)
3. Verify PySceneDetect works in bundled environment

**Week 2:**
4. Implement basic video import with thumbnail generation
5. Implement scene detection with progress reporting

**Week 3:**
6. Build clip library UI with virtual scrolling
7. Build video preview component

---

## Documentation Plan

- [ ] README with quick start guide
- [ ] CLAUDE.md with project conventions
- [ ] Architecture decision records (ADRs) for major choices
- [ ] API documentation for Tauri commands
- [ ] User manual with screenshots
- [ ] Video tutorial for common workflows

---

## References & Research

### Internal References

- This plan document
- Future: CLAUDE.md conventions

### External References

- [PySceneDetect Documentation](https://www.scenedetect.com/)
- [Tauri v2 Documentation](https://v2.tauri.app/)
- [Tauri Sidecar Guide](https://v2.tauri.app/develop/sidecar/)
- [Tauri IPC & Channels](https://v2.tauri.app/concept/inter-process-communication/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Hugging Face Video Classification](https://huggingface.co/docs/transformers/en/tasks/video_classification)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FAISS Index Selection Guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)
- [TanStack Virtual](https://tanstack.com/virtual/latest)
- [TanStack Query](https://tanstack.com/query/latest)
- [Zustand Documentation](https://zustand.docs.pmnd.rs/)
- [Zundo (Undo/Redo)](https://github.com/charkour/zundo)
- [Remotion Timeline Guide](https://www.remotion.dev/docs/building-a-timeline)

### Inspiration

- [Runway ML](https://runwayml.com/) - AI video tools
- [Davinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve) - Professional editing
- [Lumen5](https://lumen5.com/) - Algorithmic video creation
- [Auto-editor](https://github.com/WyattBlue/auto-editor) - Automated editing CLI

---

*Generated: 2026-01-24*
*Deepened: 2026-01-24*
