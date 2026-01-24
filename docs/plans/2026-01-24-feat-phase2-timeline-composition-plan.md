---
title: "Phase 2: Timeline & Composition"
type: feat
date: 2026-01-24
priority: high
---

# Phase 2: Timeline & Composition

## Overview

Add timeline editing and algorithmic remixing to Scene Ripper. This phase transforms the app from "scene extractor" to "video collage tool" by enabling creative recombination of detected clips.

**Goal**: Enable artists to arrange clips on a timeline and generate sequences algorithmically.

## Current State

Phase 1 complete:
- Video import (local files + YouTube/Vimeo)
- Scene detection with configurable sensitivity
- Thumbnail grid browser
- Video preview player
- Clip export

**What's missing**: No way to combine clips into new sequences.

## Proposed Solution

Add three interconnected features:

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN WINDOW                               │
├─────────────────────────────────────────────────────────────┤
│  [Import] [Import URL] [Sensitivity: ─●──] [Detect] [Export]│
├─────────────────┬───────────────────────────────────────────┤
│                 │                                            │
│   CLIP BROWSER  │           VIDEO PREVIEW                   │
│   (existing)    │           (existing)                       │
│                 │                                            │
├─────────────────┴───────────────────────────────────────────┤
│                    TIMELINE (NEW)                            │
│  ┌──────┬────────────────────────────────────────────────┐  │
│  │Track1│ [Clip A  ] [Clip C] [Clip B    ]              │  │
│  │Track2│      [Clip D]                                  │  │
│  └──────┴────────────────────────────────────────────────┘  │
│  [+Track] [Remix: Shuffle ▼] [Generate] [Clear] [Export Seq]│
└─────────────────────────────────────────────────────────────┘
```

## Technical Approach

### Architecture

Use Qt's QGraphicsView framework for the timeline:

```
ui/
├── main_window.py          # Add timeline panel
└── timeline/
    ├── __init__.py
    ├── timeline_widget.py  # Main container
    ├── timeline_scene.py   # QGraphicsScene with tracks
    ├── timeline_view.py    # QGraphicsView with zoom/scroll
    ├── track_item.py       # Horizontal track container
    ├── clip_item.py        # Draggable clip rectangle
    ├── playhead.py         # Vertical scrubber line
    └── ruler.py            # Time ruler at top

core/
├── remix/
│   ├── __init__.py
│   ├── shuffle.py          # Constrained random shuffle
│   ├── similarity.py       # Similarity chaining (greedy TSP)
│   └── rhythm.py           # Motion-based sequencing
└── sequence_export.py      # Export timeline to video
```

### Data Model Extension

```python
# models/sequence.py

@dataclass
class SequenceClip:
    """A clip placed on the timeline."""
    id: str
    source_clip_id: str      # Reference to original Clip
    track_index: int
    start_frame: int         # Position on timeline
    in_point: int            # Trim start (frames into source clip)
    out_point: int           # Trim end (frames into source clip)

@dataclass
class Track:
    """A horizontal track containing clips."""
    id: str
    name: str
    clips: list[SequenceClip]

@dataclass
class Sequence:
    """A timeline composition."""
    id: str
    name: str
    fps: float
    tracks: list[Track]
```

## Implementation Phases

### Phase 2.1: Timeline Foundation

**Goal**: Display clips on a timeline, drag to reposition

**Deliverables**:
- [x] `TimelineWidget` container with ruler and tracks
- [x] `TimelineScene` (QGraphicsScene) holding all items
- [x] `TimelineView` (QGraphicsView) with zoom (Ctrl+scroll)
- [x] `TrackItem` horizontal lane for clips
- [x] `ClipItem` draggable rectangle with thumbnail
- [x] `Playhead` vertical line, draggable
- [x] Drag clips from browser to timeline
- [x] Drag clips to reposition on timeline
- [x] Snap to adjacent clip edges

**Files to create**:
```
ui/timeline/__init__.py
ui/timeline/timeline_widget.py
ui/timeline/timeline_scene.py
ui/timeline/timeline_view.py
ui/timeline/track_item.py
ui/timeline/clip_item.py
ui/timeline/playhead.py
models/sequence.py
```

**Success criteria**:
- Drag 5 clips to timeline
- Reposition clips by dragging
- Zoom in/out with Ctrl+scroll
- Playhead moves when dragged

---

### Phase 2.2: Timeline Playback

**Goal**: Preview the sequence in real-time

**Deliverables**:
- [ ] Play button starts playback at playhead position
- [ ] Playhead advances during playback
- [ ] Video player shows current frame from correct source
- [ ] Seamless transitions between clips on timeline
- [ ] Stop at end of sequence or on click

**Files to modify**:
```
ui/timeline/timeline_widget.py   # Add playback controls
ui/video_player.py               # Support multi-source playback
ui/main_window.py                # Wire up signals
```

**Success criteria**:
- Press play, watch 3-clip sequence
- Playhead tracks playback position
- Preview switches sources at clip boundaries

---

### Phase 2.3: Algorithmic Remix

**Goal**: Generate sequences automatically

**Deliverables**:
- [x] **Shuffle**: Random order, no same-source back-to-back
- [ ] **Similarity Chain**: Each clip similar to previous (greedy TSP)
- [ ] **Building Energy**: Sort by motion intensity (requires optical flow)
- [x] Remix panel with algorithm selector
- [x] "Generate" button populates timeline
- [x] Configurable clip count / target duration

**Files to create**:
```
core/remix/__init__.py
core/remix/shuffle.py
core/remix/similarity.py
core/remix/rhythm.py
core/motion.py              # Optical flow motion extraction
ui/remix_panel.py           # Algorithm selection UI
```

**Algorithm implementations**:

**Shuffle** (no same-source consecutive):
```python
def constrained_shuffle(clips, max_consecutive=1):
    """Fisher-Yates with rejection sampling."""
    for _ in range(1000):
        shuffled = clips.copy()
        random.shuffle(shuffled)
        if _check_no_consecutive(shuffled):
            return shuffled
    return _greedy_repair(clips)  # Fallback
```

**Similarity Chain** (greedy nearest neighbor):
```python
def similarity_chain(clips, similarity_fn):
    """Each clip followed by most similar unvisited."""
    visited = {0}
    path = [0]
    while len(visited) < len(clips):
        current = path[-1]
        best = max(
            (i for i in range(len(clips)) if i not in visited),
            key=lambda i: similarity_fn(clips[current], clips[i])
        )
        visited.add(best)
        path.append(best)
    return [clips[i] for i in path]
```

**Success criteria**:
- "Shuffle" generates random sequence, no same-source consecutive
- "Similarity" creates smooth visual flow
- "Building" orders low-to-high motion

---

### Phase 2.4: Sequence Export

**Goal**: Render timeline to video file

**Deliverables**:
- [x] "Export Sequence" button
- [ ] Export dialog (format, quality, resolution)
- [x] FFmpeg concat demuxer for fast export
- [x] Re-encode option for mixed sources
- [x] Progress bar during export
- [x] Open output folder on completion

**Files to create/modify**:
```
core/sequence_export.py      # New: timeline → video
ui/export_dialog.py          # New: export options
core/ffmpeg.py               # Add concat_clips method
```

**Success criteria**:
- Export 5-clip sequence to MP4
- Progress bar shows export progress
- Output plays correctly in VLC

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│  Scene Ripper - project.srp                              [─][□][×]
├─────────────────────────────────────────────────────────────────┤
│ [Import] [Import URL] Sensitivity: [──●────] 3.0 [Detect]       │
│                                                    [Export Clips]│
├───────────────────────┬─────────────────────────────────────────┤
│                       │                                          │
│  ┌─────┐ ┌─────┐     │      ┌───────────────────────────┐       │
│  │ 001 │ │ 002 │     │      │                           │       │
│  │ 2.3s│ │ 4.1s│     │      │      VIDEO PREVIEW        │       │
│  └─────┘ └─────┘     │      │                           │       │
│  ┌─────┐ ┌─────┐     │      │                           │       │
│  │ 003 │ │ 004 │     │      └───────────────────────────┘       │
│  │ 1.8s│ │ 3.2s│     │       00:12 / 02:34  [▶] ──●─────        │
│  └─────┘ └─────┘     │                                          │
│                       │                                          │
├───────────────────────┴─────────────────────────────────────────┤
│  TIMELINE                           Remix: [Shuffle    ▼]       │
│  ──────────────────────────────────  [Generate] [Clear]         │
│                                                                  │
│  0:00    0:10    0:20    0:30    0:40    0:50    1:00           │
│  │────────│────────│────────│────────│────────│────────│        │
│  ▼                                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │V1│ [Clip 001 ▓▓▓] [Clip 003] [Clip 002 ▓▓▓▓▓]            │ │
│  │A1│ [░░░░░░░░░░░░] [░░░░░░░░] [░░░░░░░░░░░░░░░]            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  [+ Track]                                    [Export Sequence]  │
└─────────────────────────────────────────────────────────────────┘
```

## Acceptance Criteria

### Functional Requirements

- [x] Drag clips from browser to timeline
- [x] Reposition clips by dragging on timeline
- [ ] Delete clips from timeline (Delete key or right-click)
- [x] Add/remove tracks
- [x] Zoom timeline (Ctrl+scroll)
- [x] Scrub playhead to preview position
- [ ] Play sequence in preview player
- [x] Generate sequence with "Shuffle" algorithm
- [ ] Generate sequence with "Similarity" algorithm
- [x] Export sequence to MP4

### Non-Functional Requirements

- [ ] Timeline renders smoothly with 50+ clips
- [ ] Playback preview <100ms latency between clips
- [ ] Zoom/scroll responsive (no lag)
- [ ] Export 2-minute sequence in <30 seconds (re-encode) or <5 seconds (concat)

## Dependencies

**Existing**:
- PySide6 (QGraphicsView, QGraphicsScene)
- FFmpeg (export)
- OpenCV (motion analysis for rhythm remix)

**No new dependencies required.**

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| QGraphicsView performance with many clips | Medium | Medium | Virtual scrolling, item caching |
| Seamless playback across sources | Medium | High | Pre-buffer next clip, or accept brief pause |
| Motion analysis too slow | Low | Low | Make rhythm remix optional, show progress |
| Mixed codec export issues | Low | Medium | Always re-encode for sequence export |

## Success Metrics

| Metric | Target |
|--------|--------|
| Clips on timeline | Supports 100+ |
| Remix generation | <2 seconds for 20 clips |
| Playback latency | <100ms between clips |
| Export speed | 2x realtime (re-encode) |

## Immediate Next Steps

**Week 1**:
1. Create `TimelineWidget` with basic QGraphicsView
2. Implement `TrackItem` and `ClipItem` with drag-drop
3. Add playhead with drag support

**Week 2**:
4. Wire timeline to video player for preview
5. Implement shuffle remix algorithm
6. Add "Generate" button

**Week 3**:
7. Add sequence export (FFmpeg concat)
8. Polish UI, add keyboard shortcuts
9. Test with real footage

---

*Generated: 2026-01-24*
