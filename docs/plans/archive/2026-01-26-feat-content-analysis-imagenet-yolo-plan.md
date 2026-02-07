---
title: "Content Analysis: ImageNet Classification & YOLOv8 Object Detection"
type: feat
date: 2026-01-26
---

# Content Analysis Features Implementation Plan

## Overview

Add three content analysis capabilities to Scene Ripper:

1. **Frame Classification (ImageNet/MobileNet)** - Tag clips with object labels (car, dog, person, tree, etc.)
2. **Object Detection (YOLOv8)** - Count and locate objects per clip
3. **Person Detection** - Count people in clips (specialized YOLO application)

This plan follows the established patterns:
- Analysis modules in `core/analysis/`
- QThread workers in `ui/main_window.py`
- Agent tools in `core/chat_tools.py` with both CLI and GUI-live variants
- CLI commands in `cli/commands/analyze.py`

---

## Problem Statement

Users currently lack:
- Automatic content tagging (what objects appear in clips)
- Object counting per scene (how many people, cars, etc.)
- Person detection for interview/crowd scene identification
- Content-based filtering ("find clips with dogs")

Existing analysis provides color palette and shot type, but not semantic content understanding.

---

## Proposed Solution

### Model Selection (CPU-First Priority)

| Feature | Model | Size | Rationale |
|---------|-------|------|-----------|
| Frame Classification | MobileNetV3-Small | ~5MB | CPU-optimized, 70%+ ImageNet accuracy |
| Object Detection | YOLOv8n (nano) | ~6MB | Fastest YOLO, real-time on CPU |
| Person Detection | YOLOv8n (COCO classes) | ~6MB | Reuse YOLO with person class filter |

**Alternative for higher accuracy** (optional GPU path):
- Classification: EfficientNet-B0 (~21MB)
- Detection: YOLOv8s (small) (~22MB)

### Data Model Changes

**File**: `models/clip.py`

```python
@dataclass
class Clip:
    # ... existing fields ...

    # NEW: Content analysis fields
    object_labels: Optional[list[str]] = None  # ImageNet labels, e.g., ["dog", "car", "tree"]
    detected_objects: Optional[list[dict]] = None  # [{label, confidence, bbox}]
    person_count: Optional[int] = None  # Number of people detected
```

Serialization format for `detected_objects`:
```json
{
  "detected_objects": [
    {"label": "person", "confidence": 0.92, "bbox": [100, 50, 200, 300]},
    {"label": "car", "confidence": 0.87, "bbox": [300, 100, 500, 250]}
  ]
}
```

---

## Technical Approach

### Directory Structure (New Files)

```
core/
├── analysis/
│   ├── color.py          (existing)
│   ├── shots.py          (existing)
│   ├── classification.py  # NEW: ImageNet classification
│   ├── detection.py       # NEW: YOLOv8 object detection
│   └── models.py          # NEW: Shared model loading utilities
cli/
└── commands/
    └── analyze.py         # EXTEND: Add classify, detect subcommands
ui/
└── main_window.py         # EXTEND: Add workers
```

### Lazy Model Loading Pattern

Following the pattern in `core/analysis/shots.py`:

```python
# core/analysis/models.py
"""Shared model loading utilities with lazy initialization."""

import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model singletons and locks
_mobilenet_model = None
_mobilenet_lock = threading.Lock()

_yolo_model = None
_yolo_lock = threading.Lock()


def get_mobilenet_model():
    """Lazy load MobileNetV3-Small (thread-safe)."""
    global _mobilenet_model

    if _mobilenet_model is not None:
        return _mobilenet_model

    with _mobilenet_lock:
        if _mobilenet_model is None:
            logger.info("Loading MobileNetV3-Small model...")
            import torch
            from torchvision import models

            _mobilenet_model = models.mobilenet_v3_small(pretrained=True)
            _mobilenet_model.eval()
            logger.info("MobileNetV3-Small loaded")

    return _mobilenet_model


def get_yolo_model(model_size: str = "n"):
    """Lazy load YOLOv8 model (thread-safe)."""
    global _yolo_model

    if _yolo_model is not None:
        return _yolo_model

    with _yolo_lock:
        if _yolo_model is None:
            logger.info(f"Loading YOLOv8{model_size} model...")
            from ultralytics import YOLO

            _yolo_model = YOLO(f"yolov8{model_size}.pt")
            logger.info(f"YOLOv8{model_size} loaded")

    return _yolo_model
```

---

## Phase 1: Frame Classification (ImageNet/MobileNet)

### 1.1 Core Analysis Module

**File**: `core/analysis/classification.py`

```python
"""Frame classification using MobileNetV3."""

import logging
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet class labels (top-level categories)
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Lazy load
_model = None
_model_lock = threading.Lock()
_labels = None


def _load_model():
    """Lazy load MobileNetV3-Small (thread-safe)."""
    global _model, _labels

    if _model is not None:
        return _model, _labels

    with _model_lock:
        if _model is None:
            logger.info("Loading MobileNetV3-Small...")
            import torch
            from torchvision import models, transforms

            _model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            _model.eval()

            # Load labels
            import urllib.request
            with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
                _labels = [line.decode("utf-8").strip() for line in f.readlines()]

            logger.info("MobileNetV3-Small loaded")

    return _model, _labels


def classify_frame(
    image_path: Path,
    top_k: int = 5,
    threshold: float = 0.1,
) -> list[tuple[str, float]]:
    """
    Classify objects in a frame using MobileNetV3.

    Args:
        image_path: Path to image file
        top_k: Number of top predictions to return
        threshold: Minimum confidence threshold

    Returns:
        List of (label, confidence) tuples
    """
    import torch
    from torchvision import transforms

    model, labels = _load_model()

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            confidence = prob.item()
            if confidence >= threshold:
                label = labels[idx.item()]
                results.append((label, confidence))

        return results

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return []
```

### 1.2 QThread Worker

**File**: `ui/main_window.py` (add new worker class)

```python
class ClassificationWorker(QThread):
    """Background worker for frame classification using MobileNet."""

    progress = Signal(int, int)  # current, total
    labels_ready = Signal(str, list)  # clip_id, [(label, confidence), ...]
    finished = Signal()

    def __init__(self, clips: list[Clip], top_k: int = 5, threshold: float = 0.1):
        super().__init__()
        self.clips = clips
        self.top_k = top_k
        self.threshold = threshold
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        logger.info("ClassificationWorker.run() STARTING")
        from core.analysis.classification import classify_frame

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    results = classify_frame(
                        clip.thumbnail_path,
                        top_k=self.top_k,
                        threshold=self.threshold,
                    )
                    labels = [label for label, _ in results]
                    self.labels_ready.emit(clip.id, results)
            except Exception as e:
                logger.warning(f"Classification failed for {clip.id}: {e}")
            self.progress.emit(i + 1, total)

        self.finished.emit()
        logger.info("ClassificationWorker.run() COMPLETED")
```

### 1.3 Agent Tool (GUI-Live)

**File**: `core/chat_tools.py`

```python
@tools.register(
    description="Classify frame content using ImageNet labels. Identifies objects like 'dog', 'car', 'tree' in clips.",
    requires_project=True,
    modifies_gui_state=True
)
def classify_content_live(main_window, clip_ids: list[str], top_k: int = 5) -> dict:
    """Classify content in clips with live GUI update."""
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    if main_window.classification_worker and main_window.classification_worker.isRunning():
        return {"success": False, "error": "Classification already in progress"}

    return {
        "_wait_for_worker": "classification",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
        "top_k": top_k,
    }
```

### 1.4 CLI Command

**File**: `cli/commands/analyze.py` (extend existing)

```python
@analyze.command("classify")
@click.argument("project_file", type=click.Path(exists=True, path_type=Path))
@click.option("--clip", "-c", "clip_ids", multiple=True)
@click.option("--top-k", "-k", type=int, default=5)
@click.option("--threshold", "-t", type=float, default=0.1)
@click.option("--force", "-f", is_flag=True)
@click.pass_context
def classify(ctx, project_file, clip_ids, top_k, threshold, force):
    """Classify frame content using ImageNet labels."""
    # Similar pattern to colors/shots commands
```

---

## Phase 2: Object Detection (YOLOv8)

### 2.1 Core Analysis Module

**File**: `core/analysis/detection.py`

```python
"""Object detection using YOLOv8."""

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# YOLOv8 COCO classes (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

_model = None
_model_lock = threading.Lock()


def _load_yolo():
    """Lazy load YOLOv8n (thread-safe)."""
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info("Loading YOLOv8n model...")
            from ultralytics import YOLO
            _model = YOLO("yolov8n.pt")  # Downloads on first use (~6MB)
            logger.info("YOLOv8n loaded")

    return _model


def detect_objects(
    image_path: Path,
    confidence_threshold: float = 0.5,
    classes: Optional[list[int]] = None,
) -> list[dict]:
    """
    Detect objects in an image using YOLOv8.

    Args:
        image_path: Path to image
        confidence_threshold: Minimum detection confidence
        classes: Optional list of class IDs to filter (None = all)

    Returns:
        List of detections: [{label, confidence, bbox: [x1, y1, x2, y2]}]
    """
    model = _load_yolo()

    try:
        results = model(str(image_path), verbose=False, conf=confidence_threshold, classes=classes)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                detections.append({
                    "label": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}",
                    "confidence": round(conf, 3),
                    "bbox": [int(x) for x in bbox],
                })

        return detections

    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        return []


def count_people(image_path: Path, confidence_threshold: float = 0.5) -> int:
    """Count people in an image using YOLOv8."""
    # Person is class 0 in COCO
    detections = detect_objects(image_path, confidence_threshold, classes=[0])
    return len(detections)
```

### 2.2 QThread Worker

**File**: `ui/main_window.py` (add new worker class)

```python
class ObjectDetectionWorker(QThread):
    """Background worker for object detection using YOLOv8."""

    progress = Signal(int, int)  # current, total
    objects_ready = Signal(str, list, int)  # clip_id, detections, person_count
    finished = Signal()

    def __init__(self, clips: list[Clip], confidence: float = 0.5, detect_all: bool = True):
        super().__init__()
        self.clips = clips
        self.confidence = confidence
        self.detect_all = detect_all  # False = persons only
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        logger.info("ObjectDetectionWorker.run() STARTING")
        from core.analysis.detection import detect_objects, count_people

        total = len(self.clips)
        for i, clip in enumerate(self.clips):
            if self._cancelled:
                break
            try:
                if clip.thumbnail_path and clip.thumbnail_path.exists():
                    if self.detect_all:
                        detections = detect_objects(clip.thumbnail_path, self.confidence)
                        person_count = sum(1 for d in detections if d["label"] == "person")
                    else:
                        detections = []
                        person_count = count_people(clip.thumbnail_path, self.confidence)

                    self.objects_ready.emit(clip.id, detections, person_count)
            except Exception as e:
                logger.warning(f"Detection failed for {clip.id}: {e}")
            self.progress.emit(i + 1, total)

        self.finished.emit()
        logger.info("ObjectDetectionWorker.run() COMPLETED")
```

### 2.3 Agent Tools

**File**: `core/chat_tools.py`

```python
@tools.register(
    description="Detect and count objects in clips using YOLO. Returns object labels, counts, and bounding boxes.",
    requires_project=True,
    modifies_gui_state=True
)
def detect_objects_live(main_window, clip_ids: list[str], confidence: float = 0.5) -> dict:
    """Detect objects in clips with live GUI update."""
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    if main_window.detection_worker and main_window.detection_worker.isRunning():
        return {"success": False, "error": "Object detection already in progress"}

    return {
        "_wait_for_worker": "object_detection",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
    }


@tools.register(
    description="Count people in clips. Faster than full object detection when you only need person counts.",
    requires_project=True,
    modifies_gui_state=True
)
def count_people_live(main_window, clip_ids: list[str]) -> dict:
    """Count people in clips with live GUI update."""
    valid_ids = [cid for cid in clip_ids if cid in main_window.project.clips_by_id]

    if not valid_ids:
        return {"success": False, "error": "No valid clips found"}

    return {
        "_wait_for_worker": "person_detection",
        "clip_ids": valid_ids,
        "clip_count": len(valid_ids),
    }
```

---

## Phase 3: Filter Integration

### 3.1 Extend filter_clips Tool

**File**: `core/chat_tools.py` (modify existing `filter_clips`)

Add new filter criteria:
```python
@tools.register(
    description="Filter clips by criteria including content analysis. New filters: has_object (object label), min_people, max_people.",
    requires_project=True,
    modifies_gui_state=False
)
def filter_clips(
    project,
    shot_type: Optional[str] = None,
    has_speech: Optional[bool] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    search_query: Optional[str] = None,
    # NEW FILTERS
    has_object: Optional[str] = None,     # Filter by object label
    min_people: Optional[int] = None,      # Minimum person count
    max_people: Optional[int] = None,      # Maximum person count
) -> list[dict]:
    """Filter clips by various criteria including content analysis."""
    results = []

    for clip in project.clips:
        # ... existing filters ...

        # Object label filter
        if has_object is not None:
            labels = clip.object_labels or []
            detected_labels = [d["label"] for d in (clip.detected_objects or [])]
            all_labels = set(labels + detected_labels)
            if has_object.lower() not in [l.lower() for l in all_labels]:
                continue

        # Person count filters
        if min_people is not None:
            if (clip.person_count or 0) < min_people:
                continue
        if max_people is not None:
            if (clip.person_count or 0) > max_people:
                continue

        results.append(clip_to_dict(clip))

    return results
```

---

## Dependencies

### requirements.txt additions

```
# Content Analysis
ultralytics>=8.0.0  # YOLOv8
torchvision>=0.15.0  # MobileNet (already have torch)
```

### Optional GPU support

The models automatically use GPU if available. For explicit CUDA support:
```
torch>=2.0+cu118  # CUDA 11.8 variant
```

---

## Tool Timeouts

**File**: `core/chat_tools.py` (update TOOL_TIMEOUTS)

```python
TOOL_TIMEOUTS = {
    # ... existing ...
    "classify_content": 300,      # 5 minutes
    "classify_content_live": 300,
    "detect_objects": 300,
    "detect_objects_live": 300,
    "count_people": 300,
    "count_people_live": 300,
}
```

---

## Acceptance Criteria

### Functional Requirements
- [ ] `classify_content_live` returns top-k ImageNet labels per clip
- [ ] `detect_objects_live` returns COCO object labels with bounding boxes
- [ ] `count_people_live` returns person count per clip
- [ ] CLI commands: `scene_ripper analyze classify`, `objects`, `people`
- [ ] `filter_clips` supports `has_object`, `min_people`, `max_people`
- [ ] Models load lazily on first use
- [ ] CPU fallback works when GPU unavailable

### Non-Functional Requirements
- [ ] Classification: <1s per frame on M1 Mac CPU
- [ ] YOLO detection: <0.5s per frame on M1 Mac CPU
- [ ] Memory: <2GB RAM for model loading
- [ ] First-run model download shows progress

### Testing
- [ ] Unit tests for each core analysis function
- [ ] Worker cancellation tests
- [ ] CLI command tests with JSON output
- [ ] Integration test: classify -> filter by object

---

## Implementation Order

### Week 1: Phase 1 - Classification
1. Create `core/analysis/classification.py`
2. Add `ClassificationWorker` to `ui/main_window.py`
3. Add `classify_content_live` tool
4. Add CLI command
5. Update Clip model

### Week 2: Phase 2 - Object Detection
1. Create `core/analysis/detection.py`
2. Add `ObjectDetectionWorker`
3. Add `detect_objects_live` and `count_people_live` tools
4. Add CLI commands
5. Update Clip model

### Week 3: Phase 3 - Integration
1. Extend `filter_clips` with new filters
2. Add UI buttons to Analyze tab
3. Testing and documentation
4. Performance optimization

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large model downloads on first use | Poor UX | Show download progress, cache models locally |
| Slow CPU inference | Bad performance | Use nano models, batch processing, optional GPU |
| PyTorch conflicts with existing torch | Import errors | Use same torch version, test compatibility |
| Memory pressure with large projects | OOM crashes | Process clips in batches, unload models when idle |
| YOLO model license (AGPL) | Legal concern | Document license, consider alternative models |

---

## References

### Internal
- `core/analysis/shots.py:42-85` - CLIP model loading pattern
- `core/analysis/color.py` - Analysis module pattern
- `ui/main_window.py:325-359` - ColorAnalysisWorker pattern
- `core/chat_tools.py:2578-2604` - analyze_colors_live pattern
- `cli/commands/analyze.py` - CLI command pattern

### External
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
- [COCO Dataset Classes](https://cocodataset.org/#explore)

---

*Generated: 2026-01-26*
