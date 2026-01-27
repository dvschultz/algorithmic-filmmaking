---
title: "feat: Add missing analysis tools to GUI for agent parity"
type: feat
date: 2026-01-27
---

# Add Missing Analysis Tools to GUI

## Overview

The agent chat has several analysis tools that are not accessible via the GUI. This creates a feature gap where users must use the chat agent to access capabilities like ImageNet classification, YOLO object detection, and VLM content description. This plan adds the missing GUI buttons and wiring to achieve full parity.

## Problem Statement

**Current State:**
| Analysis Tool | Agent Chat | GUI Analyze Tab |
|---------------|------------|-----------------|
| Color extraction | `analyze_colors_live` | "Extract Colors" button |
| Shot classification | `analyze_shots_live` | "Classify Shots" button |
| Transcription | `transcribe_live` | "Transcribe" button |
| ImageNet classification | `classify_content_live` | **MISSING** |
| YOLO object detection | `detect_objects_live` | **MISSING** |
| Person counting | `count_people_live` | **MISSING** |
| VLM description | `describe_content_live` | **MISSING** |

**Impact:** Users cannot access 4 analysis capabilities from the GUI that are available to the agent.

**Root Cause:** Workers exist in `main_window.py` for all operations, but signals and buttons were never added to the Analyze tab for the newer analysis types.

## Proposed Solution

Add new buttons to the Analyze tab that trigger the existing workers, following the established pattern used by Extract Colors, Classify Shots, and Transcribe.

### Design Decisions

1. **Consolidate Count People into Detect Objects** - Don't add a separate button; person count is always included in object detection results.

2. **Keep "Analyze All" as-is** - The current 3-operation scope (colors, shots, transcribe) is fast and well-understood. Adding 4 more would change runtime from ~2 minutes to ~20+ minutes.

3. **Use Settings for tier/model selection** - The "Describe" button uses the Vision Description settings already in the Settings dialog. No inline tier selector needed.

4. **Add model download confirmation** - When first using a feature that requires model download, show a confirmation dialog.

## Technical Approach

### File Changes

#### 1. `ui/tabs/analyze_tab.py` - Add signals and buttons

**Add signals** (after line 43):
```python
classify_requested = Signal()
detect_objects_requested = Signal()
describe_requested = Signal()
```

**Add buttons** in `_create_controls()` (after Transcribe button, before Analyze All):
```python
# Classify Content button
self.classify_btn = QPushButton("Classify")
self.classify_btn.setToolTip(
    "Classify frame content using ImageNet labels\n"
    "(dog, car, tree, person, etc.)"
)
self.classify_btn.setEnabled(False)
self.classify_btn.clicked.connect(self._on_classify_click)
controls.addWidget(self.classify_btn)

# Detect Objects button
self.detect_btn = QPushButton("Detect Objects")
self.detect_btn.setToolTip(
    "Detect and locate objects using YOLO\n"
    "Includes bounding boxes and person count"
)
self.detect_btn.setEnabled(False)
self.detect_btn.clicked.connect(self._on_detect_click)
controls.addWidget(self.detect_btn)

# Describe Content button
self.describe_btn = QPushButton("Describe")
self.describe_btn.setToolTip(
    "Generate AI descriptions of frame content\n"
    "Uses model configured in Settings > Vision Description"
)
self.describe_btn.setEnabled(False)
self.describe_btn.clicked.connect(self._on_describe_click)
controls.addWidget(self.describe_btn)
```

**Add click handlers**:
```python
def _on_classify_click(self):
    """Handle Classify button click."""
    self.classify_requested.emit()

def _on_detect_click(self):
    """Handle Detect Objects button click."""
    self.detect_objects_requested.emit()

def _on_describe_click(self):
    """Handle Describe button click."""
    self.describe_requested.emit()
```

**Update `set_analyzing()`** to handle new operation types:
```python
def set_analyzing(self, operation: str, analyzing: bool):
    """Set the analyzing state for a specific operation."""
    button_map = {
        "colors": self.colors_btn,
        "shots": self.shots_btn,
        "transcribe": self.transcribe_btn,
        "classify": self.classify_btn,      # NEW
        "detect": self.detect_btn,          # NEW
        "describe": self.describe_btn,      # NEW
        "all": self.analyze_all_btn,
    }
    # ... rest of method
```

**Update `_update_button_states()`** to include new buttons:
```python
def _update_button_states(self):
    """Enable/disable buttons based on clip selection."""
    has_clips = len(self._clip_ids) > 0
    self.colors_btn.setEnabled(has_clips)
    self.shots_btn.setEnabled(has_clips)
    self.transcribe_btn.setEnabled(has_clips)
    self.classify_btn.setEnabled(has_clips)      # NEW
    self.detect_btn.setEnabled(has_clips)        # NEW
    self.describe_btn.setEnabled(has_clips)      # NEW
    self.analyze_all_btn.setEnabled(has_clips)
    self.clear_btn.setEnabled(has_clips)
```

#### 2. `ui/main_window.py` - Add signal connections and handlers

**Add signal connections** in `_connect_signals()` (around line 245):
```python
# Existing connections
self.analyze_tab.analyze_colors_requested.connect(self._on_analyze_colors_from_tab)
self.analyze_tab.analyze_shots_requested.connect(self._on_analyze_shots_from_tab)
self.analyze_tab.transcribe_requested.connect(self._on_transcribe_from_tab)
self.analyze_tab.analyze_all_requested.connect(self._on_analyze_all_from_tab)

# NEW connections
self.analyze_tab.classify_requested.connect(self._on_classify_from_tab)
self.analyze_tab.detect_objects_requested.connect(self._on_detect_objects_from_tab)
self.analyze_tab.describe_requested.connect(self._on_describe_from_tab)
```

**Add handler methods**:
```python
def _on_classify_from_tab(self):
    """Handle classify request from Analyze tab."""
    clip_ids = list(self.analyze_tab._clip_ids)
    if not clip_ids:
        return

    clips = [self.clips_by_id[cid] for cid in clip_ids if cid in self.clips_by_id]
    if not clips:
        return

    self.analyze_tab.set_analyzing("classify", True)
    self._run_classification_worker(clips, source="gui")

def _on_detect_objects_from_tab(self):
    """Handle detect objects request from Analyze tab."""
    clip_ids = list(self.analyze_tab._clip_ids)
    if not clip_ids:
        return

    clips = [self.clips_by_id[cid] for cid in clip_ids if cid in self.clips_by_id]
    if not clips:
        return

    self.analyze_tab.set_analyzing("detect", True)
    self._run_object_detection_worker(clips, source="gui")

def _on_describe_from_tab(self):
    """Handle describe request from Analyze tab."""
    clip_ids = list(self.analyze_tab._clip_ids)
    if not clip_ids:
        return

    clips = [self.clips_by_id[cid] for cid in clip_ids if cid in self.clips_by_id]
    if not clips:
        return

    # Check for API key if cloud tier
    settings = load_settings()
    if settings.description_model_tier == "cloud":
        # Check for appropriate API key based on model
        # Show error dialog if missing
        pass

    self.analyze_tab.set_analyzing("describe", True)
    self._run_description_worker(clips, source="gui")
```

**Update worker completion handlers** to reset button state:
```python
def _on_classification_finished(self, results: dict):
    """Handle classification worker completion."""
    self.analyze_tab.set_analyzing("classify", False)
    # ... existing result handling

def _on_object_detection_finished(self, results: dict):
    """Handle object detection worker completion."""
    self.analyze_tab.set_analyzing("detect", False)
    # ... existing result handling

def _on_description_finished(self, results: dict):
    """Handle description worker completion."""
    self.analyze_tab.set_analyzing("describe", False)
    # ... existing result handling
```

## Acceptance Criteria

### Functional Requirements

- [x] "Classify" button appears in Analyze tab after Transcribe button
- [x] "Detect Objects" button appears after Classify button
- [x] "Describe" button appears after Detect Objects button
- [x] All three new buttons are disabled when no clips are present
- [x] All three new buttons are enabled when clips are present
- [x] Clicking "Classify" runs `ClassificationWorker` on all clips in tab
- [x] Clicking "Detect Objects" runs `ObjectDetectionWorker` on all clips in tab
- [x] Clicking "Describe" runs `DescriptionWorker` on all clips in tab
- [x] Buttons show "running" state while worker is active
- [x] Results are stored in clip model (`object_labels`, `detected_objects`, `person_count`, `description`)
- [x] Project saves include new analysis data

### Non-Functional Requirements

- [ ] Button layout does not overflow on 1280px wide window
- [x] Tooltips explain what each button does
- [x] Workers can be cancelled via existing cancel mechanism

### Quality Gates

- [ ] No new regressions in existing analysis buttons
- [x] Workers started from GUI vs agent are distinguished for result handling

## Implementation Notes

### Existing Workers (no changes needed)

All workers already exist in `ui/main_window.py`:
- `ClassificationWorker` (lines 477-519)
- `ObjectDetectionWorker` (lines 522-571)
- `DescriptionWorker` (lines 574-616)

### Existing Agent Tools (reference only)

The agent tools in `core/chat_tools.py` show the expected behavior:
- `classify_content_live()` - line 2789
- `detect_objects_live()` - line 2826
- `count_people_live()` - line 2863
- `describe_content_live()` - line 411

### Model Download Handling

First-time use of these features requires model downloads:
- **Classification**: MobileNetV3-Small (~100MB)
- **Object Detection**: YOLOv8-nano (~6MB)
- **Description (CPU)**: Moondream 2B (~1.6GB)

The workers handle this transparently, but a progress indicator would improve UX. This is a future enhancement.

## Future Enhancements (Out of Scope)

1. **Model download progress dialog** - Show progress when downloading models
2. **Settings for thresholds** - Add classification/detection confidence settings
3. **ClipBrowser filters for new metadata** - Filter by detected objects, person count
4. **Result display in clip cards** - Show labels/objects/descriptions in thumbnails
5. **Extended "Analyze All"** - Option to include all 7 analysis types

## References

### Internal References

- Analyze tab UI: `ui/tabs/analyze_tab.py`
- Worker definitions: `ui/main_window.py:477-616`
- Agent tools: `core/chat_tools.py:411-2898`
- Clip model: `models/clip.py`
- Settings: `core/settings.py`

### Related Plans

- `docs/plans/2026-01-26-feat-content-analysis-imagenet-yolo-plan.md`
- `docs/plans/2026-01-26-feat-video-description-vision-models-plan.md`
- `docs/plans/2026-01-25-feat-agent-accessible-gui-features-plan.md`
