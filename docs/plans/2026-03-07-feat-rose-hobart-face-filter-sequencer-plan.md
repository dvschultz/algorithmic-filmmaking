---
title: "feat: Add Rose Hobart face-filter sequencer"
type: feat
status: completed
date: 2026-03-07
origin: docs/brainstorms/2026-03-07-rose-hobart-sequencer-brainstorm.md
---

# feat: Add Rose Hobart face-filter sequencer

## Overview

Add a new dialog-based sequencer algorithm called "Rose Hobart" that filters clips by person identity using face recognition. The user provides 1-3 reference images, and the system uses InsightFace/ArcFace face embeddings to keep only clips where that person's face appears. Face embedding extraction is also available as a standalone "Detect Faces" analysis operation in the Analyze tab.

Named after Joseph Cornell's 1936 film where he re-edited *East of Borneo* to isolate shots of actress Rose Hobart.

## Problem Statement / Motivation

Users working with multi-source footage often want to create supercuts focused on a specific person. Currently there is no way to automatically filter clips by person identity. Users must manually scrub through all clips to find appearances, which is tedious and error-prone for large projects.

## Proposed Solution

Face recognition via InsightFace/ArcFace, chosen over VLM and CLIP alternatives for the best balance of speed, cost, and accuracy (see brainstorm: `docs/brainstorms/2026-03-07-rose-hobart-sequencer-brainstorm.md`).

**Accepted limitation**: The person's face must be at least partially visible. This aligns with the Rose Hobart concept.

## Technical Approach

### Architecture

```
User clicks Rose Hobart card
        |
        v
RoseHobartDialog opens (modal)
        |
        v
User selects 1-3 reference images + sensitivity + ordering
        |
        v
[Generate clicked]
        |
        v
RoseHobartWorker (QThread inside dialog)
    |-- Extract reference face embeddings from images
    |-- For each clip:
    |     |-- If face_embeddings cached on Clip: use cache
    |     |-- Else: sample frames every 1s, extract faces, cache on Clip
    |     |-- Compare clip faces against reference embedding
    |     |-- Record match confidence (max cosine similarity)
    |-- Filter: keep clips where max_similarity >= threshold
    |-- Order filtered clips per user selection
    |-- Emit sequence_ready signal
        |
        v
SequenceTab._apply_dialog_sequence() places clips on timeline
```

### New Files

| File | Purpose |
|------|---------|
| `core/analysis/faces.py` | InsightFace model loading, face detection, embedding extraction, comparison |
| `core/remix/rose_hobart.py` | Filter + order logic (called by dialog worker) |
| `ui/dialogs/rose_hobart_dialog.py` | Dialog UI with inner QThread worker |
| `tests/test_rose_hobart.py` | Unit tests for face matching and filtering logic |

### Modified Files

| File | Change |
|------|--------|
| `models/clip.py` | Add `face_embeddings` field, update `to_dict`/`from_dict` |
| `core/analysis_operations.py` | Add `"face_embeddings"` operation |
| `core/feature_registry.py` | Add `"face_detect"` feature dependency |
| `ui/algorithm_config.py` | Add `"rose_hobart"` algorithm entry |
| `ui/dialogs/__init__.py` | Import and export `RoseHobartDialog` |
| `ui/tabs/sequence_tab.py` | Add `_show_rose_hobart_dialog`, routing in `_on_card_clicked` and `_on_confirm_generate` |
| `ui/main_window.py` | Add `_launch_face_detection_worker` in `_launch_analysis_worker` |
| `core/chat_tools.py` | Add `rose_hobart` to `list_sorting_algorithms`, add dedicated `generate_rose_hobart` tool |
| `core/project.py` | Bump `SCHEMA_VERSION` to `"1.3"` |
| `requirements.txt` | Add `insightface`, `onnxruntime` (or `onnxruntime-silicon` for Apple Silicon) |

### Implementation Phases

#### Phase 1: Core Face Analysis Module

**Deliverable**: `core/analysis/faces.py` -- standalone, testable face extraction and comparison.

**Tasks:**
- [x] Create `core/analysis/faces.py` with module-level model caching (following `core/analysis/detection.py` pattern: `_model = None`, `_model_lock = threading.Lock()`, double-check locking)
- [x] Implement `_load_insightface()` -- lazy model loading with auto-download on first use. Use `settings.model_cache_dir` for weights storage. Handle download progress via callback parameter.
- [x] Implement `extract_faces_from_image(image_path: Path) -> list[dict]` -- detect faces in a single image, return list of `{"bbox": [x, y, w, h], "embedding": list[float], "confidence": float}`
- [x] Implement `extract_faces_from_clip(source_path: Path, start_frame: int, end_frame: int, fps: float, sample_interval: float = 1.0) -> list[dict]` -- sample frames every N seconds using OpenCV `VideoCapture` (following `core/analysis/color.py` pattern), extract faces from each frame, return deduplicated list of face dicts with `frame_number` added to each
- [x] Implement `compare_faces(reference_embeddings: list[list[float]], clip_faces: list[dict], threshold: float) -> tuple[bool, float]` -- returns `(is_match, max_similarity)`. Compare each clip face embedding against each reference embedding via cosine similarity. Match if any pair exceeds threshold.
- [x] Implement `average_embeddings(embeddings: list[list[float]]) -> list[float]` -- average multiple reference embeddings for robust identity representation (used when user provides 2-3 reference images)
- [x] Implement `is_model_loaded() -> bool` and `unload_model()` for memory management
- [x] Add `insightface` and `onnxruntime` to `requirements.txt`
- [x] Register `"face_detect"` in `core/feature_registry.py`

**Sensitivity threshold values** (cosine similarity, ArcFace):
- **Strict**: >= 0.50 (very high confidence, frontal faces only)
- **Balanced**: >= 0.35 (good accuracy, allows angled faces)
- **Loose**: >= 0.25 (permissive, may include ambiguous matches)

These are starting values based on ArcFace literature. Tune empirically.

**Frame sampling detail**: Use OpenCV `VideoCapture` with `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)`. For clips shorter than `sample_interval`, sample a single frame at the midpoint (following the `_sample_frame_positions()` pattern from `core/analysis/color.py`).

**Multiple faces in reference image**: If `extract_faces_from_image()` finds multiple faces, use the one with the highest detection confidence. The dialog will show a bounding box overlay on the reference image preview so the user can verify which face was selected.

**Per-clip match confidence**: Maximum cosine similarity across all faces in all sampled frames. This is used for "By confidence" ordering.

**Test file**: `tests/test_rose_hobart.py`
- [x] Test `compare_faces` with matching embeddings above threshold
- [x] Test `compare_faces` with non-matching embeddings below threshold
- [x] Test `average_embeddings` produces correct mean vector
- [x] Test `extract_faces_from_image` returns empty list for image with no faces (mock InsightFace)
- [x] Test threshold preset values against known similarity scores

#### Phase 2: Clip Model + Analysis Operation

**Deliverable**: `face_embeddings` field on Clip, "Detect Faces" in Analyze tab.

**Tasks:**
- [x] Add `face_embeddings: Optional[list[dict]]` to `Clip` in `models/clip.py` (after `detected_objects` field, ~line 222). Default `None`.
- [x] Update `Clip.to_dict()` (~line 350): serialize `face_embeddings` only when not None. Embeddings are lists of floats, bboxes are lists of ints -- serialize directly (same pattern as `detected_objects`)
- [x] Update `Clip.from_dict()` (~line 420): deserialize `face_embeddings` with validation. Each entry must have `bbox` (list of 4 numbers), `embedding` (list of 512 floats), `confidence` (float). Discard malformed entries with a warning log.
- [x] Bump `SCHEMA_VERSION` to `"1.3"` in `core/project.py`. Old projects load cleanly since `face_embeddings` defaults to `None`.

**Analysis operation registration:**
- [x] Add `face_embeddings` to `ANALYSIS_OPERATIONS` in `core/analysis_operations.py`:

```python
AnalysisOperation(
    key="face_embeddings",
    label="Detect Faces",
    tooltip="Extract face embeddings for person identification",
    phase="sequential",  # Needs video file access for frame sampling
    default_enabled=False,  # Specialized operation, not in default analysis set
)
```

- [x] Add metadata check in `core/cost_estimates.py` `METADATA_CHECKS`:

```python
"face_embeddings": lambda clip: bool(clip.face_embeddings),
```

- [x] Add time estimate in `TIME_PER_CLIP`:

```python
"face_embeddings": {"local": 2.0},  # ~2s per clip (30 frames x ~50ms each + overhead)
```

- [x] Add `_launch_face_detection_worker` in `ui/main_window.py` `_launch_analysis_worker` dispatch (~line 2948). Follow the YOLO `ObjectDetectionWorker` pattern.
- [x] Create `ui/workers/face_detection_worker.py` extending `BatchProcessingWorker`. Process clips sequentially (video file access). Emit `progress(int, int)` per clip. Mutate `clip.face_embeddings` in-place (partial results persist on cancel, matching existing brightness/volume pattern).

**Worker guard pattern** (from learnings): Use boolean guard flag (`_face_analysis_finished_handled`), `Qt.UniqueConnection`, and `@Slot()` decorators on completion handlers.

**Test file**: `tests/test_rose_hobart.py` (append)
- [x] Test `Clip.to_dict()` round-trips face_embeddings correctly
- [x] Test `Clip.from_dict()` handles missing face_embeddings (old projects)
- [x] Test `Clip.from_dict()` discards malformed face embedding entries

#### Phase 3: Rose Hobart Dialog

**Deliverable**: Dialog UI with face matching and sequence generation.

**Tasks:**
- [x] Create `ui/dialogs/rose_hobart_dialog.py`:
  - Inherits `QDialog`
  - Declares `sequence_ready = Signal(list)` -- emits `list[(Clip, Source)]`
  - Uses `QStackedWidget` with pages: **Config** (reference images, settings) and **Progress** (progress bar, match count, action buttons)

**Config page layout:**
- [x] Reference image section:
  - "Add Reference Image" button opens `QFileDialog.getOpenFileName` with filter `"Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"`
  - Display up to 3 image thumbnails in a horizontal layout
  - Each thumbnail has a remove button (X)
  - Show detected face bounding box overlay on each thumbnail (run `extract_faces_from_image` immediately on selection)
  - If no face detected in an image, show warning icon and tooltip "No face detected in this image"
  - If multiple faces detected, use highest confidence; show that face's bounding box highlighted
- [x] Sensitivity dropdown: `QComboBox` with items "Strict", "Balanced" (default), "Loose"
- [x] Ordering dropdown: `QComboBox` with items "Original Order" (default), "By Duration", "By Color", "By Brightness", "By Confidence", "Random"
- [x] Sampling interval: `QDoubleSpinBox`, range 0.25-5.0, default 1.0, step 0.25, suffix " sec"
- [x] "Generate" button (disabled until at least 1 reference image with a detected face)
- [x] "Cancel" button

**Progress page layout:**
- [x] Progress bar + label: "Processing clip 47 of 200..."
- [x] Match count label: updates live as matches are found
- [x] "Cancel" button to abort extraction

**Match results behavior:**
- [x] After processing, show match count prominently: "Found 23 clips matching reference person"
- [x] If 0 matches: show "No clips matched" with option to adjust sensitivity or cancel. Switch sensitivity dropdown to enabled, add "Retry" button. Do NOT auto-close.
- [x] If > 0 matches: emit `sequence_ready` and `self.accept()`

**Inner worker** (`RoseHobartWorker`, QThread inside dialog):
- [x] Receives: reference image paths, clips list, sensitivity preset, ordering mode, sample interval
- [x] Step 1: Extract reference face embeddings (1-3 images). If multiple images, average embeddings.
- [x] Step 2: For each clip, check cache (`clip.face_embeddings`). If cached, compare directly. If not, extract faces and cache on clip object.
- [x] Step 3: Filter clips where `compare_faces` returns `is_match=True`
- [x] Step 4: Order matched clips per user selection:
  - "Original Order": sort by `(source.file_path.name, clip.start_frame)`
  - "By Duration": sort by `clip.duration_seconds(source.fps)`
  - "By Color": use `generate_sequence("color", ...)` on matched clips
  - "By Brightness": auto-compute brightness (call `_auto_compute_brightness`), sort
  - "By Confidence": sort by match confidence descending
  - "Random": `random.shuffle`
- [x] Step 5: Emit `finished_sequence(list)` with ordered `(Clip, Source)` tuples
- [x] Signals: `progress(str)`, `match_found(int)` (running count), `finished_sequence(list)`, `error(str)`
- [x] Support cancellation via `_cancelled` flag checked between clips

**Model download handling**: If InsightFace model is not downloaded, the worker's first call to `_load_insightface()` triggers download. Emit `progress("Downloading face recognition model...")` before the download starts. The download is handled by InsightFace's own model hub (not subprocess), so no subprocess cleanup needed.

#### Phase 4: Sequence Tab + Algorithm Config Integration

**Deliverable**: Rose Hobart wired into the sequence tab UI and algorithm grid.

**Tasks:**
- [x] Add `rose_hobart` to `ALGORITHM_CONFIG` in `ui/algorithm_config.py`:

```python
"rose_hobart": {
    "icon": "\U0001f464",  # bust_in_silhouette
    "label": "Rose Hobart",
    "description": "Isolate clips featuring a specific person",
    "allow_duplicates": False,
    "required_analysis": [],  # Dialog handles its own prerequisites
    "is_dialog": True,
},
```

- [x] Add to grid layout in `ui/widgets/sorting_card_grid.py` (row 3, next available column)
- [x] Import `RoseHobartDialog` in `ui/dialogs/__init__.py`
- [x] Import `RoseHobartDialog` in `ui/tabs/sequence_tab.py` (line 24)
- [x] Add `_show_rose_hobart_dialog(self, clips)` method in `sequence_tab.py`:

```python
def _show_rose_hobart_dialog(self, clips):
    clip_objects = [clip for clip, source in clips]
    sources_by_id = {source.id: source for clip, source in clips}
    dialog = RoseHobartDialog(clip_objects, sources_by_id, parent=self)
    dialog.sequence_ready.connect(self._apply_dialog_sequence)
    dialog.exec()
```

- [x] Add routing in `_on_card_clicked` (~line 506-518): `if algorithm == "rose_hobart": self._show_rose_hobart_dialog(clips); return`
- [x] Add routing in `_on_confirm_generate` (~line 406-418): `if algorithm == "rose_hobart": self._apply_chromatic_bar_to_sequence("rose_hobart"); self._show_rose_hobart_dialog(clips); return`

#### Phase 5: Chat Tools Integration

**Deliverable**: Agent can trigger Rose Hobart via chat.

**Tasks:**
- [x] Add `rose_hobart` entry to `list_sorting_algorithms` in `core/chat_tools.py` (~line 3672):

```python
{
    "key": "rose_hobart",
    "name": "Rose Hobart",
    "description": "Isolate clips featuring a specific person (requires reference image)",
    "available": has_face_embeddings,  # True if any clip has face_embeddings
    "reason": None if has_face_embeddings else "Run face detection analysis first",
    "parameters": [
        {"name": "reference_image_path", "type": "string", "description": "Path to reference image of the person"},
        {"name": "sensitivity", "type": "string", "options": ["strict", "balanced", "loose"], "default": "balanced"},
        {"name": "ordering", "type": "string", "options": ["original", "duration", "color", "brightness", "confidence", "random"], "default": "original"},
    ]
}
```

- [x] Create dedicated `generate_rose_hobart` tool (following `generate_reference_guided` pattern at line 3810):

```python
@tools.register(
    description="Generate a Rose Hobart sequence filtering clips by a specific person's face.",
    requires_project=True,
    modifies_gui_state=True
)
def generate_rose_hobart(
    project, main_window,
    reference_image_path: str,
    sensitivity: str = "balanced",
    ordering: str = "original",
) -> dict:
```

- [x] The tool extracts the reference face embedding from the image, iterates clips (using cached face_embeddings where available), filters, orders, and applies to timeline via `sequence_tab.generate_and_apply` or direct timeline manipulation.

#### Phase 6: Tests

- [x] `tests/test_rose_hobart.py` -- core logic tests (phases 1-2)
- [x] `tests/test_rose_hobart_dialog.py` -- dialog UI tests with QApplication fixture (offscreen)
  - Test dialog disables Generate button when no reference image selected
  - Test dialog enables Generate button when reference image with detected face is added
  - Test sensitivity dropdown has 3 options
  - Test ordering dropdown has 6 options

## System-Wide Impact

### Interaction Graph

```
RoseHobartDialog.sequence_ready
  -> SequenceTab._apply_dialog_sequence()
    -> timeline.clear_timeline()
    -> timeline.add_clip() per clip
    -> algorithm_dropdown.setCurrentText()
    -> chromatic_bar_controls updated
    -> STATE_TIMELINE set
```

Face detection worker (Analyze tab path):
```
MainWindow._launch_face_detection_worker()
  -> FaceDetectionWorker.run()
    -> core.analysis.faces.extract_faces_from_clip() per clip
    -> clip.face_embeddings mutated in-place
  -> Worker.completed
    -> MainWindow updates clip browser
    -> Project auto-saved
```

### Error Propagation

- InsightFace model download failure -> `error` signal from worker -> dialog shows error message
- Corrupt video file -> `extract_faces_from_clip` returns empty list (skip clip, log warning)
- No face in reference image -> dialog shows warning, Generate stays disabled
- No clips match -> dialog shows count, lets user adjust sensitivity

### State Lifecycle Risks

- **Partial face extraction on cancel**: Face embeddings are written to `clip.face_embeddings` in-place during processing. Cancellation preserves partial results. Next run uses cache for completed clips. No orphaned state.
- **Project file growth**: Face embeddings add ~4KB per face per clip in JSON. A project with 500 clips averaging 5 faces each = ~10MB additional. Acceptable for v1. Monitor.

### API Surface Parity

- `list_sorting_algorithms` chat tool: must include `rose_hobart`
- `generate_rose_hobart` chat tool: dedicated tool (not through `generate_remix` since it has unique parameters)
- GUI state tracking: `gui_state.py` should report face analysis availability

## Acceptance Criteria

### Functional Requirements

- [x] User can select 1-3 reference images via file picker dialog
- [x] Dialog shows face detection bounding box overlay on reference image thumbnails
- [x] Dialog shows warning when no face detected in a reference image
- [x] Three sensitivity presets (Strict/Balanced/Loose) produce different filter thresholds
- [x] Six ordering options work correctly: Original, Duration, Color, Brightness, Confidence, Random
- [x] Clips without detectable faces are excluded from results
- [x] Face embeddings are cached on Clip objects and persisted in project file
- [x] Subsequent runs use cached face embeddings (no re-extraction)
- [x] Progress bar shows per-clip progress during extraction
- [x] Match count is displayed after processing
- [x] Zero-match result lets user adjust sensitivity without reopening dialog
- [x] Cancel during extraction preserves partial face embedding results
- [x] "Detect Faces" available as standalone operation in Analyze tab
- [x] InsightFace model auto-downloads on first use with progress indicator
- [x] Old projects load cleanly with `face_embeddings = None`

### Non-Functional Requirements

- [x] Face extraction processes at ~2 seconds per clip (30 frames at 1fps)
- [x] Cached face comparison completes in < 100ms per clip
- [x] Project file size increase < 50MB for a 500-clip project
- [x] InsightFace model loads in < 5 seconds after download

### Quality Gates

- [x] Unit tests for face comparison, filtering, and Clip serialization
- [x] Dialog UI tests with offscreen QApplication
- [x] Manual testing on Apple Silicon (onnxruntime compatibility)

## Dependencies & Prerequisites

- **InsightFace** Python package (`pip install insightface`)
- **onnxruntime** (or `onnxruntime-silicon` on Apple Silicon) -- InsightFace's inference backend
- **OpenCV** (`cv2`) -- already in project for frame extraction
- No cloud API dependencies -- fully local

**Platform concern — Apple Silicon**: InsightFace's pip wheel fails to build on M2/M3 Macs. The fix is installing `onnxruntime-silicon` as a drop-in replacement for `onnxruntime` (`pip install onnxruntime-silicon`). This also enables `CoreMLExecutionProvider` for GPU acceleration. The `_load_insightface()` function should detect Apple Silicon and select the appropriate execution provider automatically.

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| InsightFace install fails on Apple Silicon | Users can't use feature | Test early; document workaround; consider fallback to dlib |
| Project file bloat from face embeddings | Slow save/load | Monitor size; future: binary storage or separate embedding file |
| False positives at Loose threshold | Wrong clips in sequence | Tunable thresholds; default to Balanced |
| Model download during creative flow | Disrupts workflow | Progress indicator; consider Settings pre-download button |
| Memory pressure with InsightFace + DINOv2 loaded | OOM on low-RAM machines | Add `unload_model()` call after extraction completes |

## Alternative Approaches Considered

| Approach | Why Rejected |
|----------|-------------|
| VLM-based matching (GPT-4o/Claude Vision) | Too slow ($6-18/run), non-deterministic, requires API key |
| CLIP/SigLIP person crop similarity | High false positive rate -- matches similar-looking people, not same person |
| YOLO person detection only (no face) | Detects "a person" but can't distinguish between different people |

(see brainstorm: `docs/brainstorms/2026-03-07-rose-hobart-sequencer-brainstorm.md`)

## Design Decisions Carried Forward from Brainstorm

| Decision | Rationale |
|----------|-----------|
| Strict filter (keep/discard) | True to Cornell concept; simpler than ranking |
| One person per run | Keeps dialog simple; run multiple times for different people |
| Face embeddings (InsightFace) | Best speed/cost/accuracy balance |
| Dialog-based + standalone analysis | Dialog for one-click workflow; Analyze tab for pre-computation |
| 1-second sampling interval | Good recall without excessive processing |
| Named sensitivity presets | More intuitive than raw threshold slider |
| Up to 3 reference images | Averaged embeddings improve robustness |
| No match preview | Generate directly; user can undo/re-run |
| Show count on 0 matches | Let user adjust sensitivity; no hard error |
| Auto-download model on first use | Matches YOLO/DINOv2 pattern |
| Persist face embeddings on Clip | Enables reuse across runs and Analyze tab integration |

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-07-rose-hobart-sequencer-brainstorm.md](docs/brainstorms/2026-03-07-rose-hobart-sequencer-brainstorm.md) -- Key decisions: face embeddings over VLM, dialog-based UX, 1-3 reference images, named sensitivity presets

### Internal References

- Dialog pattern: `ui/dialogs/signature_style_dialog.py` (closest analog)
- Analysis module pattern: `core/analysis/detection.py` (YOLO model loading)
- Frame sampling pattern: `core/analysis/color.py:200` (OpenCV VideoCapture)
- Worker pattern: `ui/workers/object_detection_worker.py`
- Algorithm config: `ui/algorithm_config.py`
- Clip model: `models/clip.py:189-421`
- Chat tools (dedicated tool pattern): `core/chat_tools.py:3810` (generate_reference_guided)

### Institutional Learnings

- **Circular import prevention**: Algorithm config must live only in `ui/algorithm_config.py` (`docs/solutions/logic-errors/circular-import-config-consolidation.md`)
- **Worker guard flags**: Must use boolean guard + `Qt.UniqueConnection` + `@Slot()` (`docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`)
- **Source ID reconciliation**: Workers returning model objects must reconcile IDs (`docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`)
- **Subprocess cleanup**: Verify InsightFace download mechanism; wrap any subprocess in try/finally (`docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`)
