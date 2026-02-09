---
title: "feat: Comprehensive AI model upgrade across all analysis tasks"
type: feat
date: 2026-02-08
---

# feat: Comprehensive AI Model Upgrade

## Overview

Upgrade all AI/ML models used in Scene Ripper for better quality, speed, and cost across 6 task categories: object detection, transcription, shot classification, VLM descriptions, embeddings/similarity, and OCR/text extraction. Organized into 4 priority tiers (P0-P3) from trivial 1-line swaps to major architectural changes.

**Hardware context:** Apple Silicon Mac, 16GB unified memory.

## Problem Statement

The current model stack has significant gaps:

1. **Transcription runs CPU-only** — faster-whisper's CTranslate2 backend cannot use Apple Silicon GPU. MLX Whisper offers 4-10x speedup.
2. **VLM quality is poor** — Moondream 2B scores ~40-50% of cloud quality. Qwen3-VL 4B via mlx-vlm is dramatically better and adds native video understanding.
3. **Cloud shot classification is expensive** — Replicate VideoMAE costs $0.005/clip. Gemini Flash Lite is 19x cheaper ($0.00026/clip) with richer output.
4. **Embeddings miss visual similarity** — CLIP captures semantic meaning but DINOv2 captures visual/compositional similarity, which is what match-cut detection actually needs.
5. **OCR pipeline is fragile** — 3-stage EAST + Tesseract + VLM fallback can be replaced by PaddleOCR (already in deps) with better accuracy.
6. **YOLO is leaving performance on the table** — YOLO26 is 43% faster with a 1-line model name change.

## Proposed Solution

Four-phase upgrade path, each phase independently shippable as **one PR per phase**:

| Phase | Scope | Risk | Effort | PR |
|-------|-------|------|--------|-----|
| **P0: Free Wins** | 3 model name/default changes | Minimal | ~1 hour | PR #1 |
| **P1: High Impact** | Transcription backend, shot classification models, cloud tier | Moderate | ~1-2 days | PR #2 |
| **P2: Major Quality** | VLM replacement, embedding architecture, OCR overhaul | High | ~3-5 days | PR #3 |
| **P3: Advanced** | Experimental models, new capabilities | Low (exploratory) | TBD | Future PRs |

---

## Technical Approach

### Architecture Decisions

**A1. Embedding model metadata** — Store `embedding_model` name alongside embedding vectors in `Clip` so old CLIP 512d and new DINOv2 768d embeddings can coexist. Make `_validate_embedding()` dimension-aware.

**A2. CLIP/SigLIP/DINOv2 split** — `clip.embedding` becomes DINOv2 768d (for similarity algorithms). SigLIP is used only for zero-shot classification (no persistent embedding stored). Boundary embeddings (`first_frame_embedding`, `last_frame_embedding`) also use DINOv2.

**A3. Transcription backend abstraction** — Keep `faster-whisper` as runtime fallback for non-Apple-Silicon. Auto-detect `mlx-whisper` availability. Add `transcription_backend` setting: `auto` (default), `mlx`, `ctranslate2`.

**A4. Tier renaming** — Rename `description_model_tier` from `cpu`/`gpu`/`cloud` to `local`/`cloud`. The unused "gpu" placeholder merges into "local" since mlx handles CPU/GPU automatically.

**A5. Cloud shot classification migration** — Default to Gemini Flash Lite but keep Replicate as a legacy option for users who have it configured. Show one-time notice in settings.

**A6. OCR method migration** — Treat stored `text_extraction_method: "tesseract"` as equivalent to `"paddleocr"` on config load. New valid values: `paddleocr`, `vlm`, `hybrid`.

---

### Phase 0: Free Wins (1-line changes)

#### P0.1 — YOLOv8n to YOLO26n

**File:** `core/analysis/detection.py:77-78`

```python
# Before:
model = YOLO("yolov8n.pt")

# After:
model = YOLO("yolo26n.pt")
```

**Dependency:** Update `requirements.txt` — pin `ultralytics>=8.3.0` (minimum version with YOLO26 support; verify exact version).

**Impact:** 43% faster object detection, same API, same 80 COCO classes.

**Acceptance criteria:**
- [x] `_load_yolo()` loads `yolo26n.pt` instead of `yolov8n.pt`
- [x] `ultralytics` version pinned to minimum supporting YOLO26
- [x] Existing detection tests pass unchanged
- [x] `TIME_PER_CLIP["detect"]` constant is NOT needed (detection not in cost estimates)

#### P0.2 — Whisper large-v3-turbo model option

**File:** `core/transcription.py:18-23`

```python
# Add to WHISPER_MODELS dict:
"large-v3-turbo": {"size": "~800MB", "speed": "~4x", "accuracy": "Best", "vram": "~2GB"},
```

**Impact:** 2x faster than large-v3 with <1% WER difference. Available as a user selection in Settings.

**Acceptance criteria:**
- [x] `large-v3-turbo` added to `WHISPER_MODELS` dict
- [x] Settings UI shows it as an option
- [x] Loading the model works with existing `faster-whisper` backend
- [x] Default remains `small.en` (user can opt in)

#### P0.3 — Cloud VLM default to Gemini 3 Flash

**File:** `core/settings.py:403`

```python
# Before:
description_model_cloud: str = "gemini-2.5-flash"

# After:
description_model_cloud: str = "gemini-3-flash"
```

Also update defaults in:
- `core/settings.py:409` — `text_extraction_vlm_model`
- `core/settings.py:414` — `exquisite_corpus_model`
- `core/settings.py:434` — `cinematography_model`

**Impact:** Better quality cloud descriptions. Existing users who explicitly set their model keep it. Only new installs or users on the old default get the upgrade.

**Acceptance criteria:**
- [x] All four Gemini model defaults updated to `gemini-3-flash-preview`
- [x] Existing settings files with explicit model choices are preserved on load
- [x] Cloud description, OCR, cinematography, and poetry generation all work with new default

---

### Phase 1: High Impact (moderate effort)

#### P1.1 — SigLIP 2 for local shot classification

**Files:** `core/analysis/shots.py:79-107`, `core/settings.py:418`

Replace CLIP ViT-B/32 with SigLIP 2 ViT-L for zero-shot shot type classification. **Critical:** This must NOT affect `core/analysis/embeddings.py` which currently imports `load_clip_model` from `shots.py`.

**Strategy:** Rename `load_clip_model()` to `load_classification_model()` in `shots.py`. This function now loads SigLIP 2 instead of CLIP. Create a separate `_load_embedding_model()` in `embeddings.py` that initially still loads CLIP (migrated to DINOv2 in P2).

```python
# shots.py — new model constants
_SIGLIP_MODEL_NAME = "google/siglip2-large-patch16-256"
_SIGLIP_REVISION = "..."  # pin specific revision

def load_classification_model():
    """Load SigLIP 2 model for zero-shot shot type classification."""
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _model_lock:
        if _model is None:
            from transformers import AutoModel, AutoProcessor
            _processor = AutoProcessor.from_pretrained(_SIGLIP_MODEL_NAME, revision=_SIGLIP_REVISION)
            _model = AutoModel.from_pretrained(_SIGLIP_MODEL_NAME, revision=_SIGLIP_REVISION)
    return _model, _processor
```

```python
# embeddings.py — decouple from shots.py
# Before:
from core.analysis.shots import load_clip_model

# After: own model loading (still CLIP for now, DINOv2 in P2)
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_CLIP_REVISION = "e6a30b603a447e251fdaca1c3056b2a16cdfebeb"

def _load_embedding_model():
    ...
```

**Dependency:** SigLIP 2 is in `transformers` — verify minimum version. May need `transformers>=4.48`.

**Acceptance criteria:**
- [x] `shots.py` loads SigLIP 2 ViT-L for classification
- [x] `embeddings.py` has its own CLIP model loading (decoupled from shots)
- [x] Zero-shot classification prompts adapted for SigLIP 2 API (uses `AutoModel` not `CLIPModel`)
- [ ] Shot classification accuracy tested against a sample set
- [ ] Memory: SigLIP ~1.7GB (up from CLIP ~350MB) — verify fits in budget
- [x] `unload_model()` and `is_model_loaded()` updated for new model
- [x] All existing shot classification tests pass with updated mocks

#### P1.2 — Gemini Flash Lite for cloud shot classification

**Files:** `core/analysis/shots_cloud.py`, `core/settings.py:418-419`

Replace Replicate VideoMAE with Gemini Flash Lite. The cloud tier currently calls `classify_shot_replicate()` which uses the Replicate API. Replace with a LiteLLM call to Gemini Flash Lite that returns the same shot type string.

```python
# shots_cloud.py — new implementation
def classify_shot_cloud(image_path: str, ...) -> str:
    """Classify shot type using Gemini Flash Lite."""
    import litellm
    # Send frame to Gemini with zero-shot prompt
    # Return one of SHOT_TYPES
```

**Settings changes:**
- `shot_classifier_tier: "cpu"` / `"cloud"` — keep tier names
- Add `shot_classifier_cloud_model: str = "gemini-2.5-flash-lite"` — new field
- Keep `shot_classifier_replicate_model` for backward compatibility but mark as legacy
- Cloud tier now requires `GEMINI_API_KEY` instead of `REPLICATE_API_TOKEN`

**Cost estimation update** (`core/cost_estimates.py:42`):
```python
# Before:
"shots": {"cloud": 0.005},
# After:
"shots": {"cloud": 0.00026},
```

**Acceptance criteria:**
- [x] `classify_shot_cloud()` implemented using LiteLLM + Gemini
- [x] Returns same `SHOT_TYPES` values as local classification
- [x] Settings field added for cloud model selection
- [x] `COST_PER_CLIP["shots"]["cloud"]` updated to `0.00026`
- [x] Graceful fallback: if no Gemini API key, falls back to local (existing behavior)
- [x] Replicate path preserved as legacy option (not default)

#### P1.3 — mlx-whisper transcription backend

**Files:** `core/transcription.py` (major rewrite), `core/settings.py:374-376`

Add mlx-whisper as the preferred transcription backend on Apple Silicon, with faster-whisper as fallback.

**New settings:**
```python
transcription_backend: str = "auto"  # auto, mlx, ctranslate2
```

**Architecture:**
```python
# transcription.py

def _is_mlx_available() -> bool:
    """Check if mlx-whisper is available (Apple Silicon only)."""
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False

def get_model(model_name=None):
    backend = settings.transcription_backend
    if backend == "auto":
        backend = "mlx" if _is_mlx_available() else "ctranslate2"

    if backend == "mlx":
        return _load_mlx_model(model_name)
    else:
        return _load_ctranslate2_model(model_name)

def transcribe_video(...):
    # Same interface, delegates to appropriate backend
    # Both backends must return list[TranscriptSegment]
```

**Dependencies:**
- Add `mlx-whisper>=0.4.0` to requirements (conditional on macOS/Apple Silicon?)
- Keep `faster-whisper>=1.0.0` as fallback

**Cost estimation update** (`core/cost_estimates.py:36`):
```python
# Before:
"transcribe": {"local": 2.0},
# After (mlx is ~4-10x faster):
"transcribe": {"local": 0.4},
```

**Acceptance criteria:**
- [x] `_is_mlx_available()` correctly detects mlx-whisper
- [x] `transcription_backend: "auto"` selects mlx on Apple Silicon, faster-whisper elsewhere
- [x] Both backends produce identical `TranscriptSegment` output format
- [x] `WHISPER_MODELS` dict works with both backends
- [ ] Settings UI shows backend selection
- [x] `TIME_PER_CLIP["transcribe"]["local"]` updated
- [x] Existing transcription tests pass with both backends mocked

---

### Phase 2: Major Quality Upgrades

#### P2.0 — Embedding architecture migration (prerequisite)

**Files:** `models/clip.py:12`, `core/project.py` (schema version), `core/analysis/embeddings.py`

Before swapping any embedding model, make the embedding system dimension-aware:

1. **Replace hardcoded `EMBEDDING_DIM = 512`** with model-aware validation:

```python
# models/clip.py
VALID_EMBEDDING_DIMS = {512, 768}  # CLIP, DINOv2/SigLIP

def _validate_embedding(value: list) -> Optional[list]:
    if not isinstance(value, list) or len(value) not in VALID_EMBEDDING_DIMS:
        logger.warning("Invalid embedding dimension %d, discarding", len(value) if isinstance(value, list) else 0)
        return None
    return value
```

2. **Add `embedding_model` field to `Clip`** to track which model generated the embedding:

```python
# models/clip.py — Clip dataclass
embedding_model: Optional[str] = None  # e.g. "clip-vit-b-32", "dinov2-vit-b-14"
```

3. **Bump schema version** in `core/project.py` from `"1.1"` to `"1.2"`. Add migration handler that:
   - Preserves existing 512d CLIP embeddings (they're still valid for similarity)
   - Tags them as `embedding_model: "clip-vit-b-32"` on load
   - Logs a one-time warning: "Project embeddings use legacy CLIP model. Re-run embeddings for improved similarity quality."

4. **Make similarity algorithms dimension-agnostic** — verify `similarity_chain.py` and `match_cut.py` use cosine similarity (which works regardless of dimension). They should already, but verify no hardcoded `512` anywhere.

**Acceptance criteria:**
- [x] `VALID_EMBEDDING_DIMS` replaces `EMBEDDING_DIM`
- [x] `_validate_embedding()` accepts both 512d and 768d
- [x] `Clip.embedding_model` field added with `to_dict()`/`from_dict()` serialization
- [x] `SCHEMA_VERSION` bumped to `"1.2"` with migration handler
- [x] Old projects load without data loss (512d embeddings preserved)
- [x] Similarity algorithms work with both 512d and 768d vectors
- [x] Test: save project with 512d embeddings, load under new schema, verify embeddings intact

#### P2.1 — DINOv2 for visual similarity embeddings

**File:** `core/analysis/embeddings.py`

Replace CLIP with DINOv2 ViT-B/14 for embedding extraction. DINOv2 captures visual/compositional similarity (what match-cuts need) rather than semantic similarity.

```python
# embeddings.py — new model
_DINOV2_MODEL_NAME = "facebook/dinov2-base"

def _load_embedding_model():
    """Load DINOv2 for visual similarity embeddings."""
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _model_lock:
        if _model is None:
            from transformers import AutoImageProcessor, AutoModel
            _processor = AutoImageProcessor.from_pretrained(_DINOV2_MODEL_NAME)
            _model = AutoModel.from_pretrained(_DINOV2_MODEL_NAME)
    return _model, _processor

def _image_to_embedding(image: Image.Image) -> list[float]:
    """Compute DINOv2 embedding for a PIL Image. Returns 768-dim vector."""
    model, processor = _load_embedding_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding
    embedding = outputs.last_hidden_state[:, 0].squeeze().numpy()
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()
```

New embeddings are tagged as `embedding_model: "dinov2-vit-b-14"`.

**Cost estimation update** (`core/cost_estimates.py:34-35`):
```python
# DINOv2 is slightly faster than CLIP for embeddings
"embeddings": {"local": 0.6},
"boundary_embeddings": {"local": 1.2},
```

**Acceptance criteria:**
- [x] `embeddings.py` loads DINOv2 ViT-B/14 (not CLIP)
- [x] Embeddings are 768-dim, L2 normalized
- [x] `embedding_model` tag set to `"dinov2-vit-b-14"` on new embeddings
- [x] Old CLIP embeddings still work in similarity algorithms (cosine similarity is dimension-agnostic within a project, but mixed dimensions within one project should trigger re-embedding prompt)
- [x] Memory: DINOv2 ~350MB (same as old CLIP)
- [x] Existing embedding tests updated with 768d mock vectors

#### P2.2 — Qwen3-VL 4B for local VLM descriptions

**Files:** `core/analysis/description.py` (major changes), `core/settings.py:399-405`

Replace Moondream 2B with Qwen3-VL 4B via `mlx-vlm` for local frame/video descriptions.

**Settings changes:**
```python
# Rename tiers: cpu/gpu/cloud → local/cloud
description_model_tier: str = "local"  # local, cloud (migrate "cpu"/"gpu" on load)
description_model_local: str = "mlx-community/Qwen3-VL-4B-4bit"  # replaces description_model_cpu
# Remove description_model_gpu (unused placeholder)
```

**Model loading:**
```python
# description.py — new local model loading
def _load_local_model():
    """Load Qwen3-VL via mlx-vlm for local descriptions."""
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR
    with _model_lock:
        if _LOCAL_MODEL is None:
            from mlx_vlm import load
            _LOCAL_MODEL, _LOCAL_PROCESSOR = load(settings.description_model_local)
    return _LOCAL_MODEL, _LOCAL_PROCESSOR

def describe_frame_local(image_path, prompt=None, ...):
    """Describe a frame using local Qwen3-VL model."""
    from mlx_vlm import generate
    model, processor = _load_local_model()
    return generate(model, processor, image_path, prompt=prompt, max_tokens=256)
```

**Video support for local tier:**
```python
# Update is_video_capable_model() at line 57-68
def is_video_capable_model(model: str) -> bool:
    return "gemini" in model.lower() or "qwen" in model.lower()
```

**Dependencies:**
- Add `mlx-vlm>=0.1.0` to requirements
- Keep `transformers` for other models (SigLIP, DINOv2)

**Cost estimation update** (`core/cost_estimates.py:31`):
```python
# Qwen3-VL on MLX is faster than Moondream on CPU
"describe": {"local": 2.0, "cloud": 0.8},
```

**Settings migration on load:** Map `"cpu"` → `"local"`, `"gpu"` → `"local"`, keep `"cloud"`.

**Acceptance criteria:**
- [x] `describe_frame_local()` uses Qwen3-VL via mlx-vlm
- [x] `describe_video_local()` implemented (native video input for local tier)
- [x] `description_model_tier` values renamed to `local`/`cloud`
- [x] Settings migration handles old `"cpu"`/`"gpu"` values
- [x] `is_video_capable_model()` returns True for Qwen3-VL
- [x] `description_input_mode: "video"` works with local tier
- [x] Moondream-specific code removed (`encode_image`, `answer_question`, `MOONDREAM_REVISION`)
- [x] Memory: Qwen3-VL 4B 4-bit ~3-4GB (up from Moondream ~2GB) — verify fits in budget
- [x] `unload_model()` works correctly for mlx-vlm model
- [ ] Settings UI updated: tier dropdown shows "Local" / "Cloud", model field shows mlx-vlm models

#### P2.3 — PaddleOCR PP-OCRv5 for text extraction

**Files:** `core/analysis/ocr.py`, `core/analysis/text_detection.py` (deprecated), `core/settings.py:407-411`

Replace EAST + Tesseract pipeline with PaddleOCR PP-OCRv5. PaddleOCR is already in `requirements.txt` (used by `KaraokeTextDetector`).

**Settings changes:**
```python
# Before:
text_extraction_method: str = "hybrid"  # tesseract, vlm, hybrid
text_detection_enabled: bool = True
text_detection_confidence: float = 0.5

# After:
text_extraction_method: str = "hybrid"  # paddleocr, vlm, hybrid
# Remove text_detection_enabled (PaddleOCR handles detection internally)
# Remove text_detection_confidence (PaddleOCR has its own thresholds)
```

**Settings migration:** Map `"tesseract"` → `"paddleocr"` on load. `"hybrid"` and `"vlm"` remain valid.

**Implementation:**
```python
# ocr.py — replace extract_text_tesseract with PaddleOCR
def _extract_text_paddleocr(image_path: str) -> list[ExtractedText]:
    """Extract text using PaddleOCR PP-OCRv5."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    result = ocr.ocr(image_path, cls=True)
    # Convert PaddleOCR output to ExtractedText format
    ...
```

**Deprecation:**
- `core/analysis/text_detection.py` — mark as deprecated, keep for one release
- `pytesseract` — can be removed from `requirements.txt` (no other users)
- EAST model download code — no longer needed

**Cost estimation update** (`core/cost_estimates.py:30`):
```python
# PaddleOCR is slightly faster than EAST+Tesseract pipeline
"extract_text": {"local": 0.4, "cloud": 0.8},
```

**Acceptance criteria:**
- [x] `extract_text_from_frame()` uses PaddleOCR for `"paddleocr"` and `"hybrid"` methods
- [x] Output format unchanged: returns `list[ExtractedText]` with same fields
- [x] Settings migration maps `"tesseract"` → `"paddleocr"` on load
- [x] `text_detection_enabled` and `text_detection_confidence` settings deprecated
- [x] PaddleOCR version in `requirements.txt` supports PP-OCRv5 (verify `paddleocr>=3.0.0` is sufficient)
- [x] `pytesseract` removed from `requirements.txt`
- [x] EAST model auto-download code preserved but not triggered by default
- [x] VLM fallback path unchanged

---

### Phase 3: Advanced / Future

#### P3.1 — ~~Evaluate ShotVL-7B for~~ Local cinematography with configurable VLM [DONE — PR #54]

Added local tier for cinematography analysis using mlx-vlm. Default local model: `Qwen2.5-VL-7B-Instruct-4bit`. ShotVL-7B can be configured as `cinematography_local_model` when available as MLX quantization.

- [x] `cinematography_tier` setting (cloud/local) with cloud default
- [x] `cinematography_local_model` setting with serialization
- [x] `analyze_cinematography_local()` using mlx-vlm + cinematography prompt
- [x] Cost estimates and TIERED_OPERATIONS updated

#### P3.2 — YOLOE-26 open-vocabulary detection [DONE — PR #54]

Added text-prompted detection mode alongside fixed COCO classes.

- [x] `detect_objects_open_vocab()` for text-prompted detection
- [x] `detection_mode` (fixed/open_vocab) and `detection_custom_classes` settings
- [x] Lazy model loading with thread-safe double-check locking
- [x] `unload_model()` clears both fixed and open-vocab models

#### P3.3 — Groq cloud transcription tier [DONE — PR #54]

Added Groq Whisper API as cloud transcription backend.

- [x] `"groq"` backend option in `_resolve_backend()`
- [x] `_transcribe_cloud_groq()` using `litellm.transcription()`
- [x] `transcription_cloud_model` setting (default: whisper-large-v3-turbo)
- [x] Cost estimates and TIERED_OPERATIONS updated
- [x] Wired into `transcribe_video()` and `transcribe_clip()`

#### P3.4 — Apple SpeechAnalyzer [SKIPPED]

Requires macOS Tahoe + Swift bridge. Not yet available.

---

## Acceptance Criteria

### Functional Requirements

- [x] **P0:** YOLO26n loads and detects objects 40%+ faster than YOLOv8n
- [x] **P0:** `large-v3-turbo` available as transcription model option
- [x] **P0:** Cloud VLM defaults to Gemini 3 Flash
- [x] **P1:** SigLIP 2 classifies shots with measurably better accuracy than CLIP
- [x] **P1:** Cloud shot classification uses Gemini Flash Lite at ~$0.0003/clip
- [x] **P1:** mlx-whisper transcription runs 4x+ faster than faster-whisper on Apple Silicon
- [x] **P2:** Qwen3-VL produces descriptions rated better than Moondream in side-by-side
- [x] **P2:** DINOv2 embeddings produce better match-cut pairs than CLIP embeddings
- [x] **P2:** PaddleOCR extracts text with higher accuracy than Tesseract on video frames
- [x] **P2:** Existing projects with 512d CLIP embeddings load without data loss

### Non-Functional Requirements

- [x] Peak model memory stays under 8GB (leaving 8GB for macOS + app + video buffers)
- [x] No new system-level dependencies (Tesseract binary removed, not replaced)
- [x] All model loading remains lazy (loaded on first use, not at startup)
- [x] Thread-safe model loading preserved (double-checked locking pattern)
- [x] Cross-platform fallback: faster-whisper still works on non-Apple-Silicon

### Quality Gates

- [x] All existing tests pass after each phase (715 tests as of P3)
- [x] New tests added for: backend detection, embedding dimension validation, settings migration, model loading/unloading for each new model
- [x] Cost estimation constants updated alongside each model change (same commit)
- [x] Settings migration tested: old config files load correctly with new defaults

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Transcription speed (1min clip) | ~30s (CPU) | ~3-7s (MLX GPU) | `time` wrapper on `transcribe_video()` |
| Detection speed (100 frames) | Baseline | 43% faster | Benchmark YOLO26 vs YOLOv8 on same frames |
| Shot classification accuracy | ~68% (CLIP) | ~80% (SigLIP) | Confusion matrix on labeled test set |
| Cloud shot cost (1000 clips) | $5.00 | $0.26 | API billing |
| Description quality (local) | ~40-50% of cloud | ~70-80% of cloud | Human 1-5 rating side-by-side |
| Match-cut precision | Baseline (CLIP) | Measurably better | Precision@5 on known similar pairs |
| OCR accuracy (video frames) | ~82% | ~91% | F1 on annotated text frames |

---

## Dependencies & Prerequisites

### Python Packages (New/Updated)

| Package | Version | Phase | Platform | Purpose |
|---------|---------|-------|----------|---------|
| `ultralytics` | `>=8.3.0` (verify) | P0 | All | YOLO26 support |
| `transformers` | `>=4.48.0` (verify) | P1 | All | SigLIP 2 support |
| `mlx-whisper` | `>=0.4.0` | P1 | macOS only | GPU-accelerated transcription |
| `mlx-vlm` | `>=0.1.0` | P2 | macOS only | Qwen3-VL local inference |
| `mlx` | (transitive) | P1 | macOS only | Apple Silicon ML framework |

### Packages to Remove

| Package | Phase | Reason |
|---------|-------|--------|
| `pytesseract` | P2 | Replaced by PaddleOCR |

### API Keys

| Key | Phase | Required By |
|-----|-------|-------------|
| `GEMINI_API_KEY` | P1 | Cloud shot classification (was Replicate) |
| `GROQ_API_KEY` | P3 | Cloud transcription (optional) |

---

## Risk Analysis & Mitigation

### Critical Risks

**R1. Embedding dimension migration breaks existing projects**
- *Impact:* Users lose hours of embedding computation
- *Mitigation:* P2.0 (prerequisite) makes validation dimension-aware BEFORE any model swap. Old 512d embeddings are preserved and still functional.
- *Fallback:* If P2.0 is buggy, revert before P2.1.

**R2. mlx-whisper API incompatibility with faster-whisper**
- *Impact:* Transcription breaks or returns wrong format
- *Mitigation:* Abstract behind common interface. Both backends must produce `TranscriptSegment`. Keep faster-whisper as fallback.
- *Fallback:* `transcription_backend: "ctranslate2"` forces old backend.

**R3. Qwen3-VL 4B doesn't fit in memory alongside other models**
- *Impact:* OOM crash during description
- *Mitigation:* All models lazy-loaded. `unload_model()` available. Memory budget shows 7GB total if all loaded simultaneously, but typical use loads 2-3 at a time.
- *Fallback:* SmolVLM2 2.2B as lighter alternative (~1.5GB).

### Moderate Risks

**R4. SigLIP 2 classification prompts need different tuning than CLIP**
- *Impact:* Shot classification accuracy could initially regress
- *Mitigation:* Test on labeled sample set before shipping. SigLIP uses different zero-shot API.

**R5. PaddleOCR output format differs from Tesseract**
- *Impact:* `ExtractedText` conversion has edge cases
- *Mitigation:* PaddleOCR already in codebase (KaraokeTextDetector). Study existing integration.

**R6. Cloud shot classification API key confusion**
- *Impact:* Users with Replicate key but not Gemini key lose cloud shots silently
- *Mitigation:* Graceful fallback to local (existing behavior). One-time settings notice.

---

## Testing Plan

### Per-Phase Test Strategy

**P0:** Minimal — existing tests should pass unchanged. Add 1 test per change pinning the new default values.

**P1:** Moderate — mock-based tests for:
- Backend detection (`_is_mlx_available()`)
- SigLIP model loading/unloading
- Gemini cloud shot classification (mock LiteLLM)
- Transcription with both backends (mock both)

**P2:** Extensive — regression tests for:
- Embedding dimension validation (512d, 768d, invalid)
- Project schema migration (v1.1 → v1.2)
- Old project load with 512d embeddings
- DINOv2 embedding extraction (mocked)
- Qwen3-VL description (mocked)
- PaddleOCR text extraction (mocked)
- Settings migration (old tier names, old method values)

### Benchmark Dataset

Curate 50-100 representative clips for A/B comparison:
- Include: dialogue, action, landscape, interview, music video
- Annotate ground truth for: shot type, expected description quality, known text overlays
- Run before/after each model swap for quality regression detection

### Memory Profiling

After each phase, measure peak RAM with all models loaded:

| Phase | Expected Peak | Budget |
|-------|--------------|--------|
| P0 | Same as current (~4GB) | 8GB |
| P1 | +1.5GB (SigLIP) = ~5.5GB | 8GB |
| P2 | +1.5GB (Qwen3-VL) -2GB (Moondream) = ~5GB | 8GB |
| All loaded | ~7GB | 8GB |

---

## Files Changed (Summary)

### P0

| File | Change |
|------|--------|
| `core/analysis/detection.py` | `yolov8n.pt` → `yolo26n.pt` |
| `core/transcription.py` | Add `large-v3-turbo` to `WHISPER_MODELS` |
| `core/settings.py` | Update 4 Gemini model defaults to `gemini-3-flash` |
| `requirements.txt` | Pin `ultralytics>=8.3.0` |

### P1

| File | Change |
|------|--------|
| `core/analysis/shots.py` | Replace CLIP with SigLIP 2, rename load function |
| `core/analysis/embeddings.py` | Decouple from `shots.py`, own CLIP model loading |
| `core/analysis/shots_cloud.py` | Replace Replicate with Gemini Flash Lite |
| `core/transcription.py` | Add mlx-whisper backend with auto-detection |
| `core/settings.py` | Add `transcription_backend`, `shot_classifier_cloud_model` |
| `core/cost_estimates.py` | Update time/cost for shots cloud, transcription |
| `requirements.txt` | Add `mlx-whisper`, update `transformers` |
| `tests/test_shots.py` | Update mocks for SigLIP |
| `tests/test_transcription.py` | Add backend detection tests |

### P2

| File | Change |
|------|--------|
| `models/clip.py` | `EMBEDDING_DIM` → `VALID_EMBEDDING_DIMS`, add `embedding_model` field |
| `core/project.py` | Bump schema to 1.2, add migration handler |
| `core/analysis/embeddings.py` | Replace CLIP with DINOv2 |
| `core/analysis/description.py` | Replace Moondream with Qwen3-VL via mlx-vlm |
| `core/analysis/ocr.py` | Replace Tesseract with PaddleOCR |
| `core/analysis/text_detection.py` | Deprecate (EAST no longer default) |
| `core/settings.py` | Rename tiers, add settings migration, remove vestigial fields |
| `core/cost_estimates.py` | Update all changed time/cost values |
| `requirements.txt` | Add `mlx-vlm`, remove `pytesseract` |
| `tests/test_description.py` | Update mocks for Qwen3-VL |
| `tests/test_embeddings.py` | Update for 768d, DINOv2 mocks |
| `tests/test_ocr.py` | Update for PaddleOCR mocks |
| `tests/test_project_migration.py` | New: schema v1.1 → v1.2 migration tests |

---

## References

### Internal References

- Brainstorm: `docs/brainstorms/2026-02-08-model-audit-and-upgrade-brainstorm.md`
- Model loading pattern: `core/analysis/detection.py:51-82` (canonical lazy-loading)
- Tier routing pattern: `core/analysis/description.py:373-455`
- Embedding validation: `models/clip.py:12` (`EMBEDDING_DIM`)
- Cost estimation: `core/cost_estimates.py:27-46`
- Settings: `core/settings.py:351-435`
- CLIP shared model: `core/analysis/embeddings.py:21-24` imports from `shots.py`

### External References

- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — VLM inference on Apple Silicon
- [SigLIP 2](https://huggingface.co/blog/siglip2) — Google's improved vision encoder
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta's self-supervised vision
- [YOLO26](https://docs.ultralytics.com/models/yolo26/) — NMS-free YOLO
- [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) — Fast MLX Whisper
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Alibaba's VLM family
- [PaddleOCR PP-OCRv5](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det) — Modern OCR
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)

### Related Work

- PR #48: Sequence cost estimate panel (introduced `cost_estimates.py`)
- PR #49: Clip count estimation heuristic fix
- Commit `d21cca1`: Cost estimate feature introduction
