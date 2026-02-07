---
title: "feat: Add Video Description using Vision-Language Models"
type: feat
date: 2026-01-26
---

# feat: Add Video Description using Vision-Language Models

## Overview

Add a video frame description/captioning feature using vision-language models (VLMs). This generates natural language descriptions of video frames for searchability and AI editing assistance. The feature supports multiple model tiers (CPU-fast, GPU-quality, Cloud API) with configurable settings for speed, quality, and hardware requirements.

## Problem Statement / Motivation

Currently, Scene Ripper can:
- Classify frames with ImageNet labels (1000 object categories)
- Detect objects with YOLO (80 COCO classes with bounding boxes)
- Extract dominant colors

However, these provide structured data, not natural language understanding. Users need:
- **Searchable descriptions**: "Find clips with a person walking on a beach"
- **AI editing context**: Agent needs to understand *what's happening* in scenes, not just what objects exist
- **Rich metadata**: Narrative descriptions complement structured labels

## Proposed Solution

Add a multi-tier vision-language model system for generating frame descriptions:

| Tier | Models | Use Case | Requirements |
|------|--------|----------|--------------|
| **CPU-fast** | Moondream 2B | Quick local analysis, low-end hardware | ~4GB RAM, CPU-only |
| **GPU-quality** | LLaVA-OneVision 7B, Qwen2.5-VL 7B | High-quality local analysis | ~8-16GB VRAM |
| **Cloud API** | GPT-4o, Claude Vision, Gemini | Best quality, no local resources | API key + internet |

### Analysis Modes

1. **Frame-based** (default): Analyze single representative frame (thumbnail)
2. **Temporal**: Analyze 4+ frames across clip duration for motion/action understanding

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Settings                                  │
│  description_model_tier: "cpu" | "gpu" | "cloud"                │
│  description_model_cpu: "moondream-2b"                          │
│  description_model_gpu: "llava-onevision-7b"                    │
│  description_model_cloud: "gpt-4o"                              │
│  description_temporal_frames: 4                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                core/analysis/description.py                      │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ CPU Tier         │  │ GPU Tier         │  │ Cloud Tier     │ │
│  │ - Moondream      │  │ - LLaVA-OneVision│  │ - LiteLLM      │ │
│  │ - transformers   │  │ - transformers   │  │ - OpenAI/etc   │ │
│  │ - 4-bit quant    │  │ - 4-bit quant    │  │ - async batch  │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│            describe_frame(image_path, tier, temporal) -> str     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ CLI      │    │ GUI      │    │ Agent    │
       │ analyze  │    │ Worker   │    │ Tool     │
       │ describe │    │          │    │          │
       └──────────┘    └──────────┘    └──────────┘
```

### Data Model Changes

Extend `models/clip.py`:

```python
@dataclass
class Clip:
    # ... existing fields ...

    # Video description fields
    description: Optional[str] = None  # Natural language description
    description_model: Optional[str] = None  # Model that generated it (e.g., "moondream-2b", "gpt-4o")
    description_frames: Optional[int] = None  # 1 for single frame, N for temporal
```

### New Files

| File | Purpose |
|------|---------|
| `core/analysis/description.py` | VLM loading and inference |
| `tests/test_description.py` | Unit tests |

### Modified Files

| File | Changes |
|------|---------|
| `core/settings.py` | Add description model settings |
| `models/clip.py` | Add description fields |
| `core/chat_tools.py` | Add `describe_content_live` tool, extend `filter_clips` |
| `ui/main_window.py` | Add `DescriptionWorker` |
| `cli/commands/analyze.py` | Add `describe` command |

## Implementation Phases

### Phase 1: Foundation (Core Module + Settings)

**Tasks:**
- [x] Add description settings to `core/settings.py`
  - `description_model_tier: str = "cpu"` (cpu, gpu, cloud)
  - `description_model_cpu: str = "moondream-2b"`
  - `description_model_gpu: str = "llava-onevision-7b-q4"`
  - `description_model_cloud: str = "gpt-4o"`
  - `description_temporal_frames: int = 4`
- [x] Create `core/analysis/description.py` with:
  - Lazy model loading (thread-safe singleton pattern)
  - `describe_frame(image_path, tier, temporal_frames) -> str`
  - Support for CPU tier (Moondream via transformers)
  - Support for Cloud tier (LiteLLM with existing API keys)
- [x] Extend `models/clip.py` with description fields
- [x] Update `to_dict()` / `from_dict()` for persistence

**Files:**
- `core/settings.py`
- `core/analysis/description.py`
- `models/clip.py`

### Phase 2: CLI Integration

**Tasks:**
- [x] Add `scene_ripper analyze describe` CLI command
  - `--tier cpu|gpu|cloud` (override settings)
  - `--temporal` flag for multi-frame analysis
  - `--force` to re-describe
  - `--clip/-c` for specific clips
  - `--prompt` for custom description prompt
- [x] JSON output support
- [x] Progress display with ProgressContext

**Files:**
- `cli/commands/analyze.py`

### Phase 3: Agent Tools + Search

**Tasks:**
- [x] Add `describe_content_live` agent tool
  - Parameters: `clip_ids`, `tier`, `temporal`
  - Returns `_wait_for_worker: "description"`
- [x] Extend `filter_clips` with `search_description` parameter
  - Case-insensitive substring matching
- [x] Extend `list_clips` to include description in output
- [x] Add timeout to `TOOL_TIMEOUTS` (600s for local, 300s for cloud)

**Files:**
- `core/chat_tools.py`

### Phase 4: GUI Worker

**Tasks:**
- [x] Add `DescriptionWorker(QThread)` to `ui/main_window.py`
  - Signals: `progress(int, int)`, `description_ready(str, str)`, `finished()`
  - Support cancellation
- [x] Connect worker to tool executor dispatch
- [x] Add signal handlers for updating clips

**Files:**
- `ui/main_window.py`

### Phase 5: GPU Tier (Optional Enhancement)

**Tasks:**
- [ ] Add GPU model support (LLaVA-OneVision via transformers)
- [ ] Automatic GPU detection
- [ ] 4-bit quantization support
- [ ] Memory management (unload other models when loading GPU VLM)

**Files:**
- `core/analysis/description.py`

### Phase 6: Testing

**Tasks:**
- [x] Unit tests for `description.py` (mocked models)
- [x] Tests for `filter_clips` with `search_description`
- [x] Tests for Clip serialization with description fields
- [ ] Integration tests for CLI command (optional - CLI uses same code paths as unit tests)

**Files:**
- `tests/test_description.py`

## Acceptance Criteria

### Functional Requirements

- [ ] Can generate descriptions for clips via CLI, GUI, and agent
- [ ] Descriptions are stored in Clip model and persisted to project file
- [ ] Can search clips by description content using `filter_clips`
- [ ] Settings allow choosing between CPU, GPU, and Cloud tiers
- [ ] CPU tier works without GPU or internet (after initial model download)
- [ ] Cloud tier works with existing LLM API keys (OpenAI, Anthropic, Google)
- [ ] Temporal mode analyzes multiple frames for action understanding

### Non-Functional Requirements

- [ ] CPU tier inference < 10 seconds per frame on modern CPU
- [ ] Cloud tier respects rate limits with exponential backoff
- [ ] Model loading is lazy (only when first needed)
- [ ] Thread-safe model access for GUI worker

### Quality Gates

- [ ] All existing tests pass
- [ ] New unit tests cover core functionality
- [ ] No regressions in existing analysis features

## Dependencies & Prerequisites

### Python Dependencies (add to requirements.txt)

```
# Vision-Language Models (CPU tier)
# Note: transformers already in requirements for other models

# For Moondream specifically (if needed)
einops>=0.7.0

# For GPU tier (optional)
bitsandbytes>=0.45.0  # For 4-bit quantization
accelerate>=0.30.0    # For device_map="auto"
```

### External Dependencies

- **CPU tier**: Moondream model (~1.5GB download on first use)
- **GPU tier**: LLaVA-OneVision (~4GB download for Q4 quantized)
- **Cloud tier**: Existing API keys (OpenAI, Anthropic, or Google)

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Moondream model too slow on CPU | Medium | Medium | Benchmark during dev; consider ONNX optimization |
| Cloud API costs for batch operations | Medium | Low | Document costs; no automatic batch without user action |
| Model download fails mid-way | Low | Medium | Use huggingface_hub with resume support |
| Memory pressure with multiple models | Medium | Medium | Unload other analysis models when loading VLM |
| API key doesn't have vision access | Low | Low | Catch error and provide helpful message |

## Model Recommendations Summary

### CPU Tier: Moondream 2B

- **Size**: ~1.5GB (FP16), ~800MB (INT8)
- **Speed**: ~2-5 seconds/frame on CPU
- **Quality**: Good for basic descriptions
- **Installation**: `transformers` library, auto-downloads from HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
```

### GPU Tier: LLaVA-OneVision 7B (Q4)

- **Size**: ~4GB (4-bit quantized)
- **Speed**: ~1-2 seconds/frame on RTX 3080
- **Quality**: Excellent detailed descriptions
- **Installation**: `transformers` + `bitsandbytes`

```python
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_4bit=True)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    quantization_config=config,
    device_map="auto"
)
```

### Cloud Tier: GPT-4o / Claude / Gemini

- **Cost**: ~$0.01-0.05 per image
- **Speed**: ~1-3 seconds/frame (network latency)
- **Quality**: State-of-the-art
- **Integration**: LiteLLM (already in project)

```python
import litellm
response = litellm.completion(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video frame in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
```

## Default Prompt

```
Describe this video frame in detail. Focus on:
- Main subjects and their actions
- Setting and environment
- Mood and atmosphere
- Any notable objects or elements

Keep the description concise (2-3 sentences) but informative.
```

## References & Research

### Internal References
- Existing analysis pattern: `core/analysis/classification.py`
- Existing analysis pattern: `core/analysis/detection.py`
- Settings pattern: `core/settings.py:model_cache_dir`
- Worker pattern: `ui/main_window.py:ClassificationWorker`
- Agent tool pattern: `core/chat_tools.py:classify_content_live`

### External References
- [Moondream GitHub](https://github.com/vikhyat/moondream)
- [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
- [LiteLLM Vision Docs](https://docs.litellm.ai/docs/completion/vision)
- [Transformers VLM Guide](https://huggingface.co/docs/transformers/en/model_doc/llava_onevision)

### Research Findings
- Moondream 2B is the best CPU-friendly option (~1.5GB, good quality)
- LLaVA-OneVision 7B with 4-bit quantization fits in 8GB VRAM
- Cloud APIs (GPT-4o, Claude, Gemini) all support frame-based analysis
- Gemini 2.5 Pro has native video upload, but frame extraction works for all
