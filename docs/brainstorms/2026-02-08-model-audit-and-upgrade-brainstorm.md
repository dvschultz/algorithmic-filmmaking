# Brainstorm: Comprehensive AI Model Audit & Upgrade Plan

**Date:** 2026-02-08
**Status:** Research complete — ready for planning

## What We're Building

A comprehensive upgrade of all AI/ML models used in Scene Ripper, optimizing for quality, speed, and cost across every task: descriptions, shot classification, transcription, text extraction, embeddings, object detection, and cinematography analysis.

**Hardware context:** Apple Silicon Mac, 16GB unified memory.

---

## Current Stack vs Recommended Upgrades

### 1. VIDEO/IMAGE DESCRIPTIONS

| | Current | Recommended Local | Recommended Cloud |
|---|---|---|---|
| **Model** | Moondream 2B | **Qwen3-VL 4B** (via mlx-vlm) | **Gemini 3 Flash** |
| **Video support** | No (frame only) | Yes (native) | Yes (native) |
| **Quality** | ~40-50% of cloud | ~70-80% of cloud | Frontier |
| **Speed** | Slow (CPU only) | Fast (MLX GPU) | Fast |
| **Cost** | Free | Free | ~$0.002/clip |
| **Integration** | HF Transformers | mlx-vlm | LiteLLM (model name change) |

**Key finding:** Moondream 2B is dramatically outclassed. Qwen3-VL 4B via MLX adds video understanding, much better reasoning, and GPU acceleration. SmolVLM2 2.2B is a lighter alternative that still adds video support.

**Specialized option:** Tarsier2-7B (ByteDance) outperforms GPT-4o on video description (+8.6% in human eval) but needs 4-bit quantization to fit in 16GB.

### 2. SHOT CLASSIFICATION

| | Current Local | Current Cloud | Recommended Local | Recommended Cloud |
|---|---|---|---|---|
| **Model** | CLIP ViT-B/32 | VideoMAE (Replicate) | **SigLIP 2 ViT-L** | **Gemini 2.5 Flash Lite** |
| **Accuracy** | ~68% | ~85% | ~80% | ~72% (but full cinematography) |
| **Output** | 5 shot types | 5 shot types | 5 shot types | 20+ cinematography fields |
| **Cost** | Free | $0.005/clip | Free | $0.00026/clip |

**Key finding:** Gemini Flash Lite is **19x cheaper** than Replicate VideoMAE and returns full cinematography analysis (shot size + angle + movement + composition + lighting) vs just shot type. You already have the Gemini cinematography code — consolidate.

**Game-changer:** ShotVL-7B is a purpose-built cinematography VLM covering all 8 dimensions. At 4-bit quantization (~4-5GB) it could run locally, replacing both CLIP and Gemini for cinematography.

### 3. TRANSCRIPTION

| | Current | Recommended (Immediate) | Recommended (Best) |
|---|---|---|---|
| **Model** | faster-whisper (CTranslate2) | **Whisper large-v3-turbo** (faster-whisper) | **mlx-whisper / lightning-whisper-mlx** |
| **Backend** | CPU only (no GPU!) | CPU only | **MLX GPU acceleration** |
| **Speed** | Baseline | ~2x faster (turbo model) | **4-10x faster** (GPU + turbo) |
| **Quality** | large-v3 | Within 1% WER of large-v3 | Same |
| **Cost** | Free | Free | Free |

**Critical finding:** faster-whisper's CTranslate2 backend **cannot use Apple Silicon GPU** — it runs CPU-only. Switching to MLX Whisper is the single highest-impact performance improvement available. Lightning-whisper-mlx claims 10x faster than whisper.cpp.

**Cloud option:** Groq Whisper API at $0.04/hour (large-v3-turbo at 216x realtime) is remarkably cheap if you want a cloud tier.

**Draft mode:** Distil-large-v3 is 6.3x faster with quality within 1-2.4% WER — good for preview transcripts.

### 4. TEXT EXTRACTION (OCR)

| | Current | Recommended |
|---|---|---|
| **Pipeline** | EAST detect → Tesseract → VLM fallback | **PaddleOCR PP-OCRv5** → VLM fallback |
| **Accuracy** | ~82% | ~91% |
| **Stages** | 3 (detect + OCR + fallback) | 2 (unified detect+OCR + fallback) |
| **Scene text** | Weak | Strong (signs, titles, subtitles) |

**Key finding:** PaddleOCR replaces both EAST and Tesseract in a single model with better accuracy on video scene text. Keep the VLM fallback for edge cases.

### 5. EMBEDDINGS (Visual Similarity)

| | Current | Recommended (Similarity) | Recommended (Classification) |
|---|---|---|---|
| **Model** | CLIP ViT-B/32 (512d) | **DINOv2 ViT-B/14** (768d) | **SigLIP 2 ViT-B** (768d) |
| **Strength** | Semantic similarity | **Visual/compositional** similarity | Text-guided classification |
| **Use case** | Both similarity + classification | Match-cut, shuffle | Shot type zero-shot |

**Key finding:** DINOv2 is dramatically better than CLIP for visual similarity (5x improvement on fine-grained tasks). CLIP captures "what things mean" — DINOv2 captures "what things look like." For match-cut detection, DINOv2 is the right model.

**Dual-model approach:** DINOv2 for similarity algorithms + SigLIP 2 for shot classification. Both are ~86M params, ~350MB each.

### 6. OBJECT DETECTION

| | Current | Recommended (Immediate) | Recommended (Open-Vocab) |
|---|---|---|---|
| **Model** | YOLOv8 nano | **YOLO26 nano** | **YOLOE-26** |
| **Speed** | Baseline | **43% faster** (NMS-free) | Similar to YOLO26 |
| **Classes** | 80 COCO fixed | 80 COCO fixed | **Any class via text prompt** |
| **Change** | — | 1-line model name swap | New detection mode |

**Key finding:** YOLO26 is a free 43% speedup with zero API changes. YOLOE-26 adds open-vocabulary detection — detect "person holding camera" or "exterior building" via text prompts without retraining.

---

## Cloud Providers Summary

| Provider | Best For | Key Pricing |
|---|---|---|
| **Gemini 2.5 Flash** | VLM (image+video), cheapest quality | $0.30/$2.50 per 1M tokens |
| **Gemini 2.5 Flash Lite** | High-volume VLM, cinematography | $0.10/$0.40 per 1M tokens |
| **GPT-5 nano** | Ultra-cheap VLM | $0.05/$0.40 per 1M tokens |
| **GPT-5.2** | Highest quality vision | $1.25/$10.00 per 1M tokens |
| **Groq** | Whisper transcription | $0.04/hour (large-v3-turbo) |
| **Replicate** | Custom models (VideoMAE) | $0.005/clip (current) |
| **Modal** | Custom models (alternative) | $0.59/hr T4 GPU |

**Batch APIs:** All major providers (OpenAI, Anthropic, Google) offer 50% batch discounts with 24-hour SLA. Use for non-interactive processing of 500+ clips.

**Recommendation:** Drop OpenRouter (5.5% markup, LiteLLM already provides multi-provider access). Consider replacing Replicate VideoMAE with Gemini Flash Lite (19x cheaper, richer output).

---

## Priority-Ordered Upgrade Path

### P0: Free Wins (1-line changes)
1. **YOLOv8n → YOLO26n** — 43% faster detection, same API
2. **Whisper large-v3 → large-v3-turbo** in faster-whisper — 2x faster, negligible quality loss
3. **Cloud default → Gemini 3 Flash** — better quality, same cost

### P1: High Impact (moderate effort)
4. **faster-whisper → mlx-whisper** — 4-10x transcription speedup (GPU acceleration)
5. **CLIP ViT-B/32 → SigLIP 2 ViT-L** for shot classification — +12pp accuracy
6. **Replicate VideoMAE → Gemini Flash Lite** for cloud shots — 19x cheaper, richer output

### P2: Major Quality Upgrades
7. **Moondream 2B → Qwen3-VL 4B** via mlx-vlm — massive quality jump + video support
8. **Add DINOv2 ViT-B** for similarity embeddings — much better match-cut detection
9. **EAST + Tesseract → PaddleOCR PP-OCRv5** — better accuracy, simpler pipeline

### P3: Advanced / Future
10. **Evaluate ShotVL-7B** for local cinematography analysis
11. **Add YOLOE-26** open-vocabulary detection mode
12. **Add Groq** as cloud transcription tier
13. **Evaluate Apple SpeechAnalyzer** (requires macOS Tahoe + Swift bridge)

---

## Testing Plan

For each model upgrade, run a standardized evaluation:

### Test Dataset
- Curate 50-100 representative clips from existing projects
- Include variety: dialogue, action, landscape, interview, music video
- Annotate ground truth for shot type, expected description quality, known text overlays

### Per-Task Evaluation

**Descriptions (VLM):**
- Side-by-side comparison: Moondream vs Qwen3-VL vs Gemini Flash
- Human rating (1-5) on: accuracy, detail, relevance
- Measure: tokens/second, time-to-first-token, total latency

**Shot Classification:**
- Confusion matrix: CLIP ViT-B/32 vs SigLIP 2 vs ShotVL vs Gemini
- Measure: accuracy, per-class precision/recall
- Use existing labeled clips as ground truth

**Transcription:**
- WER comparison: faster-whisper vs mlx-whisper vs Groq
- Test on: clean dialogue, noisy scenes, music-heavy content
- Measure: WER, processing time, time-to-complete for full video

**OCR:**
- Test on frames with: titles, signs, subtitles, stylized text
- Compare: Tesseract vs PaddleOCR detection rate and accuracy
- Measure: precision, recall, F1 on text extraction

**Embeddings:**
- Cosine similarity matrix for known similar/dissimilar clips
- Compare: CLIP vs DINOv2 vs SigLIP 2 on match-cut pair detection
- Measure: precision@k for similarity retrieval

**Object Detection:**
- mAP comparison on video frames: YOLOv8 vs YOLO26
- Open-vocab test: YOLOE-26 on film-specific queries
- Measure: speed (FPS), detection quality

### Memory Profiling
- Measure peak RAM for each model combination
- Ensure total stays under 12GB (leaving 4GB for macOS + app)

---

## Memory Budget (16GB Mac)

| Model | RAM | Notes |
|---|---|---|
| Qwen3-VL 4B (4-bit) | ~3-4 GB | Replaces Moondream (~2GB) |
| SigLIP 2 ViT-L | ~1.7 GB | Replaces CLIP ViT-B/32 (~350MB) |
| DINOv2 ViT-B | ~350 MB | New addition |
| YOLO26 nano | ~12 MB | Same as YOLOv8 nano |
| PaddleOCR PP-OCRv5 | ~150 MB | Replaces EAST + Tesseract |
| mlx-whisper (large-v3-turbo) | ~1.5 GB | Same as faster-whisper |
| **Total** | **~7 GB** | Leaves ~9GB for macOS + app |

All models lazy-loaded — only active model consumes RAM.

---

## Open Questions

1. **Qwen3-VL vs SmolVLM2:** Need hands-on testing to determine if the 4B model is too slow for interactive use on 16GB, in which case SmolVLM2 2.2B is the fallback
2. **ShotVL-7B viability:** Does the 4-bit quantized model produce useful cinematography analysis, or is quality too degraded?
3. **PaddleOCR dependency weight:** PaddlePaddle is a heavy dependency — worth the install size?
4. **Embedding migration:** Switching from 512d CLIP to 768d DINOv2/SigLIP requires re-embedding all existing clips in projects. Need a migration strategy.

---

## Sources

Full source URLs are documented in the individual research reports. Key references:
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — VLM inference on Apple Silicon
- [SigLIP 2](https://huggingface.co/blog/siglip2) — Google's improved vision encoder
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta's self-supervised vision model
- [ShotVL / ShotBench](https://vchitect.github.io/ShotBench-project/) — Cinematography VLM benchmark
- [YOLO26](https://docs.ultralytics.com/models/yolo26/) — Latest YOLO with NMS-free architecture
- [YOLOE](https://docs.ultralytics.com/models/yoloe/) — Open-vocabulary YOLO
- [PaddleOCR PP-OCRv5](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det) — Modern OCR pipeline
- [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) — Fast MLX Whisper
- [mlx-whisper benchmark](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Alibaba's VLM family
- [SmolVLM2](https://huggingface.co/blog/smolvlm2) — HuggingFace's small video VLM
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Groq Whisper API](https://console.groq.com/docs/speech-to-text)
