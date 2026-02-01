# Shot Type Classifier Comparison

**Date**: 2026-02-01
**Purpose**: Evaluate shot type classification methods to replace inaccurate CLIP-based detection

## Summary

Tested 3 classification methods against 8 labeled reference images from [LearnAboutFilm](https://learnaboutfilm.com/film-language/picture/shotsize/). All methods struggled with fine-grained shot types, but VLM (GPT-4o) performed best at 62% accuracy.

**Recommendation**: Deploy custom VideoMAE model to Replicate for best accuracy (~80%) at lowest cost (~$0.0005/clip).

## Methods Tested

| Method | Description | Cost | Speed |
|--------|-------------|------|-------|
| **Baseline CLIP** | Current implementation, simple prompts | Free | ~1s |
| **Improved CLIP** | Cinematography-aware ensemble prompts | Free | ~1s |
| **VLM (GPT-4o)** | Vision language model classification | ~$0.01/frame | ~1-2s |

## Test Dataset

Reference images from LearnAboutFilm with canonical shot type labels:

| File | Ground Truth | Description |
|------|--------------|-------------|
| 01_extreme_long.jpg | Extreme Long Shot | Setting dominates, figures tiny |
| 02_long.jpg | Long Shot | Full body head to toe |
| 03_medium_long.jpg | Medium Long Shot | Three-quarter shot |
| 04_medium.jpg | Medium Shot | Waist to head |
| 05_medium_closeup.jpg | Medium Close-up | Chest to head |
| 06_closeup.jpg | Close-up | Head and shoulders |
| 07_big_closeup.jpg | Big Close-up | Face fills frame |
| 08_extreme_closeup.jpg | Extreme Close-up | Eyes or mouth only |

## Results

### Raw Classifications

| Image | Ground Truth | Baseline CLIP | Improved CLIP | VLM (GPT-4o) |
|-------|--------------|---------------|---------------|--------------|
| 01_extreme_long | Extreme Long | medium (59%) | Long/Wide (18%) | Long/Wide (95%) |
| 02_long | Long Shot | medium (66%) | Full (12%) | Full (90%) |
| 03_medium_long | Medium Long | medium (46%) | Full (14%) | Full (95%) |
| 04_medium | Medium | medium (51%) | Full (6%) | Medium (85%) |
| 05_medium_closeup | Medium CU | medium (71%) | Close-up (20%) | Medium CU (90%) |
| 06_closeup | Close-up | medium (53%) | Close-up (23%) | Medium CU (90%) |
| 07_big_closeup | Big Close-up | extreme CU (94%) | Close-up (23%) | Close-up (85%) |
| 08_extreme_closeup | Extreme CU | extreme CU (98%) | Extreme CU (15%) | Extreme CU (95%) |

### Accuracy Scores

| Method | Correct | Accuracy |
|--------|---------|----------|
| Baseline CLIP | 2/8 | **25%** |
| Improved CLIP | 4/8 | **50%** |
| VLM (GPT-4o) | 5/8 | **62%** |

## Analysis

### Baseline CLIP Problems

1. **Heavy bias toward "medium shot"** - 6/8 images classified as medium shot
2. **Simple prompts don't capture cinematography concepts** - "a medium shot of a scene" is too generic
3. **Only 4 shot types** - Missing Full Shot, Medium Long, Big Close-up

### Improved CLIP Observations

1. **Better distribution** - No single class dominates
2. **Very low confidence** - Ensemble averaging dilutes scores (6-23%)
3. **Confusion between Long/Full** - These are adjacent in the scale
4. **2x accuracy over baseline** - Prompt engineering helps

### VLM (GPT-4o) Observations

1. **Best accuracy** at 62%
2. **High confidence** but not calibrated (always 85-95%)
3. **Confuses Close-up vs Medium Close-up** - Adjacent classes
4. **Expensive** at ~$0.01/frame

### Common Failure Modes

All methods struggle with:
- **Long Shot vs Full Shot** - Subtle difference (environment vs body focus)
- **Close-up vs Medium Close-up** - Requires precise framing judgment
- **8 fine-grained classes** - Too many similar categories

## Cost Comparison

| Method | Per Frame | 1000 Clips | Notes |
|--------|-----------|------------|-------|
| Baseline CLIP | Free | Free | 25% accuracy |
| Improved CLIP | Free | Free | 50% accuracy |
| VLM (GPT-4o) | $0.01 | $10.00 | 62% accuracy |
| VideoMAE (Replicate) | $0.0005 | $0.50 | ~80% accuracy (estimated) |

## Recommendation

Deploy the custom **VideoMAE model** to Replicate:

1. **Best expected accuracy** (~80%) - trained specifically on cinematographic shot types
2. **Lowest cost** after deployment - ~$0.0005/clip on T4 GPU
3. **Video-aware** - Uses 16 frames vs single frame
4. **Industry-standard classes** - LS, FS, MS, CS, ECS (5 types)

## Files

- Test script: `scripts/compare_shot_classifiers.py`
- Reference images: Downloaded from LearnAboutFilm
- Plan document: `docs/plans/2026-02-01-feat-videomae-shot-type-classification-plan.md`

## Next Steps

1. Deploy VideoMAE model to Replicate
2. Integrate Replicate API into Scene Ripper
3. Add tier selection (CPU/Cloud) in settings
4. Deprecate CLIP-based classification
