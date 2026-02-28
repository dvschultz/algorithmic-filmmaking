---
title: "Sequence Cost Estimate Panel"
date: 2026-02-07
status: brainstorm-complete
---

# Sequence Cost Estimate Panel

## What We're Building

An inline cost estimate panel in the Sequence tab that appears after the user selects a sorting algorithm. It shows which analysis operations the chosen algorithm requires, how many clips still need that analysis, and the estimated time and dollar cost — broken down by operation. Each operation that supports multiple inference tiers (local vs cloud) gets its own tier dropdown, scoped to this sequence only (doesn't change global settings).

The estimate updates live as clips are added or removed from the sequence.

## Why This Approach

- **Contextual**: Different algorithms need different metadata. Showing cost after algorithm selection means the estimate is always relevant to what's about to happen.
- **Actionable**: Users see exactly which operations are needed and can toggle tiers per-operation to optimize cost vs speed.
- **Non-blocking**: It's an inline panel, not a modal. Users can glance at it and proceed without extra clicks.
- **Per-sequence tier overrides**: Keeps global settings untouched. A user might want cloud inference for one sequence but local for another.

## Key Decisions

1. **Scope**: Analysis operations only (describe, classify shots, transcribe, extract text, cinematography, detect objects, colors). Does NOT include render/export time.

2. **UI placement**: Collapsible panel in the parameter area, visible after algorithm selection and before "Apply". Lives alongside the direction dropdown and other algorithm parameters.

3. **Tier switching**: Per-sequence override. Each tiered operation (description, shot classification, text extraction) gets its own local/cloud dropdown in the cost panel. These don't modify `core/settings.py` values.

4. **Dollar costs**: Hardcoded default pricing per provider (Gemini Flash, Claude, Replicate, etc.). Updated with app releases. Shown as estimated totals.

5. **Time estimates**: Conservative defaults assuming CPU-only, no GPU. Displayed with a "varies by hardware" caveat.

6. **Live updates**: Recalculates when clips change, algorithm changes, or tier dropdowns change.

## Operations & Tier Support

| Operation | Local Tier | Cloud Tier | Cost Driver |
|-----------|-----------|-----------|-------------|
| Colors | k-means (free) | N/A | Per clip |
| Shot Classification | CLIP (free) | Replicate VideoMAE | Per clip ($/sec video) |
| Object Detection | YOLO (free) | N/A | Per clip |
| Text Extraction | Tesseract (free) | Gemini VLM | Per clip ($/token) |
| Transcription | faster-whisper (free) | N/A | Per clip duration |
| Description | Moondream2B (free) | Gemini/Claude/GPT | Per clip ($/token) |
| Cinematography | N/A | Gemini only | Per clip ($/token) |

## Algorithm -> Required Operations Mapping

| Algorithm | Required Metadata |
|-----------|------------------|
| color, color_cycle | dominant_colors (Colors) |
| brightness | average_brightness (Colors) |
| volume | rms_volume (no analysis — computed on demand) |
| shot_type | shot_type (Shot Classification) |
| proximity | shot_type (Shot Classification) |
| similarity_chain | CLIP embeddings (computed on demand) |
| match_cut | boundary CLIP embeddings (computed on demand) |
| exquisite_corpus | extracted_text (Text Extraction) |
| storyteller | description (Description) |
| shuffle, sequential, duration | None (use existing clip data) |

## Cost Panel Contents

```
+-----------------------------------------------+
| Cost Estimate                          [v Hide]|
+-----------------------------------------------+
| Algorithm: Storyteller                         |
|                                                |
| Operation     | Clips | Tier        | Est.    |
| ------------- | ----- | ----------- | ------- |
| Description   | 24/50 | [Cloud  v]  | $0.03   |
|               |       |             | ~2 min  |
| ------------- | ----- | ----------- | ------- |
| Total         | 24    |             | $0.03   |
|               |       |             | ~2 min  |
+-----------------------------------------------+
```

- "24/50" means 24 clips out of 50 still need this operation
- Tier dropdown only appears for operations with alternatives
- Time and cost update instantly when tier changes
- Clips that already have the metadata are skipped (no re-analysis cost)

## Open Questions

1. **Should the panel trigger analysis directly?** Could add an "Analyze Now" button that runs the needed operations before applying the algorithm. Or keep it informational and let users go to the Analyze tab.

2. **Embedding computation**: similarity_chain and match_cut compute embeddings on demand during sequencing. Should these show up in the cost panel even though they're always local/free? They do take time.

3. **Volume computation**: Also computed on demand (ffmpeg). Free but time-consuming for many clips. Include in time estimate?

## Next Steps

Run `/workflows:plan` to design the implementation.
