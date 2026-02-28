---
date: 2026-02-28
topic: signature-style
---

# Signature Style — Drawing-Based Sequencer

## What We're Building

A new sequencing algorithm called "Signature Style" that takes a visual drawing as input and interprets it left-to-right as a guide for how clips are arranged into a sequence. Users either draw directly on a built-in canvas or import an image file. The drawing is then read to determine clip selection, pacing, and color matching.

Two interpretation modes share a single canvas:

- **Parametric mode**: A direct pixel-level reading. The X-axis maps to time across the sequence. The Y-axis maps to pacing — higher means faster cuts (shorter clips), lower means longer holds. If the drawing uses color, the color at each X position drives dominant-color matching against clip metadata (closest color distance wins).

- **VLM mode**: The full drawing is sent to a vision-language model for overall mood/theme interpretation. The drawing is also sliced vertically at points where significant visual changes occur (color shifts, new shapes, line breaks). Each slice is sent to the VLM for local interpretation — informing cut timing, color, shot type, and other analysis-matching properties.

## Why This Approach

**Unified canvas with mode toggle** was chosen over separate algorithm cards or a parametric-only approach because:

- One drawing surface keeps the experience cohesive — the drawing is the centerpiece, not the mode
- Users can sketch once and try both interpretations without redrawing
- A toggle is simpler to discover than two separate cards in the grid
- The two modes complement each other: parametric for precise control, VLM for expressive/abstract input

## Key Decisions

- **Name**: "Signature Style" — appears as a card in the Sequence tab alongside existing algorithms
- **UI placement**: Algorithm card in the existing Sequence tab (not a new tab, not a dialog)
- **Input methods**: Both a built-in drawing canvas AND image import. Import is the faster path to build; the canvas adds convenience for quick sketches
- **Parametric mapping**: X = time, Y = pacing (higher = faster cuts). Color at each X position = dominant color match against clip metadata using color distance
- **Color matching strategy**: Direct dominant-color matching (closest color distance), not hue-band or mood proxy
- **VLM slicing**: Adaptive — detect where the drawing changes significantly and slice at those boundaries. Fewer calls for simple drawings, more for complex ones
- **Analysis prerequisites**: Auto-analyze on demand. If clips are missing needed metadata (colors, shot types), trigger analysis as part of the sequencing workflow
- **Algorithm output**: Returns a standard sorted `(Clip, Source)` list, feeding into the existing timeline like all other algorithms
- **Output duration**: Users set a target duration (e.g., "2:30") and FPS via controls above the canvas. The canvas width maps to that total duration. FPS is required when using individual frames; clips use native FPS by default
- **Source material**: Supports both clips (from scene detection) and individual frames (from frame collections). When using frames, the FPS setting determines hold duration. When using clips, native FPS is used and the duration setting determines how many clips/how long each plays

## Resolved Questions

- **Canvas tools**: Start minimal — pen, eraser, color picker, undo/redo. Single line thickness. Expand later if users ask for more. YAGNI.
- **Parametric B&W regions**: If a section is drawn in black/white, first look for B&W footage in the clip pool. If none exists, fall back to ignoring color matching and selecting clips based on pacing only.
- **Clip reuse**: Allow clip reuse. Clips can repeat to fill the drawing's full timeline. The drawing is always honored in full.

## Open Questions

None — all questions resolved.

## VLM Prompt Strategy

The VLM receives two calls per drawing:

1. **Whole-image call**: Send the full drawing with a prompt asking for overall mood, theme, and narrative arc interpretation. This establishes context for slice-level interpretation.

2. **Per-slice calls**: Each adaptive slice is sent with the whole-image context. The VLM is prompted to:
   - First, describe what it sees in natural language (free-form reasoning about mood, energy, visual qualities)
   - Then, return structured JSON mapping to clip metadata filters:
     ```json
     {
       "shot_type": "close-up",
       "color_mood": "warm",
       "energy": "high",
       "pacing": "fast",
       "brightness": "dark"
     }
     ```

The natural language description serves as chain-of-thought reasoning. The structured JSON is the actionable output used for clip matching. This gives the VLM freedom to interpret expressively while producing machine-readable selection criteria.

## Next Steps

-> `/workflows:plan` for implementation details
