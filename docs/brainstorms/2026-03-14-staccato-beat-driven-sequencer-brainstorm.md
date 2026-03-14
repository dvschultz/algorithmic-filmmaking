# Staccato: Beat-Driven Sequencer

**Date:** 2026-03-14
**Status:** Brainstorm

## What We're Building

A new sequencer algorithm called **Staccato** that cuts video clips to the rhythm of a user-provided music track. The user uploads an audio file (MP3/WAV/FLAC) in a dialog, and the algorithm detects onsets/beats in the music, then assigns clips to each beat interval based on **onset strength driving visual contrast** — stronger transients trigger bigger visual jumps between consecutive clips, measured by DINOv2 embedding distance.

### Core mechanic

1. Analyze the music file for onsets (or beats/downbeats depending on strategy)
2. Measure onset strength at each point
3. For each beat interval, select the next clip such that the visual distance from the previous clip is proportional to the onset strength at that cut point
4. Trim each clip to fit exactly within its beat interval. If a clip is shorter than the required duration, it loops

## Why This Approach

- **Onset strength → visual contrast** is a genuinely creative mapping that goes beyond simple beat-snapping. It makes the edit *feel* like the music — hard hits get jarring cuts, soft transitions get visually similar clips.
- **DINOv2 embeddings** already exist in the codebase (used by Human Centipede and Match Cut). Reusing them avoids new infrastructure.
- **librosa** beat/onset detection is already implemented in `core/analysis/audio.py` with `analyze_music_file()` supporting standalone audio files.
- **`core/remix/audio_sync.py`** has alignment utilities that can be leveraged.

## Key Decisions

1. **Name:** Staccato
2. **Audio input:** File picker in a modal dialog (MP3, WAV, FLAC, etc.)
3. **Clip selection logic:** Onset strength determines visual contrast. Stronger onset → select a clip with greater DINOv2 embedding distance from the previous clip. Weaker onset → select a visually similar clip.
4. **Embedding system:** DINOv2 embeddings (existing infrastructure)
5. **Clip timing:** One clip per beat interval, trimmed to fit. If clip is shorter than the interval, it loops.
6. **Clip preference:** Prefer clips that are longer than the required beat interval (to avoid looping when possible)
7. **Clip source:** Uses all clips in the project (from the Analyze/Cut pipeline)
8. **Dialog controls:**
   - Music file picker (browse for audio file)
   - Onset sensitivity slider (controls how many cut points are detected — fewer = longer clips, more = faster cuts)
   - Beat strategy dropdown (beats, downbeats, onsets)
9. **Dialog type:** `is_dialog: True` — opens a modal like Hatchet Job, Rose Hobart, etc.
10. **Required analysis:** `embeddings` (DINOv2) — auto-compute if missing, like Human Centipede does

## Algorithm Flow

```
User uploads music file
    ↓
Analyze music: librosa onset/beat detection
    ↓
Get onset strengths (librosa onset_strength)
    ↓
Normalize onset strengths to [0, 1]
    ↓
Generate cut schedule: list of (time, onset_strength) pairs
    ↓
For each beat interval:
    ↓
    Calculate target visual distance = f(onset_strength)
    ↓
    From remaining clips, find the one whose embedding distance
    from the previous clip is closest to the target distance
    ↓
    Assign clip to this interval, trim to fit (or loop if too short)
    ↓
Emit sequence
```

## Onset Strength Details

librosa provides `onset_strength()` which returns a continuous envelope of onset strength over time. At each detected onset, we sample this envelope to get the strength value. Normalize across all onsets so the strongest is 1.0 and weakest is ~0.0.

The mapping from onset strength to target embedding distance:
- Strength 1.0 → maximum embedding distance (most visually different clip)
- Strength 0.0 → minimum embedding distance (most visually similar clip)
- Linear interpolation in between

## Resolved Questions

1. **Clip reuse:** Yes, clips can repeat. Ensures the sequence always fills the full music duration.
2. **Missing embeddings:** Auto-compute DINOv2 embeddings if missing, in the dialog's background worker before generating. Consistent with Human Centipede pattern.
3. **Waveform visualization:** Yes, the dialog should show the audio waveform with detected beat/onset markers overlaid so the user can verify detection quality before generating.

## Open Questions

None — all key decisions resolved during brainstorm.
