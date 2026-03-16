# Staccato: Source Separation for Stem-Based Beat Detection

**Date:** 2026-03-15
**Status:** Brainstorm

## What We're Building

An optional source separation feature in the Staccato dialog that lets users isolate individual stems (drums, bass, vocals, other) from a music file using Meta's Demucs model, then run beat/onset detection on a specific stem instead of the full mix.

This means you can cut your video to just the drum hits, or sync to the vocal rhythm, or follow the bass line — giving much more precise rhythmic control than analyzing the full mix.

## Why This Approach

- **Demucs (htdemucs)** is the state-of-the-art source separation model, maintained by Meta. It separates into 4 stems: drums, bass, vocals, other.
- **Drums-only onset detection** produces much cleaner beat markers than full-mix analysis, where instruments overlap and muddy the transients.
- **Optional checkbox** keeps the UX simple — users who don't need separation just skip it. No extra complexity for the common case.
- **Cached stems** avoid re-running the ~30-60s separation on subsequent uses of the same file.

## Key Decisions

1. **Model:** Demucs (htdemucs) from Meta — `pip install demucs` or `torchaudio`
2. **Available stems:** All 4 — Drums, Bass, Vocals, Other
3. **UX:** Optional "Separate stems" checkbox in the Staccato dialog. When checked, a dropdown appears to select which stem to analyze.
4. **Separation runs as part of the analyze step** — after the user selects a file and enables stem separation, the worker separates first, then runs beat/onset detection on the selected stem.
5. **Caching:** Separated stems cached in the app's cache directory. Keyed by source file hash. ~50MB per song for all 4 stems. Skip separation if cached stems exist.
6. **Scope:** Staccato only for now. Can be extracted to a shared module later if needed.
7. **Waveform update:** After separation, the waveform widget should display the selected stem's waveform (not the full mix), so the user can visually confirm they're analyzing the right thing.

## Dialog UX Flow (Updated)

```
User selects music file
    ↓
[Optional] User checks "Separate stems"
    ↓
Stem dropdown appears: Drums / Bass / Vocals / Other
    ↓
User clicks or analysis auto-triggers
    ↓
Worker: Check cache for separated stems
    ↓
If not cached: run Demucs separation (~30-60s) → cache results
    ↓
Load selected stem audio
    ↓
Run librosa beat/onset detection on the stem
    ↓
Display stem waveform with beat markers
    ↓
(rest of Staccato flow unchanged)
```

## Demucs Integration Notes

- Demucs can be invoked via Python API (`demucs.separate`) or CLI (`python -m demucs`)
- Output: 4 WAV files (drums.wav, bass.wav, vocals.wav, other.wav) in an output directory
- Model download: htdemucs is ~80MB, downloaded on first use
- Requires PyTorch (already a dependency via DINOv2/transformers)
- CPU inference takes ~30-60s per song; GPU is ~5-10s

## Resolved Questions

1. **Model choice:** Demucs (htdemucs) from Meta
2. **Stem selection:** All 4 stems available via dropdown
3. **UX placement:** Optional checkbox + stem dropdown in Staccato dialog
4. **Caching:** Yes, cache separated stems in app cache dir keyed by file hash
5. **Scope:** Staccato only for now

## Open Questions

None — all key decisions resolved during brainstorm.
