---
title: "feat: Add source separation to Staccato sequencer"
type: feat
status: completed
date: 2026-03-15
origin: docs/brainstorms/2026-03-15-staccato-source-separation-brainstorm.md
---

# feat: Add Source Separation to Staccato Sequencer

## Overview

Add optional Demucs-based source separation to the Staccato dialog, letting users isolate individual stems (drums, bass, vocals, other) and run beat/onset detection on a specific stem instead of the full mix.

## Problem Statement / Motivation

Full-mix beat detection can be noisy — overlapping instruments muddy the transients. Isolating the drum track before onset detection produces much cleaner, more musically meaningful beat markers. This gives users precise rhythmic control: cut to drum hits, sync to vocals, or follow the bass line.

(see brainstorm: docs/brainstorms/2026-03-15-staccato-source-separation-brainstorm.md)

## Proposed Solution

Add a "Separate stems" checkbox to the Staccato dialog. When checked, a dropdown appears with 4 stem options (Drums, Bass, Vocals, Other). The analyze worker runs Demucs separation first, caches the stems, then runs librosa beat/onset detection on the selected stem. The waveform widget updates to show the stem's waveform.

## Technical Considerations

### Package: `demucs-infer` (NOT `demucs`)

The original `demucs` package from Meta is **archived** (Jan 2025) and has a hard `torchaudio<2.1` ceiling that conflicts with modern PyTorch. Use `demucs-infer` instead — it's a maintained fork with identical API, supports PyTorch 2.0+, and has fewer dependencies.

```python
from demucs.api import Separator, save_audio  # same import path as original
```

### Demucs API Pattern

```python
separator = Separator(
    model="htdemucs",
    device="cpu",        # or "cuda" if available
    callback=progress_fn,  # for progress reporting
    progress=False,
)
original, separated = separator.separate_audio_file("song.mp3")
# separated = {"drums": Tensor, "bass": Tensor, "vocals": Tensor, "other": Tensor}

# Save a single stem to WAV
save_audio(separated["drums"], "drums.wav", samplerate=separator.samplerate)
```

- Abort: raise an exception inside the callback
- Progress: callback receives `{"state": "start"|"end", "segment_offset": int, "audio_length": int}`
- Model download: htdemucs is ~80MB, downloaded on first use to torch cache

### Optional Dependency via feature_registry

Follow the existing pattern for optional heavy packages:
1. Add `"stem_separation"` entry to `FEATURE_DEPS` in `core/feature_registry.py`
2. Add `demucs-infer` to `core/package_manifest.json`
3. In the dialog, call `check_feature("stem_separation")` when the checkbox is first checked
4. Prompt user to install if unavailable (same pattern as `face_detect`, `ocr`, etc.)

### Stem Caching

Cache separated stems in `settings.stems_cache_dir` (new field on `Settings`):
- Default path: `_get_cache_dir() / "stems"`
- Key by file content hash (first 16 bytes of SHA-256)
- Directory structure: `stems/{hash}/drums.wav`, `stems/{hash}/bass.wav`, etc.
- ~50MB per song for all 4 stems
- On subsequent loads, check if cache dir exists with all 4 stems → skip separation

### Dialog UX Changes

Current `StaccatoDialog` config page gets two new controls after the file picker:

1. **"Separate stems" checkbox** — unchecked by default
2. **Stem dropdown** — hidden when unchecked. Shows: Drums, Bass, Vocals, Other

When the checkbox is checked and a file is loaded:
- The analyze worker runs Demucs first (with progress), then librosa on the selected stem
- Waveform shows the stem's audio, not the full mix
- Info label shows "Analyzing drums stem..." or similar

When the user changes the stem dropdown (with stems already cached):
- Re-run librosa on the new stem (fast, ~1-2s)
- Update waveform to show the new stem

### StaccatoAnalyzeWorker Changes

The worker needs a new mode: if `stem_name` is provided, it:
1. Checks stem cache → loads cached stem if available
2. If not cached: runs Demucs separation → caches all 4 stems
3. Loads the selected stem's audio
4. Runs librosa beat/onset detection on the stem
5. Emits the stem's samples for waveform display

Add parameters to worker: `stem_name: str | None = None` (None = full mix, no separation)

## Acceptance Criteria

- [x] "Separate stems" checkbox added to Staccato dialog config page
- [x] Stem dropdown (Drums, Bass, Vocals, Other) appears when checkbox is checked
- [x] `demucs-infer` registered as optional dependency via `feature_registry.py`
- [x] User prompted to install `demucs-infer` on first use if not available
- [x] Demucs separation runs in the analyze worker with progress reporting
- [x] Separated stems cached in `settings.stems_cache_dir` keyed by file hash
- [x] Cached stems reused on subsequent loads of the same file
- [x] librosa beat/onset detection runs on the selected stem (not full mix)
- [x] Waveform widget displays the selected stem's waveform after separation
- [x] Changing stem dropdown re-analyzes with cached stem (no re-separation)
- [x] Full mix analysis still works when checkbox is unchecked (no regression)
- [x] Tests cover stem caching, worker with stem mode, and feature check

## Implementation Phases

### Phase 1: Dependency Registration

**Files:**
- `core/feature_registry.py` — Add `stem_separation` feature
- `core/package_manifest.json` — Add `demucs-infer` specifier
- `core/settings.py` — Add `stems_cache_dir` field

**Tasks:**
- [x] Add `"stem_separation": FeatureDeps(binaries=[], packages=["demucs-infer"], size_estimate_mb=2000)` to `FEATURE_DEPS`
- [x] Add `"demucs-infer"` entry to `package_manifest.json` with version pin
- [x] Add `stems_cache_dir: Path` field to `Settings` dataclass: `_get_cache_dir() / "stems"`

### Phase 2: Stem Separation Module

**Files:**
- `core/analysis/stem_separation.py` — New file

**Tasks:**
- [x] Implement `separate_stems(music_path, output_dir, progress_cb=None) -> dict[str, Path]`
  - Lazy import `demucs.api` (same pattern as librosa in `audio.py`)
  - Run `Separator.separate_audio_file()` with progress callback
  - Save all 4 stems as WAV to `output_dir/`
  - Return dict mapping stem names to file paths
- [x] Implement `get_cached_stems(music_path, cache_dir) -> dict[str, Path] | None`
  - Hash file → check `cache_dir/{hash}/` for 4 stem WAVs
  - Return paths dict if all present, None if cache miss
- [x] Implement `get_stem_cache_key(music_path) -> str`
  - SHA-256 first 64KB of the file → hex[:16]
- [x] Write tests with mocked Demucs (don't require actual model in CI)

### Phase 3: Worker Enhancement

**Files:**
- `ui/dialogs/staccato_dialog.py` — Modify `StaccatoAnalyzeWorker`

**Tasks:**
- [x] Add `stem_name: str | None = None` parameter to `StaccatoAnalyzeWorker.__init__`
- [x] When `stem_name` is set:
  - Check feature availability via `check_feature("stem_separation")`
  - Check stem cache → use cached stems if available
  - If not cached: run `separate_stems()` with progress
  - Load the selected stem's WAV via librosa
  - Run `analyze_music_file()` on the stem WAV instead of the original
- [x] Emit progress messages: "Separating stems...", "Analyzing {stem} track..."
- [x] When `stem_name` is None: existing behavior (full mix analysis, unchanged)

### Phase 4: Dialog UI

**Files:**
- `ui/dialogs/staccato_dialog.py` — Modify `StaccatoDialog`

**Tasks:**
- [x] Add "Separate stems" checkbox below file picker row
- [x] Add stem dropdown (Drums, Bass, Vocals, Other) — hidden when unchecked
- [x] Wire checkbox toggled → show/hide dropdown
- [x] Wire dropdown change → re-analyze with new stem (if stems already cached, fast path)
- [x] Pass `stem_name` to worker when checkbox is checked
- [x] Show installation prompt if `check_feature("stem_separation")` returns missing deps
- [x] Update info label to show stem name when in stem mode

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| `demucs-infer` may not be installed | Optional dependency via `feature_registry.py`. User prompted to install on first use. |
| Demucs model is ~80MB, downloaded on first use | Show progress during model download (Demucs handles this internally). |
| Separation takes 30-60s on CPU | Show progress bar. Cache stems so re-runs are instant. |
| ~50MB disk per song for cached stems | Acceptable tradeoff. Can add cache cleanup later if needed. |
| Original `demucs` package is archived | Using `demucs-infer` maintained fork with same API. |
| PyTorch version conflicts | `demucs-infer` requires `torch>=2.0` — already in requirements.txt. |

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-15-staccato-source-separation-brainstorm.md](docs/brainstorms/2026-03-15-staccato-source-separation-brainstorm.md) — Key decisions: Demucs htdemucs model, all 4 stems, optional checkbox + dropdown, cached in app cache dir, Staccato-only scope.

### Internal References

- Staccato dialog: `ui/dialogs/staccato_dialog.py` (StaccatoAnalyzeWorker at line 40)
- Feature registry: `core/feature_registry.py` (FEATURE_DEPS, check_feature, install_for_feature)
- Package manifest: `core/package_manifest.json`
- Settings cache dir: `core/settings.py:326` (_get_cache_dir)
- Librosa lazy import pattern: `core/analysis/audio.py:32` (_get_librosa)

### External References

- demucs-infer PyPI: https://pypi.org/project/demucs-infer/
- Demucs Python API docs: https://github.com/facebookresearch/demucs/blob/main/docs/api.md
- Original demucs archived repo: https://github.com/facebookresearch/demucs
