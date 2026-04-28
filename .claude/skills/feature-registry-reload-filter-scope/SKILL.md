---
name: feature-registry-reload-filter-scope
description: |
  Diagnose silent "old version stuck" failures in Scene Ripper's on-demand
  feature installer. Use when: (1) a VLM/ML feature fails with
  "Unrecognized processing class" or "Can't instantiate a processor /
  tokenizer / image processor" from transformers AutoProcessor after
  feature_registry runs an install, (2) the log shows "Skipping reinstall
  of already-loaded packages [...]" and the skipped list contains a
  pure-Python package like transformers, mlx_vlm, ultralytics, or
  lightning_whisper_mlx, (3) every clip in a batch fails with the same
  model-load error in a tight loop. Root cause is scope creep in
  core/feature_registry.py's _UNSAFE_TO_RELOAD_IF_LOADED set.
author: Claude Code
version: 1.0.0
date: 2026-04-21
---

# Feature-Registry Reload Filter Scope

## Problem

Scene Ripper's on-demand dependency installer skips reinstalling packages
whose top-level module is already in `sys.modules`, to avoid corrupting
C-extension state (e.g. torch's `"function '_has_torch_function' already has
a docstring"` crash). The protection list `_UNSAFE_TO_RELOAD_IF_LOADED` in
`core/feature_registry.py` must be precisely scoped to native-code packages.

If the list includes a **pure-Python** package (transformers, ultralytics,
mlx_vlm, lightning_whisper_mlx, etc.), users get silently stuck on whatever
version they first loaded. When a new feature ships that requires a newer
version of that pure-Python dep — for example, `mlx-vlm` pulling a
`transformers` new enough to understand Qwen3-VL's processor class — the
upgrade never happens and every model load fails with an obscure message.

## Context / Trigger Conditions

- User log shows `core.feature_registry - INFO - Skipping reinstall of
  already-loaded packages ['torch', 'torchvision', 'transformers']` (or
  similar, with a pure-Python package in the list)
- Followed by repeated failures loading a VLM: `Failed to load local VLM
  'mlx-community/Qwen3-VL-4B-Instruct-4bit': Unrecognized processing class
  in ...`
- The error message mentions `Can't instantiate a processor, a tokenizer,
  an image processor or a feature extractor for this model. Make sure the
  repository contains the files of at least one of those processing
  classes.`
- Affects every clip in a batch — custom_query / describe_local /
  shot_classify all exhibit the same per-clip failure in a loop

## Solution

1. Open `core/feature_registry.py` and locate `_UNSAFE_TO_RELOAD_IF_LOADED`
2. Keep **only** packages that bundle native code that breaks on reinstall
   under a live interpreter:
   - torch / torchvision / torchaudio (C extensions + `_has_torch_function`
     docstring crash)
   - onnxruntime (native inference runtime)
   - mlx (Apple Metal native bindings)
   - insightface (C extension for detection models)
   - paddleocr (depends on paddlepaddle C extension)
3. Remove any pure-Python packages (transformers, mlx_vlm, ultralytics,
   lightning_whisper_mlx) — these MUST be reinstallable so pip can upgrade
   them when a feature pins a newer version
4. `install_packages` already evicts stale entries from `sys.modules` after
   install (see `test_install_packages_refreshes_sys_path_and_clears_stale_modules`
   in `tests/test_dependency_manager_progress.py`), so the next
   `import transformers` picks up the new version — no app restart needed

## Verification

After the fix, reproduce the user's flow:
1. Clear the HuggingFace model cache
2. Trigger custom_query with a query against a recently-released VLM
3. Confirm the install log does NOT skip the pure-Python dep
4. Confirm the VLM loads and the first clip produces a result

Run the dep-filter tests: `pytest tests/test_dependency_manager_progress.py -q`
(32 tests, all must pass).

## Example

Before (broken):
```python
_UNSAFE_TO_RELOAD_IF_LOADED = {
    "torch", "torchvision", "torchaudio", "onnxruntime",
    "transformers",             # pure-Python — wrongly included
    "insightface",
    "ultralytics",              # pure-Python — wrongly included
    "mlx",
    "mlx_vlm",                  # pure-Python — wrongly included
    "lightning_whisper_mlx",    # pure-Python — wrongly included
    "paddleocr",
}
```

After (fixed, commit `cecc299`):
```python
_UNSAFE_TO_RELOAD_IF_LOADED = {
    "torch", "torchvision", "torchaudio", "onnxruntime",
    "insightface", "mlx", "paddleocr",
}
```

## Notes

- The principle generalizes: any "reload-safety" guard over an on-demand
  dep installer should be scoped narrowly. Pure-Python packages are
  exactly the ones that get version-pinned by ML feature deps, so
  protecting them creates a silent-failure mode far worse than a
  hypothetical reload crash.
- If a pure-Python package ever *does* need reload protection in the
  future, prefer a targeted install-time prompt ("restart required") over
  silently skipping the upgrade.
- Tokenizers is Rust-backed but its reload behavior is stable in practice;
  it stays out of the set for now. Revisit if reports surface.
- Related commit context: `9c271b6` (original over-broad fix) → `cecc299`
  (scope correction).
