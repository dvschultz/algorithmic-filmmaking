---
module: Scene Ripper
date: 2026-05-12
problem_type: runtime_error
component: transcription_dependencies
symptoms:
  - "Feature 'transcribe_mlx' has partial package coverage; reinstalling the full package set"
  - "Native pip install failed for lightning-whisper-mlx>=0.0.10,<1.0, tiktoken>=0.7.0,<0.8"
  - "Skipping analysis operations with unavailable dependencies: ['transcribe']"
root_cause: transcription_auto_stuck_on_failed_mlx_candidate
resolution_type: code_fix
severity: medium
tags: [transcription, mlx, groq, dependencies, macos]
---

# Troubleshooting: Transcription Auto Mode Stuck on Failed MLX Install

## Problem

On Apple Silicon, transcription in Auto mode could get stuck trying to repair the
preferred `transcribe_mlx` feature. When `lightning-whisper-mlx` or `tiktoken`
installation failed, the analysis run skipped transcription even though the
operation had fallback candidates.

The log pattern looked like:

```text
Feature 'transcribe_mlx' has partial package coverage; reinstalling the full package set
Native pip install failed for lightning-whisper-mlx>=0.0.10,<1.0, tiktoken>=0.7.0,<0.8
Skipping analysis operations with unavailable dependencies: ['transcribe']
```

Groq cloud transcription had a separate usability gap: selecting the Groq
backend could launch work without a configured Groq API key, causing runtime
transcription failures instead of an early configuration error.

## Root Cause

`get_operation_feature_candidates("transcribe", settings)` correctly returned
multiple candidates for Auto mode on Apple Silicon:

```python
["transcribe_mlx", "transcribe"]
```

But `MainWindow._ensure_analysis_operation_available()` only prompted for the
first missing candidate. If the preferred MLX install failed and no alternate was
already installed, the gate returned false without trying `transcribe`.

For Groq, `transcribe_cloud` only requires FFmpeg at the dependency level, so the
dependency gate did not validate that a Groq API key existed before launching
workers.

## Solution

1. Add Groq backend and Groq model controls to `Settings > Models > Transcription`.
2. Add a Groq API key field to `Settings > API Keys`, stored in keyring.
3. Make Groq transcription read `get_groq_api_key()` instead of requiring users
   to launch the app with `GROQ_API_KEY`.
4. Update the dependency gate to try alternate feature candidates when the
   preferred install fails.
5. Add an early configuration gate for Groq transcription when no Groq API key
   is configured.
6. Keep MLX availability checks import-safe by using `importlib.util.find_spec()`.

## Verification

```bash
python -m pytest tests/test_dependency_widgets.py tests/test_analysis_dependency_gates.py tests/test_analysis_dependencies.py tests/test_p3_advanced_models.py tests/test_settings.py -q
ruff check ui/main_window.py tests/test_dependency_widgets.py core/settings.py core/transcription.py ui/settings_dialog.py tests/test_settings.py tests/test_p3_advanced_models.py
python -m py_compile ui/main_window.py core/settings.py core/transcription.py ui/settings_dialog.py
```
