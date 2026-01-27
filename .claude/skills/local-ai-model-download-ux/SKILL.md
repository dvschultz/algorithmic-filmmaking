---
name: local-ai-model-download-ux
description: |
  Best practices for handling large local AI model downloads (LLMs, VLMs) in GUI applications with Agent integration.
  Use when: (1) Integrating models like Moondream/Llama that require heavy downloads, (2) User reports "app frozen" during first run,
  (3) Agent tools timeout while waiting for model loading, (4) Concurrent requests fail with "worker busy".
author: Gemini
version: 1.0.0
date: 2026-01-27
---

# Local AI Model Download UX

## Problem
Local AI models (LLMs, VLMs) often require downloading gigabytes of data (e.g., 1.6GB for Moondream) on first use.
Standard synchronous calls block the UI. Asynchronous workers keep the UI responsive but leave the Agent waiting,
often causing timeouts or confusing "Busy" errors if the user retries too soon.

## Context / Trigger Conditions
- First-time execution of a feature (e.g., "Describe this clip").
- `transformers.from_pretrained()` or similar calls that block.
- Agent tool returns `_wait_for_worker` but the worker takes >60s to start processing due to download.
- User sees "Generating..." but no progress bar for the download itself.

## Solution

### 1. Lazy Loading with Dependency Checks
Don't load models at app startup. Load inside the worker thread on first use.
Explicitly check for all dependencies (including transitive ones like `torchvision`) and fail fast with instructions.

```python
def _load_model():
    try:
        import transformers
        import torch
        import torchvision  # Check transitive dependency!
        return transformers.AutoModel.from_pretrained(...)
    except ImportError:
        raise RuntimeError("Missing dependencies. Run: pip install ...")
```

### 2. Explicit User Feedback
Update the UI immediately *before* starting the blocking download.

```python
# In Worker
self.status_bar.showMessage("Downloading model weights (may take several minutes)...")
# Start download
model = load_model()
```

### 3. Agent "Busy" Guard
When the Agent requests a tool, check if the worker is already running (downloading).
Return a descriptive error instead of a generic failure.

```python
# In MainWindow._on_gui_tool_requested
if worker.isRunning():
    return {
        "success": False,
        "error": "Task in progress. First-time setup is downloading model weights. Please wait."
    }
```

### 4. Extended Timeouts
Increase Agent tool timeouts for operations that might trigger downloads.

```python
# core/chat_tools.py
TOOL_TIMEOUTS = {
    "describe_content": 600,  # 10 minutes for download + inference
}
```

### 5. Project Save Safety
**Crucial:** Do not auto-save `project.save()` in the worker's `finished` handler if the project hasn't been saved to disk yet (new project).

```python
# In _on_worker_finished
if self.project.path:  # Check existence!
    self.project.save()
```

## Verification
1. Delete local model cache (e.g., `~/.cache/huggingface`).
2. Trigger feature via Agent.
3. Verify Status Bar says "Downloading...".
4. Try to trigger again immediately -> Verify Agent reports "Busy".
5. Wait for finish -> Verify Agent reports success.

## References
- [PySide6 QThread](https://doc.qt.io/qtforpython/class.html?name=QThread)
- [HuggingFace Transformers Local Cache](https://huggingface.co/docs/transformers/installation#cache-setup)
