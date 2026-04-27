---
name: llm-api-key-by-model-not-provider
description: |
  Fix misleading "API_KEY_INVALID" / "API key not valid" errors in Scene Ripper's
  LLM-driven sequencers (Storyteller, Exquisite Corpus, Free Association) and any
  other surface that uses a per-feature model setting separate from
  `settings.llm_provider`. Use when: (1) user has a valid Gemini/Anthropic/OpenAI
  key in keyring, (2) the feature still errors with API_KEY_INVALID against the
  cloud provider, (3) the feature's model is set via a per-feature setting like
  `exquisite_corpus_model` or `description_model_cloud` rather than the chat
  provider. Root cause: `get_llm_api_key()` resolves the key from
  `settings.llm_provider` (the chat provider), not from the model being called,
  so a Gemini model gets a non-Gemini key when chat is set to Anthropic / OpenAI
  / local Ollama.
author: Claude Code
version: 1.0.0
date: 2026-04-27
---

# LLM API Key Must Match the Model, Not the Configured Chat Provider

## Problem

Scene Ripper has multiple LLM-using surfaces with their own model settings:

- `settings.exquisite_corpus_model` — Storyteller, Exquisite Corpus, Free Association
- `settings.description_model_cloud` — Describe / OCR
- `settings.cinematography_local_model` and the cloud counterpart
- `settings.llm_model` — chat agent

The chat agent's provider is selected by `settings.llm_provider`. The other
surfaces pick a model independently, which may belong to a *different* cloud
provider than the chat agent.

`core/settings.py` exposes `get_llm_api_key()` which routes by
`settings.llm_provider`. If a sequencer uses
`settings.exquisite_corpus_model = "gemini-3-flash-preview"` while
`settings.llm_provider = "local"` (or `"anthropic"` / `"openai"`),
`get_llm_api_key()` returns the wrong-provider key (or the legacy keyring
fallback). The Gemini endpoint then rejects it with:

```
litellm.AuthenticationError: GeminiException - {
  "error": { "code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT", ... }
}
```

The error is misleading: the user *does* have a valid Gemini key in keyring —
the call site just isn't asking for it.

## Context / Trigger Conditions

- A feature that calls `litellm.completion(...)` errors with
  `API_KEY_INVALID` / `API key not valid` against a cloud provider.
- The user confirms they have a key configured for that provider in keyring or
  env var.
- Search shows the call site uses `get_llm_api_key()` from `core/settings.py`.
- The feature picks its model from a per-feature setting (e.g.,
  `exquisite_corpus_model`) rather than `settings.llm_model`.

## Solution

**Don't use `get_llm_api_key()` for per-feature LLM calls.** Use one of:

1. **`get_api_key_for_model(model)`** in `core/settings.py` — resolves the key
   from the model name's provider prefix. Works for `gemini`, `claude/anthropic`,
   `gpt/openai`, `openrouter`, and returns `""` for local/Ollama.

2. **`complete_with_local_fallback(...)`** in `core/llm_client.py` — wraps
   `litellm.completion` with both correct key resolution *and* automatic
   fallback to `ollama/{settings.ollama_model}` (default `qwen3:8b`) on
   `AuthenticationError`, `RateLimitError`, `ServiceUnavailableError`, or
   `APIConnectionError`. Prefer this for sequencers/analysis where a local
   fallback is acceptable.

The pattern was already correct in `core/analysis/ocr.py` and
`core/analysis/cinematography.py` (they pick the key based on model substring).
The bug lived in `core/remix/storyteller.py`, `exquisite_corpus.py`, and
`free_association.py`.

### Replacing a broken call site

Before:

```python
import litellm
from core.settings import load_settings, get_llm_api_key

settings = load_settings()
model = model or settings.exquisite_corpus_model
api_key = get_llm_api_key()  # WRONG — keys off settings.llm_provider

if "gemini" in model.lower() and not model.startswith("gemini/"):
    model = f"gemini/{model}"

response = litellm.completion(model=model, messages=messages, api_key=api_key, ...)
```

After:

```python
from core.llm_client import complete_with_local_fallback
from core.settings import load_settings

settings = load_settings()
model = model or settings.exquisite_corpus_model

response = complete_with_local_fallback(
    model=model,
    messages=messages,
    temperature=temperature,
)
```

`complete_with_local_fallback` handles model-prefix normalization, key
resolution, and Ollama fallback internally.

## Verification

1. Set `settings.llm_provider = "local"` (or anything not matching
   `exquisite_corpus_model`).
2. Set `exquisite_corpus_model = "gemini-3-flash-preview"`.
3. Ensure a valid Gemini key is in keyring.
4. Run the affected sequencer (e.g., Storyteller).
5. Observe: cloud call now succeeds because the Gemini key is passed.
6. Disable the Gemini key (or use an invalid one) — observe a fallback warning
   (`Cloud LLM call failed ... falling back to local Ollama`) and a successful
   completion via the local model (assuming Ollama is running).

## Notes

- The reverse risk: if a future feature reads `settings.llm_model` (the chat
  model), `get_llm_api_key()` is still the right call there — it's matched to
  `settings.llm_provider` by design.
- Don't treat every `litellm.completion` site mechanically. Check what model
  the site uses *and* what setting the model comes from. If model and provider
  share the same setting (chat agent), `get_llm_api_key()` is fine.
- `complete_with_local_fallback` only catches a fixed set of LiteLLM exceptions.
  If a new failure mode emerges (e.g., a content filter raising a different
  exception), extend the except tuple — don't catch bare `Exception`, since
  programming errors should propagate.
- Check for sibling bugs by grepping
  `grep -rn "get_llm_api_key" core/ --include="*.py" | grep -v settings.py`.
  Each call site should be reviewed for whether the model it uses matches
  `settings.llm_provider`.

## References

- `core/settings.py:get_api_key_for_model`
- `core/llm_client.py:complete_with_local_fallback`
- `core/analysis/ocr.py:230` and `cinematography.py:512` — pre-existing
  correct pattern
