---
name: litellm-gemini-api-routing
description: |
  Fix "DefaultCredentialsError" or "Cloud SDK credentials not found" when using Gemini models via LiteLLM with an API key. Use when: (1) You have a valid `GEMINI_API_KEY` set, (2) LiteLLM defaults to Vertex AI authentication instead of using the API key, (3) Model names are provided without prefixes (e.g., "gemini-2.0-flash").
author: Claude Code
version: 1.0.0
date: 2026-01-27
---

# LiteLLM Gemini API Routing

## Problem
When using LiteLLM with Gemini models, providing a bare model name (e.g., `gemini-2.0-flash`) often causes the library to default to Google Vertex AI, which requires Google Cloud SDK credentials (`gcloud auth`). This fails if you intended to use Google AI Studio with a simple API Key (`GEMINI_API_KEY`).

## Context / Trigger Conditions
- **Error Message**: `google.auth.exceptions.DefaultCredentialsError: Your default credentials were not found.`
- **Configuration**: `GEMINI_API_KEY` is set in environment/keyring.
- **Input**: Model string is "gemini-pro", "gemini-1.5-flash", etc., without a provider prefix.

## Solution
Explicitly prepend the `gemini/` provider prefix to the model name. This forces LiteLLM to route the request to Google AI Studio using the API key.

### Python Implementation
Normalize the model string before passing it to `litellm.completion`:

```python
model = "gemini-2.5-flash"

# Fix: Force AI Studio routing if prefix is missing
if "gemini" in model and not any(model.startswith(p) for p in ["gemini/", "vertex_ai/"]):
    model = f"gemini/{model}"

# Now calls Google AI Studio (using GEMINI_API_KEY)
response = litellm.completion(model=model, messages=...)
```

## Verification
1. Ensure `GEMINI_API_KEY` is set.
2. Run the completion with the prefixed model name (e.g., `gemini/gemini-2.5-flash`).
3. It should succeed without checking for `application_default_credentials.json`.

## Notes
- `vertex_ai/gemini-...` routes to Vertex AI (requires Cloud IAM).
- `gemini/gemini-...` routes to AI Studio (requires API Key).
- Some older LiteLLM versions might default differently, but explicit prefixes are always safer.

## References
- [LiteLLM Provider Documentation - Gemini](https://docs.litellm.ai/docs/providers/gemini)
- [Google AI Studio vs Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/migrate/migrate-google-ai)
