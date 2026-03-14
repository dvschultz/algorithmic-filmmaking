---
title: "feat: Add user documentation and Help menu"
type: feat
status: completed
date: 2026-03-14
origin: docs/brainstorms/2026-03-14-user-documentation-brainstorm.md
---

# feat: Add User Documentation and Help Menu

## Overview

Add user-facing documentation to Scene Ripper with two initial guides (sequencer algorithms, API keys) and a Help menu in the app that opens them on GitHub in the user's browser.

## Problem Statement / Motivation

Scene Ripper has no user-facing documentation. Users encountering the 15 sequencer algorithms have no way to learn what each one does, what analysis it requires, or what the direction options mean. Getting API keys from different providers is a friction point for non-technical users. A Help menu provides discoverability within the app.

## Proposed Solution

1. Create `docs/user-guide/sequencers.md` - practical, concise guide to all 15 sequencer algorithms
2. Create `docs/user-guide/api-keys.md` - detailed per-provider API key walkthroughs
3. Add a `Help` menu to the menu bar with items that open these docs on GitHub

(see brainstorm: docs/brainstorms/2026-03-14-user-documentation-brainstorm.md)

## Technical Considerations

### URL Construction

Reuse `_GITHUB_OWNER` and `_GITHUB_REPO` from `core/update_checker.py` to construct GitHub URLs. Promote these to a shared location or import directly. URLs target the `main` branch:

```
https://github.com/{owner}/{repo}/blob/main/docs/user-guide/sequencers.md
```

Since docs and Help menu ship in the same PR/merge, links resolve immediately.

### Menu Implementation

All imports already exist in `ui/main_window.py` (line 27): `QDesktopServices`, `QUrl`, `QAction`. Insert Help menu after the View menu in `_create_menu_bar()` (after line 1275).

**macOS note**: Do NOT set `QAction.AboutRole` on any Help menu items — that relocates them to the app menu. Keep both items as plain actions in the Help menu. An About dialog is **out of scope** for this plan.

### Algorithm Config as Source of Truth

Algorithm names, descriptions, and required analysis come from `ui/algorithm_config.py` (`ALGORITHM_CONFIG`). Direction options come from `ui/tabs/sequence_tab.py` (`_DIRECTION_OPTIONS`). The docs should match these exactly.

### Cross-Referencing

The sequencer doc should link to the API keys doc when mentioning features that require cloud APIs. The API keys doc should include a "Which keys do I need?" table mapping features to providers.

## Acceptance Criteria

- [x] `docs/user-guide/sequencers.md` exists with documentation for all 15 algorithms
- [x] Each algorithm section includes: display name, what it does, required analysis, direction options (if any), dialog workflow (if applicable), auto-compute note (if applicable)
- [x] `docs/user-guide/api-keys.md` exists with detailed walkthroughs for all 6 providers
- [x] API keys doc includes "Which keys do I need?" quick-reference table at the top
- [x] Each provider section includes: account creation steps, navigating to API key page, generating the key, free tier/pricing notes, where to enter it in Scene Ripper (Settings UI primary, env var secondary)
- [x] Replicate section explicitly notes the env var is `REPLICATE_API_TOKEN` (not `_API_KEY`)
- [x] Help menu appears in the menu bar after View menu
- [x] Help > Sequencer Guide opens `sequencers.md` on GitHub in the default browser
- [x] Help > API Keys Guide opens `api-keys.md` on GitHub in the default browser
- [x] GitHub URLs are constructed from shared constants (not hardcoded strings)
- [x] Help menu items have brief tooltips

## Implementation Phases

### Phase 1: Documentation Files

Create both markdown files in `docs/user-guide/`.

#### `docs/user-guide/sequencers.md`

Structure per algorithm:
- **Heading**: Display name (e.g., "Chromatics")
- **One-liner**: What it does
- **Required analysis**: What analysis must be run first (or "None")
- **Direction options**: Table/list if applicable (5 algorithms have these)
- **Dialog workflow**: Brief 3-5 sentence walkthrough for dialog-based algorithms (6 algorithms)
- **Auto-compute note**: For algorithms that auto-compute missing analysis (brightness, volume, similarity_chain, match_cut)
- **Tips**: One practical tip per algorithm (optional)

Group algorithms by category for readability:
1. **Sorting algorithms** (Chromatics, Tempo Shift, Into the Dark, Crescendo, Focal Ladder, Up Close and Personal)
2. **Relationship algorithms** (Human Centipede, Match Cut)
3. **Randomization** (Hatchet Job, Time Capsule)
4. **AI-powered** (Exquisite Corpus, Storyteller, Reference Guide, Signature Style, Rose Hobart)

Cross-link to API keys doc for AI-powered algorithms.

#### `docs/user-guide/api-keys.md`

Structure:
1. **Introduction**: Brief explanation of how API keys work in Scene Ripper (keyring storage, env var override)
2. **Which keys do I need?** Quick-reference table:

| Feature | Required Provider |
|---------|------------------|
| Chat agent | Any LLM provider (Anthropic, OpenAI, Gemini, or OpenRouter) |
| Describe clips | Any VLM provider (Gemini, OpenAI, or Anthropic) |
| Storyteller sequencer | Any LLM provider |
| Exquisite Corpus | Any VLM provider (for text extraction) |
| Signature Style | Any VLM provider |
| YouTube search/download | YouTube Data API |
| Replicate models | Replicate |

3. **Per-provider sections** (one per provider, detailed walkthrough):
   - Anthropic
   - OpenAI
   - Google Gemini
   - OpenRouter
   - Replicate
   - YouTube Data API (most complex — requires Google Cloud project setup)

Each provider section:
- **Create an account**: Step-by-step
- **Generate an API key**: Navigate dashboard, create key
- **Enter in Scene Ripper**: Settings > API Keys tab instructions
- **Environment variable alternative**: `export PROVIDER_KEY=sk-...`
- **Pricing notes**: Free tier info, link to pricing page (avoid specific dollar amounts — they change)

### Phase 2: Help Menu

Add to `ui/main_window.py`:

1. Import `_GITHUB_OWNER`, `_GITHUB_REPO` from `core/update_checker.py`
2. Construct base URL: `f"https://github.com/{_GITHUB_OWNER}/{_GITHUB_REPO}/blob/main"`
3. Add `_create_help_menu()` method or extend `_create_menu_bar()`
4. Two actions:
   - "Sequencer Guide" → opens `{base}/docs/user-guide/sequencers.md`
   - "API Keys Guide" → opens `{base}/docs/user-guide/api-keys.md`
5. Add tooltips: "Opens sequencer documentation on GitHub"

#### Key files to modify

| File | Change |
|------|--------|
| `docs/user-guide/sequencers.md` | **New file** |
| `docs/user-guide/api-keys.md` | **New file** |
| `ui/main_window.py` | Add Help menu after View menu in `_create_menu_bar()` |

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| API key walkthrough steps become outdated as providers change their UIs | Link to provider pricing/docs pages rather than quoting specific prices. Keep step descriptions generic enough to survive minor UI changes. |
| Rose Hobart algorithm only exists on current branch | Already merged to main as of PR #68 — verified. |
| Offline users see browser error | Low frequency, acceptable tradeoff. No in-app fallback needed for v1. |
| Repo rename breaks Help URLs | URLs constructed from `_GITHUB_OWNER`/`_GITHUB_REPO` constants — update in one place. |

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-14-user-documentation-brainstorm.md](docs/brainstorms/2026-03-14-user-documentation-brainstorm.md) — Key decisions carried forward: markdown in `docs/user-guide/`, Help menu opens GitHub in browser, practical tone for sequencers, detailed walkthroughs for API keys, Ollama excluded.

### Internal References

- Algorithm config: `ui/algorithm_config.py` (all 15 algorithms)
- Direction options: `ui/tabs/sequence_tab.py:1053` (`_DIRECTION_OPTIONS`)
- Menu bar: `ui/main_window.py:1155` (`_create_menu_bar()`)
- GitHub constants: `core/update_checker.py:18-19` (`_GITHUB_OWNER`, `_GITHUB_REPO`)
- API key settings: `ui/settings_dialog.py:800` (`_create_api_keys_tab()`)
- API key env vars: `core/settings.py` (all `ENV_*` constants)
- LLM providers: `core/llm_client.py:16-65` (providers, models)
- Existing `QDesktopServices.openUrl()` usage: `ui/main_window.py:1078`
- Learnings: `docs/solutions/logic-errors/circular-import-config-consolidation.md` (import from `algorithm_config.py`, not `sequence_tab.py`)
