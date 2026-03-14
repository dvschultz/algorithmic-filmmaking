# User Documentation for Scene Ripper

**Date:** 2026-03-14
**Status:** Brainstorm

## What We're Building

A user-facing documentation section (`docs/user-guide/`) with markdown files hosted on GitHub. The app's menu bar gets a new **Help** menu with items that open specific doc pages in the user's default browser.

### Initial doc set:
1. **Sequencer Guide** - One page covering all 15 sequencer algorithms
2. **API Keys Guide** - How API keys work in Scene Ripper, plus detailed per-provider walkthroughs for obtaining keys

## Why This Approach

- **Markdown in repo** - Zero infrastructure, version-controlled alongside code, renders well on GitHub. No build step or hosting to maintain.
- **Browser-based help** - Adding `Help > Sequencer Guide` and `Help > API Keys` menu items that call `QDesktopServices.openUrl()` is trivial to implement and keeps the app simple.
- **Practical & concise tone** - Users want to understand what each sequencer does and when to use it, not read a textbook. Brief descriptions, required analysis, direction options, one-sentence "when to use."
- **Detailed API key walkthroughs** - Getting API keys from different providers is a real friction point for non-technical users. Step-by-step instructions per provider reduce support burden.

## Key Decisions

1. **Docs location**: `docs/user-guide/` in the repo (new directory)
2. **Format**: Markdown, rendered on GitHub
3. **App integration**: New `Help` menu in menu bar with items that open GitHub URLs in browser
4. **Sequencer doc structure**: Practical & concise - what it does, when to use it, required analysis, direction options
5. **API key doc structure**: Detailed per-provider walkthroughs with account creation steps, dashboard navigation, key generation
6. **Scope exclusion**: Local/Ollama setup is out of scope for now - only cloud API providers (Anthropic, OpenAI, Gemini, OpenRouter, Replicate, YouTube)

## Sequencer Algorithms to Document

| Algorithm | Display Name | Required Analysis | Has Dialog |
|-----------|-------------|-------------------|------------|
| color | Chromatics | colors | No |
| duration | Tempo Shift | none | No |
| brightness | Into the Dark | brightness | No |
| volume | Crescendo | volume | No |
| shuffle | Hatchet Job | none | Yes |
| sequential | Time Capsule | none | No |
| shot_type | Focal Ladder | shots | No |
| proximity | Up Close and Personal | shots | No |
| similarity_chain | Human Centipede | embeddings | No |
| match_cut | Match Cut | boundary_embeddings | No |
| exquisite_corpus | Exquisite Corpus | extract_text | Yes |
| storyteller | Storyteller | describe | Yes |
| reference_guided | Reference Guide | dynamic | Yes |
| signature_style | Signature Style | colors | Yes |
| rose_hobart | Rose Hobart | none (face detection) | Yes |

## API Providers to Document

| Provider | Env Var | Used For |
|----------|---------|----------|
| Anthropic | `ANTHROPIC_API_KEY` | Chat agent, Storyteller, descriptions |
| OpenAI | `OPENAI_API_KEY` | Chat agent, descriptions, VLM |
| Gemini | `GEMINI_API_KEY` | Chat agent, VLM, descriptions |
| OpenRouter | `OPENROUTER_API_KEY` | Multi-model access |
| Replicate | `REPLICATE_API_TOKEN` | (model hosting) |
| YouTube | `YOUTUBE_API_KEY` | Video search and metadata |

Each provider section needs: account creation, navigating to API key page, generating the key, free tier / pricing notes, where to paste it in Scene Ripper settings.

## Doc File Structure

```
docs/user-guide/
├── sequencers.md          # All 15 algorithms
└── api-keys.md            # API key setup for all providers
```

## Menu Structure

```
Help
├── Sequencer Guide        → opens docs/user-guide/sequencers.md on GitHub
├── API Keys Guide         → opens docs/user-guide/api-keys.md on GitHub
└── ─────────────
    About Scene Ripper     → (existing or new about dialog)
```

## Open Questions

None - all key decisions resolved during brainstorm.
