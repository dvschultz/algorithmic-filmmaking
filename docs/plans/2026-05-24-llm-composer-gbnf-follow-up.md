# LLM Composer Tier 2 Constrained Decoding Follow-Up

## Problem

The current LLM Word Composer uses JSON-schema enum constraints through the selected provider. This works for short corpora, but large vocabularies can be slow enough that a dialog-driven workflow times out or feels stuck.

## Proposed Tier 2 Backend

Add a local grammar-backed constrained decoding backend for large vocabularies:

- Preferred path: `llama-cpp-python` with GBNF grammar constraints.
- Keep the existing provider path as Tier 1 for small corpora and cloud/local provider compatibility.
- Add a threshold that switches to Tier 2 when the unique word vocabulary or recent Tier 1 latency exceeds a configurable limit.

## Integration Points

- `core/spine/words.py`: keep vocabulary normalization and inventory construction provider-agnostic.
- `core/remix/word_llm_composer.py`: choose Tier 1 vs Tier 2 before composition.
- `core/llm_client.py`: keep provider error normalization, but route grammar decoding through a separate local backend.
- `ui/dialogs/word_llm_composer_dialog.py`: surface backend status and the reason for any fallback.
- `core/feature_registry.py`: use the reserved `constrained_decoding_grammar` feature key for optional dependency checks and install prompts.

## Validation

- Benchmark representative 100, 1,000, and 5,000 unique-word corpora.
- Verify generated words are always in the corpus after normalization.
- Verify cancellation returns control to the dialog within the configured close/cancel timeout.
- Keep the MCP/agent path able to identify already-aligned clips through `list_clips[*].has_word_alignment`.
