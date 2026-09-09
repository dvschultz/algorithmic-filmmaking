---
paths:
  - "core/remix/**"
  - "ui/dialogs/**"
  - "ui/algorithm_config.py"
  - "ui/tabs/sequence_tab.py"
---

# Sequencer Algorithms

Source of truth: `ui/algorithm_config.py`. Dialog-based algorithms have `is_dialog: True`.

| Key | Label | Required Analysis | Dialog |
|-----|-------|-------------------|--------|
| `color` | Chromatics | colors | |
| `duration` | Tempo Shift | — | |
| `brightness` | Into the Dark | brightness | |
| `volume` | Crescendo | volume | |
| `shuffle` | Hatchet Job | — | yes |
| `sequential` | Time Capsule | — | |
| `shot_type` | Focal Ladder | shots | |
| `proximity` | Up Close and Personal | shots | |
| `similarity_chain` | Human Centipede | embeddings | |
| `match_cut` | Match Cut | boundary_embeddings | |
| `exquisite_corpus` | Exquisite Corpus | extract_text | yes |
| `storyteller` | Storyteller | describe | yes |
| `reference_guided` | Reference Guide | dynamic | yes |
| `signature_style` | Signature Style | colors | yes |
| `rose_hobart` | Rose Hobart | — | yes |
| `staccato` | Staccato | embeddings | yes |
| `cassette_tape` | Cassette Tape | transcribe | yes |
| `gaze_sort` | Gaze Sort | gaze | |
| `gaze_consistency` | Gaze Consistency | gaze | |
| `eyes_without_a_face` | Eyes Without a Face | gaze | yes |
| `free_association` | Free Association | describe, embeddings | yes |
| `word_sequencer` | Word Sequencer | transcription_with_words | yes |
| `word_llm_composer` | LLM Word Composer | transcription_with_words | yes |

Algorithm implementations: `core/remix/` (one module per algorithm).
Dialog UIs: `ui/dialogs/` (one file per dialog algorithm).

## Registry (U11+)

`core/remix/engine.py` holds the Qt-free `AlgorithmDefinition` base, `ParameterSpec`,
`SequenceProposal`, `run_algorithm`, and the seed contract (explicit non-negative
seed, `0` valid, `None` = draw and record). `core/remix/registry.py` registers
definitions; `models/recipe.py` is the persisted `SequenceRecipe`. All 23
algorithms run through the registry on every surface and store a recipe on the
sequence; dialogs collect parameters and hand already-computed provider results
back through `resources` so committing never repeats a model call.
Production code calls `core.remix.run_registry_algorithm` or the spine; the old
`generate_sequence` keyword dispatcher lives only in `tests/remix_compat.py`. New
algorithms: add a definition (`prepare` for prerequisites/provider work,
`generate` pure), register it, keep labels/icons/dialog hints in
`ui/algorithm_config.py`, add it to `tests/test_recipe_reconstruction.py`, and
update `docs/architecture/surface-compatibility.md`.

## Dialog Pattern

Dialog-based algorithms use modal `QDialog` subclasses in `ui/dialogs/`. They build their own sequence and set it on the project. **Watch for sequence overwrite** — dialog sequences can be clobbered by generic handlers if the algorithm isn't excluded from fallback paths.
