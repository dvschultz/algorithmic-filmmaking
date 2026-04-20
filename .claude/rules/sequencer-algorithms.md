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
| `gaze_sort` | Gaze Sort | gaze | |
| `gaze_consistency` | Gaze Consistency | gaze | |
| `eyes_without_a_face` | Eyes Without a Face | gaze | yes |
| `free_association` | Free Association | describe, embeddings | yes |

Algorithm implementations: `core/remix/` (one module per algorithm).
Dialog UIs: `ui/dialogs/` (one file per dialog algorithm).

## Dialog Pattern

Dialog-based algorithms use modal `QDialog` subclasses in `ui/dialogs/`. They build their own sequence and set it on the project. **Watch for sequence overwrite** — dialog sequences can be clobbered by generic handlers if the algorithm isn't excluded from fallback paths.
