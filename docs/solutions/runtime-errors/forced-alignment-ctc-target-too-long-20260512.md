---
module: Scene Ripper
date: 2026-05-12
problem_type: runtime_error
component: forced_alignment
symptoms:
  - "Forced alignment reported errors: targets length is too long for CTC."
  - "ctc-forced-aligner raises when transcript target tokens exceed audio emissions"
root_cause: whole_clip_transcript_too_dense_for_ctc_alignment
resolution_type: code_fix
severity: medium
tags: [forced-alignment, ctc, word-sequencer, transcription]
---

# Troubleshooting: Forced Alignment CTC Target Too Long

## Problem

Word-level alignment could fail for a clip with:

```text
Forced alignment reported errors: targets length is too long for CTC.
```

Nearby Hugging Face log lines such as this are usually harmless optional-file
probes and are not the root cause:

```text
GET /api/models/.../tree/main/additional_chat_templates?... 404
```

## Root Cause

`ctc-forced-aligner` rejects an alignment request when the target token sequence
is longer than the available CTC emission sequence for the audio. Scene Ripper
was aligning the whole clip transcript in one pass, so a dense or hallucinated
transcript could fail the entire clip even if most transcript segments were
alignable.

## Solution

When whole-clip alignment raises the CTC target-length error, retry alignment
segment-by-segment:

1. Extract each transcript segment's audio subrange.
2. Align only that segment's text against that shorter audio.
3. Offset returned word times back into clip-relative time.
4. If an individual segment is still too dense, synthesize evenly spaced
   approximate word timings inside that segment and keep processing the
   remaining segments.

This keeps one bad segment from failing an entire selected batch and avoids
leaving word-sequencer gaps for very short/dense transcript segments. The
approximate fallback has `probability=None` so downstream code can distinguish
it from model-aligned words if needed.

## Verification

```bash
python -m pytest tests/test_alignment.py tests/test_forced_alignment_worker.py tests/test_word_sequencer_dialog.py tests/test_word_llm_composer.py tests/test_transcription.py -q
ruff check core/analysis/alignment.py tests/test_alignment.py ui/workers/forced_alignment_worker.py
python -m py_compile core/analysis/alignment.py ui/workers/forced_alignment_worker.py ui/tabs/analyze_tab.py
```
