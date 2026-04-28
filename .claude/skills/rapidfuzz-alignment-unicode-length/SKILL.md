---
name: rapidfuzz-alignment-unicode-length
description: |
  Fix wrong-substring highlighting when using `rapidfuzz.fuzz.partial_ratio_alignment`
  with manually case-folded text. Use when: (1) a text-search-with-highlight feature
  shows the bold span landing on the wrong characters, (2) the bug only appears with
  Turkish dotted-I (İ→i̇), German ß↔ss, or other Unicode chars whose `.lower()` form
  has a different length than the original, (3) you call `text.lower()` before passing
  to rapidfuzz and slice the original-case string with the returned indices. Root
  cause: the alignment indices are valid against whatever string you passed in — if
  you pre-lowercased, they index into the lowercased copy, not your original.
author: Claude Code
version: 1.0.0
date: 2026-04-28
---

# RapidFuzz alignment + Unicode-length pitfall

## Problem

A text-search feature scores a phrase against a corpus using
`rapidfuzz.fuzz.partial_ratio` and uses
`rapidfuzz.fuzz.partial_ratio_alignment` to highlight the matched substring in
the rendered text. To make matching case-insensitive, the code pre-lowercases
both the phrase and the segment text before calling rapidfuzz:

```python
phrase_lc = phrase.lower()
text_lc = segment_text.lower()
score = fuzz.partial_ratio(phrase_lc, text_lc)
alignment = fuzz.partial_ratio_alignment(phrase_lc, text_lc)
match_start = alignment.dest_start
match_end = alignment.dest_end

# Later, the UI does:
highlight = original_segment_text[match_start:match_end]
```

For ASCII / common Latin text this works. For text containing characters whose
lowercase form has a **different length** than the original, the highlight
silently lands on the wrong characters:

- Turkish capital İ (U+0130) lowercases to `i̇` (2 chars). Every İ in the
  text shifts subsequent indices by +1.
- German ß ↔ SS / ss under Unicode casefold rules in some libraries.
- Other Unicode chars with single-to-multi or multi-to-single case mappings.

The bounds check `0 <= start < end <= len(original_text)` typically still
passes (lengths drift by 1–2 chars, not enough to overflow), so the fallback
to plain rendering doesn't fire and the user sees the wrong substring boldened
without any error.

## Context / Trigger Conditions

- Code uses `rapidfuzz.fuzz.partial_ratio` and / or
  `partial_ratio_alignment` for fuzzy text search.
- Code lowercases (or otherwise normalizes) inputs before passing to rapidfuzz.
- Code uses `alignment.dest_start` / `dest_end` to slice the **original**
  (un-normalized) text for display.
- Symptom: highlight visually shifts on text containing İ, ß, or similar
  variable-length-cased characters. ASCII / pure-Latin content looks fine.

## Solution

Pass `processor=str.lower` (or any other normalizer) to rapidfuzz instead of
pre-normalizing the inputs yourself. The processor is applied **internally
for scoring only** — `partial_ratio_alignment` returns indices into your
original strings, exactly as the UI needs them.

```python
from rapidfuzz import fuzz

# WRONG: indices index into the lowercased copy
phrase_lc = phrase.lower()
text_lc = segment_text.lower()
alignment = fuzz.partial_ratio_alignment(phrase_lc, text_lc)
# original_segment_text[alignment.dest_start:alignment.dest_end] -> shifted on Turkish text

# RIGHT: pass processor; alignment indexes into the original
alignment = fuzz.partial_ratio_alignment(
    phrase, segment_text, processor=str.lower
)
score = int(round(alignment.score))   # alignment carries the score too
match_start = alignment.dest_start    # index into segment_text (original)
match_end = alignment.dest_end        # index into segment_text (original)
```

Two further wins from this change:

1. **Drop the redundant scoring call.** `partial_ratio_alignment` already
   returns `.score`. Calling `partial_ratio` separately doubles per-pair
   work in hot loops.
2. **Use `processor=str.casefold` instead of `str.lower` for stricter
   case-insensitive matching.** `casefold` handles ß→ss, dotless-I, etc.
   correctly. Use `lower` only if you specifically want locale-naive
   ASCII-style folding.

## Verification

Add a test using a Unicode-length-shifting character:

```python
def test_alignment_indices_map_into_original_case_text():
    from rapidfuzz import fuzz
    text = "Welcome to İstanbul tonight"
    alignment = fuzz.partial_ratio_alignment("istanbul", text, processor=str.lower)
    assert alignment.score >= 80
    # Bold span maps back to the case-preserved word, not a shifted fragment.
    assert "İstanbul" in text[alignment.dest_start:alignment.dest_end]
```

Without the fix, the assertion fails because `dest_start` / `dest_end` index
into the lowercased copy where `İ` became 2 characters.

## Example

In Scene Ripper's Cassette Tape sequencer
(`core/remix/cassette_tape.py:_score_phrase_against_segment`), this fix was
applied as part of the code-review pass on the feature branch. The dialog
renders the matched substring in bold via `dest_start:dest_end` slicing of
`segment.text`, so the indices must remain valid against the original-case
string. See PR #93 (commit `6df756e`).

## Notes

- This pitfall applies to **any** rapidfuzz scorer that returns alignments
  (`partial_ratio_alignment`, `ratio_alignment`, etc.), not just
  `partial_ratio`. The general rule: if you want indices that map into your
  original strings, never pre-normalize the inputs — use `processor=`.
- The processor is applied to **both** strings, so `str.lower` is symmetric
  case-folding. If you need different normalizers for query vs. target, use
  `processor1=` / `processor2=` (rapidfuzz ≥ 3.0).
- The issue is invisible during development on English-only test data. Add
  at least one Unicode test case to any text-highlight feature that uses
  rapidfuzz alignments.
- `rapidfuzz` ≥ 3.0 supports the `processor` kwarg on alignment functions.
  Older versions may not — verify with `python -c "from rapidfuzz import fuzz; help(fuzz.partial_ratio_alignment)"`.

## Sibling pitfall: `partial_ratio` is asymmetric on length

`partial_ratio` aligns the **shorter** of the two strings against substrings
of the **longer** and returns the best ratio. That's exactly what you want
when you have a short query and a long body of text — but it inverts the
relationship when the query is longer than the candidate. Concrete failure
mode in a "find clips that say X" search:

| Phrase (query) | Segment (candidate) | `partial_ratio` |
|---|---|---|
| `"I love you"` | `"I"` | **100** ← wrong direction |
| `"thank you very much"` | `"a"` | **100** ← wrong direction |
| `"thank you"` | `"well thank you for coming"` | 100 ✓ |

The shorter string ("I") is found as a 1-char substring of the longer ("I
love you") and rapidfuzz returns 100. To a user this surfaces as
"match with no transcription text" — the segment is technically non-empty,
but it's so short the score is meaningless.

**Fix pattern:**

```python
if len(candidate) < len(query):
    score = fuzz.ratio(query, candidate, processor=str.lower)  # length-penalised
else:
    alignment = fuzz.partial_ratio_alignment(query, candidate, processor=str.lower)
    score = alignment.score
```

Plus a noise floor (e.g., reject candidates under ~3 chars) so single-char
artifacts from upstream sources (Whisper hallucinations on silent audio,
OCR misreads, etc.) can't surface even with `ratio`'s length penalty.

See PR #93 in Scene Ripper, commit `70e9622`, for the full implementation.

## References

- [RapidFuzz documentation — `fuzz.partial_ratio_alignment`](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html#partial-ratio-alignment)
- [Unicode TR #21 — Case Mappings](https://unicode.org/reports/tr21/) (background on
  why some lowercase forms have different length than the original)
- This skill's surfacing PR: `feat/cassette-tape` (Scene Ripper repo) — the
  adversarial reviewer in `/ce-code-review` flagged the bug class with a
  Turkish-text scenario before any user hit it in production.
