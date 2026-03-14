---
name: continuous-learning
description: Track recurring Scene Ripper errors, normalize them into stable signatures, store fix memory, and auto-generate reusable playbook skills for frequent failures. Use when debugging repeated crashes/warnings, when a user asks to "remember this fix," or when converting solved incidents into durable troubleshooting workflows.
---

# Continuous Learning

## Overview

Capture error evidence, match it against known fixes, and update a persistent error memory catalog. Promote recurring solved patterns into generated skills under `skills/error-playbooks/` so future agents can apply fixes quickly.

## Workflow

1. Normalize the error into a deterministic signature.
2. Match it against existing memory and `docs/solutions/`.
3. Record the incident with symptom text and fix references.
4. Promote repeated solved signatures into generated playbook skills.

## Commands

Use `scripts/error_learning.py` from repo root:

```bash
# Record an error from traceback text file
python .codex/skills/continuous-learning/scripts/error_learning.py record \
  --error-file /tmp/error.log \
  --solution-ref docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md \
  --problem-type runtime_error

# Record by explicit signature (no traceback available)
python .codex/skills/continuous-learning/scripts/error_learning.py record \
  --signature "thumbnail-source-id-mismatch|ui/main_window.py" \
  --title "Thumbnail source ID mismatch clears clip grid" \
  --solution-ref docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md

# Bootstrap memory from existing docs/solutions frontmatter
python .codex/skills/continuous-learning/scripts/error_learning.py bootstrap
```

Promotion behavior:
- Default threshold is `3` occurrences.
- Error memory file is `docs/learning/error_memory.json`.
- Promotion target is `skills/error-playbooks/`.
- Each promoted playbook is a standalone skill with `SKILL.md` and `agents/openai.yaml`.

## Implementation Notes

- Keep solution references pointed at canonical docs in `docs/solutions/`.
- Prefer one signature per root cause, not per stack trace line number.
- Include at least one concrete symptom excerpt when recording.
- Run bootstrap after adding new docs in `docs/solutions/` so memory stays aligned.

## References

- Error memory schema example: `references/error_memory.json`
- Frequent known incidents: `references/frequent-errors.md`
