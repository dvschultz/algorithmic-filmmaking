---
status: complete
priority: p1
issue_id: "026"
tags: [code-review, security, project-model]
dependencies: []
---

# Add Field Allowlist to Project.update_source()

## Problem Statement

`Project.update_source()` uses `setattr` with only a `hasattr` guard, meaning any existing attribute on a `Source` object can be overwritten — including `id`, `file_path`, `clips`, or internal state. The agent tool in `chat_tools.py` has its own allowlist, but the model-layer method is unprotected. If any other caller uses `update_source()` directly, it's wide open.

## Findings

**Security Sentinel (F1)**: Unrestricted `setattr` in `Project.update_source` — no field allowlist at model layer. Similar pattern in `update_frame`.

**Python Reviewer**: `update_source` silently ignores invalid field names via `hasattr` but doesn't restrict which valid fields can be set.

## Proposed Solutions

### Option A: Add Allowlist to Project.update_source() (Recommended)

```python
_UPDATABLE_SOURCE_FIELDS = {"color_profile", "fps", "analyzed", "name"}

def update_source(self, source_id, **fields):
    invalid = set(fields) - _UPDATABLE_SOURCE_FIELDS
    if invalid:
        raise ValueError(f"Cannot update fields: {invalid}")
    source = self.sources_by_id.get(source_id)
    if not source:
        raise KeyError(f"Source {source_id} not found")
    for key, value in fields.items():
        setattr(source, key, value)
    self._notify_observers("source_updated", source)
```

**Pros:** Defense in depth — protects at model layer regardless of caller
**Cons:** Must maintain allowlist as Source fields evolve
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] `Project.update_source()` rejects fields not in the allowlist
- [ ] Attempting to set `id`, `file_path`, or other protected fields raises ValueError
- [ ] Agent tool `update_source` still works for allowed fields
- [ ] Test added for allowlist enforcement

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Security Sentinel + Python Reviewer findings | setattr with hasattr is not a security boundary |
