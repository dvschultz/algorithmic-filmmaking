---
status: complete
priority: p2
issue_id: "065"
tags: [code-review, security, rose-hobart]
dependencies: []
---

# Tighten face_embeddings Deserialization Validation

## Problem Statement

`Clip.from_dict` validates structural properties (dict type, bbox length=4, embedding length=512, confidence is numeric) but does not validate that bbox/embedding values are actually numeric. Extra keys in entry dicts are preserved as-is, which could cause unexpected memory growth. Non-numeric embedding values would raise ValueError when fed to numpy in `compare_faces`.

## Findings

**Security Sentinel (Medium)**: A crafted project file could include extra keys with large data. Tightening validation also strips unexpected keys.

## Proposed Solutions

### Option A: Validate Element Types and Strip Extra Keys (Recommended)

```python
if (isinstance(bbox, list) and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
        and isinstance(emb, list) and len(emb) == 512
        and all(isinstance(v, (int, float)) for v in emb)
        and isinstance(conf, (int, float))):
    face_embeddings.append({
        "bbox": bbox, "embedding": emb,
        "confidence": conf, "frame_number": entry.get("frame_number"),
    })
```

**Pros:** Defense-in-depth, strips unexpected keys
**Cons:** Element-wise validation on 512 items adds minor overhead
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `models/clip.py` lines 392-409

## Acceptance Criteria

- [ ] Non-numeric bbox/embedding values are rejected
- [ ] Extra keys in face entries are stripped
- [ ] Existing valid project files load without issues

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Security Sentinel | Always validate element types in deserialization, not just structure |
