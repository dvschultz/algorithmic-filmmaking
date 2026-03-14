---
status: complete
priority: p3
issue_id: "072"
tags: [code-review, security, rose-hobart]
dependencies: []
---

# Dependency Version Pins Too Loose

## Problem Statement

New dependencies use floor-only `>=` pinning without upper bounds: `insightface>=0.7.3`, `onnxruntime>=1.16.0`, `onnxruntime-silicon>=0.0.3`. A future major version bump could introduce breaking changes or, worst case, a compromised release.

## Findings

**Security Sentinel (Low)**: `onnxruntime-silicon>=0.0.3` is a community-maintained package with a very low version floor.

## Proposed Solutions

### Option A: Add Upper-Bound Pins

```
insightface>=0.7.3,<1.0
onnxruntime>=1.16.0,<2.0
onnxruntime-silicon>=0.0.3,<1.0
```

**Effort:** Small | **Risk:** Low

## Technical Details

- **File:** `requirements.txt` lines 28-30

## Acceptance Criteria

- [ ] All new dependencies have upper-bound version pins

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Security Sentinel | Pin upper bounds on community-maintained packages |
