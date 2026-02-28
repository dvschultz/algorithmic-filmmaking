---
status: complete
priority: p2
issue_id: "029"
tags: [code-review, bug, ui-integration, project-adapter]
dependencies: ["026"]
---

# Add source_updated Handler in ProjectAdapter

## Problem Statement

`Project.update_source()` emits a `source_updated` observer event, but `ProjectAdapter` has no handler for it. The GUI never refreshes when source metadata is updated via the agent tool — the change is persisted but invisible until the user navigates away and back.

## Findings

**Python Reviewer**: `source_updated` event has no handler in `ProjectAdapter`. The observer notification fires into the void.

**UI Integration concern**: This is a "silent action" anti-pattern — agent makes a change that doesn't reflect in the UI.

## Proposed Solutions

### Option A: Add Handler to ProjectAdapter (Recommended)

```python
def _on_source_updated(self, source):
    # Refresh source browser thumbnail/metadata display
    self.source_browser.refresh_source(source.id)
```

**Pros:** Completes the update_source feature end-to-end
**Cons:** May need to add refresh_source method to source browser
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] Source browser reflects metadata changes after agent calls update_source
- [ ] No UI flicker or unnecessary full repaints
- [ ] Observer pattern correctly wired in ProjectAdapter

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Python Reviewer finding | Every observer event needs a handler or it's dead code |
