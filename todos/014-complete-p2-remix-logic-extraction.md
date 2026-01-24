---
status: complete
priority: p2
issue_id: "014"
tags: [code-review, architecture, solid]
dependencies: []
---

# Extract Remix Logic from UI Widget

## Problem Statement

The `_on_generate()` method in `TimelineWidget` contains business logic (algorithm selection, clip ordering, timeline population) that should not live in a UI component. This violates Single Responsibility and Open/Closed principles.

**Why it matters:** Adding new algorithms requires modifying the UI file, mixing concerns, and making testing harder.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/timeline/timeline_widget.py` lines 179-223

```python
def _on_generate(self):
    algorithm = self.remix_combo.currentText().lower()
    # ... 40+ lines of algorithm-specific logic
    if algorithm == "shuffle":
        shuffled = constrained_shuffle(...)
    elif algorithm == "similarity":
        # stub
    elif algorithm == "building":
        # stub
```

**Found by:** architecture-strategist agent

## Proposed Solutions

### Option A: Strategy Pattern with Registry (Recommended)
Create `RemixAlgorithm` base class and register algorithms:
```python
# core/remix/base.py
class RemixAlgorithm(ABC):
    name: str

    @abstractmethod
    def generate(self, clips, params) -> list:
        pass

# core/remix/shuffle.py
class ShuffleAlgorithm(RemixAlgorithm):
    name = "Shuffle"
    def generate(self, clips, params):
        return constrained_shuffle(clips, ...)
```
- **Pros:** Clean architecture, easy to add algorithms, testable
- **Cons:** More files
- **Effort:** Medium
- **Risk:** Low

### Option B: Simple function dispatch
Create a `generate_sequence()` function in core:
```python
# core/remix/__init__.py
def generate_sequence(algorithm: str, clips, params) -> list:
    algorithms = {"shuffle": _shuffle_generate, ...}
    return algorithms[algorithm](clips, params)
```
- **Pros:** Simple, quick to implement
- **Cons:** Less extensible than Strategy pattern
- **Effort:** Small
- **Risk:** Low

## Technical Details

**Affected files:**
- `ui/timeline/timeline_widget.py` - reduce to UI coordination only
- `core/remix/` - add algorithm abstraction

## Acceptance Criteria

- [ ] `TimelineWidget._on_generate()` is < 15 lines
- [ ] Algorithm logic lives in `core/remix/`
- [ ] Adding a new algorithm doesn't require modifying UI
- [ ] Existing shuffle functionality works unchanged

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | SOLID violation identified by architecture-strategist |

## Resources

- PR: Phase 2 Timeline & Composition
- Strategy Pattern documentation
