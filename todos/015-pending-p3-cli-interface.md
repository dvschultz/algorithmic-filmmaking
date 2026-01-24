---
status: pending
priority: p3
issue_id: "015"
tags: [code-review, agent-native, cli]
dependencies: []
---

# Add CLI Interface for Headless Operation

## Problem Statement

The application has well-designed core modules that are fully accessible programmatically, but lacks any CLI or API entry points. Agents cannot invoke timeline operations without GUI interaction.

**Why it matters:** The core architecture is well-factored but inaccessible to automated workflows, scripts, or agent systems.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/main.py`

Current state:
- `main.py` only launches the GUI application
- No `argparse` or CLI mode exists
- Core modules (`scene_detect`, `remix/shuffle`, `sequence_export`) are headless-ready
- 0/8 capabilities have CLI entry points

**Found by:** agent-native-reviewer agent

## Proposed Solutions

### Option A: Add CLI mode to main.py (Recommended)
Add subcommands for key operations:
```bash
python main.py --cli detect --input video.mp4 --output clips.json
python main.py --cli shuffle --clips clips.json --count 10 --output sequence.json
python main.py --cli export --sequence sequence.json --output remix.mp4
python main.py --cli pipeline --input video.mp4 --output remix.mp4
```
- **Pros:** Single entry point, leverages existing code
- **Cons:** Mixes GUI and CLI in one file
- **Effort:** Medium
- **Risk:** Low

### Option B: Separate cli.py module
Create dedicated CLI module:
```python
# cli.py
import click

@click.group()
def cli(): pass

@cli.command()
@click.argument('input_video')
def detect(input_video): ...
```
- **Pros:** Clean separation, CLI-focused
- **Cons:** Duplicate entry point
- **Effort:** Medium
- **Risk:** Low

### Option C: Add sequence serialization only
Just add JSON serialization to models for external tooling:
- **Pros:** Minimal change, enables scripting
- **Cons:** No CLI convenience
- **Effort:** Small
- **Risk:** Low

## Technical Details

**New files needed:**
- `cli.py` or modifications to `main.py`
- JSON serialization for `Sequence`, `Clip`, `Source` models

**Affected files:**
- `models/sequence.py` - add `to_dict()` / `from_dict()`
- `models/clip.py` - add serialization

## Acceptance Criteria

- [ ] Can run scene detection from command line
- [ ] Can generate shuffled sequence from command line
- [ ] Can export sequence to video from command line
- [ ] Pipeline command runs full workflow
- [ ] Progress output is structured (JSON or text)

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Core modules are headless-ready, just need CLI wrapper |

## Resources

- PR: Phase 2 Timeline & Composition
- Click library documentation
