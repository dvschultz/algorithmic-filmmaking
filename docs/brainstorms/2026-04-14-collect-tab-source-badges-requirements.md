---
title: "fix: Accurate source badges in Collect tab (CUT + ANALYZED)"
type: fix
status: draft
date: 2026-04-14
---

# Accurate source badges in Collect tab

## Problem

Sources in the Collect tab show an "Analyzed" badge as soon as scene detection (cutting) completes. This is misleading: cutting and analysis are distinct steps, and a source that has only been cut hasn't been analyzed yet.

## Requirements

- R1. Replace the single `source.analyzed` boolean with two independent flags: `source.cut` (scenes detected) and `source.has_analysis` (any analysis operation completed on its clips).
- R2. Display both badges on the source thumbnail when applicable. A source that has been cut AND analyzed shows both "CUT" and "ANALYZED" side by side.
- R3. "CUT" badge appears after scene detection completes for that source.
- R4. "ANALYZED" badge appears after **any** analysis operation (colors, shots, brightness, volume, describe, objects, faces, OCR, transcribe, embeddings, cinematography, gaze) has been run on at least one of the source's clips.
- R5. A source with neither step completed shows no badge (remove the current "Not Analyzed" badge — it adds noise without value).
- R6. Badge state must persist through save/load (serialized on the Source model).

## Scope Boundaries

- No changes to the Analyze tab workflow — this is purely about the Collect tab's display accuracy.
- No "progress bar" or per-operation breakdown — just a binary "has any analysis been done" flag.
- The existing `source.analyzed` field is renamed/replaced, not kept alongside new fields.

## Relevant Code

- `models/clip.py` — `Source` dataclass: `analyzed: bool` field (line 72), `to_dict()` (line 107), `from_dict()` (line 162)
- `ui/source_thumbnail.py` — `_update_badge()` (line 103), `set_analyzed()` (line 98)
- `ui/source_browser.py` — `update_source_analyzed()` (line 293), `get_unanalyzed_sources()` (line 282)
- `ui/tabs/collect_tab.py` — `update_source_analyzed()` (line 175), button text uses `unanalyzed_count()`
- `ui/main_window.py` — sets `source.analyzed = True` after detection (line 4905)
- Analysis completion handlers in `ui/main_window.py` — need to set the new `has_analysis` flag

## Success Criteria

- After cutting a video: source shows "CUT" badge only
- After running any analysis on its clips: source shows "CUT" + "ANALYZED" badges
- Old project files with `"analyzed": true` load correctly (backward compat: treat as `cut=True`)
