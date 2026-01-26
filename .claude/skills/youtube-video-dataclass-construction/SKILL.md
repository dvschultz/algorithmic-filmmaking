---
name: youtube-video-dataclass-construction
description: |
  Fix TypeError when constructing YouTubeVideo objects in Scene Ripper. Use when:
  (1) "got an unexpected keyword argument 'duration_str'" error, (2) constructing
  YouTubeVideo from tool output dicts, (3) syncing search results between agent
  and GUI. The duration_str is a computed property, not an init param, and
  description is a required field.
author: Claude Code
version: 1.0.0
date: 2026-01-25
---

# YouTubeVideo Dataclass Construction

## Problem

When constructing `YouTubeVideo` objects from dict data (e.g., from tool output),
you get a TypeError because the dataclass has different fields than the dict keys.

## Context / Trigger Conditions

- Error: `TypeError: YouTubeVideo.__init__() got an unexpected keyword argument 'duration_str'`
- Converting search tool output to YouTubeVideo objects
- Working with `_on_agent_youtube_search` handler or similar sync code
- The dict has `duration` as a string like "5:23" but dataclass expects timedelta

## Solution

The `YouTubeVideo` dataclass (`core/youtube_api.py`) has:

**Required init params:**
- `video_id: str`
- `title: str`
- `description: str` ← Often forgotten!
- `channel_title: str`
- `thumbnail_url: str`

**Optional init params:**
- `duration: Optional[timedelta]` ← NOT a string!
- `view_count: Optional[int]`
- `definition: Optional[str]`
- `published_at: Optional[str]`

**Computed properties (NOT init params):**
- `youtube_url` - computed from video_id
- `duration_str` - computed from duration timedelta

### Correct Construction Pattern

```python
from datetime import timedelta

def parse_duration(duration_str: str) -> Optional[timedelta]:
    """Parse '5:23' or '1:02:30' to timedelta."""
    if not duration_str:
        return None
    parts = duration_str.split(":")
    try:
        if len(parts) == 2:
            return timedelta(minutes=int(parts[0]), seconds=int(parts[1]))
        elif len(parts) == 3:
            return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
    except (ValueError, IndexError):
        pass
    return None

# Correct construction from tool output dict
video = YouTubeVideo(
    video_id=v.get("video_id", ""),
    title=v.get("title", ""),
    description="",  # Required but often not in tool output
    channel_title=v.get("channel", ""),
    thumbnail_url=v.get("thumbnail", ""),
    duration=parse_duration(v.get("duration", "")),
    view_count=v.get("view_count"),
)
```

## Verification

After fixing, the YouTubeVideo objects should construct without error and
`video.duration_str` should return the formatted string.

## Example

Tool output dict:
```python
{
    "video_id": "abc123",
    "title": "Nature Video",
    "channel": "NatureChannel",
    "duration": "5:23",
    "thumbnail": "https://...",
    "view_count": 1000
}
```

Note the mismatches:
- `channel` in dict → `channel_title` in dataclass
- `duration` is string → needs conversion to timedelta
- `thumbnail` in dict → `thumbnail_url` in dataclass
- `description` missing → must provide empty string

## Notes

- The `search_youtube` tool in `core/chat_tools.py` outputs slightly different
  keys than the dataclass uses (e.g., `channel` vs `channel_title`)
- Always check `core/youtube_api.py` for the actual dataclass definition
- The `duration_str` property formats the timedelta back to "M:SS" or "H:MM:SS"

## Related Skills

- `tool-executor-result-format` - How to unwrap tool results from ToolExecutor
