---
title: "feat: SRT Export for Sequence Metadata"
type: feat
date: 2026-02-04
---

# SRT Export for Sequence Metadata

## Overview

Add SRT (subtitle) export functionality to the Render tab that outputs sequence type-specific metadata as subtitles synchronized to the video timeline. Each sequence algorithm exports its relevant data: Storyteller exports descriptions, Color exports hex codes, Shot Type exports classifications, and Exquisite Corpus exports OCR text.

## Problem Statement / Motivation

Filmmakers using the intention-first workflow need a way to document what's in their sequences for reference. After creating a Storyteller narrative sequence, Color-sorted montage, or text-driven Exquisite Corpus edit, there's no way to export a readable document of the metadata that drove the sequence.

**User story:** As a filmmaker, I want to export an SRT file alongside my video so I can reference the visual descriptions (Storyteller), color values (Color), shot classifications (Shot Type), or extracted text (Exquisite Corpus) while reviewing in an NLE or media player.

## Proposed Solution

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      RENDER TAB                              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Export Settings                                      │    │
│  │ [Quality dropdown] [Resolution dropdown] [FPS]       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Sequence                                             │    │
│  │ Duration: 00:05:23    Clips: 42                      │    │
│  │              [Export Sequence]                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Other Exports                                        │    │
│  │ [Export Selected] [Export All] [Export JSON]         │    │
│  │ [Export SRT] <-- NEW                                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### SRT Content by Sequence Type

| Sequence Type | Metadata Source | Example SRT Text |
|---------------|-----------------|------------------|
| Storyteller | `clip.description` | "A woman walks through a sunlit forest" |
| Exquisite Corpus | `clip.combined_text` | "CHAPTER ONE \| THE BEGINNING" |
| Color | `clip.dominant_colors` | "#FF5733, #C70039, #900C3F" |
| Shot Type | `clip.shot_type` | "close-up" |
| Duration/Shuffle/Sequential | `clip.description` (fallback) | "Wide establishing shot of city" |
| Unknown/None | `clip.description` (fallback) | Falls back to description if available |

### SRT Format

```srt
1
00:00:00,000 --> 00:00:03,500
A woman walks through a sunlit forest

2
00:00:03,500 --> 00:00:07,200
close-up of hands picking wildflowers

3
00:00:07,200 --> 00:00:12,800
#FF5733, #C70039
```

## Technical Considerations

### Architecture

**New Components:**
- `core/srt_export.py` - SRT export logic with config dataclass
- Render tab UI addition - "Export SRT" button

**Modified Components:**
- `models/sequence.py` - Add `algorithm: Optional[str]` field for persistence
- `ui/tabs/render_tab.py` - Add export button and signal
- `ui/main_window.py` - Add handler for SRT export

### Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ SRT Export Flow                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sequence.get_all_clips() → list[SequenceClip]                   │
│      │                                                           │
│      ├── seq_clip.source_clip_id → clips_by_id[id] → Clip        │
│      │       └── clip.description / clip.dominant_colors / etc.  │
│      │                                                           │
│      ├── seq_clip.source_id → sources[id] → Source               │
│      │       └── source.fps (for fallback timecode)              │
│      │                                                           │
│      └── seq_clip.start_frame / in_point / out_point             │
│              └── Timeline position and duration                  │
│                                                                  │
│  Sequence.algorithm → Determines which metadata field to export  │
│  Sequence.fps → Master fps for timecode calculation              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

**Timecode Calculation:**
```python
# Use sequence.fps as master timeline fps
start_time = seq_clip.start_frame / sequence.fps
end_time = (seq_clip.start_frame + seq_clip.duration_frames) / sequence.fps

# SRT format: HH:MM:SS,mmm
def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

**Metadata Extraction by Algorithm:**
```python
ALGORITHM_METADATA_MAP = {
    "storyteller": lambda clip: clip.description,
    "exquisite_corpus": lambda clip: clip.combined_text,
    "color": lambda clip: format_colors_hex(clip.dominant_colors),
    "shot_type": lambda clip: clip.shot_type,
    "duration": lambda clip: clip.description,  # fallback
    "shuffle": lambda clip: clip.description,   # fallback
    "sequential": lambda clip: clip.description,  # fallback
}
```

**Color Formatting:**
```python
def format_colors_hex(colors: list[tuple[int, int, int]]) -> str:
    if not colors:
        return None
    return ", ".join(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors)
```

### Sequence Model Changes

Add `algorithm` field to `Sequence` for persistence:

```python
# models/sequence.py
@dataclass
class Sequence:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Sequence"
    fps: float = 30.0
    tracks: list[Track] = field(default_factory=list)
    algorithm: Optional[str] = None  # NEW: "storyteller", "color", etc.

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "fps": self.fps,
            "tracks": [track.to_dict() for track in self.tracks],
        }
        if self.algorithm:
            data["algorithm"] = self.algorithm
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Sequence":
        # ... existing code ...
        seq.algorithm = data.get("algorithm")
        return seq
```

### Empty Result Handling

When all clips lack the required metadata:
1. Show warning dialog: "No clips have [metadata type]. Export anyway?"
2. If user confirms, export valid SRT with no entries (empty but valid file)
3. If user cancels, return to Render tab

## Acceptance Criteria

### Core Functionality
- [x] New `core/srt_export.py` module with `SRTExportConfig` and `export_srt()` function
- [x] Export function returns `bool` (success/failure) following existing patterns
- [x] SRT uses standard format with numbered entries
- [x] Timecodes in `HH:MM:SS,mmm` format
- [x] Clips without metadata are skipped silently
- [x] UTF-8 encoding for file output

### Sequence Type Metadata
- [x] Storyteller exports `clip.description`
- [x] Exquisite Corpus exports `clip.combined_text` (OCR text)
- [x] Color exports `clip.dominant_colors` as hex codes (e.g., "#FF5733, #C70039")
- [x] Shot Type exports `clip.shot_type`
- [x] Duration/Shuffle/Sequential fall back to `clip.description`
- [x] Unknown/None algorithm falls back to `clip.description`

### Persistence
- [x] Add `algorithm: Optional[str]` field to `Sequence` model
- [x] Algorithm persisted in `to_dict()` / `from_dict()`
- [x] SequenceTab sets `sequence.algorithm` when generating sequence
- [x] SRT export works correctly after project save/load

### UI Integration
- [x] "Export SRT" button in Render tab "Other Exports" group
- [x] `export_srt_requested` signal from RenderTab
- [x] Button enabled when sequence has clips (same rule as Export Sequence)
- [x] Save dialog with `.srt` filter
- [x] Default filename uses `sequence.name` + ".srt"
- [x] Success feedback via status bar

### Edge Cases
- [x] Empty sequence: button disabled
- [x] All clips lack metadata: warning dialog, allow export of empty SRT
- [x] Mixed metadata: clips without data skipped, others exported
- [x] Multi-track sequence: all tracks exported (sorted by start_frame)
- [x] Different source fps: use `sequence.fps` for all timecodes

## Dependencies & Risks

### Dependencies
- Existing `Sequence` and `SequenceClip` models
- Clip metadata fields populated by Analyze tab
- RenderTab UI infrastructure

### Risks

| Risk | Mitigation |
|------|------------|
| Algorithm not set before export | Fallback to description; warn user if all empty |
| Large sequences slow export | SRT export is string manipulation (fast), no FFmpeg needed |
| Special characters break SRT | Use UTF-8 encoding, sanitize newlines within subtitle text |
| User confusion about empty output | Warning dialog before exporting when no metadata |

## References & Research

### Internal References
- Export pattern: `core/edl_export.py:42-118` (EDLExportConfig, export_edl)
- Timecode conversion: `core/edl_export.py:24-39` (frames_to_timecode)
- Render tab integration: `ui/tabs/render_tab.py:154-175` (Other Exports group)
- Main window handler: `ui/main_window.py:4682-4729` (_on_export_dataset_click pattern)
- Sequence model: `models/sequence.py:113-188`
- Clip metadata: `models/clip.py:159-360` (description, dominant_colors, shot_type, combined_text)
- Storyteller algorithm: `core/remix/storyteller.py:1-100`

### SRT Format Specification
Standard SRT format:
```
1
00:00:00,000 --> 00:00:05,000
Subtitle text line 1
Optional second line

2
00:00:05,000 --> 00:00:10,000
Next subtitle
```

Key rules:
- Entry number starts at 1
- Timecode format: `HH:MM:SS,mmm` (comma before milliseconds, not period)
- Arrow: ` --> ` (spaces around)
- Blank line between entries
- UTF-8 encoding preferred

## MVP Implementation

### core/srt_export.py

```python
"""SRT (SubRip) export for sequence metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from models.clip import Clip, Source
from models.sequence import Sequence


@dataclass
class SRTExportConfig:
    """Configuration for SRT export."""
    output_path: Path


# Metadata extraction by algorithm
ALGORITHM_METADATA_EXTRACTORS = {
    "storyteller": lambda clip: clip.description,
    "exquisite_corpus": lambda clip: clip.combined_text,
    "color": lambda clip: _format_colors_hex(clip.dominant_colors),
    "shot_type": lambda clip: clip.shot_type,
    "duration": lambda clip: clip.description,
    "shuffle": lambda clip: clip.description,
    "sequential": lambda clip: clip.description,
}


def _format_colors_hex(colors: Optional[list[tuple[int, int, int]]]) -> Optional[str]:
    """Format RGB tuples as hex color codes."""
    if not colors:
        return None
    return ", ".join(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timecode format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _sanitize_srt_text(text: str) -> str:
    """Sanitize text for SRT format (remove problematic characters)."""
    # SRT text should not contain the --> arrow or entry numbers at line start
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple newlines to prevent format breaking
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text.strip()


def export_srt(
    sequence: Sequence,
    clips_by_id: dict[str, Clip],
    sources_by_id: dict[str, Source],
    config: SRTExportConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[bool, int, int]:
    """Export sequence metadata as SRT subtitle file.

    Args:
        sequence: The sequence to export
        clips_by_id: Dict mapping clip_id to Clip
        sources_by_id: Dict mapping source_id to Source
        config: Export configuration
        progress_callback: Optional callback (progress 0-1, message)

    Returns:
        Tuple of (success, exported_count, skipped_count)
    """
    seq_clips = sequence.get_all_clips()
    if not seq_clips:
        return False, 0, 0

    # Determine metadata extractor based on algorithm
    algorithm = sequence.algorithm or "description_fallback"
    extractor = ALGORITHM_METADATA_EXTRACTORS.get(
        algorithm,
        lambda clip: clip.description  # Default fallback
    )

    entries = []
    entry_num = 1
    skipped = 0
    total = len(seq_clips)

    for i, seq_clip in enumerate(seq_clips):
        if progress_callback:
            progress_callback(i / total, f"Processing clip {i + 1}/{total}")

        # Get source clip for metadata
        clip = clips_by_id.get(seq_clip.source_clip_id)
        if not clip:
            skipped += 1
            continue

        # Extract metadata
        text = extractor(clip)
        if not text:
            skipped += 1
            continue

        # Calculate timecodes using sequence fps
        start_time = seq_clip.start_frame / sequence.fps
        duration = seq_clip.duration_frames / sequence.fps
        end_time = start_time + duration

        start_tc = _seconds_to_srt_time(start_time)
        end_tc = _seconds_to_srt_time(end_time)

        # Build SRT entry
        sanitized_text = _sanitize_srt_text(text)
        entry = f"{entry_num}\n{start_tc} --> {end_tc}\n{sanitized_text}\n"
        entries.append(entry)
        entry_num += 1

    # Write to file
    try:
        output_path = config.output_path
        if not output_path.suffix.lower() == ".srt":
            output_path = output_path.with_suffix(".srt")

        content = "\n".join(entries)
        output_path.write_text(content, encoding="utf-8")

        if progress_callback:
            progress_callback(1.0, "Export complete")

        return True, entry_num - 1, skipped

    except (OSError, IOError):
        return False, 0, 0
```

### ui/tabs/render_tab.py additions

```python
# Add signal
export_srt_requested = Signal()

# In _create_content_area(), add to other_layout:
self.export_srt_btn = QPushButton("Export SRT")
self.export_srt_btn.setToolTip("Export sequence metadata as subtitles")
self.export_srt_btn.clicked.connect(self._on_export_srt_click)
other_layout.addWidget(self.export_srt_btn)

# Add handler
def _on_export_srt_click(self):
    """Handle export SRT button click."""
    self.export_srt_requested.emit()

# Update set_sequence_info to enable/disable
self.export_srt_btn.setEnabled(clip_count > 0)

# Update clear() to reset
self.export_srt_btn.setEnabled(False)
```

### ui/main_window.py additions

```python
# Connect signal in _connect_signals:
self.render_tab.export_srt_requested.connect(self._on_export_srt_click)

# Add handler:
def _on_export_srt_click(self):
    """Export sequence metadata as SRT subtitle file."""
    sequence = self.sequence_tab.get_sequence()
    all_clips = sequence.get_all_clips()

    if not all_clips:
        QMessageBox.information(self, "Export SRT", "No clips in timeline to export")
        return

    # Build lookups
    sources = self.sequence_tab.timeline.get_sources_lookup()
    clips_lookup = {}
    for clip in self.clips:
        clips_lookup[clip.id] = clip

    # Fallback to project data
    if not sources and self.sources_by_id:
        sources = dict(self.sources_by_id)

    # Get default filename
    default_name = f"{sequence.name}.srt"
    if sequence.name == "Untitled Sequence" and self.project_metadata:
        default_name = f"{self.project_metadata.name}.srt"

    file_path, _ = QFileDialog.getSaveFileName(
        self,
        "Export SRT",
        default_name,
        "SRT Subtitle Files (*.srt);;All Files (*)",
    )
    if not file_path:
        return

    output_path = Path(file_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".srt")

    config = SRTExportConfig(output_path=output_path)
    success, exported, skipped = export_srt(
        sequence, clips_lookup, sources, config
    )

    if success:
        if exported == 0:
            # All clips skipped - warn user
            result = QMessageBox.warning(
                self,
                "Export SRT",
                f"No clips have the required metadata for this sequence type.\n"
                f"An empty SRT file was created.\n\n"
                f"Skipped: {skipped} clips",
                QMessageBox.Ok,
            )
        else:
            self.status_bar.showMessage(
                f"SRT exported: {exported} entries ({skipped} skipped)"
            )
    else:
        QMessageBox.critical(self, "Export Error", "Failed to export SRT file")
```

### models/sequence.py additions

```python
# Add field to Sequence dataclass:
algorithm: Optional[str] = None  # "storyteller", "color", "shot_type", etc.

# Update to_dict:
if self.algorithm:
    data["algorithm"] = self.algorithm

# Update from_dict:
seq.algorithm = data.get("algorithm")
```

### ui/tabs/sequence_tab.py changes

```python
# When generating sequence, set algorithm on the Sequence object:
def _generate_sequence(self, algorithm: str, clips: list, direction: str = None):
    # ... existing code ...
    sequence = self._timeline.get_sequence()
    sequence.algorithm = algorithm
    # ... rest of generation ...
```
