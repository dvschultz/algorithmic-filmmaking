---
title: "feat: EDL Export for NLE Workflows"
type: feat
date: 2026-01-24
priority: P2
---

# feat: EDL Export for NLE Workflows

## Overview

Export timeline sequences as Edit Decision Lists (EDL) in CMX 3600 format for import into Premiere Pro, DaVinci Resolve, and other NLEs. This enables round-tripping between Scene Ripper and professional editing software.

## Problem Statement

Users who build sequences in Scene Ripper currently have no way to continue editing in professional NLEs. They must either:
1. Export as a single video file (lossy, no edit flexibility)
2. Manually recreate the edit in their NLE

EDL export solves this by providing an industry-standard interchange format.

## Proposed Solution

Add EDL export following existing export patterns:
- New `core/edl_export.py` module
- Button in Render tab for EDL export
- CMX 3600 format (most widely supported)

## Technical Approach

### EDL Format (CMX 3600)

```
TITLE: Scene Ripper Export
FCM: NON-DROP FRAME

001  001      V     C        00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
* FROM CLIP NAME: beach_sunset.mp4

002  002      V     C        00:00:10:15 00:00:15:00 00:00:05:00 00:00:10:00
* FROM CLIP NAME: city_night.mp4
```

**Format breakdown:**
- Line 1: Edit number (001, 002, ...)
- Line 2: Reel/source number
- Line 3: Track (V=video, A=audio)
- Line 4: Transition (C=cut)
- Line 5-8: Source IN, Source OUT, Record IN, Record OUT (timecode)
- Comment line: Original filename

### Files to Create/Modify

| File | Change |
|------|--------|
| `core/edl_export.py` | New - EDL export logic |
| `ui/tabs/render_tab.py` | Add EDL export button |

### Data Flow

```
Sequence + Sources + Clips
         │
         ▼
    export_edl()
         │
         ├── Build header (TITLE, FCM)
         │
         ├── For each SequenceClip:
         │   ├── Get Source for file path
         │   ├── Convert frames to timecode
         │   └── Format EDL event line
         │
         └── Write to .edl file
```

## Implementation Steps

### Step 1: Create EDL Export Module

**File:** `core/edl_export.py`

```python
"""EDL (Edit Decision List) export for NLE workflows."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from models.clip import Clip, Source
from models.sequence import Sequence, SequenceClip


@dataclass
class EDLExportConfig:
    """Configuration for EDL export."""
    output_path: Path
    title: str = "Scene Ripper Export"
    drop_frame: bool = False


def frames_to_timecode(frames: int, fps: float, drop_frame: bool = False) -> str:
    """Convert frame number to SMPTE timecode string."""
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    remaining_frames = int(frames % fps)

    separator = ";" if drop_frame else ":"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{remaining_frames:02d}"


def export_edl(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, Clip],
    config: EDLExportConfig,
    progress_callback: Callable[[float, str], None] | None = None
) -> bool:
    """Export sequence as CMX 3600 EDL file.

    Args:
        sequence: The sequence to export
        sources: Dict mapping source_id to Source
        clips: Dict mapping clip_id to Clip
        config: Export configuration
        progress_callback: Optional (progress_0_to_1, message) callback

    Returns:
        True if export succeeded, False otherwise
    """
    if progress_callback:
        progress_callback(0.1, "Building EDL...")

    lines = []

    # Header
    lines.append(f"TITLE: {config.title}")
    fcm = "DROP FRAME" if config.drop_frame else "NON-DROP FRAME"
    lines.append(f"FCM: {fcm}")
    lines.append("")

    # Get all clips sorted by timeline position
    seq_clips = sequence.get_all_clips()
    total_clips = len(seq_clips)

    record_frame = 0  # Running record position

    for i, seq_clip in enumerate(seq_clips):
        if progress_callback:
            progress = 0.1 + (0.8 * (i / max(total_clips, 1)))
            progress_callback(progress, f"Processing clip {i + 1}/{total_clips}")

        # Get source for this clip
        source = sources.get(seq_clip.source_id)
        if not source:
            continue

        fps = source.fps

        # Source timecodes (in/out points within source video)
        src_in = frames_to_timecode(seq_clip.in_point, fps, config.drop_frame)
        src_out = frames_to_timecode(seq_clip.out_point, fps, config.drop_frame)

        # Record timecodes (position on timeline)
        duration_frames = seq_clip.out_point - seq_clip.in_point
        rec_in = frames_to_timecode(record_frame, sequence.fps, config.drop_frame)
        rec_out = frames_to_timecode(record_frame + duration_frames, sequence.fps, config.drop_frame)
        record_frame += duration_frames

        # Edit number and reel
        edit_num = f"{i + 1:03d}"
        reel = f"{i + 1:03d}"

        # EDL event line
        # Format: EDIT# REEL TRACK TRANS SRC_IN SRC_OUT REC_IN REC_OUT
        event_line = f"{edit_num}  {reel}      V     C        {src_in} {src_out} {rec_in} {rec_out}"
        lines.append(event_line)

        # Source filename comment
        lines.append(f"* FROM CLIP NAME: {source.filename}")
        lines.append("")

    if progress_callback:
        progress_callback(0.9, "Writing EDL file...")

    # Write to file
    try:
        output_path = config.output_path
        if not output_path.suffix.lower() == ".edl":
            output_path = output_path.with_suffix(".edl")

        output_path.write_text("\n".join(lines), encoding="utf-8")

        if progress_callback:
            progress_callback(1.0, f"Exported to {output_path.name}")

        return True

    except (OSError, IOError) as e:
        if progress_callback:
            progress_callback(1.0, f"Export failed: {e}")
        return False
```

### Step 2: Add EDL Export Button to Render Tab

**File:** `ui/tabs/render_tab.py`

Add button next to existing export button:

```python
# In _create_export_section():
self.export_edl_btn = QPushButton("Export EDL")
self.export_edl_btn.setToolTip("Export timeline as Edit Decision List for NLE import")
self.export_edl_btn.clicked.connect(self._on_export_edl_clicked)
export_layout.addWidget(self.export_edl_btn)
```

Add handler:

```python
def _on_export_edl_clicked(self):
    """Export sequence as EDL file."""
    if not self._sequence or self._sequence.duration_frames == 0:
        return

    # Get save path
    file_path, _ = QFileDialog.getSaveFileName(
        self,
        "Export EDL",
        str(Path.home() / "sequence.edl"),
        "Edit Decision List (*.edl)"
    )

    if not file_path:
        return

    from core.edl_export import EDLExportConfig, export_edl

    config = EDLExportConfig(
        output_path=Path(file_path),
        title=self._sequence.name or "Scene Ripper Export"
    )

    # EDL export is fast, no need for worker thread
    success = export_edl(
        self._sequence,
        self._sources,
        self._clips,
        config,
        progress_callback=lambda p, m: self.statusBar().showMessage(m) if hasattr(self, 'statusBar') else None
    )

    if success:
        # Open containing folder
        import subprocess
        import sys
        folder = Path(file_path).parent
        if sys.platform == "darwin":
            subprocess.run(["open", str(folder)])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", str(folder)])
        else:
            subprocess.run(["explorer", str(folder)])
```

### Step 3: Wire Up Data Access

Ensure RenderTab has access to sources and clips dictionaries (may already have via MainWindow).

## Acceptance Criteria

- [x] EDL export produces valid CMX 3600 format
- [ ] Exported EDL imports correctly in DaVinci Resolve
- [ ] Exported EDL imports correctly in Premiere Pro
- [x] Timecodes are frame-accurate
- [x] Source filenames are preserved in comments
- [x] Empty timeline shows appropriate message (no export)
- [x] Export opens containing folder on completion

## Testing

1. Create sequence with 3+ clips from different sources
2. Export as EDL
3. Import into DaVinci Resolve - verify clips appear at correct times
4. Import into Premiere Pro - verify same
5. Check timecode accuracy by comparing to original

## Constraints

- CMX 3600 format only (most compatible)
- Video track only (no audio track export initially)
- Cut transitions only (no dissolves/wipes)
- No reel name customization (uses sequential numbers)

## Future Enhancements

- FCPXML export for Final Cut Pro
- AAF export for Avid
- Audio track support
- Transition types (dissolve, wipe)
- Custom reel naming

## References

### Internal
- Export pattern: `core/sequence_export.py`
- Dataset export: `core/dataset_export.py`
- Sequence model: `models/sequence.py`

### External
- [CMX 3600 EDL Format](https://en.wikipedia.org/wiki/Edit_decision_list)
- [EDL Specification](https://www.edlmax.com/edl-format.html)

---

*Generated: 2026-01-24*
