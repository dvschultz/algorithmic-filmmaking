# Export Project Feature

**Date:** 2026-02-28
**Status:** Brainstorm complete

## What We're Building

A project export feature that bundles a Scene Ripper project file and all its referenced assets into a self-contained folder. The exported folder can be opened on another machine (or by another person) and the project loads seamlessly with all data intact.

### Core Behavior

- Creates a structured folder with the project file and organized subdirectories for assets
- Copies source videos, thumbnails, and extracted frames into the bundle
- Rewrites paths in the exported project file to point to the bundled assets
- Option to exclude source videos for a lightweight export (metadata + thumbnails only)
- Opening an exported project file works seamlessly — relative paths resolve from the bundle folder

### Bundle Structure

```
<ProjectName>-export/
├── project.srp              # Project JSON file (paths rewritten)
├── sources/                 # Source video files (optional)
│   ├── video1.mp4
│   └── video2.mp4
├── thumbnails/              # All referenced thumbnails
│   ├── src_thumb1.jpg
│   └── clip_thumb1.jpg
└── frames/                  # Extracted frames (if any)
    └── frame001.png
```

### Lightweight Export (No Videos)

When the user opts to exclude source videos:
- The `sources/` directory is omitted
- Thumbnails and frames are still included
- When opened, a dialog prompts the user to re-link missing source files
- Leverages the existing `missing_source_callback` in `load_project()`

## Why This Approach

**Subdirectory structure over flat or mirrored:**
- Clean and inspectable — users can see exactly what's in the bundle
- Predictable layout regardless of where original files lived
- Avoids deep nested paths from scattered source locations
- Avoids filename collisions that a flat structure would require renaming to solve

**Plain folder over ZIP:**
- Easy to inspect and modify without extraction
- No compression overhead for large video files (which are already compressed)
- Simpler implementation — no archive library dependency

**Relative paths (existing pattern):**
- Project files already store paths relative to the project file's parent directory
- Placing assets alongside the project file in known subdirectories makes "open and it works" trivial
- No special "bundle mode" detection needed in the loader

## Key Decisions

1. **Export format:** Plain folder with organized subdirectories (`sources/`, `thumbnails/`, `frames/`)
2. **Video inclusion:** User chooses per-export — full bundle (with videos) or lightweight (without)
3. **Path strategy:** Rewrite all paths in the exported project file to be relative to the bundle folder
4. **Re-import:** Seamless — opening the project file from the bundle folder resolves paths naturally
5. **Missing sources:** Prompt-to-relink dialog when opening a lightweight export with missing videos
6. **Use cases:** Both archival/backup and sharing with collaborators

## Scope

### In Scope

- Export action accessible from the UI (menu or toolbar)
- Folder creation with subdirectory structure
- Copying source videos (optional), thumbnails, and frames
- Rewriting project file paths for the bundle
- Progress indication for large exports (video copying can be slow)
- Handling filename collisions (e.g., two sources named `video.mp4` from different directories)
- Validation that referenced assets exist before starting export

### Out of Scope

- ZIP/archive compression
- Incremental or differential exports
- Cloud upload or network transfer
- Project merge (combining two exported projects)
- CLI export command (GUI only for now)

## Open Questions

None — all key decisions resolved during brainstorming.
