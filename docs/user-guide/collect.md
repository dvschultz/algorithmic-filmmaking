# Collect Tab

The **Collect** tab is your source video library. It's where you import videos (local files or YouTube downloads), see which have been cut into clips, and manage what's in the project.

You can also import **audio files** here — they appear in a separate Audio Library section below the videos. Audio sources feed the Staccato sequencer and audio transcription. See [Audio Sources](audio-sources.md) for details.

## Source thumbnails

Each imported video appears as a thumbnail card. The filename is shown below the image. Status badges below the filename indicate what's been done with the source.

### Status badges

| Badge | Meaning |
|-------|---------|
| (no badge) | The source has been imported but not processed |
| **CUT** | Scene detection has been run — the source has been split into clips |
| **ANALYZED** | At least one analysis operation (colors, shots, describe, etc.) has been run on the source's clips |
| **CUT + ANALYZED** | Both steps completed |

A source moves from (none) → CUT → CUT + ANALYZED as you work through the pipeline.

## Selecting sources

- **Click** a thumbnail to toggle selection
- **Cmd+Click** additional thumbnails to add to the selection
- **Marquee drag** in the empty space of the grid to rubber-band-select multiple sources
- **Shift+Marquee** to add to the existing selection

Selected thumbnails are highlighted. The selection feeds into the **Cut N Videos** button at the top, which runs scene detection on everything selected.

## Deleting sources

Right-click any thumbnail (or press **Delete** / **Backspace** on a focused thumbnail) to open a context menu with a "Delete [filename]" option.

If multiple sources are selected when you right-click, the menu offers to delete all of them in one action.

### What gets deleted

Deleting a source removes:

- The source itself from the project
- All clips derived from it (in the Cut and Analyze tabs)
- All frames extracted from it (in the Frames tab)

The underlying video file on disk is **not** touched — only the project's reference to it.

### Sequence guard

If any of the source's clips are used in a sequence, deletion is blocked. Scene Ripper shows an error listing which sequences contain the clips so you know what to clean up first:

> Cannot delete "my-video.mp4": clips are used in Chromatics, Tempo Shift. Delete those sequences first.

Delete the blocking sequence(s) from the Sequence tab first, then retry the source deletion.

### Confirmation dialog

For unguarded deletions, a confirmation dialog appears: "Delete "[name]" and its N clips? This cannot be undone." This is especially important for populated sources — deletion removes all derived work (descriptions, colors, transcripts, etc.).

## Adding sources

- **Drag & drop** video files onto the grid
- Use **Add Video** in the toolbar for a file picker
- Use the **YouTube** search panel to download from YouTube
- Click **Cut All New Videos** to run scene detection on any CUT-less sources

See [Analysis Operations](analysis.md) for what you can do with clips once videos are cut.
