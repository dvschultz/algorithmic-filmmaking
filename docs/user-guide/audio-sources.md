# Audio Sources

Scene Ripper lets you import audio files (music, podcasts, voiceovers) into a project alongside your videos. Audio sources are **not** cut into clips and never appear in the sequencer output — they exist to feed audio-consuming tools like the Staccato sequencer and audio transcription.

## Supported formats

`.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, `.ogg`

## Importing audio

You have three ways to bring an audio file into a project:

1. **Click "Import Audio…"** in the Collect tab toolbar and pick one or more audio files in the file dialog.
2. **Drag and drop** audio files onto the Collect tab. Files with audio extensions are routed to the audio library; everything else flows to the video import path as before.
3. **Pick a new file from inside the Staccato dialog** — the dialog's "Import new…" item runs the same import flow and auto-selects the new audio source.

Imported audio is referenced in place — Scene Ripper does not copy the audio file into the project directory. If you move or delete the file on disk, the audio source becomes unusable until the file is restored.

## The audio library

Imported audio files appear as rows in the **Audio Library** section at the bottom of the Collect tab. Each row shows:

- **Filename**
- **Duration** (formatted as `MM:SS` or `H:MM:SS`)
- **Transcribe** button — runs Whisper transcription on the audio (see below). Becomes a disabled "Transcribed" badge once transcripts exist; hover for the segment count.
- **Remove** button — removes the audio source from the project. The underlying file on disk is not deleted.

## Using audio sources

### Staccato sequencer

The Staccato sequencer uses a project audio source as its music track. When you open the Staccato dialog, the audio picker is populated from your project's audio library, so you don't have to re-select the same file every session. If the project has no audio sources yet, the dialog shows an "Import new…" item that runs the import worker and auto-selects the imported audio.

### Audio transcription

Click the **Transcribe** button on any audio library row to run Whisper on the file. Scene Ripper uses the same transcription backend it uses for video clips (faster-whisper, mlx-whisper, or Groq cloud — whichever is configured in Settings). The resulting transcript segments are stored on the audio source and persist across project save/load.

Transcribing an audio source does **not** transcribe its clips, since audio sources have none. The transcript belongs to the audio source itself.

## Persistence

Audio sources are saved as part of the project file (`.sceneripper`). They round-trip cleanly through save/load and survive across sessions. Older project files that predate audio sources continue to load — they just have an empty audio library.

## Agent and MCP access

The chat agent and external MCP clients can interact with audio sources via:

- `list_audio_sources` — list all audio sources in the project
- `get_audio_source(audio_source_id)` — full record including transcript segments
- `import_audio_source(file_path)` — synchronously import an audio file (chat tool only)

See [Agent Tools Reference](agent-tools.md) for the full list.

## What's not yet supported

The following are intentionally deferred and may land in a future release:

- **Stem separation as a standalone tool.** Stem separation is currently embedded inside the Staccato dialog's flow.
- **Sequence music selection by audio source ID.** The export-time music mux still takes a raw path — it does not yet consume an audio source.
- **YouTube audio-only download.** All YouTube downloads currently include video.
- **Waveform thumbnails in the audio library list.** A generic icon is used for now; per-file waveforms are still rendered inside the Staccato dialog.
