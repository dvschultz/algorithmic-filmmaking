# Reusing legacy analysis

Older projects may contain analysis values without a record of the inputs or
model that produced them. These values remain visible, but they do not count as
completed analysis until you recompute them or explicitly accept reuse.

For colors, brightness, volume, classification, object detection, gaze, shot types, OCR, descriptions, cinematography, transcription, word alignment, and compatible
DINO thumbnail or boundary embeddings, the CLI provides an explicit reuse command:

```sh
scene_ripper analyze accept-legacy project.json --operation colors --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation embeddings --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation brightness --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation volume --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation classify --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation detect_objects --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation boundary_embeddings --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation gaze --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation shots --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation extract_text --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation describe --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation cinematography --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation transcribe --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation align_words --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation custom_query --query 'Is there a person?' --clip-id CLIP_ID
```

Omit `--clip-id` to process all clips. The command reports accepted and failed
clip IDs and saves accepted decisions. It reads and fingerprints the current
media but does not run inference. Color reuse binds the default five-color
analysis configuration. Embeddings must have a compatible recorded model and
valid vector; an unknown model requires recomputation.
Brightness binds the default five-sample configuration. Zero brightness or
volume is a valid measurement. An absent legacy volume value requires
recomputation because it cannot distinguish missing analysis from no audio.
Classification retains saved label names without inventing confidence scores.
Object detection requires valid detections and a matching person count. Empty
label lists and empty detections with a zero person count can be accepted.
Both operations bind their current default settings and require a readable
thumbnail.
Boundary reuse requires both endpoint vectors, a compatible recorded DINO model,
and a valid source range. The pair is accepted together; a missing or invalid
endpoint requires recomputation. Later trim changes invalidate the decision.
Gaze requires both finite angles and a known category, and binds the default
sampling interval. Missing legacy gaze values require recomputation; they do
not prove that no gaze was detected.
Shot reuse requires a known label and readable thumbnail. It captures the
configured backend/model when the request starts. The desktop uses its current
settings, including unsaved settings changes. Acceptance does not establish the
old label's confidence; changing the selected model invalidates completion.
OCR accepts stored text observations or an explicitly stored empty list. Missing
results and observations outside the clip range require recomputation. It binds
the default three-keyframe OCR configuration and captures the selected fallback
model from current settings before execution.
Description reuse binds the current model, input mode, and default prompt while
preserving the old text, recorded model, and frame count. Missing model or frame
count metadata remains unknown. Empty descriptions, saved error messages, and
invalid frame counts require recomputation. Changing the selected model or
prompt invalidates reuse; acceptance does not claim the old text was generated
using the current settings.
Cinematography reuse validates saved categories, confidence, and metadata and
requires the clip's shot label to agree with the cinematography result. It
preserves the recorded model and analysis mode while binding the decision to
current settings. Missing analysis, invalid values, or conflicting shot labels
require recomputation.

Transcription reuse preserves saved text, segment confidence, and word timings.
An explicitly stored empty transcript can be accepted; missing transcripts and
timings outside the clip or containing segment require recomputation. Acceptance
captures current transcription settings and leaves separate word-alignment
records unchanged. It does not establish that saved words were force-aligned.
Word-alignment acceptance requires saved word timings for every spoken segment.
It preserves their distribution and the separate transcription record. Empty
transcripts are valid; missing transcripts, missing word timings, and timings
outside the clip or containing segment require recomputation. Accepted records
make no claim about which alignment model or fallback produced the words.
Changes to media, editorial text, segment boundaries, or the current alignment
runtime invalidate the decision.
Custom-query reuse requires the exact saved question (`--query` in the CLI or
`query` in agent tools). It accepts the latest answer to that question, including
a negative answer with zero confidence, while preserving the complete history.
Other questions keep their own verification state. Missing answers, invalid
confidence/match values, and unsupported record formats require recomputation.
The desktop dialog shows a question field when Custom query is selected.

An accepted value retains **unknown provenance**. Acceptance records your
decision to use it with the current inputs; it does not establish how the old
value was computed. Changes to media, trim, model, or analysis settings can
invalidate that decision. Verified or failed records cannot be relabelled as
legacy, and unknown future record formats are preserved.

MCP agents can use `start_accept_legacy_analysis` with `project_path`, `operation`,
and optional `clip_ids`, then poll `get_job_status` and `get_job_result`. Invoke
it only after the user explicitly chooses legacy reuse. The job fingerprints
media off the server event loop, supports cancellation, and rejects project or
selected-media changes while queued. Accepted decisions are saved under the
project writer lock.

In the desktop app, select clips in Cut or Analyze and choose **Edit → Reuse
Legacy Analysis…**. Choose an operation, review the explanation,
and click **Reuse selected values**. Media checks run in the background. Cancel
discards pending decisions; changed clips or a replaced project reject late
results. Save the project to keep accepted decisions.

The built-in desktop agent can use `accept_legacy_analysis` with `operation`
and exact `clip_ids` after you explicitly request reuse. It waits for the media
checks and reports accepted and failed IDs. Decisions remain unsaved until you
save the project. Cancelling the request cancels its worker.

Imported audio sources have a separate CLI command:

```sh
scene_ripper analyze accept-legacy-audio project.json --audio-source-id AUDIO_ID
```

Omit `--audio-source-id` to process all imported audio sources. This accepts
stored whole-file transcripts, including explicitly stored silence, using current
transcription settings. Missing transcripts, invalid durations, and timestamps
outside the audio or containing segment require recomputation. Saved text and
word timings remain unchanged, and provenance remains unknown.

MCP agents can call `start_accept_legacy_audio_transcripts` with `project_path`
and optional `audio_source_ids`, then poll the usual job tools. Use it only after
an explicit reuse choice. Settings are captured before queueing; changes to the
project, audio, or transcription runtime while queued reject the request.
Accepted decisions are saved.

In Collect, right-click an imported audio row and choose **Reuse Legacy
Transcript…**. Review the explanation and choose Yes to begin the media checks.
The built-in agent can use `accept_legacy_audio_transcript` with an exact
`audio_source_id` after you explicitly request reuse. Both paths use current
desktop settings and discard cancelled or stale results. Save the project to
persist an accepted desktop decision.

Acceptance for other analysis operations is not available yet. Recompute those
results through their usual analysis commands.
