# Reusing legacy analysis

Older projects may contain analysis values without a record of the inputs or
model that produced them. These values remain visible, but they do not count as
completed analysis until you recompute them or explicitly accept reuse.

For colors, brightness, volume, classification, object detection, and compatible
DINO thumbnail or boundary embeddings, the CLI provides an explicit reuse command:

```sh
scene_ripper analyze accept-legacy project.json --operation colors --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation embeddings --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation brightness --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation volume --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation classify --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation detect_objects --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation boundary_embeddings --clip-id CLIP_ID
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

Acceptance for other analysis operations is not available yet. Recompute those
results through their usual analysis commands.
