# Analysis provenance and derived artifacts

The U10 implementation covers color palettes, thumbnail and boundary DINOv2 embeddings,
object detection, OCR, ImageNet and shot classification, and gaze across the desktop, shared spine, CLI, MCP jobs, and embedding
prerequisites used by sequencing. This document describes that bounded scope.
The complete U10 contract remains in the shared editing engine plan.

## Reuse

`AnalysisIdentity` identifies source content, source frame range, operation and
schema versions, model revision and package versions, normalized parameters,
sampling policy, and optional prompt digest. Its key excludes paths and target
IDs so moving unchanged media does not require inference. `AnalysisInput`
separately captures target bindings, paths, and file stamps for queued-work and
owner-thread checks. Workers verify full source hashes before reuse and reject
media changed during computation.

`AnalysisRecord` distinguishes successful, failed, and missing results. A valid
empty value is a successful result. Legacy values have unknown provenance and
do not automatically satisfy migrated operations. Unrecognized record versions
survive save/load without becoming reusable. Existing clip/frame fields remain
read projections for display and consumers that have not migrated.

Colors include the palette size and extraction policy in their identity.
Thumbnail embeddings use the DINOv2 revision declared in
`core/analysis_model_identity.py`; both the loader and the identity use that pin.
A changed thumbnail path invalidates the cheap completion check even when the
previous thumbnail still exists. Content-identical relocated inputs can be
revalidated by a worker without inference.

Semantic reuse and job recovery are separate. Durable jobs still journal
computation before guarded model publication, and GUI results retain their
one-use delivery guards. An artifact-backed embedding can be reused after the
old job cache is removed. Ordinary projects without artifacts or job receipts
do not initialize either cache during save.

Object detection records both successful empty observations and failed attempts.
Its identity distinguishes confidence thresholds and full detection from
people-only counting. Changing those options, the source range, image, or model
requires revalidation. Failed attempts are persisted without successful job
receipts and never satisfy completion checks. A verified result remains reusable
after its old job cache is removed. CLI analysis-image selection validates the
record's source binding before handing the image to worker-side content checks.
GUI delivery carries the entire immutable outcome, including the record, and
checks it against the computation journal or the worker's exact transient result.

OCR records clip video content and frame ranges (including FPS), or individual
frame image content. Its identity includes resolved VLM options, the exact prompt,
sampling policy, and local runtime versions. Empty text observations remain
reusable after job-cache removal. Failed attempts retain the previous displayed
text but invalidate automatic reuse; the next request can retry. GUI and saved
jobs publish those failure records without creating successful job receipts.

ImageNet classification records the MobileNet weight identity, source and image
content, clip range, label limit, and confidence threshold. Reuse restores label
names without inventing confidence scores absent from the saved projection.
Failed refreshes keep displayed labels while invalidating completion. CLI image
binding and cache-independent reuse follow the same rules as object detection.
Combined clip/frame controllers publish successful, reused, and failed records
through their owner-bound applications. They verify transient reuse/failure
outcomes against the worker, and only successful journal receipts enter project
job history. Frame completion checks require provenance for migrated operations.
Reuse checks run with the requested settings, including people-only detection.
Reused embeddings publish refreshed input bindings without a new inference call.
Frame classification depends on its image and can run with the original video
offline.

Shot classification records the pinned SigLIP revision, label vocabulary and
prompts, requested cloud model, and the backend that actually produced the result.
A local fallback remains visible but cannot satisfy a later cloud request.
Verified results remain reusable after job-cache removal. Failed refreshes retain
the displayed shot label and record the failed attempt for retry. Atomic shot
jobs retain their all-or-nothing publication behavior; failed non-atomic jobs and
GUI runs publish failure records without successful receipts. Frame inference
uses its image even when the original source video is offline.

Boundary embeddings identify source content, frame range and FPS, the pinned
DINOv2 runtime, FFmpeg, and first/last-frame sampling. Verified pairs remain
reusable after job-cache removal and source relocation. A damaged pair triggers
targeted recomputation while preserving thumbnail embeddings and editorial data.
Match-cut verifies its private prerequisites and excludes stale vectors when
refresh fails. It cannot relabel thumbnail embeddings from another model.

Gaze records source content, frame range and FPS, sampling interval, the
FaceLandmarker version and detector settings, and the angle-classification
constants. Reuse compares angles at the existing two-decimal project precision.
An empty observation is a successful record and can be reused without loading
the model, including after job-cache removal. The legacy headless response still
reports `no_gaze_detected` for an empty computed observation. Failed refreshes
preserve displayed angles and record the failed attempt; unprocessed items after
a model-load failure remain distinct. GUI failure and reuse delivery uses the
same owner and journal guards as successful results.

## Storage and ownership

The configured cache directory contains `artifacts/`, with a SQLite reference
index and registered payload files. References contain a digest, size, and media
type. Writes stream into staging files, verify content, and pin the registered
payload before publishing its reference in an atomic project save. Thumbnail
and boundary embedding vectors are stored as JSON artifacts rather than repeated in project
JSON. Loading hydrates their read projections. Missing or corrupt payloads mark
only the affected operation missing, preserving editorial notes and sequences.
The shared embedding model label survives damage to either vector family.
Unsaved legacy vectors are staged with unknown provenance. Old display vectors
retained after a failed refresh are also managed, while the record remains failed
through save, load, payload loss, and restoration. GUI save acknowledgments compare
the stored payload and its record rather than requiring inline arrays.

Known saved-project manifests retain payloads after projects close. Live
projects retain current results and source-removal undo history. Detached save
and bundle-export snapshots hold leases until their consumers retire. Unchanged
reference sets do not cause database writes on ordinary project edits. Garbage
collection callbacks use nonblocking pin release and retry after store
transactions, avoiding a wait on a lock held by the same thread.

Collection removes only unreferenced registered files whose inode, content
stamp, and single-link status still match. It preserves source files, unknown
files, replaced files, and conservative pins left by uncertain publication.
Successful replacement saves reconcile pending manifest pins for that path.
Portable bundle export copies and verifies referenced artifacts before writing
the bundle manifest; loading a bundle can restore them into a fresh cache.
Portable records retain semantic identity but discard original file bindings.
A worker revalidates the relocated media before reuse, without inference when
the content is unchanged. Missing thumbnails fail their own embedding target
without stopping analysis of valid neighboring clips.

New durable computed-result receipts externalize spec and payload bodies larger
than 16 KiB into the artifact store beside `jobs.db`. SQLite keeps references and
a durable pin; callers still receive the exact original strings and digests.
Files are staged before the receipt is inserted. Proven duplicate inserts release
their temporary pins, while uncertain publication conservatively retains them.
Reads verify payload checksums, and pending-receipt recovery resolves files after
closing the job database connection. Existing inline receipts and session-only
stores retain their original representation. Receipt pins survive checkpointing
and restart: terminal-job deletion does not establish that a computed receipt is
unreferenced by saved projects. Receipt pruning and reconciliation of abandoned
publication pins remain to be implemented.

New large job arguments and operation specifications share a managed input body;
large final/intermediate results use a separate body. Job reads hydrate the
original JSON columns without changing safe status projections. Readers acquire
temporary ownership while the job row is locked, then load files outside the job
transaction. This prevents deletion or result replacement from collecting a file
mid-read. Successful history deletion and terminal-job pruning release these
owners after committing the database change. Result replacement releases the old
body, and rejected updates or duplicate inserts release their unpublished bodies.
Computed receipts have separate owners and survive history pruning. Uncertain
publication or deletion retains files conservatively; abandoned-pin reconciliation
remains outstanding. Session-only histories keep their JSON entirely in memory.

New job inputs, operation specifications, results, and computed receipts retain
referenced artifacts as well as their serialized bodies. Reference discovery
includes structured data inside embedded `*_json` documents, while ordinary user
strings remain text. This applies to inline and external bodies alike. Queued and
running jobs keep inputs available after the producing project retires; result
replacement and history deletion release their own references. Session-only jobs
hold temporary pins on existing managed files and release them when the session
store closes, without persisting job arguments. The store retains its selected
artifact root for those releases if settings subsequently change. Existing stored
rows are not eagerly backfilled with the new reference ownership.

Continuous sequence previews now use a registered media cache. Worker-side
identity includes source/music/still content, timeline settings, and the FFmpeg
runtime stamp. Rendering uses a private staging directory and publishes only a
successful, nonempty output after checking that the inputs stayed unchanged.
Cache hits verify the managed payload; damaged output is recomputed. The UI
accepts worker-verified results and checks input/output file stamps before reuse,
instead of trusting a nonempty file at a predicted path.

The preview cache retains its latest entries per sequence. Pruning releases only
registered owners and uses managed collection; unknown and legacy MP4 files are
preserved and do not qualify for automatic reuse. Reader leases bridge lookup,
worker completion, active preview selection, and playback. The video player also
retains its last preview lease until another preview replaces it or the player is
destroyed, so invalidating the timeline does not remove a file still loaded by
MPV. Uncertain cache publication remains conservatively retained.

Transformed-clip prerender production also uses verified managed media. Its key
includes source content, exclusive source-frame range and exact rate, requested
and applied transforms, reverse safety policy, encoder settings, and FFmpeg runtime
stamp. Editor clip IDs are not content identity. Batches share a thread-safe
fingerprint cache, so each unchanged source is read once per batch. Rendering
stages output privately and rejects failed, empty, source-changed, or runtime-changed
results. Legacy filename-only outputs are preserved without automatic reuse.
Project saves copy managed prerenders instead of hard-linking them, preventing
project-local edits from changing shared cache bytes.

Sequence entries now carry explicit prerender artifact references. Managed paths
are read projections resolved after checksum verification; missing media keeps its
reference and leaves the path unavailable. Malformed references remain opaque and
round-trip without preventing the rest of the project from loading. Unregistered
legacy prerenders retain their original path-based representation.

Live projects retain references in every sequence, sequence collection history,
timeline placement history, and reversible source removal. Trim/transform edits
clear both the path and reference while retaining the previous pair for undo.
Saved manifests retain media after projects close. Save snapshots and bundle
exports acquire leases, and bundles restore prerenders into a fresh artifact store.
Batch results retain produced files until sequence binding or explicit retirement;
their worker pool prunes registered cache entries while those leases are active.
Consumers of the low-level path-returning helper must use a retained batch or
attach a reference before unrelated cache eviction. Ordinary project saving no
longer duplicates managed prerenders into the legacy transformed-clips folder.

## Description execution metadata

Description providers now report their actual model/backend and frame/video input
mode through an optional execution callback. A local Qwen-to-Moondream fallback
returns the Moondream model name, using the same model-resolution rule as the
loader and durable job runtime snapshot. Cloud video extraction failures report
frame execution, and metadata is available before inference so failed attempts
can retain their execution identity.

The shared description operation and direct headless entry point now use verified
records. Their identity covers image/source contents, clip range and frame rate,
prompt digest, requested model/input mode, actual provider execution, installed
runtime versions, and video-extraction runtime when applicable. Parallelism is
excluded. Relocating identical files reuses the value and refreshes its bindings;
changed media, prompts, models, or read projections require computation. A video
request that fell back to frame input does not satisfy a later video request.
Failures retain the displayed description while marking the record failed.
Publication rejects changed media, prior values/records, and target/session bindings.

GUI worker task construction/delivery, durable headless job receipts, completion
projections, and saved-result checkpoints still need migration. Those paths retain
their existing reuse rules; the description family is not yet complete.

## Remaining U10 work

- Migrate the other U7 analysis families to
  semantic reuse. Extend operation-owned failure records beyond object detection,
  OCR, ImageNet and shot classification, gaze, and boundary embeddings.
- Expose the explicit legacy-reuse decision through user and agent flows; the
  current record model supports the decision but the flows are not wired.
- Finish the cross-consumer reuse/projection audit, including legacy prerender playback.
  Add safe computed-receipt pruning and audit legacy job-row ownership;
  existing rows are not migrated eagerly.
- Complete end-to-end retention and recovery coverage for those additional
  consumers. Abandoned save pins without a later replacement remain retained.
