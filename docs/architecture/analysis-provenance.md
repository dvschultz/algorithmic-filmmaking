# Analysis provenance and derived artifacts

The U10 implementation covers color palettes, thumbnail DINOv2 embeddings,
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
the stored payload and its record rather than requiring inline arrays. Boundary
array storage is implemented; boundary analysis reuse still needs migration.

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

## Remaining U10 work

- Migrate the other U7 analysis families, including boundary embeddings, to
  semantic reuse. Extend operation-owned failure records beyond object detection,
  OCR, ImageNet and shot classification, and gaze.
- Expose the explicit legacy-reuse decision through user and agent flows; the
  current record model supports the decision but the flows are not wired.
- Move preview/prerender media and durable job array payloads into managed
  storage, with pins spanning execution, playback, export, and recovery.
- Complete end-to-end retention and recovery coverage for those additional
  consumers. Abandoned save pins without a later replacement remain retained.
