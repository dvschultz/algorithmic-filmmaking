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

GUI workers and clip/frame delivery now carry the complete verified outcome.
Combined analysis sends populated descriptions through worker verification.
Successful computations are journaled before publication; verified reuse and
failures use exact transient-outcome guards without inventing successful receipts.
Saved records support reuse after the receipt cache is lost. Save checkpoints
require the exact published analysis record as well as its displayed fields.
Local model loading occurs after reuse verification and checks media again before
inference, so a valid local description does not require loading model weights.

Durable headless jobs now use the same verified records. Missing old receipt rows
do not prevent semantic reuse; present corrupt receipts are rejected. Failed-save
recovery preserves computed records, and failed attempts publish failure records
without successful receipts. A committed frame fallback cannot satisfy a later
video request. GUI and headless receipt identities exclude parallelism, so changing
scheduling does not repeat completed inference after a failed save. Job and GUI
computation share their full-content fingerprint cache across clips.

CLI high-resolution analysis images remain separate from display thumbnails.
Later jobs may select that recorded image while its source/range/frame-rate binding
remains current, then verify its content and semantic identity before reuse.
Description completion indicators now require the verified record, current default
prompt/model/input mode, runtime metadata, displayed fields, and media bindings.
The picker and quick-run menu pass current source lookups, including frame rate;
missing source context cannot establish completion. UI checks do not import VLM
runtimes. An installed but unprobed local backend remains available for worker
verification, and a source lookup refresh updates quick-run availability.

## Custom-query validation checkpoint

Custom queries now distinguish an explicit negative answer from an unparseable
response. The provider parser requires a leading yes/no decision and validates
explicit confidence percentages. The shared operation rejects invalid match,
confidence, and model values before publication. Local fallback results report
the actual Moondream model. GUI and durable receipt identities include the response
parser version, preventing reuse of receipts from the earlier permissive parser.
Existing saved query history remains intact.

Shared custom-query operations and the direct headless entry point now keep one
record per trimmed, case-sensitive query in the existing analysis-record map. The
key is `custom_query:<sha256(query)>`; the identity also retains the query and full
prompt digest, media content, range/frame rate, model/runtime, and parser version.
Each record verifies the latest history entry for its own query. Other queries do
not invalidate it. Verified reuse refreshes bindings without appending duplicate
history; an explicit repeat request still computes and appends. Reuse remains
opt-in, preserving the existing query-request contract.

Failures update only that query's record and preserve all history. Invalidated
media or settings require recomputation. Per-query records survive project save/load,
while old aggregate query projections remain provenance-unknown.

GUI custom-query workers now capture those records for clip and AnalysisTarget
inputs, and queued delivery preserves successful, reused, and failed records on
the project owner thread. Reuse is verified in the worker before local model
loading; local loading and inference stay on the same worker thread. Frame
custom-query storage remains unsupported.

The GUI journal checks full runtime and media identity, carries records through
failed-save recovery, and ignores scheduling parallelism when matching receipts.
Missing old receipt rows do not prevent verification of saved project records;
present corrupt rows still fail validation. Reused and failed outcomes use an
exact delivery guard without creating successful computation receipts. Explicit
save acknowledges a query receipt only when both its history prefix and its
per-query record are present. A superseded record's receipt stays uncommitted and
retained.

Durable headless query jobs now publish the same verified records and save failed
attempts without adding successful receipts or changing query history. Receipt
identity includes the full runtime, previous query history, and previous per-query
record; scheduling parallelism is excluded. A failed project save recovers the
cached answer and its record. A failed receipt checkpoint is reconciled only when
the saved history and record match; a later explicit request computes and appends
again. Missing old receipt rows do not block new requests, while corrupt present
rows still fail validation. Saved headless records can be reused by the direct
query operation with `skip_existing=True`. Public job results keep their existing
query-answer fields. The remaining cross-consumer and legacy-reuse audit applies
to custom queries as well; U10 remains incomplete.

## Cinematography execution checkpoint

The provider can now report its actual model, backend, and input mode before
inference. Video extraction fallback reports a subsequent frame attempt, including
when that attempt fails. Authentication errors retain the video attempt and do
not trigger fallback. Local execution reports the selected cinematography model
after confirming MLX availability.

The shared operation and direct headless entry point now record media content,
clip range/frame rate, model and package identity, prompt/schema digests, sampling,
and actual execution mode. A record verifies both the rich analysis and its derived
shot type. Clip and frame tasks can reuse verified records; parallelism changes do
not invalidate them, and local reuse does not enter inference. Video requests do
not reuse a previous frame fallback. Failed attempts retain their actual execution
path, replace verification state, and preserve the earlier display values.

Owner-thread publication checks the prior record, displayed values, media stamps,
editorial bindings, requested settings, and prompt digest. Legacy deliveries clear
prior verification rather than implicitly carrying it forward.

GUI clip/frame workers now capture those records and perform reuse checks on the
worker thread. Construction no longer probes MLX on the UI thread. Queued delivery
and the combined frame pipeline preserve success, reuse, and failure records;
reused and failed outcomes have exact delivery guards without successful receipts.
GUI recovery checks full media/runtime identity, excludes parallelism from receipt
matching, tolerates missing old receipts, and requires the exact record at explicit
save. The combined pipelines queue existing cinematography for verification.
Durable headless jobs now carry the records through result receipts and project
saves. Legacy or stale display values require recomputation; valid project records
can be reused even after old receipt rows are removed. Identical media with new
timestamps refreshes record bindings without inference. Failures preserve previous
display values while saving failed verification state. Forced refreshes publish as
one batch and recover from save/checkpoint failures without repeated inference;
parallelism is excluded from receipt matching. A frame fallback does not satisfy a
later video request.

Completion checks now compare the current media/source binding, settings, runtime,
prompt/schema, sampling, and both displayed values. The picker, quick-run menu, and
MCP cinematography status use verified completion. Missing source context and an
installed but unprobed local runtime cannot establish completion; UI checks do not
probe or load VLM runtimes. Clip and frame completion use their actual input mode.
The remaining cross-consumer and legacy-reuse audit still applies; U10 remains
incomplete.

## Transcription extraction checkpoint

Clip transcription and whole-video MLX/Groq transcription now raise an error
when FFmpeg extraction fails or produces an empty file. Confirmed media without
audio still produces a valid empty transcript. Temporary extraction files are
removed on failure, and whole-video MLX model loading starts only after successful
extraction. Saved jobs report extraction failure without publishing an empty
transcript or a successful receipt.

The focused transcription, audio transcription, recovery, GUI recovery, and import
regression run passed 98 tests. This corrects the distinction between failed
extraction and successful silence before the record migration.

Clip and whole-video providers now expose an optional execution callback reporting
the resolved backend and actual model before extraction or model loading. This
includes MLX-to-faster-whisper fallback and MLX model-name mapping. Confirmed
video-only media reports an audio-probe result without resolving or loading a
model. Groq model selection is frozen before extraction; callers can provide an
explicit cloud model for a queued request. Provider callbacks and the API request
use that same selection even if settings change during the call.

The provider regression run passed 135 tests, followed by 15 execution tests with
expanded auto-backend and extraction-failure coverage. Scoped provider typing and
Ruff passed.

Shared transcription options now include the frozen Groq model. Clip and audio
job specifications, GUI workers and recovery journals, and shared computation
carry that selection to the provider. A batch snapshots settings once, so later
clips cannot silently switch cloud models. Retrying a failed save with the same
frozen model reuses the computed result; a new request selecting another model
does not recover that result. The combined regression run, including all MCP
tests, passed 312 tests; scoped typing passed for all seven changed modules.
Shared clip transcription and the direct headless operation now support verified
records. The identity includes media content, frame range/frame rate, resolved
backend and actual model, provider packages, FFmpeg/ffprobe identity, language,
segmentation, and inference configuration. Batches share media hashes. Verified
empty transcripts are reusable; old display values require recomputation. New
timestamps on identical media refresh bindings without inference. Model and
settings changes invalidate reuse, while parallelism does not.

Publication checks the prior transcript/record, source and clip objects, range,
media stamps, requested settings, and exact detached input binding. Failures retain
the earlier displayed transcript and publish failed verification state. Malformed
inference output is a failure; valid negative segment log probabilities remain
supported. Legacy raw delivery clears verification. Runtime or media changes
during computation discard the result.

The migration regression run passed 357 tests including MCP coverage, with 74
follow-up tests after guard refinements and 26 final record tests including shared
hashing and binding refresh. Scoped operation typing and changed-file Ruff passed.
GUI clip workers and journals now carry verified transcript records, including
the combined analysis pipeline. Existing transcripts are queued for verification;
preflight and model loading run only after a reuse miss. One preflight serves the
batch, and MLX preload/inference remain on the same worker thread. Journals verify
runtime identity, exclude parallelism from recovery matching, tolerate missing old
receipts, and close their stores on all exit paths. Queued object outcomes retain
records; reuse and failures have exact transient-delivery guards without success
receipts. Delivery rejects cancellation, replaced requests, changed inputs, and
modified payloads. Save checkpoints require the exact transcript record.
The GUI regression run passed 392 tests including MCP and combined-pipeline
coverage; 11 follow-up worker/delivery tests passed, including MLX thread affinity.
Scoped typing passed for six migrated modules, and changed-file Ruff passed.

Durable headless jobs now persist transcript records with their result receipts.
Reuse verifies current media/range/runtime/options and the displayed transcript;
missing old receipt rows do not prevent reuse of a valid project record. Present
corrupt receipts still fail validation. Identical media with new timestamps
refreshes bindings without inference. Failure publication uses the guarded
application and preserves the prior displayed transcript without creating a
successful receipt.

Forced refreshes stage one batch and recover matching transcript/record pairs
after save or checkpoint failure. Parallelism does not affect receipt identity.
Manually edited managed transcripts still require force. The CLI now sends
populated transcripts through verification, and generic analysis recomputes a
transcript when the requested model differs. Public result fields remain stable.
Clip transcription completion now checks the current source binding, range, media
stamps, settings, runtime, and displayed value against its verified record. The
picker and quick-run availability paths use that check, and MCP transcript status
counts verified empty transcripts as complete. Confirmed no-audio records remain
complete for unchanged inputs without re-running ffprobe. Completion does not
load models or probe media. The completion regression run passed 249 tests, with
34 focused follow-up tests and clean scoped typing/Ruff.
Standalone audio, alignment, and the remaining cross-consumer audit still need
migration or verification.
The durable transcription regression run passed 273 tests including CLI, combined
analysis recovery, and all MCP tests. Scoped job typing and changed-file Ruff
passed. This does not complete the remaining transcription consumers or U10.

## Standalone audio transcription checkpoint

The shared audio operation can now capture verified tasks and publish success,
reuse, and failure records. Identity includes whole-file content, audio metadata,
actual backend/model, decoder or extraction path, package/binary identity, and
transcription settings. Whole-file decoding is distinct from clip extraction.
Verified empty transcripts can be reused; changes to media, metadata, settings,
or displayed text invalidate reuse. Cancellation and stale delivery do not publish
verification, and failed refreshes preserve the displayed transcript.

`Project.set_audio_transcript()` accepts a matching record and publishes it with
the transcript before notifying observers. It rejects mismatched records before
mutation. Raw callers continue to create unknown-provenance values. The audio
task factory retains its legacy default for unmigrated consumers. GUI workers
and durable audio jobs now request verified tasks.

The regression run passed 294 tests including audio GUI, ownership, agent, project,
and MCP coverage. Changed-file Ruff passed. Scoped operation typing is clean;
the existing `Project.record_analysis()` mapping typing error was confirmed
against the unchanged HEAD version.

GUI audio journals now use semantic transcription settings and whole-file runtime
identity, retain successful records for recovery, and authenticate transient reuse
and failure outcomes before owner-thread publication. Save acknowledgement checks
both transcript and record. The launcher verifies existing transcripts instead of
rejecting them based on field presence. Audio card/completion surfaces still need
migration. The GUI regression run passed 298 tests, including real queued success,
reuse, failure, cancellation, and altered-payload delivery; scoped typing passed
for all three changed worker/journal modules.

Durable audio jobs use version 2 specifications and semantic record identities.
Valid transcripts (including silence) reuse without inference; legacy values or
changed settings/media/metadata recompute. A manually edited managed transcript
requires force. Failure records preserve existing text. Recovery matches both the
saved transcript and its record; failed forced-refresh saves reuse their computed
receipt, while a successful refresh advances the generation. Missing historical
receipt rows do not invalidate an independently verified project record, but
present corrupt rows still fail validation. Cancellation during reuse does not
report success. The durable regression run passed 308 tests, plus 31 focused
follow-ups; scoped typing and changed-file Ruff passed.

Audio completion now checks the current record's input binding, whole-file range,
runtime, settings, and displayed transcript without hashing media, probing, or
running inference. Confirmed silence remains complete. The shared audio listing's
`transcribed` flag and the GUI agent's immediate skip use this predicate; legacy
transcripts remain readable but pending verification. Audio rows use the same
predicate, keep verification clickable, and refresh after settings changes. The
completion regression run passed 277 tests, with 12 focused follow-ups and clean
scoped typing. Alignment and the broader U10 audit remain unfinished.

## Alignment execution foundation

The alignment provider now offers execution callbacks for loaded CTC model
execution (including the available model revision), whole-clip versus segment
attempts, and uniform approximate fallback. Approximate timings identify no model;
empty input reports no engine execution. The provider explicitly selects the
installed library's existing model default instead of inheriting a mutable
default. Callers without callbacks retain their existing result API.

The regression run passed 257 tests covering alignment, worker delivery/recovery,
spine imports, and MCP. Changed-file Ruff passed. Scoped typing retains one
pre-existing external-provider return typing error, confirmed against unchanged
HEAD. Alignment operation records, GUI/durable recovery, and completion checks
still need migration; execution reporting alone does not establish verified reuse.

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
