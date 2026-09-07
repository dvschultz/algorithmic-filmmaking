# U7 detection migration

Detection computation now enters through `core.operations.detection.run_detection`.
`DetectionRequest.build` snapshots visual and karaoke settings as detached JSON;
each execution creates fresh backend configuration. The operation returns new
source/clip objects and does not read or mutate a project.

| Surface | Adapter | Execution |
|---|---|---|
| Desktop and live chat | `ui.workers.detection_worker.DetectionWorker` | Shared operation on the job executor; QThread relays delivery |
| Headless chat and MCP | `core.spine.detect` | Shared operation, then thumbnails and model publication |
| CLI detect | `cli.commands.detect` | Shared operation, then project save |
| CLI download with detection | `cli.commands.youtube` | Shared operation after download, then project save |

Visual requests use `SceneDetector.detect_scenes_with_progress`; karaoke requests
use its karaoke method. The older non-progress `SceneDetector.detect_scenes`
remains a compatibility API, but these application adapters no longer call it.
The backend still owns scene algorithms and media decoding.

Cancellation is checked before and after the native call, with the spine and
desktop retaining their checks before later stages or result delivery. This is
cooperative cancellation, not native-call interruption or an atomic cancellation
and publication transaction. Spine progress reserves the final portion for
thumbnails and publication. Existing result envelopes and desktop signals remain.

Requests also capture media file identity, size, modification time, and change
time. A changed stamp rejects computation or delivery. Before publication,
`DetectionGuard` checks the owning session, media stamp, and a digest of the
target sources and clips. Unrelated sources can change without invalidating the
result. Desktop delivery uses the captured source ID rather than the current
selection, and both desktop entry points reject obsolete tasks/sessions. Errors
from obsolete tasks are ignored too. Intention detection retains an existing
source ID and replaces its clips, avoiding duplicate imports on re-detection.

Desktop adapters connect only `DetectionWorker.result_ready` for guarded
publication. `detection_completed` remains available for compatibility; consumers
must not publish through both signals. Thumbnail files produced before a stale
result is rejected can remain in the cache, but are not attached to the project.

Desktop detection now submits an immutable operation specification to
`JobRuntime.for_session()`. The runtime owns task IDs, progress, cancellation,
terminal status, and the JSON result. The worker reconstructs source/clip objects
after terminal success, then emits guarded results for delivery on the GUI thread. Native
computation never receives the live project. Both manual and intention entry
points report session-only persistence; this does not save the project or make
detection restart-safe. The runtime is closed after emitting results, retaining the task
ID and terminal status on the worker. `ui.main_window.DetectionWorker` remains an
import alias for compatibility.

Verification includes configuration isolation across retries, cancellation before
and during computation, progress/error propagation, real queued desktop delivery,
CLI/MCP compatibility, and the headless import boundary. A generated 24 fps video
with cuts every 48 frames produced identical ranges from the old and shared paths
in both adaptive and content modes.

MCP bulk detection over existing projects now captures project revision, media
stamps, target state, and detached arguments at submission. Changed queued inputs
fail before decoding. Each successfully computed source is recorded in the result
ledger, then its clips and receipt are saved together before checkpointing. A
retry reuses clip IDs and computations; a failed checkpoint reconciles the saved
receipt without applying twice. Target edits made after computation or publication
are preserved and reported as conflicts. Source failures are collected; persistence
failures stop the job. Completed sources remain saved after failure or cancellation.
Each source is one publication group because it may contain many clips.

This completes the computation seam, target/session guards, desktop job lifecycle,
and durable MCP bulk detection for existing projects. U7 still requires the
new-project detection path, source import/download and thumbnail orchestration,
and ordered intention plans to use the same durable workflows.
The remaining analysis families follow those changes. Existing CLI minimum-scene
duration conversion still assumes 30 fps before detection and needs correction
as part of parameter parity; this extraction preserves that behavior.
