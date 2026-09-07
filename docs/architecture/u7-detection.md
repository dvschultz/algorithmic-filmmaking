# U7 detection migration

Detection computation now enters through `core.operations.detection.run_detection`.
`DetectionRequest.build` snapshots visual and karaoke settings as detached JSON;
each execution creates fresh backend configuration. The operation returns new
source/clip objects and does not read or mutate a project.

| Surface | Adapter | Execution |
|---|---|---|
| Desktop and live chat | `DetectionWorker` | Shared operation on the QThread |
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

Verification includes configuration isolation across retries, cancellation before
and during computation, progress/error propagation, real queued desktop delivery,
CLI/MCP compatibility, and the headless import boundary. A generated 24 fps video
with cuts every 48 frames produced identical ranges from the old and shared paths
in both adaptive and content modes.

This completes the computation seam and target/session guards. U7 still requires shared job scheduling,
durable detection receipts,
source import/download and thumbnail orchestration, and ordered intention plans.
The remaining analysis families follow those changes. Existing CLI minimum-scene
duration conversion still assumes 30 fps before detection and needs correction
as part of parameter parity; this extraction preserves that behavior.
