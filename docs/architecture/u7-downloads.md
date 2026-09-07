# U7 download migration

`core.operations.downloads.run_download` is the shared execution entry point for
desktop single downloads, URL batches, search-result batches, CLI downloads, and
the headless download spine used by MCP. Its frozen request carries the URL,
output directory, resolution, and adaptive-timeout policy. Provider behavior stays
in `VideoDownloader`.

The operation validates URLs before metadata lookup, forwards cooperative
cancellation to the native downloader, and checks cancellation between metadata,
download, and publication. A native error after cancellation is classified as
cancellation. URL batches retain adaptive timeout calculation; other adapters
retain the downloader's default timeout. CLI metadata preview remains in its
adapter. The spine keeps its succeeded/failed/cancelled result envelope and
preserves earlier successes when later work fails or is cancelled. Cancellation
can leave downloaded files that were not published as successful results.

Desktop download signals pass through an owner-thread `DownloadDelivery` relay.
It captures the project session and worker channel at submission. Results,
progress, errors, and workflow completion from replaced workers or old sessions
are discarded. Replaced workers are cancelled and retained until finished; their
cleanup cannot clear a newer worker reference. Reset and shutdown include retained
download workers. The relay does not change existing shutdown timeout policy.

MCP queued jobs snapshot their URL lists so caller mutation cannot change the
work after submission. Intention source admission accepts the actual
`DownloadResult` shape, which does not promise fps or dimensions, and keeps the
existing defaults when those fields are unavailable.

`run_download_batch` now owns both parallel desktop and serial headless scheduling.
It snapshots requests, keeps at most the configured number of downloads active,
and stops starting native work once cancellation is observed. It returns exactly
one outcome per input in input order, including duplicate URLs and cancelled
items. Previously completed successes survive later failures or cancellation.
Callbacks run on the scheduler's calling thread, not its download executor threads.
A callback failure stops further dispatch, signals cancellation, waits for active
work, and propagates the error.

The Qt adapters live in `ui.workers.download_workers`; their former MainWindow
names remain import aliases. Search-result IDs and URLs are copied at construction.
URL-batch final results now have deterministic input ordering and explicitly
include cancelled items with `success: false` and `cancelled: true`. Existing
success and failure fields and per-item signals remain. The headless spine retains
its separate succeeded/failed/cancelled lists.

Durable per-download results, shared ordered intention plans,
and the remaining analysis workflows are outstanding U7 work. A direct chat
download notification is still associated with the chat worker rather than these
download-worker relays and needs its own submission/session guard.

Tests cover timeout/resolution forwarding, cancellation at each stage, invalid
URLs, failure aggregation, frozen MCP submission arguments, actual intention
download results, and real queued Qt delivery after reset and worker replacement.
Scheduler tests also cover bounded parallelism, callback thread ownership,
ordered outcomes, cancelled pending work, per-item failure isolation, and callback
failure. Adapter tests exercise input snapshots and existing result envelopes.
