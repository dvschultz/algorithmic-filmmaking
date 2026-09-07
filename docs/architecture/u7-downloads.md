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
download workers. Reset cancels downloads without waiting or terminating their
threads; retained workers finish independently and cannot publish into the new
session. Shutdown cancels and waits for download workers, allowing receipt writes
to finish safely. Waiting can last through native-call timeouts or file hashing.

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

Desktop, CLI, and headless downloads now record verified file receipts in the additive
`download_receipts` job-store table. Receipts survive job-row retries and purges.
The request identity includes the URL, canonical directory, policy, and version;
the receipt contains result metadata plus the output's size and SHA-256 digest.
Successful new outputs are flushed and recorded before their completion is
delivered. A receipt-write failure stops further dispatch, retaining receipts
already committed for earlier items; other parallel downloads may still be active.

Retries verify file content before reusing it. Missing outputs are downloaded
again and their receipts replaced. Changed outputs are reported as
`download_output_changed` without accepting or overwriting the changed file.
Use another output directory, or intentionally remove the changed file, to request
a fresh download. Receipt corruption stops the job. This is verified local-file
reuse, not a check that the remote URL still serves the same media.

A cancellable OS lock serializes durable batches sharing a physical output
directory across processes. Submission also captures directory identity; replacing
or retargeting it while queued rejects execution. Hashing existing files incurs
local I/O, and native calls retain their existing cancellation/timeouts. A crash
before the receipt is committed can still require another downloader invocation;
missing files after restart are verified as missing and downloaded again.

`run_recoverable_batch` accepts the same immutable requests, concurrency limits,
and per-item callbacks as the download scheduler. Desktop URL/search adapters keep
their parallelism and timeout policies. All entry points use the configured cache
directory's `jobs.db`, so matching request policies can share receipts. Single
desktop downloads retain native progress, capped below 100 percent until receipt
storage finishes. CLI progress converts native percentages to its 0-to-1 scale;
cached CLI downloads skip the redundant remote metadata preview. Existing local
dependency gates remain in place.

Worker storage failures become per-item errors and completion signals while
already delivered successes remain available. Native progress callbacks originate
on executor threads; Qt adapters only emit signals there. Receipt and item
callbacks remain on the calling worker thread.

Direct chat download notifications now pass through `ChatDelivery`, which checks
the requesting project session and current chat worker before delivery. The same
guard covers text, GUI tool requests, cancellation, search results, and completion.
Replacing or clearing a conversation invalidates delivery and cooperatively stops
its workers; retained workers are released only after `finished`. Project reset
does not wait for chat threads. Shutdown waits for them, which can take until a
provider or native call returns. Ordinary chat cancellation retains the existing
completion/history behavior. Pending GUI-tool cancellation is consumed before
invalidation so its later queued signal cannot cancel a new conversation's work.

Chat workers accept GUI replies through `GuiToolMailbox`. Each request gets a
fresh transport token, independent of provider call IDs, and only the first reply
with the matching token and tool name is accepted. Timeout closes the request;
cancellation permanently closes the worker's mailbox and wakes any waiter. Replies
are copied at admission, and the original provider call ID is restored before
tool results enter LLM history. This rejects late or duplicate replies without
changing the provider conversation format.

Shared ordered intention plans and the remaining analysis workflows are outstanding
U7 work. This relay protects incoming chat signals; per-operation ownership of
asynchronous GUI tool replies and worker-side project access still need migration.
Standalone color, shot-type, description, classification, and object-detection
completion handlers now capture a `GuiToolReply` when their worker is started.
It retains the requesting chat, session, tool name, and transport token; completion
does not read or clear another operation's pending fields. An owner-thread relay
also verifies the analysis worker channel and session, ignores duplicate completion,
and prevents old-thread cleanup from clearing a replacement worker. Direct GUI
responses, plan display acknowledgments, and failed starts use captured replies.
Dispatch stops follow-up work after a nested GUI event replaces the conversation;
the five analysis starters recheck ownership after dependency availability gates.

Agent transcription now retains its captured reply across every source handoff,
stops the remaining queue when the conversation is cancelled/replaced or the
request times out, and reports progress against the original total source count.
The shared transcription launcher uses guarded completion delivery for agent and
manual-pipeline callbacks, including identity-safe cleanup when one source starts
the next. `GuiToolReply` checks the chat worker's live mailbox before follow-up work
or delivery, so an expired request cannot continue merely because its chat still
exists. Per-source computation remains in the existing transcription worker.

Desktop detection now captures its request when detection starts and carries that
reply through the thumbnail stage. Errors and completion leave unrelated pending
fields alone, and the response uses the captured source ID. An expired request
cannot publish a late detection result or start thumbnail generation. Detection
thumbnail progress, individual results, and completion use an owner-thread relay
guarded by the detection identity, project session, and thumbnail worker. Cleanup
cannot clear a replacement worker. Busy or stale detection dispatch reports a
failed start instead of leaving the agent waiting for an operation that never ran.

Download batches, exports, and combined
analysis completion still need this sender migration. Their legacy completion
handlers can construct replies from shared pending fields; mailbox validation
cannot distinguish a stale result relabeled with the current request token.
Per-item analysis mutation delivery and full computation migration remain separate
U7 work; this completion relay does not replace those operation implementations.

Tests cover timeout/resolution forwarding, cancellation at each stage, invalid
URLs, failure aggregation, frozen MCP submission arguments, actual intention
download results, and real queued Qt delivery after reset and worker replacement.
Scheduler tests also cover bounded parallelism, callback thread ownership,
ordered outcomes, cancelled pending work, per-item failure isolation, and callback
failure. Adapter tests exercise input snapshots and existing result envelopes.
Recovery tests reopen the store, remove or edit outputs, fail receipt writes and
progress callbacks, cancel lock waiters, and upgrade a database with existing jobs.
Adapter tests also verify cross-entry-point receipt reuse, resolution/timeout
preservation, desktop retries, and CLI progress before and after receipt storage.
