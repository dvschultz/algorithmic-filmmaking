# U7 source import migration

`core.spine.sources` owns shared source lookup, metadata preparation, and
owner-thread admission. Manual desktop imports, folder imports, and download
completion handlers reuse these helpers. Detection lookup and target guards use
the same identity comparison.

Lookup compares resolved paths, then physical file identity when media is
available. Relative paths, symlinks, and hard links therefore reuse an existing
source. Matching offline paths still work; inaccessible paths fall back to their
absolute spelling so an old unavailable source cannot block unrelated imports.
Separate files with identical bytes remain separate sources. Offline hard links
with different paths cannot be identified as aliases.

Admission preserves the existing source object, ID, metadata, clips, and dirty
state on retries. It rejects an ID already assigned to different media and uses
`Project.add_source` for new sources. Existing duplicate sources are not merged.

`probe_source` lazily loads the FFmpeg metadata reader and returns prepared source
data without mutating a project. Probe failures retain the desktop's default
metadata behavior. Folder import now uses this helper in place of its broken
reference to the removed `load_source` function.

Desktop manual imports and download callbacks now prepare metadata and thumbnails
through `SourceImportQueue`. One worker runs at a time; workers never receive the
project. The owner thread admits the prepared source after checking the captured
session and media stamp. A selection generation prevents an older import from
overriding a later source selection. Existing sources still select immediately.

Reset discards queued work and invalidates active results without waiting for
native media calls. The active thread remains owned until it finishes; shutdown
cancels and waits for it rather than terminating it. Cancellation is cooperative
between metadata and thumbnail stages. Native subprocess timeouts bound the wait.
Stale results can leave unreferenced thumbnail cache files. Bulk agent download
responses wait until pending source imports drain, so the agent can then find its
downloaded sources. Session changes discard deferred responses.

This is an incremental U7 migration. Folder imports still probe synchronously in
their existing tool adapter. Intention downloads use their supplied metadata.
Download-worker submission/session guards are described in `u7-downloads.md`.
Shared durable download orchestration, ordered intention plans, and the remaining
analysis families are outstanding.

Regression coverage checks alias retries, preserved edits, unavailable old paths,
all four desktop admission routes, folder retries, metadata fallback, and stale
detection results through hard-link aliases. Headless import-boundary tests ensure
the shared helpers do not import GUI or heavy analysis dependencies at load time.
Real Qt-loop tests cover background preparation, owner-thread delivery, reset and
retry during an active probe, changed media, per-item failures, and shutdown.
