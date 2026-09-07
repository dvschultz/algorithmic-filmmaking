# Project schema upgrades

`core/project_migrations.py` owns the current JSON schema version and ordered
in-memory migrations. `core.project.SCHEMA_VERSION` remains an import-compatible
alias. The first migration normalizes legacy single-sequence documents to the
multi-sequence representation. It copies timeline values verbatim: ambiguous
trim coordinates are not guessed or converted.

Loading a supported older document upgrades a detached copy. Loading alone
never writes the original file. Before an upgrade save, the shared writer saves
the exact original bytes beside the project as
`<project>.pre-v1.4-<content-hash>.bak`. Backup bytes are flushed in a temporary
file before an exclusive hard-link operation publishes the final name. On POSIX,
the directory is flushed before proceeding. Existing backups are verified rather
than overwritten and retained after a failed project write. A failed backup,
including a filesystem that cannot publish hard links, aborts the save. The
project replacement uses a flushed temporary file followed by atomic replacement.

Documents with a newer version may be inspected through the known model fields.
Their `Project.is_read_only` flag blocks supported editorial commands, analysis
application, and library mutation methods. All save paths refuse to serialize a
newer-version model, including Save As, and refuse to overwrite a newer-version
destination. Invalid versions also fail closed. These checks preserve fields
the current application cannot interpret. Direct mutation of returned model
objects is outside these supported APIs.

To roll back an upgrade, close clients using the project and restore a copy of
the matching `.bak` file. Keep the backup until the restored file has been
verified with the previous application. A backup contains the JSON document;
it does not duplicate referenced media.

This implements part of U5. Shared headless models check external revisions
before offering history; MCP exposes the retained session API described below.

Offline source files remain declared sources on load, so their clips and edits
survive in every sequence. A relink callback can supply an existing replacement;
returning `None` or an unavailable replacement keeps the original path. Callback
exceptions still cancel loading. Stills and audio references are retained too.
Media-dependent operations must check file availability and report missing input
instead of treating successful document loading as proof that media is online.

## Complete-state saves

`Project.save()` serializes a detached snapshot containing every sequence,
still, audio source, and UI field. CLI mutators load the complete `Project`
instead of unpacking the legacy tuple. Bundle export also saves a complete
snapshot and reports a refused write as an error. Synchronous progress callbacks
cannot change the snapshot; later edits stay dirty after saving.

The writer does not merge editorial data from an existing destination.
`sequences` and `active_sequence_index` come from the supplied state; the legacy
`sequence` key is a derived active-sequence projection for compatibility.
The standalone `save_project()` API treats its supplied data as complete: without
`_all_sequences`, it writes only the supplied single sequence (or an empty list).
Callers editing an existing multi-sequence document must use `Project.load()`
and `Project.save()`, within the appropriate ownership lifetime.

Prerenders from every sequence are localized before serialization, with original
model paths preserved. Filename collisions compare content rather than assuming
equal byte lengths identify equal media. Publication retries a competing filename
and uses exclusive creation for copy fallback, so it cannot overwrite a winner.
Missing sequence music stays referenced on load so a subsequent save cannot
silently delete it. Bundles copy music referenced only by alternate sequences
too, rewrite those paths, and strip machine-local fallbacks before the single
atomic project-file publication.

## Writer ownership

`core/project_lock.py` provides nonblocking ownership for cooperating processes
using the same user's app-support directory. Permanent hashed lock records live
under `project-locks`; never delete them while clients may be running. Records
use [POSIX flock](https://docs.python.org/3/library/fcntl.html) or
[Windows byte-range locking](https://docs.python.org/3/library/msvcrt.html).
These are local coordination records, not distributed or multi-user locks.

Ownership covers the resolved destination path and the existing file's device
and inode identity. This handles symlink and hard-link aliases. Replacement
acquires the new file's identity before publishing it and retains the path lock.
On macOS, case and Unicode normalization conservatively serialize path aliases;
distinct case variants on case-sensitive volumes can therefore conflict too.
Nested scopes reuse ownership only in the same process, thread, and async task.

`ProjectWriter.activate()` explicitly lends an existing lease to one save
operation, including a worker thread in the same process. Possession of the
writer object is required; inherited contexts alone do not grant access.
Borrow permission expires when the operation exits, including in copied
contexts. Concurrent borrowing and closing an active lease fail immediately.
`SaveProjectWorker` accepts an optional writer and verifies that it owns the
destination. Success and failure release the borrow before emitting completion,
while the original session lease remains held. The desktop passes this lease
and settles ownership only after worker completion.

Every shared project save holds ownership while serializing and replacing the
file. MCP mutations additionally acquire it before loading and retain it through
saving; project-scoped jobs hold it throughout execution. Contention returns
`project_busy` from MCP tools and job results. The legacy Boolean save API returns
`False`, preserving the unsaved state. A failed download-and-detect project save
reports a detection error instead of claiming a saved project filename.

CLI analysis, transcription, and sequence mutations acquire ownership before
loading and release it when the Click command context closes, including early
returns and errors. Detection owns its destination before checking overwrite
policy and computing scenes. Download-and-detect acquires the destination after
the downloaded media path is known; contention preserves the successful download
and returns a structured `project_busy` detection error. Other mutating CLI
commands exit with status 1 on contention and return the structured error in
JSON mode. Read-only project inspection and export do not acquire ownership.

Desktop projects use `Project.new(retain_writer=True)` and
`Project.load(..., retain_writer=True)`. Loading acquires before reading; new
projects acquire before their first save. Closing or clearing releases the
session lease. Save As holds the original and destination until the write
finishes: failure releases only the destination, while success transfers
ownership. Newer edits during an asynchronous save stay dirty and retain the
successful Save As destination. Pending saves block New, Open, and Close,
including agent project replacement. Opening the already-open canonical path
returns without reloading. Agent saves use the same model ownership path.

Plain `Project.load()` remains a non-owning inspection API; callers intending
to mutate must use a retained session or a surrounding headless writer scope.
Save-only locking cannot detect a stale document loaded before another writer
completed. The mtime check remains a supplementary
diagnostic with a one-second tolerance; it does not replace lifetime ownership
and does not prevent writes by applications that ignore these locks.

Newer-schema projects display a read-only title in the desktop. Save, bundle,
import, analysis, sequencing, and metadata-edit controls are disabled; selection
handlers cannot re-enable them. Delete, drag, and context-menu editing gestures
are guarded too. Browsing, playback, filters, and inspection exports remain
available. Agent dispatch rejects project mutations and project-file writes
before starting workers. Opening a supported project restores editing controls.

`load_with_mtime` binds a SHA-256 content revision to its headless model session
and verifies it after loading. `save_with_mtime_check` verifies this revision
under writer ownership, then refreshes it after a successful save. The legacy
mtime check remains for compatibility, but a same-timestamp edit or deletion
now fails closed. Session history availability, undo/redo, and command execution
check the bound revision. Once a conflict is observed, the session requires a
reload; restoring old bytes does not reactivate stale history. Clearing a
project discards the revision. Plain `Project.load` does not opt into these
headless checks. These checks detect observed changes; they cannot make an
uncooperative external writer participate in the application lock protocol.

MCP now retains explicit headless editing sessions in `core/spine/project_sessions.py`.
A transport-owned serial executor keeps their model mutations on one owner
thread. Session create/rename/delete sequence tools and undo/redo save each
successful edit under load-through-save writer ownership. The session retains
its model and history between calls, but releases the writer lease. External
revision changes reload the model and invalidate previous history before the
next operation; failed saves discard unpublished state. Shutdown drains queued
session work and closes models. Unmigrated path-based tools still load independently;
their writes trigger the same revision-driven reload. Analysis and import
operation lifecycle remains separate from editorial undo.

Retained timeline tools delegate to `core/spine/timeline.py` and shared reversible
commands, using explicit sequence IDs and validated track indices. The migrated
path-based remove/reorder/clear/shuffle tools enter the same retained session
through `edit_path`; standalone calls still acquire ownership and save on an
owner thread without retained history. Remaining analysis and import operations still require lifecycle migration.

Legacy MCP insertion, tag/note edits, and source removal now use the retained
editorial adapter too. New track creation is captured inside the insertion
command, including empty intermediate tracks, so undo/redo restores structure
atomically. Session metadata accepts validated transcript JSON and converts it
to shared model types before publication. Import/analysis jobs remain outside
editorial undo and are the next lifecycle migration surface.
