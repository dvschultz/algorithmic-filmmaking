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

This implements part of U5. Elimination of the legacy read-modify-merge writer
and full UI read-only affordances remain separate work.

Offline source files remain declared sources on load, so their clips and edits
survive in every sequence. A relink callback can supply an existing replacement;
returning `None` or an unavailable replacement keeps the original path. Callback
exceptions still cancel loading. Stills and audio references are retained too.
Media-dependent operations must check file availability and report missing input
instead of treating successful document loading as proof that media is online.

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
