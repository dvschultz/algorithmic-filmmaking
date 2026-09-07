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

Every shared project save holds ownership while serializing and replacing the
file. MCP mutations additionally acquire it before loading and retain it through
saving; project-scoped jobs hold it throughout execution. Contention returns
`project_busy` from MCP tools and job results. The legacy Boolean save API returns
`False`, preserving the unsaved state. A failed download-and-detect project save
reports a detection error instead of claiming a saved project filename.

Desktop ownership for an entire open editing session and legacy CLI ownership
from load through save remain to be implemented. Save-only locking cannot detect
a stale document loaded before another writer completed. Keep the desktop
project closed while MCP edits it. The mtime check remains a supplementary
diagnostic with a one-second tolerance; it does not replace lifetime ownership
and does not prevent writes by applications that ignore these locks.
