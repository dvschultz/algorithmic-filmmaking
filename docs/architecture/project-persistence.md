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

This is the schema portion of U5. Cross-process writer ownership, elimination
of the legacy read-modify-merge writer, and full UI read-only affordances remain
separate work. Schema checks and mtime diagnostics do not close concurrent-write
races; they do not constitute a project lock.

Offline source files remain declared sources on load, so their clips and edits
survive in every sequence. A relink callback can supply an existing replacement;
returning `None` or an unavailable replacement keeps the original path. Callback
exceptions still cancel loading. Stills and audio references are retained too.
Media-dependent operations must check file availability and report missing input
instead of treating successful document loading as proof that media is online.
