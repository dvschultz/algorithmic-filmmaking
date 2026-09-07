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

This is an incremental U7 migration. Metadata probing remains blocking at current
entry points. Moving preparation to workers, guarding import delivery by session,
durable download orchestration, and ordered intention plans remain outstanding.

Regression coverage checks alias retries, preserved edits, unavailable old paths,
all four desktop admission routes, folder retries, metadata fallback, and stale
detection results through hard-link aliases. Headless import-boundary tests ensure
the shared helpers do not import GUI or heavy analysis dependencies at load time.
