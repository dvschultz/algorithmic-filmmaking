# Project sessions and editorial history

Every `Project` has one Qt-free `ProjectSession`, created on the project's owner
thread. Clip enable/disable and manual sequence insertion/removal use reversible
commands. Reorder and timing/track/transform updates use that same history.
Browser controls, the timeline, and chat use the same model delegation; the Edit menu and chat
Undo/Redo project the same history through `SessionHistoryAdapter`.

`Project.set_clips_disabled()` captures explicit before/after values for unique
existing clip IDs. Unknown IDs retain their existing ignored/reported behavior.
Already-matching values produce no history entry, notification, or dirty change.
Undo and redo validate all affected object identities and values before applying
anything. A conflicting edit returns an error without partially changing clips.
Each accepted edit advances the monotonic mutation generation once and publishes
one `clips_updated` event, with dirty state already settled.

The save checkpoint records both the history position and an external mutation
revision. Returning to the saved history position clears dirty state only if
analysis or another non-history mutation has not changed the project since that
save. Analysis remains outside editorial history, so undo does not erase results
or repeat inference. Save-worker generation checks remain in force.

New Project clears history and rotates the session identity. Loading a different
project closes the old session and rebinds menu actions to the new one. Color
applications capture the session identity and verify it and the owner thread
before touching results, in addition to the pilot's existing input checks.
Color result application runs inside `ProjectSession.apply_external()`: observer
callbacks cannot reset or close the session or start another editorial edit
mid-application. The guard releases even when validation fails. Analysis still
uses external dirty revisions and creates no undo entry; unchanged inputs remain
valid through unrelated project or clip renames.

`insert_sequence_clips()` accepts prepared entries, and `add_to_sequence()` and
`add_frames_to_sequence()` resolve source IDs into one insertion command.
`remove_from_sequence()` removes timeline IDs across tracks. Its default ripple
behavior retains the agent contract; timeline Delete passes `ripple=False` to
preserve gaps. Undo restores the original entries, ordering, and start positions,
and rejects conflicting track edits atomically. Commands target their original
sequence even after the active sequence changes. Timeline model refreshes emit
`sequence_refreshed`, which updates views without creating an external mutation.

`reorder_sequence()` reorders the first track and rejects duplicate IDs.
`update_sequence_clip()` validates the complete timing/track/transform request
before committing it. Track moves update actual membership; undo restores cached
render references invalidated by trim or transform edits. No-op edits do not enter history.
Desktop drag/resize gestures use detached previews, leaving save snapshots stable,
and commit one command on release. Frame entries resize their hold duration.

`add_sequence(..., activate=True)` combines creation and activation in one edit.
`remove_sequence()` retains the original sequence, including clips and settings;
deleting the last sequence creates a stable empty fallback that is also undone.
`rename_sequence()` and `update_sequence_metadata()` validate and record names,
frame rates, music paths, and repeat settings. Collection/settings changes publish
`sequences_changed` followed by `active_sequence_changed` after dirty state settles.
The Qt adapter forwards these events so dropdowns and the timeline follow agent
commands and Undo/Redo. Restoring a sequence with permanently removed clips fails
atomically. `core.spine.sequences` supplies listing and management wrappers.

`clear_sequence()` clears all tracks in one reversible command, shared by chat
and the timeline Clear button. Undo retains original clip objects, gaps, trims,
and transforms. Empty clears create no history or dirty change. Generation and
project-reset paths retain their explicit history bypass.
Source references retained in both sides of clip-edit snapshots remain discoverable.

`remove_source()` and batch `remove_sources()` now retain sources, library clips,
frames, and affected entries across every sequence in one command. They never
delete media files. Undo restores original objects and ordering without repeating
import or analysis; conflicting library or track membership fails atomically.
The desktop keeps its guard against deleting sources used in live sequences,
but sources used only in Undo history can be removed: Undo restores the source
before restoring earlier edits. The `sources_changed` event refreshes library
views after commit, alongside legacy per-source notifications. Chat and MCP
delegate through `core.spine.sources`; MCP retains its response and save contract.

`core.spine.sequences.SequenceDraft` keeps generated output detached until one
`Generate sequence` command publishes it. Empty-sequence reuse and replacement
restore the original sequence on Undo; Redo restores realized output without
calling algorithms or providers. Desktop callback families share this draft
lifecycle. Agent ordering paths use `core.spine.sequences.generate_sequence / publish_recipe`, with explicit source
ranges for trimmed results. Failed population discards the draft; saved project
data never includes that partial output. Draft view updates do not mark the
project dirty. Actual analysis updates remain outside editorial history.
Worker delivery checks session identity and clip/source input ranges before
applying output. A changed replacement target is rejected without overwriting it.

`edit_metadata()` captures only the requested editorial fields, validates the
whole batch, and deep-copies both sides for Undo/Redo. Use `update_clip_metadata()`,
`update_frame_metadata()`, `update_source_metadata()`, and `rename()` for manual
corrections. The sidebar submits detached values through the current project;
chat and MCP metadata adapters use the same command. Batch tags are one edit.
The Qt adapter refreshes metadata and project names after Undo without reloading
playback or resetting unchanged fields that the user is still editing.
Automatic analysis continues to use `update_clips()`, `update_frame()`, and
`update_source()` outside history. Undo preserves unrelated analysis fields and
rejects conflicts on the fields it would restore.
Saving or choosing a new save filename preserves the project's editorial name;
it does not introduce an implicit rename or invalidate name history.

This is an incremental U4 migration.
Legacy direct model mutations are tracked as
external changes; they are not yet undoable or comprehensively thread-guarded.
MCP and CLI do not yet expose history tools; the shared spine history functions
are available to retained headless sessions. Cross-process locks, durable history,
and persistence migrations are outside this change.
