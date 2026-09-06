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
render references invalidated by transform edits. No-op edits do not enter history.
Desktop drag/resize gestures use detached previews, leaving save snapshots stable,
and commit one command on release. Frame entries resize their hold duration.

This is an incremental U4 migration. Generated sequence population, clearing,
sequence management, source removal, and metadata commands still need migration. Legacy direct model mutations are tracked as
external changes; they are not yet undoable or comprehensively thread-guarded.
MCP and CLI do not yet expose history tools; the shared spine history functions
are available to retained headless sessions. Cross-process locks, durable history,
and persistence migrations are outside this change.
