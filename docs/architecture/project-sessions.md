# Project sessions and editorial history

Every `Project` has one Qt-free `ProjectSession`, created on the project's owner
thread. Clip enable/disable is the first migrated editorial command. Browser
controls and the chat tool use the same model delegation; the Edit menu and chat
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

This is the first U4 migration in the shared-engine plan. Sequence insertion,
removal, reordering, trimming, sequence management, source removal, and metadata
commands still need migration. Legacy direct model mutations are tracked as
external changes; they are not yet undoable or comprehensively thread-guarded.
MCP and CLI do not yet expose history tools; the shared spine history functions
are available to retained headless sessions. Cross-process locks, durable history,
and persistence migrations are outside this change.
