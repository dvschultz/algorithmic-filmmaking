# Media coordinates

Sequence trims (`in_point`, `out_point`) are absolute source-video frame
indices, with an exclusive end. They are never offsets from a library clip.
`core/sequence_time.py` converts library selections to sequence entries once.
Word selections retain their floor/ceil boundary policy before that conversion.

`source_rate` and `timeline_rate` store exact rational rates. `timeline_start`
stores exact seconds; `start_frame` is its nearest timeline-frame projection.
Adjacent entries accumulate rational seconds and round shared boundaries once,
with half-frame ties going to the later frame. Reordering and ripple edits use
the same rule. Explicit frame-position edits snap to the timeline grid.
Changing the timeline frame rate requantizes positions without changing source
ranges or elapsed time, and is reversible through project history.
An entry shorter than a timeline frame can project to zero display frames. Its
positive source duration and edit remain intact; increasing the timeline rate
can make it visible again. Validation uses exact duration rather than inflating
each short entry to one frame and drifting the total length.

The media-time types distinguish video frame ranges, still holds, audio sample
ranges and timeline ranges. Still entries have no video trim coordinates; their
exact hold duration survives frame-rate changes. Audio samples use their own
sample rate. Variable-rate video requires verified presentation boundaries on
the source, including the exclusive final boundary. Entries retain only their
start/end presentation times, so the full mapping is not copied into every cut.

## Legacy documents

Schema 1.6 migrates both legacy sequence keys. A range is converted automatically
only when one interpretation is valid, or both interpretations are identical
because the library clip starts at source frame zero. If both different
interpretations are valid, the entry remains unresolved. Missing references,
invalid ranges and unavailable VFR mappings also retain diagnostics.

`legacy_timing.original` preserves the exact original entry, including unknown
fields. Loading does not rewrite the original project. The existing migration
writer saves a byte-for-byte backup before the first upgraded write.

An unresolved entry remains inspectable, but blocks affected playback, preview,
video export and EDL export. Resolve it through:

- The sequence dropdown context menu: **Resolve legacy timing...**
- CLI: `scene_ripper project resolve-timing PROJECT SEQUENCE_ID ENTRY_ID
  --convention source` (or `clip-relative`).
- MCP: `resolve_sequence_timing` with the same sequence/entry IDs and convention.
- Built-in agent: `get_sequence_state` exposes diagnostics and IDs;
  `resolve_sequence_timing` applies the same reversible choice.

Resolution uses project history and preserves the original values. It can be
undone and redone. Missing source references must be restored before a valid
coordinate choice can be applied.
