# Reusing legacy analysis

Older projects may contain analysis values without a record of the inputs or
model that produced them. These values remain visible, but they do not count as
completed analysis until you recompute them or explicitly accept reuse.

For colors and compatible DINO embeddings, the CLI provides an explicit reuse
command:

```sh
scene_ripper analyze accept-legacy project.json --operation colors --clip-id CLIP_ID
scene_ripper analyze accept-legacy project.json --operation embeddings --clip-id CLIP_ID
```

Omit `--clip-id` to process all clips. The command reports accepted and failed
clip IDs and saves accepted decisions. It reads and fingerprints the current
media but does not run inference. Color reuse binds the default five-color
analysis configuration. Embeddings must have a compatible recorded model and
valid vector; an unknown model requires recomputation.

An accepted value retains **unknown provenance**. Acceptance records your
decision to use it with the current inputs; it does not establish how the old
value was computed. Changes to media, trim, model, or analysis settings can
invalidate that decision. Verified or failed records cannot be relabelled as
legacy, and unknown future record formats are preserved.

The desktop and agent acceptance controls, and acceptance for other analysis
operations, are not available yet. Recompute those results through their usual
analysis commands.
