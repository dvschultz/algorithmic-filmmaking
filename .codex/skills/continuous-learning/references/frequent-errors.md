# Frequent Errors (Scene Ripper)

Use this file as a quick index before creating a new error pattern.

## Runtime Errors

- `QThread: Destroyed while thread is still running`  
  Reference: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- `IndexError` from missing `QGraphicsScene` track items  
  Reference: `docs/solutions/runtime-errors/qgraphicsscene-missing-items-20260124.md`

## UI Bugs

- Thumbnail/source ID mismatch clears the clip grid  
  Reference: `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
- Timeline widget/scene sequence mismatch disables export state  
  Reference: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`

## Security/Hardening

- FFmpeg concat path escaping and filename sanitization  
  Reference: `docs/solutions/security-issues/ffmpeg-path-escaping-20260124.md`
- URL scheme validation bypasses whitelist checks  
  Reference: `docs/solutions/security-issues/url-scheme-validation-bypass.md`

## Reliability

- Subprocess cleanup leak on exception paths  
  Reference: `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`
