---
date: 2026-01-24
problem_type: security_issue
component: subprocess_call
symptoms:
  - "Potential command injection via FFmpeg concat file"
  - "Path traversal possible in export filenames"
root_cause: insufficient_input_validation
severity: high
tags: [security, ffmpeg, path-traversal, subprocess]
---

# FFmpeg Path Escaping and Filename Sanitization

## Problem Statement

The sequence export functionality had two security vulnerabilities:
1. FFmpeg concat file path escaping was incomplete (only single quotes, not backslashes or newlines)
2. Export filenames derived from video file stems could contain path traversal sequences

## Symptoms

Found during code review (security-sentinel agent):
- FFmpeg concat list file only escaped single quotes, not backslashes
- Export filename constructed directly from `source.file_path.stem` without sanitization
- A video named `../../../etc/video.mp4` could write files outside the export directory

## Investigation

### Issue 1: FFmpeg Concat Path Escaping

The FFmpeg concat demuxer format uses special syntax:
```
file 'path/to/video.mp4'
```

Original code only escaped single quotes:
```python
escaped_path = str(path).replace("'", "'\\''")
```

Missing:
- Backslash escaping (needed on Windows)
- Newline rejection (would break concat file format)

### Issue 2: Path Traversal in Export Filenames

Original code:
```python
source_name = self.current_source.file_path.stem
output_file = output_path / f"{source_name}_scene_{i+1:03d}.mp4"
```

If video file was named `../../../etc/video.mp4`, stem would be `../../../etc/video` and output could escape the export directory.

## Solution

### Fix 1: Comprehensive FFmpeg Path Escaping

```python
def _concat_segments(self, segment_paths, output_path, config):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in segment_paths:
            # Validate path doesn't contain newlines (would break concat format)
            path_str = str(path.resolve())
            if "\n" in path_str or "\r" in path_str:
                raise ValueError(f"Invalid path with newline characters: {path}")
            # Escape backslashes and single quotes for FFmpeg concat format
            escaped_path = path_str.replace("\\", "\\\\").replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
```

### Fix 2: Filename Sanitization

```python
@staticmethod
def _sanitize_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal and invalid characters."""
    # Remove path separators and other dangerous/invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    # Strip leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized or "video"

# Usage
source_name = self._sanitize_filename(self.current_source.file_path.stem)
```

## Prevention

**Subprocess security checklist:**
- [ ] Use argument arrays, never shell=True
- [ ] Escape all special characters for the target format
- [ ] Validate inputs don't contain format-breaking characters (newlines)
- [ ] Use `path.resolve()` to normalize paths

**Filename security checklist:**
- [ ] Sanitize user-controlled parts of filenames
- [ ] Remove path separators (`/`, `\`, `:`)
- [ ] Remove special characters (`<>"|?*`)
- [ ] Remove control characters (`\x00-\x1f`)
- [ ] Strip leading dots (hidden files)
- [ ] Limit length
- [ ] Provide fallback for empty result

**Test cases to add:**
```python
# Path traversal
assert sanitize("../../../etc/passwd") == "_.._.._etc_passwd"
# Special characters
assert sanitize('video<>:"/\\|?*name') == "video_________name"
# Leading dots
assert sanitize("...dangerous...") == "dangerous"
# Empty input
assert sanitize("") == "video"
```

## Related

- OWASP Path Traversal: https://owasp.org/www-community/attacks/Path_Traversal
- FFmpeg concat demuxer: https://ffmpeg.org/ffmpeg-formats.html#concat

## Files Changed

- `core/sequence_export.py` - Added path validation and escaping
- `ui/main_window.py` - Added `_sanitize_filename()` helper
