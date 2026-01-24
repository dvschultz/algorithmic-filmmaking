---
title: "Subprocess Resource Leak on Exception"
category: reliability-issues
tags: [subprocess, resource-leak, exception-handling, cleanup, python]
module: downloader
symptom: "Orphaned processes and file descriptor leaks when exception occurs during subprocess output reading"
root_cause: "subprocess.Popen stdout reading loop not wrapped in try/finally"
date: 2026-01-24
---

# Subprocess Resource Leak on Exception

## Problem

When reading from a subprocess's stdout in a loop, an exception can leave the subprocess running as an orphan, leaking resources.

## Symptom

- Orphaned processes visible in `ps aux` after application errors
- File descriptor exhaustion over time
- Zombie processes accumulating
- Background downloads continue after app crashes

## Root Cause

```python
# VULNERABLE: No cleanup on exception
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

for line in process.stdout:  # Exception here leaves process orphaned
    process_line(line)

process.wait()  # Never reached if exception thrown
```

If an exception occurs in the loop (or the thread is killed), `process.wait()` is never called, leaving the subprocess running.

## Solution

Wrap the stdout reading in try/finally with proper cleanup:

```python
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

try:
    for line in process.stdout:
        line = line.strip()
        # ... process output ...

    process.wait()
finally:
    # Ensure subprocess is cleaned up even on exception
    if process.stdout:
        process.stdout.close()
    if process.poll() is None:  # Process still running
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
```

## Cleanup Sequence

1. **Close stdout pipe** - Prevents file descriptor leak
2. **Check if running** - `process.poll() is None` means still alive
3. **Graceful terminate** - `process.terminate()` sends SIGTERM
4. **Wait with timeout** - Give process 5 seconds to exit cleanly
5. **Force kill** - `process.kill()` sends SIGKILL if timeout expires
6. **Reap zombie** - Final `process.wait()` prevents zombie

## Alternative: Context Manager

For simpler cases, use the context manager (Python 3.2+):

```python
with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as process:
    for line in process.stdout:
        process_line(line)
# Automatically cleaned up on exit
```

Note: The context manager doesn't do the terminate/kill sequence, so explicit cleanup is better for long-running processes.

## Prevention

- Always use try/finally with Popen when reading stdout/stderr
- Prefer `subprocess.run()` for simple cases (handles cleanup automatically)
- Add timeout parameters to prevent indefinite hangs
- Consider using a process manager for complex subprocess handling

## Testing

```python
def test_cleanup_on_exception():
    """Verify subprocess is killed when exception occurs."""
    import os
    import signal

    # Start a long-running process
    process = subprocess.Popen(["sleep", "60"], ...)
    pid = process.pid

    try:
        raise RuntimeError("Simulated error")
    finally:
        # Cleanup code here
        ...

    # Verify process is gone
    try:
        os.kill(pid, 0)  # Check if process exists
        assert False, "Process should be terminated"
    except OSError:
        pass  # Expected - process doesn't exist
```

## References

- [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html)
- [Subprocess management best practices](https://docs.python.org/3/library/subprocess.html#subprocess-replacements)
