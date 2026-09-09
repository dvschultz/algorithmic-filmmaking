"""Task handlers executed inside the worker process.

Each handler receives ``(args, context)`` where ``context`` provides
``staging_dir`` (the only directory results may reference), ``cancel``
(threading.Event set when the host cancels), and ``progress(fraction, message)``.
Handlers return a JSON-serializable dict; any file they hand back must live
under ``staging_dir``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


class TaskCancelled(Exception):
    """Raised by a handler when it stops because the host cancelled."""


class WorkerContext:
    def __init__(self, staging_dir: Path, cancel, progress: Callable[[float, str], None]) -> None:
        self.staging_dir = staging_dir
        self.cancel = cancel
        self.progress = progress
        self.children: list[subprocess.Popen] = []

    def check_cancelled(self) -> None:
        if self.cancel.is_set():
            raise TaskCancelled()

    def run_child(self, command: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
        """Run a child (e.g. FFmpeg) that the host can tear down with the worker."""
        self.check_cancelled()
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        self.children.append(process)
        try:
            deadline = time.monotonic() + timeout if timeout else None
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if self.cancel.is_set():
                        process.kill()
                        process.communicate()
                        raise TaskCancelled()
                    if deadline is not None and time.monotonic() > deadline:
                        process.kill()
                        process.communicate()
                        raise RuntimeError(f"Child process timed out: {command[0]}")
        finally:
            self.children.remove(process)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


# --- transcription -----------------------------------------------------------


def transcribe(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    """Transcribe a media file with faster-whisper and write segments to staging."""
    media = Path(str(args["media_path"]))
    if not media.is_file():
        raise ValueError(f"Media file not found: {media}")
    model_name = str(args.get("model", "small.en"))
    language = args.get("language") or None
    ffmpeg = args.get("ffmpeg")
    context.progress(0.05, "Loading transcription model...")
    from faster_whisper import WhisperModel  # heavy runtime, imported only here

    model = WhisperModel(model_name, device=str(args.get("device", "cpu")), compute_type=str(args.get("compute_type", "int8")))
    context.check_cancelled()

    source: Path = media
    if ffmpeg:
        # Extract audio ourselves so the host can reason about one process tree.
        source = context.staging_dir / "audio.wav"
        context.progress(0.1, "Extracting audio...")
        completed = context.run_child([
            str(ffmpeg), "-y", "-i", str(media), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(source),
        ], timeout=float(args.get("extract_timeout", 600)))
        if completed.returncode != 0:
            raise RuntimeError("FFmpeg audio extraction failed: " + completed.stderr.decode("utf-8", "replace")[-500:])

    context.progress(0.2, "Transcribing...")
    segments, info = model.transcribe(
        str(source), language=language if language != "auto" else None,
        word_timestamps=bool(args.get("word_timestamps", True)), vad_filter=bool(args.get("vad_filter", True)),
    )
    results = []
    for segment in segments:
        context.check_cancelled()
        words = getattr(segment, "words", None)
        results.append({
            "start": float(segment.start), "end": float(segment.end), "text": segment.text.strip(),
            "confidence": float(segment.avg_logprob),
            "words": [
                {"start": float(w.start), "end": float(w.end), "text": w.word.strip(),
                 "probability": float(w.probability) if getattr(w, "probability", None) is not None else None}
                for w in words
            ] if words else None,
        })
        context.progress(0.2 + 0.7 * min(1.0, float(segment.end) / max(1.0, float(getattr(info, "duration", 0) or 1.0))), "Transcribing...")
    output = context.staging_dir / "transcript.json"
    output.write_text(json.dumps({
        "language": getattr(info, "language", None), "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "model": model_name, "segments": results,
    }))
    context.progress(1.0, f"Transcribed {len(results)} segments")
    return {"result_path": str(output), "segment_count": len(results), "language": getattr(info, "language", None)}


# --- test-only handlers (enabled by the host's hello.allow_test_tasks) --------


def echo(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    return {"echo": args.get("value"), "pid": os.getpid()}


def sleep(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    """Sleep in small steps; honours cancellation unless ``ignore_cancel``."""
    total = float(args.get("seconds", 1.0))
    ignore = bool(args.get("ignore_cancel", False))
    spawn_child = bool(args.get("spawn_child", False))
    if spawn_child:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        context.children.append(child)
        (context.staging_dir / "child_pid").write_text(str(child.pid))
    end = time.monotonic() + total
    while time.monotonic() < end:
        if not ignore:
            context.check_cancelled()
        time.sleep(0.05)
    return {"slept": total}


def crash(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    os._exit(int(args.get("code", 3)))


def big_output(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    return {"blob": "x" * int(args.get("size", 2_000_000))}


def escape_staging(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    outside = Path(str(args.get("path", "/etc/passwd")))
    return {"result_path": str(outside)}


def raw_stdout(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    sys.stdout.write("this is not json\n")
    sys.stdout.flush()
    return {"ok": True}


HANDLERS: dict[str, Callable[[dict[str, Any], WorkerContext], dict[str, Any]]] = {
    "transcribe": transcribe,
}

TEST_HANDLERS: dict[str, Callable[[dict[str, Any], WorkerContext], dict[str, Any]]] = {
    "echo": echo, "sleep": sleep, "crash": crash, "big_output": big_output,
    "escape_staging": escape_staging, "raw_stdout": raw_stdout,
}
