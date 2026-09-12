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
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .protocol import TRANSCRIPT_SCHEMA_VERSION, validate_transcript_payload


class TaskCancelled(Exception):
    """Raised by a handler when it stops because the host cancelled."""


class ModelLoadError(RuntimeError):
    """A model could not be loaded or downloaded (reported with kind='model')."""


class WorkerContext:
    def __init__(
        self, staging_dir: Path, cancel, progress: Callable[[float, str], None], *, allow_test_tasks: bool = False,
    ) -> None:
        self.staging_dir = staging_dir
        self.cancel = cancel
        self.progress = progress
        self.allow_test_tasks = allow_test_tasks
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

_whisper_cache: dict[tuple[str, str, str], Any] = {}


def _whisper_model(model_name: str, device: str, compute_type: str):
    """One loaded model per worker; batch transcription must not reload per clip."""
    from faster_whisper import WhisperModel  # heavy runtime, imported only here

    key = (model_name, device, compute_type)
    model = _whisper_cache.get(key)
    if model is None:
        _whisper_cache.clear()  # one warm model per family keeps memory bounded
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:  # noqa: BLE001 - download/load failures are a distinct kind
            raise ModelLoadError(f"Could not load model {model_name!r}: {exc}") from exc
        _whisper_cache[key] = model
    return model



def transcribe(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    """Transcribe a media file with faster-whisper and write segments to staging."""
    media = Path(str(args["media_path"]))
    if not media.is_file():
        raise ValueError(f"Media file not found: {media}")
    model_name = str(args.get("model", "small.en"))
    language = args.get("language") or None
    ffmpeg = args.get("ffmpeg")
    context.progress(0.05, "Loading transcription model...")
    model = _whisper_model(model_name, str(args.get("device", "cpu")), str(args.get("compute_type", "int8")))
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
    output.write_text(json.dumps(validate_transcript_payload({
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "language": getattr(info, "language", None), "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "model": model_name, "segments": results,
    })))
    context.progress(1.0, f"Transcribed {len(results)} segments")
    return {"result_path": str(output), "segment_count": len(results), "language": getattr(info, "language", None)}


# --- test-only handlers (enabled by the host's hello.allow_test_tasks) --------
#
# These ship in the worker on purpose: the protocol tests and the packaged
# `native-worker` smoke target need deterministic crash, oversize, and
# staging-escape behaviour from the *real* worker binary. They are inert
# unless the host's hello message sets allow_test_tasks, which only the smoke
# target and tests do; the production supervisor never sets it.


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


# --- runtime health -----------------------------------------------------------

# Modules a host may ask this worker to import as a health check. The worker
# is the only process that ever imports native runtimes, so a host validating
# an install never loads wheels built for another interpreter.
PROBE_MODULES: dict[str, str] = {
    "faster_whisper": "WhisperModel",
    "torch": "Tensor",
    "transformers": "AutoModel",
    "paddleocr": "PaddleOCR",
    "mlx_vlm": "load",
    "librosa": "beat",
    "ctc_forced_aligner": "load_alignment_model",
}


def _note(text: str) -> None:
    """Progress breadcrumb on stderr; the host keeps a tail of these."""
    sys.stderr.write(text.rstrip() + "\n")
    sys.stderr.flush()


def probe(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    """Import an allowlisted runtime module here and report its version."""
    name = str(args.get("module", ""))
    attribute = PROBE_MODULES.get(name)
    if attribute is None:
        raise ValueError(f"Unknown probe module {name!r}; known: {', '.join(sorted(PROBE_MODULES))}")
    # A probe that times out used to leave no trace of where it stopped: on a
    # Windows runner importing a tree pip had just written, the worker was silent
    # for the whole 600s ceiling. These land on stderr, which the host keeps.
    _note(f"probe: importing {name}")
    module = importlib.import_module(name)
    _note(f"probe: imported {name}, resolving {attribute}")
    getattr(module, attribute)  # a half-installed runtime fails here, like a real import would
    _note(f"probe: {name} ok")
    return {
        "ok": True, "module": name,
        "version": str(getattr(module, "__version__", "") or ""),
        "python": sys.executable,
        "file": str(getattr(module, "__file__", "") or ""),
    }


# --- isolated engine calls -----------------------------------------------------

def analysis(args: dict[str, Any], context: WorkerContext) -> dict[str, Any]:
    """Run one allowlisted engine call with JSON kwargs; result must be JSON.

    The engine package (``core``) is on the worker path in source mode and
    staged beside this package in frozen builds. Anything not in
    ``ISOLATED_CALLS`` is refused before any import happens.
    """
    from .calls import DISCARD_RESULT, ISOLATED_CALLS, TEST_CALLS

    name = str(args.get("call", ""))
    target = ISOLATED_CALLS.get(name)
    if target is None:
        raise ValueError(f"Unknown isolated call {name!r}")
    if name in TEST_CALLS and not context.allow_test_tasks:
        raise ValueError(f"Isolated call {name!r} is a diagnostic; the host did not enable test tasks")
    kwargs = args.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be an object")
    os.environ["SCENE_RIPPER_WORKER_PROCESS"] = "1"  # the decorator must not re-forward
    module_name, function_name = target
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    kwargs = _rebuild_arguments(function, kwargs)
    executions: list[Any] = []
    # Provenance/progress callbacks cannot cross the process boundary; the
    # worker collects them and the host replays them to its own callbacks.
    parameters = _signature_parameters(function)
    if "on_execution" in parameters:
        kwargs["on_execution"] = lambda info: executions.append(_jsonable(info))
    if "progress_cb" in parameters:
        kwargs["progress_cb"] = lambda message: context.progress(0.5, str(message))
    if "cancel_event" in parameters:
        kwargs["cancel_event"] = context.cancel
    context.check_cancelled()
    try:
        value = function(**kwargs)
    except (ModelLoadError, ImportError, TaskCancelled):
        raise  # classified by the task loop (model / dependency_missing / cancelled)
    except Exception as exc:
        # Keep the engine's failure classes: a model download/load failure must
        # surface as kind "model" so batch runners halt instead of retrying.
        if type(exc).__name__ == "ModelDownloadError":
            raise ModelLoadError(str(exc)) from exc
        raise AnalysisCallError(exc, executions) from exc
    result: dict[str, Any] = {"call": name, "executions": executions}
    encoded = None if name in DISCARD_RESULT else _jsonable(value)
    payload = json.dumps(encoded, ensure_ascii=True, allow_nan=False)
    if len(payload) > LARGE_VALUE_BYTES:
        # Big embeddings/face batches would exceed the protocol's line bound;
        # hand them over through the task's staging directory instead.
        value_file = context.staging_dir / "value.json"
        value_file.write_text(payload, encoding="utf-8")
        result["value_path"] = str(value_file)
    else:
        result["value"] = encoded
    return result


LARGE_VALUE_BYTES = 256 * 1024


class AnalysisCallError(RuntimeError):
    """An engine call failed; carries the provenance it reported before failing."""

    def __init__(self, cause: BaseException, executions: list[Any]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.executions = executions


def _signature_parameters(function: Callable[..., Any]) -> dict[str, Any]:
    import inspect

    try:
        return dict(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return {}


def _rebuild_arguments(function: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Turn JSON kwargs back into what the engine function expects.

    Paths are rebuilt from ``Path`` annotations, dataclass parameters from
    dicts, and lists of objects with ``from_dict`` from their dicts. Anything
    the host could not send (callables, events) is simply absent.
    """
    import typing

    try:
        hints = typing.get_type_hints(function)
    except Exception:  # noqa: BLE001 - string annotations may not resolve; fall back to raw values
        hints = {}
    rebuilt: dict[str, Any] = {}
    for name, value in kwargs.items():
        hint = hints.get(name)
        rebuilt[name] = _rebuild_value(hint, value)
    return rebuilt


def _rebuild_value(hint: Any, value: Any) -> Any:
    import dataclasses
    import typing

    origin = typing.get_origin(hint)
    if origin is typing.Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        for candidate in typing.get_args(hint):
            if candidate is type(None):
                continue
            rebuilt = _rebuild_value(candidate, value)
            if rebuilt is not value:
                return rebuilt
        return value
    if hint is Path and isinstance(value, str):
        return Path(value)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint) and isinstance(value, dict):
        if hasattr(hint, "from_dict"):
            return hint.from_dict(value)
        return hint(**value)
    if origin in (list, tuple) and isinstance(value, list):
        args = typing.get_args(hint)
        if args:
            return [_rebuild_value(args[0], item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    """Coerce engine results to JSON: paths, tuples, dataclasses, numpy scalars/arrays."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist") and callable(value.tolist):  # numpy scalar/array
        return _jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return _jsonable(dataclasses.asdict(value))
    return str(value)


HANDLERS: dict[str, Callable[[dict[str, Any], WorkerContext], dict[str, Any]]] = {
    "transcribe": transcribe,
    "probe": probe,
    "analysis": analysis,
}

TEST_HANDLERS: dict[str, Callable[[dict[str, Any], WorkerContext], dict[str, Any]]] = {
    "echo": echo, "sleep": sleep, "crash": crash, "big_output": big_output,
    "escape_staging": escape_staging, "raw_stdout": raw_stdout,
}
