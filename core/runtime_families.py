"""Native runtime families and the host-side isolation seam (plan U14).

A *family* is a set of native runtimes that share one warm worker process
(``core.runtime_supervisor``). Engine functions that touch a native runtime
are decorated with :func:`isolated`: when the family is enabled for worker
isolation the call is forwarded to the family's worker as JSON and the
host never imports the runtime; otherwise the function runs in-process as
before. The decorator is a no-op inside a worker process.

Cutover is per family (``Settings.native_worker_families`` or the
``SCENE_RIPPER_NATIVE_WORKER_FAMILIES`` env var, comma separated). The
transcription family is on by default; the others switch on once their
packaged worker smoke evidence lands. ``native_worker_isolation=False`` (or
``SCENE_RIPPER_NATIVE_WORKERS=0``) disables every family.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from core.runtime_worker.calls import ISOLATED_CALLS

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RuntimeFamily:
    id: str
    description: str
    probe_module: str
    """Module a worker imports to prove the family's runtime is usable."""


FAMILIES: dict[str, RuntimeFamily] = {
    "transcription": RuntimeFamily("transcription", "faster-whisper transcription", "faster_whisper"),
    "vision": RuntimeFamily("vision", "torch/transformers embeddings, SigLIP shots, YOLO objects, InsightFace faces, MediaPipe gaze", "torch"),
    "ocr": RuntimeFamily("ocr", "PaddleOCR text extraction", "paddleocr"),
    "vlm": RuntimeFamily("vlm", "local vision-language models (mlx-vlm / transformers)", "mlx_vlm"),
    "audio": RuntimeFamily("audio", "librosa audio analysis and Demucs stem separation", "librosa"),
    "alignment": RuntimeFamily("alignment", "CTC forced word alignment", "ctc_forced_aligner"),
}

ENV_FAMILIES = "SCENE_RIPPER_NATIVE_WORKER_FAMILIES"
ENV_WORKER_PROCESS = "SCENE_RIPPER_WORKER_PROCESS"


def in_worker_process() -> bool:
    return os.environ.get(ENV_WORKER_PROCESS) == "1"


def enabled_families() -> frozenset[str]:
    """Families whose calls are forwarded to workers in this process."""
    from core.transcription import native_worker_enabled

    if in_worker_process() or not native_worker_enabled():
        return frozenset()
    override = os.environ.get(ENV_FAMILIES)
    if override is not None:
        names = {part.strip() for part in override.split(",") if part.strip()}
    else:
        try:
            from core.settings import load_settings

            names = set(load_settings(read_keyring=False).native_worker_families)
        except Exception:  # noqa: BLE001 - settings problems must not change routing silently
            names = {"transcription"}
    return frozenset(name for name in names if name in FAMILIES)


def family_isolated(family: str) -> bool:
    return family in enabled_families()


class IsolatedCallError(RuntimeError):
    """The worker could not complete an isolated engine call."""


def _encode(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    return value


def isolated(family: str, call: str, *, decode: Callable[..., Any] | None = None) -> Callable[[F], F]:
    """Forward the decorated engine function to the family's worker when isolated.

    ``call`` must be listed in ``core.runtime_worker.calls.ISOLATED_CALLS``
    (checked at import time). Keyword arguments become JSON (paths as
    strings; callables such as progress callbacks are dropped because they
    cannot cross the process boundary). ``decode`` rebuilds the return value
    from JSON when the function does not return plain data; a two-argument
    ``decode(value, arguments)`` also sees the bound call arguments (used to
    check that returned files live where the host asked for them).
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown runtime family {family!r}")
    if call not in ISOLATED_CALLS:
        raise ValueError(f"{call!r} is not an allowlisted isolated call")

    def wrap(function: F) -> F:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not family_isolated(family):
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            cancel_event = arguments.pop("cancel_event", None)
            on_execution = arguments.pop("on_execution", None)
            payload = {
                name: _encode(value) for name, value in arguments.items()
                if not callable(value) and _is_jsonable(value)
            }
            outcome = run_isolated(family, call, payload, cancel_event=cancel_event, raw=True)
            if callable(on_execution):
                for info in outcome.get("executions") or []:
                    on_execution(info)
            value = outcome.get("value")
            if decode is None:
                return value
            if _decode_wants_arguments(decode):
                return decode(value, arguments)
            return decode(value)

        wrapper.__isolated__ = (family, call)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return wrap


def _decode_wants_arguments(decode: Callable[..., Any]) -> bool:
    try:
        return len(inspect.signature(decode).parameters) >= 2
    except (TypeError, ValueError):
        return False


def _is_jsonable(value: Any) -> bool:
    import json

    try:
        json.dumps(_encode(value))
    except (TypeError, ValueError):
        return False
    return True


def run_isolated(
    family: str, call: str, kwargs: dict[str, Any], *, cancel_event=None, timeout: float | None = None,
    raw: bool = False,
) -> Any:
    """Run one allowlisted engine call in the family's worker.

    Returns the JSON ``value`` (or the whole worker result with ``raw=True``,
    which also carries the collected ``executions``).
    """
    from core.runtime_supervisor import (
        WorkerCancelled, WorkerCrashed, WorkerError, WorkerTaskError, default_supervisor,
    )

    if call not in ISOLATED_CALLS:
        raise ValueError(f"{call!r} is not an allowlisted isolated call")
    try:
        result = default_supervisor().run(
            family, "analysis", {"call": call, "kwargs": kwargs}, cancel_event=cancel_event, timeout=timeout,
        )
    except WorkerCancelled:
        raise
    except TypeError as exc:  # a kwarg that cannot be encoded as JSON never left the host
        raise IsolatedCallError(f"{call}: arguments are not JSON-encodable: {exc}") from exc
    except WorkerTaskError as exc:
        # Keep the in-process exception classes so callers' outcome codes
        # (dependency_missing, model download, critical batch halts) hold.
        kind = getattr(exc, "kind", "task")
        if kind == "dependency_missing":
            raise ImportError(f"{FAMILIES[family].description} is not installed in the worker: {exc}") from exc
        if kind == "model":
            from core.errors import ModelDownloadError

            raise ModelDownloadError(str(exc)) from exc
        raise IsolatedCallError(f"{call} failed in the {family} worker: {exc}") from exc
    except WorkerCrashed as exc:
        raise IsolatedCallError(f"The {family} worker crashed during {call}: {exc}") from exc
    except WorkerError as exc:
        raise IsolatedCallError(f"{call} could not run in the {family} worker: {exc}") from exc
    return result if raw else result.get("value")


# Families that import each native runtime in-process when they are NOT isolated.
_RUNTIME_FAMILIES: dict[str, frozenset[str]] = {
    "torch": frozenset({"vision", "vlm", "alignment", "audio"}),
    "mlx": frozenset({"vlm"}),
}


def host_needs_runtime(runtime: str) -> bool:
    """Whether the GUI host must still be able to import ``runtime`` itself.

    ``main.py`` pre-imports torch (before PySide6) and MLX (on the main thread)
    to dodge Shiboken import-hook crashes; those workarounds are only needed
    while some family still runs that runtime in-process. The MLX whisper
    backend is not isolated, so MLX stays needed while that backend can be
    selected.
    """
    families = _RUNTIME_FAMILIES.get(runtime)
    if families is None:
        return True
    isolated_now = enabled_families()
    if not families.issubset(isolated_now):
        return True
    if runtime == "mlx":
        try:
            from core.settings import load_settings

            backend = str(load_settings(read_keyring=False).transcription_backend)
        except Exception:  # noqa: BLE001
            backend = "auto"
        return backend in ("auto", "mlx-whisper")
    return False
