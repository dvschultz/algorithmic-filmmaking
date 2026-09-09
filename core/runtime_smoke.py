"""Frozen runtime smoke checks for release validation."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import wave
from typing import Callable
from pathlib import Path

import cv2
import numpy as np

from core.binary_resolver import find_binary, is_bundled_binary_path, get_subprocess_kwargs
from core.project import Project
from core.scene_detect import DetectionConfig, SceneDetector
from models.clip import Clip, Source

logger = logging.getLogger(__name__)

RUNTIME_SMOKE_TARGET_ENV = "SCENE_RIPPER_RUNTIME_SMOKE_TEST_TARGET"


def get_runtime_smoke_targets() -> tuple[str, ...]:
    """Return the supported frozen runtime smoke targets."""
    return (
        "imports",
        "project",
        "scene-detect",
        "transcription",
        "updater",
        "analyze-clip",
        "sequence-build",
        "render-short",
        "mcp-stdio",
        "native-worker",
        "native-analysis",
    )


def run_runtime_smoke_target(target: str) -> str:
    """Execute a single frozen runtime smoke target."""
    normalized = (target or "").strip().lower()
    handlers = {
        "imports": _run_imports_smoke,
        "project": _run_project_smoke,
        "scene-detect": _run_scene_detect_smoke,
        "transcription": _run_transcription_smoke,
        "updater": _run_updater_smoke,
        "analyze-clip": _run_analyze_clip_smoke,
        "sequence-build": _run_sequence_build_smoke,
        "render-short": _run_render_short_smoke,
        "mcp-stdio": _run_mcp_stdio_smoke,
        "native-worker": _run_native_worker_smoke,
        "native-analysis": _run_native_analysis_smoke,
    }
    handler = handlers.get(normalized)
    if handler is None:
        valid = ", ".join(sorted(handlers))
        raise ValueError(f"Unknown runtime smoke target '{target}'. Expected one of: {valid}")

    logger.info("Running runtime smoke target: %s", normalized)
    handler()
    return normalized


def _run_imports_smoke() -> None:
    """Validate that packaged dynamic-import dependencies load."""
    import google_auth_httplib2  # noqa: F401
    import httplib2  # noqa: F401
    import httpx  # noqa: F401
    import keyring  # noqa: F401
    import litellm
    import mpv  # noqa: F401
    import scipy  # noqa: F401
    import tenacity  # noqa: F401
    from googleapiclient.discovery import build  # noqa: F401
    from litellm.exceptions import RateLimitError  # noqa: F401
    from sklearn.cluster import KMeans  # noqa: F401

    if sys.platform == "win32":
        from keyring.backends.Windows import WinVaultKeyring

        backend = WinVaultKeyring()
        if backend.priority <= 0:
            raise RuntimeError("Windows keyring backend did not initialize correctly.")

    if not hasattr(litellm, "completion"):
        raise RuntimeError("LiteLLM completion API missing from bundled runtime.")


def _run_project_smoke() -> None:
    """Validate project persistence and sequence operations."""
    with tempfile.TemporaryDirectory(prefix="scene-ripper-project-smoke-") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "placeholder.mp4"
        video_path.write_bytes(b"smoke")

        project = Project.new(name="Runtime Smoke")
        source = Source(
            file_path=video_path,
            duration_seconds=6.0,
            fps=24.0,
            width=96,
            height=72,
            cut=True,
        )
        project.add_source(source)

        clips = [
            Clip(source_id=source.id, start_frame=0, end_frame=24),
            Clip(source_id=source.id, start_frame=24, end_frame=48),
        ]
        project.add_clips(clips)
        project.add_to_sequence([clip.id for clip in clips])

        project_path = tmp_path / "runtime-smoke.sceneripper"
        project.save(project_path)
        loaded = Project.load(project_path)

        if loaded.metadata.name != "Runtime Smoke":
            raise RuntimeError("Project name was not preserved across save/load.")
        if len(loaded.sources) != 1 or len(loaded.clips) != 2:
            raise RuntimeError("Project sources/clips were not preserved across save/load.")
        if loaded.sequence is None or len(loaded.sequence.get_all_clips()) != 2:
            raise RuntimeError("Project sequence did not round-trip correctly.")


def _run_scene_detect_smoke() -> None:
    """Validate synthetic scene detection in the frozen runtime."""
    with tempfile.TemporaryDirectory(prefix="scene-ripper-detect-smoke-") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "synthetic-detect.mp4"
        _create_synthetic_scene_video(video_path)

        detector = SceneDetector(
            DetectionConfig(
                threshold=1.0,
                min_scene_length=5,
                use_adaptive=False,
                luma_only=False,
            )
        )
        source, clips = detector.detect_scenes(video_path)

        if source.width != 96 or source.height != 72:
            raise RuntimeError("Synthetic detection video metadata was not read correctly.")
        if len(clips) < 3:
            raise RuntimeError(f"Expected at least 3 clips from synthetic scene detect, got {len(clips)}.")


def _run_transcription_smoke() -> None:
    """Validate the FFmpeg path used by transcription audio extraction."""
    from core.paths import is_frozen
    from core.transcription import _require_ffmpeg

    ffmpeg = _require_ffmpeg()
    ffprobe = find_binary("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for transcription smoke but was not resolved.")

    if is_frozen():
        for name, binary in {"ffmpeg": ffmpeg, "ffprobe": ffprobe}.items():
            if not is_bundled_binary_path(binary):
                raise RuntimeError(f"Frozen app resolved {name} outside bundled runtime: {binary}")

    with tempfile.TemporaryDirectory(prefix="scene-ripper-transcription-smoke-") as tmp:
        tmp_path = Path(tmp)
        source_wav = tmp_path / "source.wav"
        extracted_wav = tmp_path / "extracted.wav"
        _create_synthetic_audio(source_wav)

        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss", "0",
                "-to", "0.5",
                "-i", str(source_wav),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(extracted_wav),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            **get_subprocess_kwargs(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg transcription smoke extraction failed: "
                f"{(result.stderr or result.stdout).strip()}"
            )
        if not extracted_wav.is_file() or extracted_wav.stat().st_size == 0:
            raise RuntimeError("FFmpeg transcription smoke produced no extracted audio.")


def _run_updater_smoke() -> None:
    """Validate bundled Windows updater availability metadata."""
    if sys.platform != "win32":
        return

    from core.windows_updater import get_status

    status = get_status(update_channel="stable")
    if not status.available:
        raise RuntimeError(f"Windows updater unavailable in frozen runtime: {status.reason}")
    if status.dll_path is None or not status.dll_path.exists():
        raise RuntimeError("Bundled WinSparkle DLL missing from updater status.")
    if not status.feed_url:
        raise RuntimeError("Bundled WinSparkle feed URL missing from updater status.")
    if not status.public_key:
        raise RuntimeError("Bundled WinSparkle public key missing from updater status.")


def _run_analyze_clip_smoke() -> None:
    """Validate the non-ML color + brightness analysis paths on a synthetic clip."""
    from core.analysis.color import extract_dominant_colors, get_average_brightness

    with tempfile.TemporaryDirectory(prefix="scene-ripper-analyze-smoke-") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "synthetic-analyze.mp4"
        _create_synthetic_scene_video(video_path)

        # The synthetic clip is three solid-color scenes of 18 frames each at
        # 24fps; analyze the first (blue) scene so brightness is well below 1.0.
        start_frame, end_frame = 0, 18

        colors = extract_dominant_colors(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            n_colors=3,
        )
        if not colors:
            raise RuntimeError("Color analysis returned no dominant colors for synthetic clip.")
        for color in colors:
            if len(color) != 3 or not all(0 <= channel <= 255 for channel in color):
                raise RuntimeError(f"Color analysis produced an out-of-range RGB tuple: {color}")

        brightness = get_average_brightness(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=24.0,
        )
        if not 0.0 <= brightness <= 1.0:
            raise RuntimeError(f"Brightness analysis produced an out-of-range value: {brightness}")


def _run_sequence_build_smoke() -> None:
    """Validate building a Sequence with SequenceClips from detected clips."""
    with tempfile.TemporaryDirectory(prefix="scene-ripper-sequence-smoke-") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "synthetic-sequence.mp4"
        _create_synthetic_scene_video(video_path)

        detector = SceneDetector(
            DetectionConfig(
                threshold=1.0,
                min_scene_length=5,
                use_adaptive=False,
                luma_only=False,
            )
        )
        source, clips = detector.detect_scenes(video_path)
        if len(clips) < 2:
            raise RuntimeError(
                f"Expected at least 2 clips to build a sequence, got {len(clips)}."
            )

        project = Project.new(name="Sequence Build Smoke")
        project.add_source(source)
        project.add_clips(clips)

        # Build the sequence through the core API (appends to track 0).
        sequence_clip_ids = [clip.id for clip in clips[:3]]
        project.add_to_sequence(sequence_clip_ids)

        sequence = project.sequence
        if sequence is None:
            raise RuntimeError("Sequence build smoke produced no sequence.")

        seq_clips = sequence.get_all_clips()
        if len(seq_clips) != len(sequence_clip_ids):
            raise RuntimeError(
                "Sequence build smoke expected "
                f"{len(sequence_clip_ids)} sequence clips, got {len(seq_clips)}."
            )

        # SequenceClips must be laid end-to-end with plausible trim points and
        # source references back to the detected clips.
        expected_start = 0
        for seq_clip in seq_clips:
            if seq_clip.source_clip_id not in {clip.id for clip in clips}:
                raise RuntimeError("Sequence clip references an unknown source clip.")
            if seq_clip.out_point <= seq_clip.in_point:
                raise RuntimeError(
                    "Sequence clip has a non-positive duration "
                    f"({seq_clip.in_point}-{seq_clip.out_point})."
                )
            if seq_clip.start_frame != expected_start:
                raise RuntimeError(
                    "Sequence clips are not laid end-to-end: expected start "
                    f"{expected_start}, got {seq_clip.start_frame}."
                )
            expected_start += seq_clip.duration_frames

        if sequence.duration_frames <= 0:
            raise RuntimeError("Sequence build smoke produced a zero-length timeline.")


def _run_render_short_smoke() -> None:
    """Validate rendering a short sequence to a playable MP4 through the export path."""
    from core.sequence_export import export_sequence

    ffprobe = find_binary("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for render-short smoke but was not resolved.")

    from core.paths import is_frozen

    if is_frozen() and not is_bundled_binary_path(ffprobe):
        raise RuntimeError(f"Frozen app resolved ffprobe outside bundled runtime: {ffprobe}")

    with tempfile.TemporaryDirectory(prefix="scene-ripper-render-smoke-") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "synthetic-render.mp4"
        _create_synthetic_scene_video(video_path)

        detector = SceneDetector(
            DetectionConfig(
                threshold=1.0,
                min_scene_length=5,
                use_adaptive=False,
                luma_only=False,
            )
        )
        source, clips = detector.detect_scenes(video_path)
        if not clips:
            raise RuntimeError("Render-short smoke detected no clips to render.")

        project = Project.new(name="Render Short Smoke")
        project.add_source(source)
        project.add_clips(clips)
        sequence = project.sequence
        if sequence is None:
            raise RuntimeError("Render-short smoke produced no sequence to render.")
        sequence.fps = source.fps or 24.0
        project.add_to_sequence([clip.id for clip in clips])

        # Keep the render well under 5 seconds: the synthetic clip is ~2.25s of
        # source at 24fps, and the sequence renders at its own fps (24).
        if sequence.duration_seconds > 5.0:
            raise RuntimeError(
                f"Render-short smoke sequence too long: {sequence.duration_seconds:.2f}s > 5s."
            )

        sources = {source.id: source}
        clip_map = {clip.id: (clip, source) for clip in clips}
        output_path = tmp_path / "render-short.mp4"

        ok = export_sequence(
            sequence=sequence,
            sources=sources,
            clips=clip_map,
            output_path=output_path,
        )
        if not ok:
            raise RuntimeError("Render-short smoke export returned failure.")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Render-short smoke produced no output file.")

        # ffprobe must parse the output and report a video stream with duration.
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration:stream=codec_type",
                "-of", "json",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            **get_subprocess_kwargs(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Render-short smoke ffprobe failed: "
                f"{(result.stderr or result.stdout).strip()}"
            )
        try:
            probe = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Render-short smoke ffprobe returned invalid JSON: {exc}") from exc

        streams = probe.get("streams", [])
        if not any(stream.get("codec_type") == "video" for stream in streams):
            raise RuntimeError("Render-short smoke output has no video stream.")

        duration = float(probe.get("format", {}).get("duration", 0.0) or 0.0)
        if duration <= 0.0:
            raise RuntimeError("Render-short smoke output reported a non-positive duration.")


def _run_mcp_stdio_smoke() -> None:
    """Validate the MCP server over stdio: initialize handshake + one read tool call.

    Speaks the newline-delimited JSON-RPC framing directly over the subprocess
    pipes (rather than pulling in the async MCP client) so the smoke run stays
    synchronous and can bound every read with a timeout. A hung server is
    surfaced as a timeout error instead of wedging the whole smoke run.
    """
    # Prefer the ``scene-ripper-mcp`` console entry point (the canonical launch
    # used by the release pipeline and documented for external agents). Fall
    # back to importing the server under its real module name for a bare source
    # checkout without the script installed. We deliberately avoid
    # ``python -m scene_ripper_mcp.server`` here: running the module as
    # ``__main__`` creates a second ``scene_ripper_mcp.server`` module when the
    # tool packages do ``from scene_ripper_mcp.server import mcp``, so the
    # decorators register on a different ``mcp`` than ``main()`` runs and the
    # server exposes zero tools.
    import shutil

    entry_point = shutil.which("scene-ripper-mcp")
    if entry_point:
        command = [entry_point, "--transport", "stdio"]
    else:
        command = [
            sys.executable,
            "-c",
            "from scene_ripper_mcp.server import main; main()",
            "--transport",
            "stdio",
        ]

    with tempfile.TemporaryDirectory(prefix="scene-ripper-mcp-smoke-") as tmp:
        # list_projects requires an existing directory under a safe root; the
        # temp dir qualifies and contains no projects, so we expect count == 0.
        search_dir = str(Path(tmp).resolve())

        env = dict(os.environ)
        # Keep the child's cache/jobs DB inside the temp dir so the smoke run
        # never touches the user's real cache (the server's lifespan opens a
        # jobs.db under the cache dir on startup).
        env["SCENE_RIPPER_CACHE_DIR"] = search_dir
        env.setdefault("MCP_TOOL_TIMEOUT", "60")

        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            env=env,
            **get_subprocess_kwargs(),
        )

        try:
            _mcp_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "runtime-smoke", "version": "1.0"},
                    },
                },
            )
            init_result = _mcp_read_response(proc, expected_id=1, timeout=60.0)
            negotiated = (init_result.get("result") or {}).get("protocolVersion")
            if not negotiated:
                raise RuntimeError(
                    f"MCP initialize did not return a protocol version: {init_result}"
                )

            # Acknowledge initialization (notification, no response expected).
            _mcp_send(
                proc,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )

            _mcp_send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "list_projects",
                        "arguments": {"directory": search_dir},
                    },
                },
            )
            call_result = _mcp_read_response(proc, expected_id=2, timeout=60.0)
            if "error" in call_result:
                raise RuntimeError(f"MCP list_projects returned an error: {call_result['error']}")

            payload = _mcp_extract_tool_json(call_result)
            if payload.get("success") is not True:
                raise RuntimeError(f"MCP list_projects reported failure: {payload}")
            if "count" not in payload or "projects" not in payload:
                raise RuntimeError(f"MCP list_projects response missing expected keys: {payload}")
        finally:
            _mcp_terminate(proc)


def _mcp_send(proc: subprocess.Popen, message: dict) -> None:
    """Write one newline-delimited JSON-RPC message to the MCP server stdin."""
    if proc.stdin is None:
        raise RuntimeError("MCP server subprocess has no stdin pipe.")
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _mcp_read_response(proc: subprocess.Popen, expected_id: int, timeout: float) -> dict:
    """Read JSON-RPC lines until the response with ``expected_id`` arrives.

    Skips notifications/requests initiated by the server (which have no ``id``
    matching our request). Raises on timeout, EOF, or a crashed subprocess.
    """
    import time

    if proc.stdout is None:
        raise RuntimeError("MCP server subprocess has no stdout pipe.")

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Timed out waiting {timeout:.0f}s for MCP response id={expected_id}."
            )

        line = _readline_with_timeout(proc, remaining)
        if line is None:
            code = proc.poll()
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"MCP server closed stdout before response id={expected_id} "
                f"(exit={code}). stderr: {stderr.strip()[:500]}"
            )

        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON logging that leaked onto stdout; ignore and keep reading.
            continue
        if message.get("id") == expected_id:
            return message
        # Otherwise it's an unrelated notification/request; keep waiting.


def _readline_with_timeout(proc: subprocess.Popen, timeout: float) -> str | None:
    """Read a single line from proc.stdout, giving up after ``timeout`` seconds.

    Returns the line (with trailing newline) or None on EOF. Uses a watchdog
    thread so a hung server can't block forever on ``readline()``.
    """
    import threading

    result: dict[str, str | None] = {"line": None}

    def _reader() -> None:
        try:
            result["line"] = proc.stdout.readline()  # type: ignore[union-attr]
        except (ValueError, OSError):
            result["line"] = None

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        # Reader is still blocked; killing the process unblocks readline() and
        # lets the daemon thread exit. Signal a timeout to the caller.
        proc.kill()
        raise RuntimeError(f"Timed out waiting {timeout:.0f}s for an MCP stdout line.")
    line = result["line"]
    if line == "":
        return None
    return line


def _mcp_extract_tool_json(response: dict) -> dict:
    """Extract the JSON payload returned by an MCP tool call.

    FastMCP wraps string tool returns in a ``content`` list of text parts; the
    tool itself returns a JSON string, so parse the first text block.
    """
    result = response.get("result") or {}
    content = result.get("content") or []
    for part in content:
        if part.get("type") == "text":
            try:
                return json.loads(part.get("text") or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"MCP tool returned non-JSON text content: {exc}"
                ) from exc
    # Some servers also surface parsed data via structuredContent.
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    raise RuntimeError(f"MCP tool response had no parseable content: {response}")


def _mcp_terminate(proc: subprocess.Popen) -> None:
    """Terminate the MCP server subprocess cleanly, escalating to kill."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except OSError:
            pass


def _create_synthetic_scene_video(path: Path) -> Path:
    """Create a small MP4 with clear hard cuts for smoke validation."""
    width, height, fps = 96, 72, 24.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create synthetic runtime smoke video: {path}")

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]
    for color in colors:
        for _ in range(18):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)

    writer.release()
    if not path.exists():
        raise RuntimeError(f"Synthetic runtime smoke video was not created: {path}")
    return path


def _create_synthetic_audio(path: Path) -> Path:
    """Create a tiny mono WAV file that FFmpeg can trim like transcription input."""
    sample_rate = 16000
    duration_seconds = 1.0
    amplitude = 12000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(int(sample_rate * duration_seconds)):
            value = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return path


def _run_native_worker_smoke() -> None:
    """Prove the managed worker launches under an explicit interpreter and can infer.

    Frozen apps must launch the worker from the staged ``runtime_worker_src``
    package under the managed Python, never the frozen executable. The
    handshake and a stdlib task are mandatory. Real transcription runs when the
    ``transcription-whisper`` profile is installed; set
    ``SCENE_RIPPER_SMOKE_INSTALL_PROFILES=1`` to install it first, otherwise a
    missing profile is a reported failure so packaged evidence stays honest.
    """
    import os

    from core.paths import get_managed_python_dir, is_frozen
    from core.runtime_profiles import install_profile, profile_status
    from core.runtime_supervisor import RuntimeSupervisor, default_launch, worker_package_root
    from core.transcription import _transcribe_in_worker

    launch = default_launch("transcription", ensure_interpreter=is_frozen())
    if is_frozen():
        if launch.interpreter.resolve() == Path(sys.executable).resolve():
            raise RuntimeError("Frozen app resolved the worker interpreter to its own executable.")
        managed_dir = get_managed_python_dir().resolve()
        if managed_dir not in launch.interpreter.resolve().parents:
            raise RuntimeError(f"Worker interpreter is not the managed Python: {launch.interpreter}")
        if worker_package_root().name != "runtime_worker_src":
            raise RuntimeError("Frozen app did not resolve the staged runtime_worker_src package.")

    supervisor = RuntimeSupervisor(allow_test_tasks=True)
    supervisor.launch_factory = lambda family: launch
    try:
        worker = supervisor.worker("transcription")
        if worker.python is None or Path(worker.python).resolve() != launch.interpreter.resolve():
            raise RuntimeError(f"Worker reported interpreter {worker.python}, expected {launch.interpreter}")
        echoed = supervisor.run("transcription", "echo", {"value": "smoke"})
        if echoed.get("echo") != "smoke" or echoed.get("pid") == os.getpid():
            raise RuntimeError("Worker echo task did not run in a separate process.")
        logger.info("Native worker handshake OK: pid=%s python=%s", worker.pid, worker.python)

        status = profile_status("transcription-whisper")
        if not status["installed"]:
            if os.environ.get("SCENE_RIPPER_SMOKE_INSTALL_PROFILES") == "1":
                status = install_profile("transcription-whisper")
            if not status["installed"]:
                raise RuntimeError(
                    "transcription-whisper profile is not installed; missing "
                    + ", ".join(status["missing"]) + ". Set SCENE_RIPPER_SMOKE_INSTALL_PROFILES=1 to install."
                )
        with tempfile.TemporaryDirectory(prefix="scene-ripper-native-worker-smoke-") as tmp:
            wav = _write_tone_wav(Path(tmp) / "tone.wav", sample_rate=16000)
            import core.runtime_supervisor as supervisor_module

            previous_env = os.environ.get("SCENE_RIPPER_NATIVE_WORKERS")
            os.environ["SCENE_RIPPER_NATIVE_WORKERS"] = "1"
            previous = supervisor_module._default
            supervisor_module._default = supervisor
            try:
                segments, language = _transcribe_in_worker(wav, "tiny.en", "en", None, extract_audio=True)
            finally:
                supervisor_module._default = previous
                if previous_env is None:
                    os.environ.pop("SCENE_RIPPER_NATIVE_WORKERS", None)
                else:
                    os.environ["SCENE_RIPPER_NATIVE_WORKERS"] = previous_env
            logger.info("Native worker transcription OK: %d segments, language=%s", len(segments), language)
    finally:
        supervisor.shutdown()


# Per-family packaged proof (plan U14). Each runtime profile is health-checked
# inside its worker; families with a model-free or tiny-model call also run a
# real isolated analysis. Profiles that are not installed are reported as
# "missing" (or installed when SCENE_RIPPER_SMOKE_INSTALL_PROFILES=1), and
# profiles unsupported on this platform as "unsupported"; both are explicit
# results, never silent passes. The target fails when any installed family
# cannot import or run.
def _analysis_smoke_families() -> frozenset[str]:
    """Every runtime family except transcription, which the native-worker target proves."""
    from core.runtime_families import FAMILIES

    return frozenset(FAMILIES) - {"transcription"}


def _write_tone_wav(path: Path, *, sample_rate: int = 22050, seconds: float = 1.0, frequency: float = 220.0) -> Path:
    """A mono 16-bit sine tone; whisper smoke uses 16 kHz, librosa smoke 22.05 kHz."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(int(sample_rate * seconds)):
            value = int(8000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            handle.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return path


def _synthetic_frame(path: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 200), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 140, 120), fill=(30, 60, 200))
    draw.text((40, 150), "SCENE RIPPER", fill=(0, 0, 0))
    image.save(path)
    return path


def _run_native_analysis_smoke() -> None:
    import platform
    import tempfile

    from core.runtime_families import FAMILIES
    from core.runtime_profiles import PROFILES, install_profile, probe_profile_runtime, profile_status

    install_allowed = os.environ.get("SCENE_RIPPER_SMOKE_INSTALL_PROFILES") == "1"
    only = {f.strip() for f in os.environ.get("SCENE_RIPPER_SMOKE_FAMILIES", "").split(",") if f.strip()}
    results: dict[str, str] = {}
    failures: list[str] = []
    previous = {key: os.environ.get(key) for key in ("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "SCENE_RIPPER_NATIVE_WORKERS")}
    os.environ["SCENE_RIPPER_NATIVE_WORKER_FAMILIES"] = ",".join(FAMILIES)
    os.environ["SCENE_RIPPER_NATIVE_WORKERS"] = "1"
    smoke_families = _analysis_smoke_families()
    try:
        with tempfile.TemporaryDirectory(prefix="scene-ripper-native-analysis-") as tmp:
            work = Path(tmp)
            for profile_id, profile in PROFILES.items():
                family = profile.family
                if family not in smoke_families or (only and family not in only):
                    continue
                if profile.probe_module == "mlx_vlm" and not (platform.system() == "Darwin" and platform.machine() == "arm64"):
                    results[profile_id] = "unsupported"
                    continue
                status = profile_status(profile_id)
                if not status["installed"] and install_allowed:
                    status = install_profile(profile_id)
                if not status.get("installed"):
                    results[profile_id] = "missing: " + ", ".join(status.get("missing") or []) or "missing"
                    continue
                try:
                    health = probe_profile_runtime(profile_id)
                    logger.info("Family %s probe OK: %s %s", family, profile.probe_module, health.get("version"))
                    detail = _run_family_call(family, work)
                    results[profile_id] = "ok" + (f" ({detail})" if detail else "")
                except Exception as exc:  # noqa: BLE001 - report every family, then fail once
                    results[profile_id] = f"failed: {exc}"
                    failures.append(profile_id)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from core.runtime_supervisor import shutdown_default_supervisor

        shutdown_default_supervisor()
    for profile_id, outcome in results.items():
        logger.info("Native analysis family result: %s -> %s", profile_id, outcome)
    if failures:
        raise RuntimeError("Isolated analysis failed for: " + ", ".join(f"{p} ({results[p]})" for p in failures))
    if not any(outcome.startswith("ok") for outcome in results.values()):
        raise RuntimeError("No native analysis family ran; results: " + ", ".join(f"{k}={v}" for k, v in results.items()))


def _smoke_audio(work: Path) -> str:
    # Through the public decorated entry point, exactly as the operations call it.
    from core.analysis.audio import analyze_audio

    analysis = analyze_audio(_write_tone_wav(work / "tone.wav"), include_onsets=False)
    if not (0.9 < analysis.duration_seconds < 1.1):
        raise RuntimeError(f"audio analysis returned an unexpected duration: {analysis.duration_seconds}")
    return "analyze_audio"


def _smoke_ocr(work: Path) -> str:
    from core.analysis.ocr import extract_text_from_frame

    text, confidence, source = extract_text_from_frame(
        _synthetic_frame(work / "frame.png"), use_vlm_fallback=False, raise_errors=True,
    )
    if source != "paddleocr":
        raise RuntimeError(f"OCR did not come from the isolated PaddleOCR engine (source={source})")
    return f"paddle text={text[:20]!r}"


def _smoke_vision(work: Path) -> str:
    from core.analysis.detection import detect_objects

    detections = detect_objects(_synthetic_frame(work / "frame.png"), confidence_threshold=0.9)
    if not isinstance(detections, list):
        raise RuntimeError("object detection returned no list")
    return f"yolo detections={len(detections)}"


# Families without an entry are probed only (their models are too large to pull in CI).
_FAMILY_SMOKE_CALLS: dict[str, Callable[[Path], str]] = {
    "audio": _smoke_audio,
    "ocr": _smoke_ocr,
    "vision": _smoke_vision,
}


def _run_family_call(family: str, work: Path) -> str:
    """One cheap real call per family through the public engine functions."""
    call = _FAMILY_SMOKE_CALLS.get(family)
    return call(work) if call is not None else "probe only"
