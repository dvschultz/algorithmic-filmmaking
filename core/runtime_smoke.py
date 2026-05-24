"""Frozen runtime smoke checks for release validation."""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import tempfile
import wave
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
    return ("imports", "project", "scene-detect", "transcription", "updater")


def run_runtime_smoke_target(target: str) -> str:
    """Execute a single frozen runtime smoke target."""
    normalized = (target or "").strip().lower()
    handlers = {
        "imports": _run_imports_smoke,
        "project": _run_project_smoke,
        "scene-detect": _run_scene_detect_smoke,
        "transcription": _run_transcription_smoke,
        "updater": _run_updater_smoke,
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
