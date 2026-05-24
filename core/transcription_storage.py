"""Storage preflight helpers for local transcription backends."""

from __future__ import annotations

import shutil
from pathlib import Path

from core.transcription_models import TranscriptionError

WHISPER_MODEL_DOWNLOAD_BYTES = {
    "tiny.en": 250 * 1024 * 1024,
    "small.en": 750 * 1024 * 1024,
    "medium.en": 2 * 1024 * 1024 * 1024,
    "large-v3": 4 * 1024 * 1024 * 1024,
    "large-v3-turbo": 2 * 1024 * 1024 * 1024,
}


def format_bytes(size_bytes: int) -> str:
    """Format a byte count for user-facing transcription diagnostics."""
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def estimate_transcription_required_disk_bytes(
    model_name: str,
    backend: str,
    min_free_disk_gb: float,
) -> int:
    """Return the free-space floor for local transcription startup."""
    if backend == "groq":
        return 0

    configured_floor = int(max(0.0, min_free_disk_gb) * 1024 * 1024 * 1024)
    model_bytes = WHISPER_MODEL_DOWNLOAD_BYTES.get(
        model_name,
        WHISPER_MODEL_DOWNLOAD_BYTES["medium.en"],
    )
    temp_audio_headroom = 512 * 1024 * 1024
    return max(configured_floor, model_bytes + temp_audio_headroom)


def validate_transcription_disk_space(
    model_name: str,
    backend: str,
    cache_dir: Path,
    min_free_disk_gb: float,
) -> None:
    """Raise ``TranscriptionError`` when model/temp storage is likely insufficient."""
    required = estimate_transcription_required_disk_bytes(model_name, backend, min_free_disk_gb)
    if required <= 0:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    if free < required:
        raise TranscriptionError(
            "Not enough free disk space for transcription. "
            f"Available: {format_bytes(free)}; required: {format_bytes(required)}. "
            "Free space or lower the transcription disk-space threshold in Settings."
        )
