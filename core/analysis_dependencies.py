"""Dependency resolution helpers for analysis operations."""

from __future__ import annotations

import platform
from typing import Any


def _is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon macOS."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_operation_feature_candidates(
    op_key: str,
    settings: Any,
    *,
    description_tier: str | None = None,
    transcription_backend: str | None = None,
    text_extraction_method: str | None = None,
    cinematography_tier: str | None = None,
) -> list[str]:
    """Return installable feature candidates for an analysis operation.

    The list is ordered by preference. Any available candidate makes the
    operation runnable.
    """
    if op_key == "shots":
        return ["shot_classify"]

    if op_key == "classify":
        return ["image_classify"]

    if op_key == "detect_objects":
        return ["object_detect"]

    if op_key == "face_embeddings":
        return ["face_detect"]

    if op_key == "gaze":
        return ["gaze_detect"]

    if op_key in ("embeddings", "boundary_embeddings"):
        return ["embeddings"]

    if op_key == "extract_text":
        method = text_extraction_method or getattr(settings, "text_extraction_method", "hybrid")
        return ["ocr"] if method in ("paddleocr", "hybrid") else []

    if op_key == "transcribe":
        backend = transcription_backend or getattr(settings, "transcription_backend", "auto")
        if backend == "groq":
            return ["transcribe_cloud"]
        if backend == "mlx-whisper":
            return ["transcribe_mlx"]
        if backend == "auto" and _is_apple_silicon():
            return ["transcribe_mlx", "transcribe"]
        return ["transcribe"]

    if op_key == "describe":
        tier = description_tier or getattr(settings, "description_model_tier", "local")
        if tier == "local":
            if _is_apple_silicon():
                return ["describe_local", "describe_local_cpu"]
            return ["describe_local_cpu"]
        return []

    if op_key == "custom_query":
        tier = description_tier or getattr(settings, "description_model_tier", "local")
        if tier == "local":
            if _is_apple_silicon():
                # Only list describe_local (mlx-vlm / Qwen3-VL) — don't fall back to
                # describe_local_cpu here so the install prompt triggers for mlx-vlm.
                # Moondream is a poor fit for yes/no visual queries.
                return ["describe_local"]
            return ["describe_local_cpu"]
        return []

    if op_key == "cinematography":
        tier = cinematography_tier or getattr(settings, "cinematography_tier", "cloud")
        if tier == "local" and _is_apple_silicon():
            return ["describe_local"]
        return []

    return []
