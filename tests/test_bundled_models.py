"""Tests for bundled model helpers used by frozen desktop builds."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from core import bundled_models


def test_seed_bundled_models_copies_runtime_and_writes_manifest(monkeypatch, tmp_path):
    """Bundled model assets should seed into the writable model cache."""
    bundled_dir = tmp_path / "bundled"
    (bundled_dir / "huggingface").mkdir(parents=True)
    (bundled_dir / "huggingface" / "weights.bin").write_text("weights", encoding="utf-8")
    (bundled_dir / "manifest.json").write_text(
        json.dumps({"bundle_version": "v1"}),
        encoding="utf-8",
    )

    target_dir = tmp_path / "target"

    monkeypatch.setattr(bundled_models, "is_frozen_macos_apple_silicon", lambda: True)
    monkeypatch.setattr(bundled_models, "get_bundled_models_dir", lambda: bundled_dir)
    monkeypatch.setattr(
        bundled_models,
        "get_bundled_model_manifest_path",
        lambda: bundled_dir / "manifest.json",
    )

    changed = bundled_models.seed_bundled_models(target_dir)

    assert changed is True
    assert (target_dir / "huggingface" / "weights.bin").read_text(encoding="utf-8") == "weights"
    assert json.loads(
        bundled_models.get_seeded_model_manifest_path(target_dir).read_text(encoding="utf-8")
    ) == {"bundle_version": "v1"}


def test_seed_bundled_models_skips_when_manifest_already_matches(monkeypatch, tmp_path):
    """Seeding should be a no-op when the same bundle version is already present."""
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir(parents=True)
    (bundled_dir / "manifest.json").write_text(
        json.dumps({"bundle_version": "v1"}),
        encoding="utf-8",
    )

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    bundled_models.get_seeded_model_manifest_path(target_dir).write_text(
        json.dumps({"bundle_version": "v1"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(bundled_models, "is_frozen_macos_apple_silicon", lambda: True)
    monkeypatch.setattr(bundled_models, "get_bundled_models_dir", lambda: bundled_dir)
    monkeypatch.setattr(
        bundled_models,
        "get_bundled_model_manifest_path",
        lambda: bundled_dir / "manifest.json",
    )

    changed = bundled_models.seed_bundled_models(target_dir)

    assert changed is False


def test_normalize_frozen_macos_settings_clamps_local_model_choices(monkeypatch):
    """Frozen macOS builds should clamp persisted settings to bundled local models."""
    settings = SimpleNamespace(
        transcription_model="small.en",
        transcription_backend="faster-whisper",
        description_model_local="vikhyatk/moondream2",
    )

    monkeypatch.setattr(bundled_models, "is_frozen_macos_apple_silicon", lambda: True)

    normalized = bundled_models.normalize_frozen_macos_settings(settings)

    assert normalized.transcription_model == "medium.en"
    assert normalized.transcription_backend == "mlx-whisper"
    assert normalized.description_model_local == "mlx-community/Qwen3-VL-4B-Instruct-4bit"


def test_normalize_frozen_macos_transcription_request_clamps_local_backend(monkeypatch):
    """Local transcription requests should honor the bundled macOS contract."""
    monkeypatch.setattr(bundled_models, "is_frozen_macos_apple_silicon", lambda: True)

    model_name, backend = bundled_models.normalize_frozen_macos_transcription_request(
        "small.en",
        "faster-whisper",
    )

    assert model_name == "medium.en"
    assert backend == "mlx-whisper"


def test_normalize_frozen_macos_transcription_request_preserves_groq(monkeypatch):
    """Cloud transcription requests should not be clamped to local bundled models."""
    monkeypatch.setattr(bundled_models, "is_frozen_macos_apple_silicon", lambda: True)

    model_name, backend = bundled_models.normalize_frozen_macos_transcription_request(
        "small.en",
        "groq",
    )

    assert model_name == "small.en"
    assert backend == "groq"


def test_get_bundled_paddleocr_kwargs_requires_seeded_dirs(tmp_path):
    """PaddleOCR local dirs should only be passed when all seeded dirs exist."""
    model_cache_dir = tmp_path / "models"

    assert bundled_models.get_bundled_paddleocr_kwargs(model_cache_dir) == {}

    for child in ("det", "rec", "cls"):
        (model_cache_dir / "paddleocr" / child).mkdir(parents=True, exist_ok=True)

    kwargs = bundled_models.get_bundled_paddleocr_kwargs(model_cache_dir)

    assert kwargs == {
        "det_model_dir": str(model_cache_dir / "paddleocr" / "det"),
        "rec_model_dir": str(model_cache_dir / "paddleocr" / "rec"),
        "cls_model_dir": str(model_cache_dir / "paddleocr" / "cls"),
    }


def test_get_missing_large_models_reports_missing_large_macos_models(tmp_path):
    """Large excluded models should be reported until explicitly downloaded."""
    missing_ids = {
        item["id"]
        for item in bundled_models.get_missing_large_models(tmp_path / "models")
    }

    assert {"whisper_medium", "siglip2_shot_classifier", "qwen3_vl_4b"} <= missing_ids


def test_get_missing_large_models_clears_when_assets_exist(tmp_path):
    """Presence checks should pass once the managed large-model files exist."""
    model_cache_dir = tmp_path / "models"
    whisper_dir = model_cache_dir / "mlx_models" / "medium"
    whisper_dir.mkdir(parents=True)
    (whisper_dir / "weights.npz").write_text("weights", encoding="utf-8")
    (whisper_dir / "config.json").write_text("{}", encoding="utf-8")

    qwen_dir = (
        model_cache_dir
        / "huggingface"
        / "models--mlx-community--Qwen3-VL-4B-Instruct-4bit"
        / "snapshots"
        / "rev"
    )
    qwen_dir.mkdir(parents=True)
    (qwen_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    siglip_dir = (
        model_cache_dir
        / "huggingface"
        / "models--google--siglip2-base-patch16-224"
        / "snapshots"
        / "rev"
    )
    siglip_dir.mkdir(parents=True)
    (siglip_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    assert bundled_models.get_missing_large_models(model_cache_dir) == []
