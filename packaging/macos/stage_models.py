#!/usr/bin/env python3
"""Stage bundled local model assets for the macOS app bundle."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bundled_models import (  # noqa: E402
    build_runtime_model_manifest,
    get_macos_bundle_manifest_entry,
)
from core.settings import ENV_CONFIG_PATH, _sync_model_cache_env  # noqa: E402


RUNTIME_DIR = PROJECT_ROOT / "packaging" / "runtime" / "models" / "macos"


@contextmanager
def _isolated_stage_environment(runtime_dir: Path):
    """Run staging with isolated cache/config paths."""
    old_env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="scene-ripper-model-stage-") as temp_dir:
        temp_root = Path(temp_dir)
        fake_home = temp_root / "home"
        fake_home.mkdir(parents=True, exist_ok=True)

        config_path = temp_root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "paths": {
                        "model_cache_dir": str(runtime_dir),
                    },
                    "transcription": {
                        "model": get_macos_bundle_manifest_entry()["whisper_model"],
                        "backend": get_macos_bundle_manifest_entry()["transcription_backend"],
                    },
                    "vision": {
                        "tier": "local",
                        "model_local": get_macos_bundle_manifest_entry()["description_model_local"],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        os.environ["HOME"] = str(fake_home)
        os.environ["XDG_CACHE_HOME"] = str(fake_home / ".cache")
        os.environ["XDG_CONFIG_HOME"] = str(fake_home / ".config")
        os.environ[ENV_CONFIG_PATH] = str(config_path)
        os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
        _sync_model_cache_env(runtime_dir)

        try:
            yield temp_root
        finally:
            os.environ.clear()
            os.environ.update(old_env)


def _ensure_clean_runtime_dir(runtime_dir: Path) -> None:
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)


def _stage_mlx_whisper_medium() -> None:
    from core.transcription import get_mlx_model

    get_mlx_model(get_macos_bundle_manifest_entry()["whisper_model"])


def _stage_mlx_vlm_local() -> None:
    from core.analysis.description import _load_local_model

    _load_local_model()


def _stage_dinov2_embeddings() -> None:
    from core.analysis.embeddings import _get_model

    _get_model()


def _stage_siglip_shot_classifier() -> None:
    from core.analysis.shots import load_classification_model

    load_classification_model()


def _stage_mobilenet_imagenet() -> None:
    from core.analysis.classification import _load_model

    _load_model()


def _stage_yolo26n_detection() -> None:
    from core.analysis.detection import ensure_default_detection_model_loaded

    ensure_default_detection_model_loaded("n")


def _stage_yoloe26s_detection() -> None:
    from core.analysis.detection import _load_yoloe

    _load_yoloe(["person"])


def _find_paddleocr_model_dirs(search_root: Path) -> dict[str, Path]:
    """Infer PaddleOCR det/rec/cls model directories from an isolated cache root."""
    found: dict[str, Path] = {}
    for role in ("det", "rec", "cls"):
        for candidate in sorted(path for path in search_root.rglob("*") if path.is_dir()):
            name = candidate.name.lower()
            if role not in name:
                continue
            contains_model = any(
                child.is_file() and child.suffix in {".pdmodel", ".json", ".pdiparams"}
                for child in candidate.rglob("*")
            )
            if contains_model:
                found[role] = candidate
                break
    return found


def _stage_paddleocr_en(temp_root: Path, runtime_dir: Path) -> None:
    from paddleocr import PaddleOCR

    PaddleOCR(use_angle_cls=True, lang="en")
    discovered = _find_paddleocr_model_dirs(temp_root)
    missing = {"det", "rec", "cls"} - set(discovered)
    if missing:
        raise RuntimeError(f"Failed to locate staged PaddleOCR model dirs for: {sorted(missing)}")

    paddle_runtime_dir = runtime_dir / "paddleocr"
    paddle_runtime_dir.mkdir(parents=True, exist_ok=True)
    for role, source_dir in discovered.items():
        shutil.copytree(source_dir, paddle_runtime_dir / role, dirs_exist_ok=True)


def _stage_insightface_buffalo_l() -> None:
    from core.analysis.faces import _load_insightface

    _load_insightface()


def _stage_demucs_htdemucs() -> None:
    from demucs_infer.pretrained import get_model

    get_model("htdemucs")


STAGE_TARGETS = {
    "mlx_whisper_medium": lambda temp_root, runtime_dir: _stage_mlx_whisper_medium(),
    "mlx_vlm_local": lambda temp_root, runtime_dir: _stage_mlx_vlm_local(),
    "dinov2_embeddings": lambda temp_root, runtime_dir: _stage_dinov2_embeddings(),
    "siglip_shot_classifier": lambda temp_root, runtime_dir: _stage_siglip_shot_classifier(),
    "mobilenet_imagenet": lambda temp_root, runtime_dir: _stage_mobilenet_imagenet(),
    "yolo26n_detection": lambda temp_root, runtime_dir: _stage_yolo26n_detection(),
    "yoloe26s_detection": lambda temp_root, runtime_dir: _stage_yoloe26s_detection(),
    "paddleocr_en": _stage_paddleocr_en,
    "insightface_buffalo_l": lambda temp_root, runtime_dir: _stage_insightface_buffalo_l(),
    "demucs_htdemucs": lambda temp_root, runtime_dir: _stage_demucs_htdemucs(),
}


def main() -> int:
    manifest_entry = get_macos_bundle_manifest_entry()
    _ensure_clean_runtime_dir(RUNTIME_DIR)

    with _isolated_stage_environment(RUNTIME_DIR) as temp_root:
        for target_name in manifest_entry["stage_targets"]:
            stage_func = STAGE_TARGETS[target_name]
            print(f"==> Staging {target_name}")
            stage_func(temp_root, RUNTIME_DIR)

    (RUNTIME_DIR / "manifest.json").write_text(
        json.dumps(build_runtime_model_manifest(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"==> Bundled model runtime staged at {RUNTIME_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
