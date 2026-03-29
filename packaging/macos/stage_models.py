#!/usr/bin/env python3
"""Stage bundled local model assets for the macOS app bundle."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bundled_models import (  # noqa: E402
    build_runtime_model_manifest,
    get_macos_bundle_manifest_entry,
)
from core.settings import ENV_CONFIG_PATH, _sync_model_cache_env, load_settings  # noqa: E402


RUNTIME_DIR = Path(
    os.environ.get(
        "SCENE_RIPPER_MODEL_RUNTIME_DIR",
        str(PROJECT_ROOT / "packaging" / "runtime" / "models" / "macos"),
    )
)
_DEMUCS_HTDEMUCS_FILENAME = "955717e8-8726e21a.th"
_DEMUCS_HTDEMUCS_URL = (
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
)
_PRIMABLE_CACHE_PATHS = (
    Path("huggingface/models--facebook--dinov2-base"),
    Path("huggingface/models--openvision--yoloe26-s-seg"),
    Path("paddleocr"),
    Path("insightface/models/buffalo_l"),
    Path("hub/checkpoints/mobilenet_v3_small-047dcff4.pth"),
    Path(f"hub/checkpoints/{_DEMUCS_HTDEMUCS_FILENAME}"),
)


@contextmanager
def _isolated_stage_environment(runtime_dir: Path):
    """Run staging with isolated cache/config paths."""
    old_env = os.environ.copy()
    stage_root = runtime_dir.parent
    stage_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="scene-ripper-model-stage-",
        dir=stage_root,
    ) as temp_dir:
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
        os.environ["TMPDIR"] = str(temp_root)
        os.environ["TMP"] = str(temp_root)
        os.environ["TEMP"] = str(temp_root)
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


def _copy_existing_cache_path(source_path: Path, target_path: Path) -> None:
    if source_path.is_dir():
        shutil.copytree(source_path, target_path, symlinks=False, dirs_exist_ok=True)
    elif source_path.is_file():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _prime_runtime_dir_from_existing_cache(runtime_dir: Path) -> None:
    try:
        model_cache_dir = load_settings().model_cache_dir
    except Exception:
        return

    if not model_cache_dir.exists():
        return

    for relative_path in _PRIMABLE_CACHE_PATHS:
        source_path = model_cache_dir / relative_path
        if not source_path.exists():
            continue
        _copy_existing_cache_path(source_path, runtime_dir / relative_path)


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
    """Infer PaddleOCR OCR model directories from an isolated cache root."""
    found: dict[str, Path] = {}
    role_patterns = {
        "text_detection": ("server_det", "text_detection", "_det"),
        "text_recognition": ("mobile_rec", "text_recognition", "_rec"),
        "textline_orientation": ("textline_ori", "textline_orientation"),
    }
    candidates = sorted(path for path in search_root.rglob("*") if path.is_dir())
    for role, patterns in role_patterns.items():
        for candidate in candidates:
            name = candidate.name.lower()
            if not any(pattern in name for pattern in patterns):
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

    PaddleOCR(use_textline_orientation=True, lang="en")
    discovered = _find_paddleocr_model_dirs(temp_root)
    missing = {"text_detection", "text_recognition", "textline_orientation"} - set(discovered)
    if missing:
        raise RuntimeError(f"Failed to locate staged PaddleOCR model dirs for: {sorted(missing)}")

    paddle_runtime_dir = runtime_dir / "paddleocr"
    paddle_runtime_dir.mkdir(parents=True, exist_ok=True)
    for role, source_dir in discovered.items():
        shutil.copytree(source_dir, paddle_runtime_dir / role, dirs_exist_ok=True)


def _stage_insightface_buffalo_l() -> None:
    from core.analysis.faces import _load_insightface

    _load_insightface()


def _download_to_path(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f"{target_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        temp_path.replace(target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _stage_demucs_htdemucs(runtime_dir: Path) -> None:
    checkpoint_path = runtime_dir / "hub" / "checkpoints" / _DEMUCS_HTDEMUCS_FILENAME
    if checkpoint_path.is_file():
        return
    _download_to_path(_DEMUCS_HTDEMUCS_URL, checkpoint_path)


def _materialize_file_symlinks(root_dir: Path) -> None:
    """Replace staged file symlinks with real files.

    Hugging Face snapshot caches use file symlinks that PyInstaller preserves as
    `SYMLINK` TOC entries. On macOS, those entries are relocated into the
    `Contents/Frameworks` side of the bundle, which conflicts with data-only
    directories like `models/...`. Materializing the files keeps the original
    paths and names while avoiding the problematic relocation path.
    """
    for path in sorted(root_dir.rglob("*")):
        if not path.is_symlink():
            continue
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_file():
            continue

        temp_path = path.with_name(f"{path.name}.tmp")
        shutil.copy2(resolved_path, temp_path)
        temp_path.replace(path)


def _prune_huggingface_blob_store(root_dir: Path) -> None:
    """Drop redundant blob stores once snapshot files are fully materialized."""
    huggingface_root = root_dir / "huggingface"
    if not huggingface_root.is_dir():
        return

    for repo_dir in sorted(path for path in huggingface_root.iterdir() if path.is_dir()):
        blobs_dir = repo_dir / "blobs"
        snapshots_dir = repo_dir / "snapshots"
        if not blobs_dir.is_dir() or not snapshots_dir.is_dir():
            continue

        snapshot_files = [path for path in snapshots_dir.rglob("*") if path.is_file()]
        if not snapshot_files:
            continue
        if any(path.is_symlink() for path in snapshot_files):
            continue

        shutil.rmtree(blobs_dir)


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
    "demucs_htdemucs": lambda temp_root, runtime_dir: _stage_demucs_htdemucs(runtime_dir),
}


def main() -> int:
    manifest_entry = get_macos_bundle_manifest_entry()
    _ensure_clean_runtime_dir(RUNTIME_DIR)
    _prime_runtime_dir_from_existing_cache(RUNTIME_DIR)

    with _isolated_stage_environment(RUNTIME_DIR) as temp_root:
        for target_name in manifest_entry["stage_targets"]:
            stage_func = STAGE_TARGETS[target_name]
            print(f"==> Staging {target_name}")
            stage_func(temp_root, RUNTIME_DIR)

    _materialize_file_symlinks(RUNTIME_DIR)
    _prune_huggingface_blob_store(RUNTIME_DIR)
    (RUNTIME_DIR / "manifest.json").write_text(
        json.dumps(build_runtime_model_manifest(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"==> Bundled model runtime staged at {RUNTIME_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
