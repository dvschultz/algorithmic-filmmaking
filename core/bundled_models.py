"""Helpers for bundled model assets in frozen desktop builds."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.paths import get_resource_path, is_frozen

logger = logging.getLogger(__name__)

_SOURCE_MANIFEST_PATH = Path(__file__).with_name("model_bundle_manifest.json")
_BUNDLED_MODELS_DIRNAME = "models"
_BUNDLED_MANIFEST_FILENAME = "manifest.json"
_SEEDED_MANIFEST_FILENAME = ".scene_ripper_model_bundle.json"
_FROZEN_MACOS_WHISPER_MODEL = "medium.en"
_FROZEN_MACOS_LOCAL_TRANSCRIPTION_BACKEND = "mlx-whisper"
_FROZEN_MACOS_LOCAL_DESCRIPTION_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
_EXPLICIT_LARGE_MODEL_DOWNLOADS = 0


def is_frozen_macos_apple_silicon() -> bool:
    """Return True when running inside the bundled macOS Apple Silicon app."""
    return is_frozen() and sys_platform_is_macos() and platform.machine() == "arm64"


def sys_platform_is_macos() -> bool:
    """Return True when the current platform is macOS."""
    return sys.platform == "darwin"


def get_source_model_bundle_manifest_path() -> Path:
    """Return the repository manifest describing bundled model policy."""
    return _SOURCE_MANIFEST_PATH


def load_source_model_bundle_manifest() -> dict[str, Any]:
    """Load the repository manifest that defines bundled model policy."""
    return json.loads(_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))


def get_macos_bundle_manifest_entry() -> dict[str, Any]:
    """Return the bundled-model entry for macOS Apple Silicon."""
    return load_source_model_bundle_manifest()["macos_arm64"]


def get_bundled_models_dir() -> Path:
    """Return the bundled runtime models directory."""
    return get_resource_path(_BUNDLED_MODELS_DIRNAME)


def get_bundled_model_manifest_path() -> Path:
    """Return the manifest path for staged bundled models."""
    return get_bundled_models_dir() / _BUNDLED_MANIFEST_FILENAME


def get_seeded_model_manifest_path(model_cache_dir: Path) -> Path:
    """Return the manifest path written after seeding bundled models."""
    return model_cache_dir / _SEEDED_MANIFEST_FILENAME


def bundled_models_available() -> bool:
    """Return True when a staged bundled-model runtime is present."""
    return get_bundled_model_manifest_path().is_file()


def load_bundled_model_manifest() -> dict[str, Any]:
    """Load the staged bundled-model manifest from app resources."""
    return json.loads(get_bundled_model_manifest_path().read_text(encoding="utf-8"))


def load_seeded_model_manifest(model_cache_dir: Path) -> dict[str, Any] | None:
    """Load the seeded manifest from the writable model cache, if present."""
    path = get_seeded_model_manifest_path(model_cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Seeded bundled-model manifest is invalid JSON: %s", path)
        return None


def get_frozen_macos_whisper_model() -> str:
    """Return the only packaged local Whisper model for frozen macOS builds."""
    return _FROZEN_MACOS_WHISPER_MODEL


def get_frozen_macos_local_description_model() -> str:
    """Return the packaged local description model for frozen macOS builds."""
    return _FROZEN_MACOS_LOCAL_DESCRIPTION_MODEL


def get_mlx_models_dir(model_cache_dir: Path) -> Path:
    """Return the managed root used for MLX Whisper downloads."""
    return model_cache_dir / "mlx_models"


def get_available_transcription_models() -> tuple[str, ...]:
    """Return the local transcription models that should be exposed in the UI."""
    if is_frozen_macos_apple_silicon():
        return (_FROZEN_MACOS_WHISPER_MODEL,)
    return ("tiny.en", "small.en", "medium.en", "large-v3")


def get_large_model_requirements() -> tuple[dict[str, str], ...]:
    """Return large models that are intentionally excluded from the app bundle."""
    entry = get_macos_bundle_manifest_entry()
    return tuple(dict(item) for item in entry.get("large_models", ()))


def get_huggingface_repo_cache_dir(model_cache_dir: Path, repo_id: str) -> Path:
    """Return the managed Hugging Face cache directory for a model repo id."""
    return model_cache_dir / "huggingface" / f"models--{repo_id.replace('/', '--')}"


def find_huggingface_snapshot_dir(
    model_cache_dir: Path,
    repo_id: str,
    revision: str | None = None,
    required_files: tuple[str, ...] = (),
) -> Path | None:
    """Return a locally available Hugging Face snapshot dir when present.

    This prefers the pinned revision directory when provided. The returned path
    is only considered valid when all required files exist as regular files.
    """
    snapshot_root = get_huggingface_repo_cache_dir(model_cache_dir, repo_id) / "snapshots"
    if not snapshot_root.is_dir():
        return None

    candidates: list[Path] = []
    if revision:
        candidates.append(snapshot_root / revision)
    candidates.extend(path for path in sorted(snapshot_root.iterdir()) if path.is_dir())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        if all((candidate / relative_path).is_file() for relative_path in required_files):
            return candidate
    return None


def find_huggingface_snapshot_file(
    model_cache_dir: Path,
    repo_id: str,
    filename: str,
    revision: str | None = None,
) -> Path | None:
    """Return a locally available Hugging Face snapshot file when present."""
    snapshot_dir = find_huggingface_snapshot_dir(
        model_cache_dir,
        repo_id,
        revision=revision,
        required_files=(filename,),
    )
    if snapshot_dir is None:
        return None
    return snapshot_dir / filename


def _qwen_snapshot_complete(model_cache_dir: Path) -> bool:
    return (
        find_huggingface_snapshot_file(
            model_cache_dir,
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            "model.safetensors",
        )
        is not None
    )


def _whisper_medium_complete(model_cache_dir: Path) -> bool:
    model_root = get_mlx_models_dir(model_cache_dir) / "medium"
    return (model_root / "weights.npz").is_file() and (model_root / "config.json").is_file()


def _siglip_snapshot_complete(model_cache_dir: Path) -> bool:
    repo_root = get_huggingface_repo_cache_dir(
        model_cache_dir,
        "google/siglip2-base-patch16-224",
    )
    if any(repo_root.rglob("*.incomplete")):
        return False

    return (
        find_huggingface_snapshot_file(
            model_cache_dir,
            "google/siglip2-base-patch16-224",
            "model.safetensors",
        )
        is not None
        or find_huggingface_snapshot_file(
            model_cache_dir,
            "google/siglip2-base-patch16-224",
            "pytorch_model.bin",
        )
        is not None
    )


def get_missing_large_models(model_cache_dir: Path) -> list[dict[str, str]]:
    """Return large models that still need an explicit user download step."""
    checkers = {
        "whisper_medium": _whisper_medium_complete,
        "siglip2_shot_classifier": _siglip_snapshot_complete,
        "qwen3_vl_4b": _qwen_snapshot_complete,
    }

    missing: list[dict[str, str]] = []
    for item in get_large_model_requirements():
        checker = checkers.get(item["id"])
        if checker is None or not checker(model_cache_dir):
            missing.append(item)
    return missing


def has_missing_large_models(model_cache_dir: Path) -> bool:
    """Return True when explicit-download large models are still missing."""
    return bool(get_missing_large_models(model_cache_dir))


def large_model_downloads_allowed() -> bool:
    """Return True while the explicit large-model setup flow is running."""
    return _EXPLICIT_LARGE_MODEL_DOWNLOADS > 0


def get_paddleocr_model_dirs(model_cache_dir: Path) -> dict[str, Path]:
    """Return explicit PaddleOCR model directories under the managed cache."""
    base_dir = model_cache_dir / "paddleocr"
    return {
        "text_detection_model_dir": base_dir / "text_detection",
        "text_recognition_model_dir": base_dir / "text_recognition",
        "textline_orientation_model_dir": base_dir / "textline_orientation",
    }


def get_bundled_paddleocr_kwargs(model_cache_dir: Path) -> dict[str, str]:
    """Return PaddleOCR kwargs for seeded local model directories when present."""
    model_dirs = get_paddleocr_model_dirs(model_cache_dir)
    if not all(path.is_dir() for path in model_dirs.values()):
        return {}
    return {key: str(path) for key, path in model_dirs.items()}


@contextmanager
def mlx_model_working_directory(model_cache_dir: Path):
    """Run MLX Whisper operations from the managed model cache root."""
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(model_cache_dir)
    try:
        yield model_cache_dir
    finally:
        os.chdir(previous_cwd)


@contextmanager
def explicit_large_model_downloads():
    """Allow large model downloads for the duration of the setup flow."""
    global _EXPLICIT_LARGE_MODEL_DOWNLOADS
    _EXPLICIT_LARGE_MODEL_DOWNLOADS += 1
    try:
        yield
    finally:
        _EXPLICIT_LARGE_MODEL_DOWNLOADS = max(0, _EXPLICIT_LARGE_MODEL_DOWNLOADS - 1)


def normalize_frozen_macos_settings(settings: Any) -> Any:
    """Clamp persisted settings to the frozen macOS bundled-model contract."""
    if not is_frozen_macos_apple_silicon():
        return settings

    if getattr(settings, "transcription_model", None) != _FROZEN_MACOS_WHISPER_MODEL:
        settings.transcription_model = _FROZEN_MACOS_WHISPER_MODEL

    backend = getattr(settings, "transcription_backend", "auto")
    if backend not in {"auto", "mlx-whisper", "groq"}:
        settings.transcription_backend = _FROZEN_MACOS_LOCAL_TRANSCRIPTION_BACKEND

    if getattr(settings, "description_model_local", None) != _FROZEN_MACOS_LOCAL_DESCRIPTION_MODEL:
        settings.description_model_local = _FROZEN_MACOS_LOCAL_DESCRIPTION_MODEL

    return settings


def normalize_frozen_macos_transcription_request(
    model_name: str,
    backend: str,
) -> tuple[str, str]:
    """Clamp local transcription requests to the frozen macOS bundle contract."""
    if not is_frozen_macos_apple_silicon():
        return model_name, backend

    if backend == "groq":
        return model_name, backend

    normalized_backend = backend
    if backend == "faster-whisper":
        normalized_backend = _FROZEN_MACOS_LOCAL_TRANSCRIPTION_BACKEND
    elif backend not in {"auto", "mlx-whisper"}:
        normalized_backend = "auto"

    return _FROZEN_MACOS_WHISPER_MODEL, normalized_backend


def download_large_models(model_cache_dir: Path, progress_callback=None) -> bool:
    """Explicitly download large models omitted from the bundled app."""
    missing = get_missing_large_models(model_cache_dir)
    if not missing:
        if progress_callback is not None:
            progress_callback(1.0, "Large models already downloaded")
        return True

    total = len(missing)
    with explicit_large_model_downloads():
        for index, item in enumerate(missing):
            start = index / total
            end = (index + 1) / total
            if progress_callback is not None:
                progress_callback(start, f"Downloading {item['label']}...")

            if item["id"] == "whisper_medium":
                from core.transcription import get_mlx_model

                with mlx_model_working_directory(model_cache_dir):
                    get_mlx_model(_FROZEN_MACOS_WHISPER_MODEL)
            elif item["id"] == "siglip2_shot_classifier":
                from core.analysis.shots import download_siglip_model

                download_siglip_model()
            elif item["id"] == "qwen3_vl_4b":
                from core.analysis.description import _load_local_model

                _load_local_model()
            else:
                raise RuntimeError(f"Unknown large model requirement: {item['id']}")

            if progress_callback is not None:
                progress_callback(end, f"{item['label']} ready")

    return True


def seed_bundled_models(model_cache_dir: Path) -> bool:
    """Seed bundled model assets into the writable model cache when needed."""
    if not is_frozen_macos_apple_silicon():
        return False
    if not bundled_models_available():
        logger.info("No bundled model runtime found at %s", get_bundled_models_dir())
        return False

    bundled_dir = get_bundled_models_dir()
    bundled_manifest = load_bundled_model_manifest()
    seeded_manifest = load_seeded_model_manifest(model_cache_dir)
    if seeded_manifest == bundled_manifest:
        return False

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(bundled_dir.iterdir()):
        if source_path.name == _BUNDLED_MANIFEST_FILENAME:
            continue
        destination = model_cache_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

    get_seeded_model_manifest_path(model_cache_dir).write_text(
        json.dumps(bundled_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Seeded bundled model runtime into %s", model_cache_dir)
    return True


def build_runtime_model_manifest() -> dict[str, Any]:
    """Return the staged runtime manifest written into the app bundle."""
    entry = dict(get_macos_bundle_manifest_entry())
    entry["platform"] = "macos-arm64"
    return entry
