"""Regression tests for macOS model staging helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_stage_models_module():
    module_path = Path("packaging/macos/stage_models.py").resolve()
    spec = importlib.util.spec_from_file_location("test_stage_models_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_demucs_htdemucs_downloads_checkpoint(monkeypatch, tmp_path):
    """Demucs staging should seed the exact checkpoint into TORCH_HOME layout."""
    stage_models = _load_stage_models_module()
    calls: list[tuple[str, Path]] = []

    def _fake_download(url: str, target_path: Path) -> None:
        calls.append((url, target_path))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(stage_models, "_download_to_path", _fake_download)

    runtime_dir = tmp_path / "runtime"
    stage_models._stage_demucs_htdemucs(runtime_dir)

    checkpoint_path = runtime_dir / "hub" / "checkpoints" / stage_models._DEMUCS_HTDEMUCS_FILENAME
    assert calls == [(stage_models._DEMUCS_HTDEMUCS_URL, checkpoint_path)]
    assert checkpoint_path.read_text(encoding="utf-8") == "weights"


def test_stage_demucs_htdemucs_skips_existing_checkpoint(monkeypatch, tmp_path):
    """Demucs staging should not re-download when the checkpoint already exists."""
    stage_models = _load_stage_models_module()
    runtime_dir = tmp_path / "runtime"
    checkpoint_path = runtime_dir / "hub" / "checkpoints" / stage_models._DEMUCS_HTDEMUCS_FILENAME
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("cached", encoding="utf-8")

    def _unexpected_download(url: str, target_path: Path) -> None:
        raise AssertionError("download should not be attempted")

    monkeypatch.setattr(stage_models, "_download_to_path", _unexpected_download)

    stage_models._stage_demucs_htdemucs(runtime_dir)

    assert checkpoint_path.read_text(encoding="utf-8") == "cached"


def test_prime_runtime_dir_from_existing_cache_copies_known_assets(monkeypatch, tmp_path):
    """Known reusable model cache paths should seed into the runtime dir."""
    stage_models = _load_stage_models_module()
    source_cache = tmp_path / "source-cache"
    runtime_dir = tmp_path / "runtime"

    dinov2_blob = (
        source_cache
        / "huggingface"
        / "models--facebook--dinov2-base"
        / "blobs"
        / "blob.bin"
    )
    dinov2_blob.parent.mkdir(parents=True, exist_ok=True)
    dinov2_blob.write_text("dinov2", encoding="utf-8")

    mobilenet_weights = source_cache / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
    mobilenet_weights.parent.mkdir(parents=True, exist_ok=True)
    mobilenet_weights.write_text("mobilenet", encoding="utf-8")

    monkeypatch.setattr(
        stage_models,
        "load_settings",
        lambda: SimpleNamespace(model_cache_dir=source_cache),
    )

    stage_models._prime_runtime_dir_from_existing_cache(runtime_dir)

    assert (
        runtime_dir
        / "huggingface"
        / "models--facebook--dinov2-base"
        / "blobs"
        / "blob.bin"
    ).read_text(encoding="utf-8") == "dinov2"
    assert (
        runtime_dir / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
    ).read_text(encoding="utf-8") == "mobilenet"


def test_prime_runtime_dir_from_existing_cache_materializes_file_symlinks(monkeypatch, tmp_path):
    """Priming should copy symlinked cache files as regular files."""
    stage_models = _load_stage_models_module()
    source_cache = tmp_path / "source-cache"
    runtime_dir = tmp_path / "runtime"

    blob_path = (
        source_cache
        / "huggingface"
        / "models--facebook--dinov2-base"
        / "blobs"
        / "blob.bin"
    )
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_text("blob-data", encoding="utf-8")

    snapshot_path = (
        source_cache
        / "huggingface"
        / "models--facebook--dinov2-base"
        / "snapshots"
        / "revision"
        / "model.safetensors"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.symlink_to("../../blobs/blob.bin")

    monkeypatch.setattr(
        stage_models,
        "load_settings",
        lambda: SimpleNamespace(model_cache_dir=source_cache),
    )

    stage_models._prime_runtime_dir_from_existing_cache(runtime_dir)

    copied_snapshot = (
        runtime_dir
        / "huggingface"
        / "models--facebook--dinov2-base"
        / "snapshots"
        / "revision"
        / "model.safetensors"
    )
    assert copied_snapshot.read_text(encoding="utf-8") == "blob-data"
    assert not copied_snapshot.is_symlink()


def test_materialize_file_symlinks_replaces_snapshot_links(tmp_path):
    """Staged file symlinks should be replaced in place with regular files."""
    stage_models = _load_stage_models_module()
    runtime_dir = tmp_path / "runtime"

    blob_path = runtime_dir / "huggingface" / "models--openvision--yoloe26-s-seg" / "blobs" / "blob.pt"
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_text("weights", encoding="utf-8")

    snapshot_path = (
        runtime_dir
        / "huggingface"
        / "models--openvision--yoloe26-s-seg"
        / "snapshots"
        / "revision"
        / "model.pt"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.symlink_to("../../blobs/blob.pt")

    stage_models._materialize_file_symlinks(runtime_dir)

    assert snapshot_path.read_text(encoding="utf-8") == "weights"
    assert not snapshot_path.is_symlink()


def test_prune_huggingface_blob_store_removes_redundant_blobs(tmp_path):
    """Blob stores should be removed once snapshots are real files."""
    stage_models = _load_stage_models_module()
    runtime_dir = tmp_path / "runtime"

    repo_dir = runtime_dir / "huggingface" / "models--facebook--dinov2-base"
    blob_path = repo_dir / "blobs" / "blob.bin"
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_text("blob-data", encoding="utf-8")

    snapshot_path = repo_dir / "snapshots" / "rev" / "model.safetensors"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text("blob-data", encoding="utf-8")

    stage_models._prune_huggingface_blob_store(runtime_dir)

    assert not (repo_dir / "blobs").exists()
    assert snapshot_path.read_text(encoding="utf-8") == "blob-data"
