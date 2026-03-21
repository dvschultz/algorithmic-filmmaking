"""Tests for optional packaging runtime helpers."""

import importlib.util
import sys
from pathlib import Path


def _load_module(name: str, relative_path: str):
    project_root = Path(__file__).resolve().parent.parent
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_support = _load_module("scene_ripper_build_support_tests", "packaging/build_support.py")
collect_macos_sparkle_datas = build_support.collect_macos_sparkle_datas
collect_windows_winsparkle_binaries = build_support.collect_windows_winsparkle_binaries
get_core_pyinstaller_metadata = build_support.get_core_pyinstaller_metadata
get_core_pyinstaller_modules = build_support.get_core_pyinstaller_modules


def test_collect_macos_sparkle_datas_returns_empty_when_not_staged(tmp_path):
    """Optional Sparkle runtime collection should be empty when no runtime is staged."""
    assert collect_macos_sparkle_datas(tmp_path) == []


def test_collect_macos_sparkle_datas_preserves_relative_layout(tmp_path):
    """Staged Sparkle assets should bundle only the canonical versioned framework tree."""
    sparkle_cli = (
        tmp_path
        / "packaging"
        / "runtime"
        / "sparkle"
        / "macos"
        / "Sparkle.framework"
        / "Versions"
        / "B"
        / "Resources"
        / "bin"
        / "sparkle"
    )
    sparkle_cli.parent.mkdir(parents=True)
    sparkle_cli.write_text("binary", encoding="utf-8")

    collected = collect_macos_sparkle_datas(tmp_path)

    assert len(collected) == 1
    source, destination = collected[0]
    assert source == str(sparkle_cli)
    assert destination.replace("\\", "/") == "Sparkle.framework/Versions/B/Resources/bin"


def test_collect_windows_winsparkle_binaries_returns_empty_when_not_staged(tmp_path):
    """Optional WinSparkle collection should be empty when no runtime is staged."""
    assert collect_windows_winsparkle_binaries(tmp_path) == []


def test_collect_windows_winsparkle_binaries_collects_staged_dll(tmp_path):
    """Staged WinSparkle DLLs should be copied to the executable directory."""
    winsparkle_dll = (
        tmp_path
        / "packaging"
        / "runtime"
        / "winsparkle"
        / "windows"
        / "WinSparkle.dll"
    )
    winsparkle_dll.parent.mkdir(parents=True)
    winsparkle_dll.write_text("binary", encoding="utf-8")

    collected = collect_windows_winsparkle_binaries(tmp_path)

    assert collected == [(str(winsparkle_dll), ".")]


def test_core_pyinstaller_modules_cover_packaged_runtime_dependencies():
    """Frozen builds should explicitly collect modules with dynamic imports."""
    modules = get_core_pyinstaller_modules()
    assert "googleapiclient" in modules
    assert "httplib2" in modules
    assert "sklearn" in modules
    assert "scipy" in modules
    assert "keyring" in modules


def test_core_pyinstaller_metadata_covers_core_requirements():
    """Frozen builds should carry distribution metadata for bundled core requirements."""
    metadata = get_core_pyinstaller_metadata()
    assert "google-api-python-client" in metadata
    assert "scikit-learn" in metadata
    assert "scipy" in metadata
    assert "Pillow" in metadata
