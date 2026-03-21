"""Tests for optional packaging runtime helpers."""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_support = _load_module("scene_ripper_build_support_tests", "packaging/build_support.py")
collect_macos_sparkle_datas = build_support.collect_macos_sparkle_datas
collect_windows_winsparkle_binaries = build_support.collect_windows_winsparkle_binaries
get_core_pyinstaller_collect_targets = build_support.get_core_pyinstaller_collect_targets
get_core_pyinstaller_metadata = build_support.get_core_pyinstaller_metadata
read_core_requirement_distributions = build_support.read_core_requirement_distributions


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


def test_core_requirement_distributions_follow_requirements_file(tmp_path):
    """Frozen bundle mappings should be derived from requirements-core.txt."""
    distributions = read_core_requirement_distributions(PROJECT_ROOT)
    assert "scikit-learn" in distributions
    assert "opencv-python" in distributions
    assert "google-api-python-client" in distributions
    assert "pyside6" in distributions


def test_core_pyinstaller_collect_targets_cover_packaged_runtime_dependencies(tmp_path):
    """Frozen builds should explicitly collect dynamic-import package trees."""
    targets = get_core_pyinstaller_collect_targets(PROJECT_ROOT)
    assert "googleapiclient" in targets
    assert "google_auth_httplib2" in targets
    assert "httplib2" in targets
    assert "sklearn" in targets
    assert "scipy" in targets
    assert "cv2" in targets
    assert "numpy" in targets
    assert "mpv" in targets


def test_core_pyinstaller_metadata_covers_core_requirements(tmp_path):
    """Frozen builds should carry distribution metadata for bundled core requirements."""
    metadata = get_core_pyinstaller_metadata(PROJECT_ROOT)
    assert "google-api-python-client" in metadata
    assert "scikit-learn" in metadata
    assert "scipy" in metadata
    assert "pillow" in metadata
    assert "opencv-python" in metadata
