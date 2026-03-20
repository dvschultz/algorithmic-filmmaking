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


def test_collect_macos_sparkle_datas_returns_empty_when_not_staged(tmp_path):
    """Optional Sparkle runtime collection should be empty when no runtime is staged."""
    assert collect_macos_sparkle_datas(tmp_path) == []


def test_collect_macos_sparkle_datas_preserves_relative_layout(tmp_path):
    """Staged Sparkle assets should keep their framework-relative destination paths."""
    sparkle_cli = (
        tmp_path
        / "packaging"
        / "runtime"
        / "sparkle"
        / "macos"
        / "Sparkle.framework"
        / "Versions"
        / "Current"
        / "Resources"
        / "bin"
        / "sparkle"
    )
    sparkle_cli.parent.mkdir(parents=True)
    sparkle_cli.write_text("binary", encoding="utf-8")

    collected = collect_macos_sparkle_datas(tmp_path)

    assert collected == [
        (
            str(sparkle_cli),
            "Sparkle.framework/Versions/Current/Resources/bin",
        )
    ]


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
