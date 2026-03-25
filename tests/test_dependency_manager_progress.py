"""Regression tests for dependency installation and progress reporting."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import ModuleType

import pytest

from core.dependency_manager import install_package, install_packages
from core.feature_registry import install_for_feature, requires_full_package_repair


def _restore_sys_module(name: str, original: object | None) -> None:
    """Restore a sys.modules entry without disturbing preloaded global modules."""
    if original is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = original


class _FakePopen:
    """Minimal subprocess.Popen stand-in for pip progress tests."""

    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = 1


def test_install_package_reports_intermediate_pip_progress(monkeypatch):
    """Large package installs should surface incremental pip progress to the UI."""
    progress_calls: list[tuple[float, str]] = []

    def _on_progress(progress: float, message: str):
        progress_calls.append((progress, message))

    def _fake_popen(cmd, **kwargs):
        assert "--progress-bar" in cmd
        return _FakePopen(
            [
                "Collecting torch>=2.0",
                "Downloading torch-2.6.0.whl (200.0 MB)",
                "Installing collected packages: torch",
                "Successfully installed torch-2.6.0",
            ]
        )

    monkeypatch.setattr("core.dependency_manager.ensure_python", lambda cb=None: Path("/tmp/python"))
    monkeypatch.setattr("core.dependency_manager.get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr("core.dependency_manager.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("core.dependency_manager._write_compat_marker", lambda: None)

    assert install_package("torch>=2.0", _on_progress) is True
    assert progress_calls[0][0] == 0.2
    assert any(0.2 < progress < 1.0 for progress, _ in progress_calls)
    assert progress_calls[-1] == (1.0, "Installed torch>=2.0")


def test_install_packages_batches_specifiers_into_one_pip_run(monkeypatch):
    """Related packages should be installed together in one pip resolver pass."""
    captured_cmd: list[str] = []

    def _fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return _FakePopen(
            [
                "Collecting torch>=2.0",
                "Collecting torchvision>=0.19,<0.22",
                "Collecting transformers>=4.50,<5",
                "Successfully installed torch-2.6.0 torchvision-0.21.0 transformers-4.52.0",
            ]
        )

    monkeypatch.setattr("core.dependency_manager.ensure_python", lambda cb=None: Path("/tmp/python"))
    monkeypatch.setattr("core.dependency_manager.get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr("core.dependency_manager.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("core.dependency_manager._write_compat_marker", lambda: None)

    assert install_packages(["torch>=2.0", "torchvision>=0.19,<0.22", "transformers>=4.50,<5"]) is True
    assert captured_cmd.count("--target") == 1
    assert "--upgrade" in captured_cmd
    assert captured_cmd[-3:] == ["torch>=2.0", "torchvision>=0.19,<0.22", "transformers>=4.50,<5"]


def test_install_packages_refreshes_sys_path_and_clears_stale_modules(monkeypatch, tmp_path):
    """Newly installed managed packages should be importable in the current app session."""
    packages_dir = tmp_path / "packages"
    package_path = str(packages_dir)
    original_path = list(sys.path)
    transformers_root = packages_dir / "transformers"
    transformers_root.mkdir(parents=True)
    transformers_init = transformers_root / "__init__.py"
    transformers_init.write_text("# managed transformers")
    processing_auto = transformers_root / "processing_auto.py"
    processing_auto.write_text("# managed auto")

    transformers_module = ModuleType("transformers")
    transformers_module.__file__ = str(transformers_init)
    transformers_auto_module = ModuleType("transformers.models.auto.processing_auto")
    transformers_auto_module.__file__ = str(processing_auto)

    if package_path in sys.path:
        sys.path.remove(package_path)

    monkeypatch.setattr("core.dependency_manager.get_managed_packages_dir", lambda: packages_dir)
    monkeypatch.setattr("core.dependency_manager.ensure_python", lambda cb=None: Path("/tmp/python"))
    monkeypatch.setattr("core.dependency_manager.get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(
        "core.dependency_manager.subprocess.Popen",
        lambda cmd, **kwargs: _FakePopen(["Successfully installed transformers-4.52.0"]),
    )
    monkeypatch.setattr("core.dependency_manager._write_compat_marker", lambda: None)

    sys.modules["transformers"] = transformers_module
    sys.modules["transformers.models.auto.processing_auto"] = transformers_auto_module

    original_transformers = sys.modules.get("transformers")
    original_processing_auto = sys.modules.get("transformers.models.auto.processing_auto")

    try:
        assert install_packages(["transformers>=4.50,<5"]) is True
        assert package_path in sys.path
        assert "transformers" not in sys.modules
        assert "transformers.models.auto.processing_auto" not in sys.modules
    finally:
        sys.path[:] = original_path
        _restore_sys_module("transformers", original_transformers)
        _restore_sys_module("transformers.models.auto.processing_auto", original_processing_auto)


def test_reset_imported_package_roots_only_evicts_managed_package_modules(monkeypatch, tmp_path):
    """Global site-packages modules must not be purged during managed-package installs."""
    from types import ModuleType

    packages_dir = tmp_path / "packages"
    managed_root = packages_dir / "transformers"
    managed_root.mkdir(parents=True)
    managed_file = managed_root / "__init__.py"
    managed_file.write_text("# managed transformers")

    site_root = tmp_path / "site-packages" / "torch"
    site_root.mkdir(parents=True)
    site_file = site_root / "__init__.py"
    site_file.write_text("# site torch")

    managed_module = ModuleType("transformers")
    managed_module.__file__ = str(managed_file)
    site_module = ModuleType("torch")
    site_module.__file__ = str(site_file)

    monkeypatch.setattr("core.dependency_manager.get_managed_packages_dir", lambda: packages_dir)
    original_transformers = sys.modules.get("transformers")
    original_torch = sys.modules.get("torch")
    sys.modules["transformers"] = managed_module
    sys.modules["torch"] = site_module

    try:
        from core.dependency_manager import _reset_imported_package_roots

        _reset_imported_package_roots(["transformers", "torch"])

        assert "transformers" not in sys.modules
        assert sys.modules["torch"] is site_module
    finally:
        _restore_sys_module("transformers", original_transformers)
        _restore_sys_module("torch", original_torch)


def test_reset_imported_package_roots_keeps_modules_without_managed_origin(monkeypatch, tmp_path):
    """Modules without a managed-packages path should not be evicted by name alone."""
    from types import ModuleType

    packages_dir = tmp_path / "packages"
    packages_dir.mkdir(parents=True)

    global_torch = ModuleType("torch")
    global_torch.__spec__ = None

    monkeypatch.setattr("core.dependency_manager.get_managed_packages_dir", lambda: packages_dir)
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = global_torch

    try:
        from core.dependency_manager import _reset_imported_package_roots

        _reset_imported_package_roots(["torch"])

        assert sys.modules["torch"] is global_torch
    finally:
        _restore_sys_module("torch", original_torch)


def test_install_for_feature_batches_missing_packages_and_validates_runtime(monkeypatch):
    """Feature installs should use one package batch and then validate runtime imports."""
    progress_calls: list[tuple[float, str]] = []
    package_batches: list[list[str]] = []
    validated: list[str] = []

    def _on_progress(progress: float, message: str):
        progress_calls.append((progress, message))

    def _fake_install(specifiers: list[str], progress_callback):
        package_batches.append(specifiers)
        progress_callback(0.0, "Starting batch")
        progress_callback(0.5, "Halfway batch")
        progress_callback(1.0, "Finished batch")
        return True

    monkeypatch.setattr(
        "core.feature_registry.check_feature",
        lambda _name: (False, ["package:torch", "package:transformers"]),
    )
    monkeypatch.setattr("core.dependency_manager.get_pip_specifier", lambda name: f"{name}>=1.0")
    monkeypatch.setattr("core.dependency_manager.install_packages", _fake_install)
    monkeypatch.setattr(
        "core.feature_registry._validate_feature_runtime",
        lambda name: validated.append(name),
    )

    assert install_for_feature("describe_local_cpu", _on_progress) is True
    assert package_batches == [["torch>=1.0", "transformers>=1.0", "tokenizers>=1.0"]]
    assert validated == ["describe_local_cpu"]
    assert [round(progress, 2) for progress, _ in progress_calls] == [0.0, 0.5, 1.0]


def test_install_for_feature_reinstalls_broken_runtime_even_when_packages_exist(monkeypatch):
    """Fragile local runtimes should be repaired if imports are broken on disk."""
    package_batches: list[list[str]] = []
    validations: list[str] = []

    def _fake_validate(name: str):
        validations.append(name)
        if len(validations) == 1:
            raise RuntimeError("Could not import module 'AutoProcessor'")

    def _fake_install(specifiers: list[str], _progress_callback):
        package_batches.append(specifiers)
        return True

    monkeypatch.setattr("core.feature_registry.check_feature", lambda _name: (True, []))
    monkeypatch.setattr("core.feature_registry._validate_feature_runtime", _fake_validate)
    monkeypatch.setattr("core.dependency_manager.get_pip_specifier", lambda name: f"{name}>=1.0")
    monkeypatch.setattr("core.dependency_manager.install_packages", _fake_install)

    assert install_for_feature("shot_classify") is True
    assert validations == ["shot_classify", "shot_classify"]
    assert package_batches == [[
        "torch>=1.0",
        "torchvision>=1.0",
        "transformers>=1.0",
        "tokenizers>=1.0",
        "einops>=1.0",
        "sentencepiece>=1.0",
        "protobuf>=1.0",
    ]]


@pytest.mark.parametrize(
    ("feature_name", "missing", "expected_packages"),
    [
        (
            "describe_local",
            ["package:mlx_vlm"],
            ["mlx_vlm", "transformers", "tokenizers", "sentencepiece", "protobuf"],
        ),
        (
            "describe_local_cpu",
            ["package:transformers"],
            ["torch", "transformers", "tokenizers"],
        ),
        (
            "shot_classify",
            ["package:sentencepiece", "package:protobuf"],
            ["torch", "torchvision", "transformers", "tokenizers", "einops", "sentencepiece", "protobuf"],
        ),
        (
            "image_classify",
            ["package:torchvision"],
            ["torch", "torchvision"],
        ),
        (
            "object_detect",
            ["package:ultralytics"],
            ["torch", "ultralytics"],
        ),
        (
            "face_detect",
            ["package:onnxruntime"],
            ["insightface", "onnxruntime"],
        ),
        (
            "ocr",
            ["package:paddleocr"],
            ["paddleocr"],
        ),
        (
            "audio_analysis",
            ["package:librosa"],
            ["librosa"],
        ),
        (
            "transcribe",
            ["package:faster_whisper"],
            ["faster_whisper"],
        ),
        (
            "transcribe_mlx",
            ["package:lightning_whisper_mlx"],
            ["lightning_whisper_mlx"],
        ),
    ],
)
def test_install_for_feature_repairs_full_runtime_stack_when_only_subset_is_missing(
    monkeypatch,
    feature_name,
    missing,
    expected_packages,
):
    """Fragile runtimes should reinstall their full repair package set."""
    package_batches: list[list[str]] = []
    cleared: list[list[str]] = []

    monkeypatch.setattr(
        "core.feature_registry.check_feature",
        lambda _name: (False, missing),
    )
    monkeypatch.setattr("core.dependency_manager.get_pip_specifier", lambda name: f"{name}>=1.0")
    monkeypatch.setattr("core.feature_registry._validate_feature_runtime", lambda _name: None)
    monkeypatch.setattr(
        "core.dependency_manager.clear_package_roots",
        lambda package_names: cleared.append(list(package_names)),
    )
    monkeypatch.setattr(
        "core.dependency_manager.install_packages",
        lambda specifiers, _progress=None: package_batches.append(list(specifiers)) or True,
    )

    assert install_for_feature(feature_name) is True
    assert cleared == [expected_packages]
    assert package_batches == [[f"{package_name}>=1.0" for package_name in expected_packages]]


@pytest.mark.parametrize(
    "feature_name",
    [
        "describe_local",
        "describe_local_cpu",
        "shot_classify",
        "image_classify",
        "object_detect",
        "ocr",
        "audio_analysis",
        "face_detect",
        "transcribe",
        "transcribe_mlx",
    ],
)
def test_runtime_validated_analysis_features_force_full_repair(feature_name):
    """Runtime-validated analysis features should prefer a full repair install."""
    assert requires_full_package_repair(feature_name, ["package:anything"]) is True
