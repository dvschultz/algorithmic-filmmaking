"""Regression tests for dependency installation and progress reporting."""

from __future__ import annotations

import io
from pathlib import Path

from core.dependency_manager import install_package, install_packages
from core.feature_registry import install_for_feature


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
                "Collecting torchvision>=0.15.0",
                "Collecting transformers>=4.36",
                "Successfully installed torch-2.6.0 torchvision-0.21.0 transformers-4.52.0",
            ]
        )

    monkeypatch.setattr("core.dependency_manager.ensure_python", lambda cb=None: Path("/tmp/python"))
    monkeypatch.setattr("core.dependency_manager.get_subprocess_kwargs", lambda: {})
    monkeypatch.setattr("core.dependency_manager.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("core.dependency_manager._write_compat_marker", lambda: None)

    assert install_packages(["torch>=2.0", "torchvision>=0.15.0", "transformers>=4.36"]) is True
    assert captured_cmd.count("--target") == 1
    assert "--upgrade" in captured_cmd
    assert captured_cmd[-3:] == ["torch>=2.0", "torchvision>=0.15.0", "transformers>=4.36"]


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
        lambda _name: (False, ["package:torch", "package:torchvision", "package:transformers"]),
    )
    monkeypatch.setattr("core.dependency_manager.get_pip_specifier", lambda name: f"{name}>=1.0")
    monkeypatch.setattr("core.dependency_manager.install_packages", _fake_install)
    monkeypatch.setattr(
        "core.feature_registry._validate_feature_runtime",
        lambda name: validated.append(name),
    )

    assert install_for_feature("describe_local_cpu", _on_progress) is True
    assert package_batches == [["torch>=1.0", "torchvision>=1.0", "transformers>=1.0"]]
    assert validated == ["describe_local_cpu"]
    assert [round(progress, 2) for progress, _ in progress_calls] == [0.0, 0.5, 1.0]
