"""Regression tests for dependency download UI and runtime-aware feature gating."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QDialog, QMessageBox

from ui.main_window import MainWindow
from ui.widgets.dependency_widgets import _DownloadWorker, prompt_feature_download


def test_download_worker_reports_failed_verification():
    events: list[tuple[str, str | None]] = []
    worker = _DownloadWorker(lambda _progress: False, lambda *_args: None)
    worker.finished_ok.connect(lambda: events.append(("ok", None)))
    worker.failed.connect(lambda message: events.append(("failed", message)))

    worker.run()

    assert events == [("failed", "Install completed but dependency verification failed.")]


def test_prompt_feature_download_rechecks_runtime_readiness(monkeypatch):
    checks = iter([
        (False, ["runtime:transformers install is incomplete"]),
        (False, ["runtime:transformers install is incomplete"]),
    ])

    class _FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature: next(checks),
    )
    monkeypatch.setattr("core.feature_registry.get_feature_size_estimate", lambda _feature: 450)
    monkeypatch.setattr(
        "core.feature_registry.install_for_feature",
        lambda _feature, _cb=None: True,
    )
    monkeypatch.setattr("ui.widgets.dependency_widgets.DependencyDownloadDialog", _FakeDialog)
    monkeypatch.setattr("ui.widgets.dependency_widgets.QMessageBox.question", lambda *args, **kwargs: QMessageBox.Yes)

    assert prompt_feature_download("shot_classify") is False


def test_prompt_feature_download_shows_full_repair_stack(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature: (False, ["package:ultralytics"]),
    )
    monkeypatch.setattr("core.feature_registry.get_feature_size_estimate", lambda _feature: 430)

    def _fake_question(_parent, _title, text, *_args, **_kwargs):
        captured["text"] = text
        return QMessageBox.No

    monkeypatch.setattr("ui.widgets.dependency_widgets.QMessageBox.question", _fake_question)

    assert prompt_feature_download("object_detect") is False
    assert "torch, ultralytics" in captured["text"]


def test_analysis_operation_gate_uses_runtime_ready_check(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace()

    monkeypatch.setattr(
        "ui.main_window.get_operation_feature_candidates",
        lambda *_args, **_kwargs: ["shot_classify"],
    )
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature: (False, ["runtime:broken"]),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda *_args, **_kwargs: False,
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "shots") is False


def test_extract_text_hybrid_prompts_for_ocr_install(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace(text_extraction_method="hybrid")

    prompts = []

    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature: (False, ["package:paddleocr"]),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda feature_name, *_args, **_kwargs: prompts.append(feature_name) or True,
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "extract_text") is True
    assert prompts == ["ocr"]


def test_description_gate_attempts_preferred_install_before_cpu_fallback(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace()

    prompts = []

    monkeypatch.setattr(
        "ui.main_window.get_operation_feature_candidates",
        lambda *_args, **_kwargs: ["describe_local", "describe_local_cpu"],
    )

    def _check_feature_ready(feature_name):
        if feature_name == "describe_local":
            return False, ["package:mlx_vlm"]
        if feature_name == "describe_local_cpu":
            return True, []
        raise AssertionError(f"unexpected feature {feature_name}")

    monkeypatch.setattr("core.feature_registry.check_feature_ready", _check_feature_ready)
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda feature_name, *_args, **_kwargs: prompts.append(feature_name) or False,
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "describe") is True
    assert prompts == ["describe_local"]


def test_description_gate_uses_fresh_preferred_install_when_available(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace()

    prompts = []

    monkeypatch.setattr(
        "ui.main_window.get_operation_feature_candidates",
        lambda *_args, **_kwargs: ["describe_local", "describe_local_cpu"],
    )
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda feature_name: ((feature_name == "describe_local"), []),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda feature_name, *_args, **_kwargs: prompts.append(feature_name) or True,
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "describe") is True
    assert prompts == []
