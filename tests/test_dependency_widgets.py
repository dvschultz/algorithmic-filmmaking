"""Regression tests for dependency download UI and runtime-aware feature gating."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QDialog, QMessageBox
from PySide6.QtCore import QTimer

from ui.main_window import MainWindow
from ui.widgets.dependency_widgets import (
    DependencyDownloadDialog,
    _DownloadWorker,
    prompt_feature_download,
)


def test_download_worker_reports_failed_verification():
    events: list[tuple[str, str | None]] = []
    worker = _DownloadWorker(lambda _progress: False, lambda *_args: None)
    worker.finished_ok.connect(lambda: events.append(("ok", None)))
    worker.failed.connect(lambda message: events.append(("failed", message)))

    worker.run()

    assert events == [("failed", "Install completed but dependency verification failed.")]


def test_download_worker_does_not_retry_internal_type_error():
    calls = []
    events = []

    def install(_progress, _cancel_event):
        calls.append(True)
        raise TypeError("installer bug")

    worker = _DownloadWorker(install, lambda *_args: None)
    worker.failed.connect(events.append)
    worker.run()

    assert calls == [True]
    assert events == ["installer bug"]


def test_download_dialog_waits_for_matching_thread_finish_before_retry(monkeypatch):
    dialog = DependencyDownloadDialog("Install", "Installing", lambda _progress: True)
    old_worker = _DownloadWorker(lambda _progress: True, lambda *_args: None)
    replacement = _DownloadWorker(lambda _progress: True, lambda *_args: None)
    dialog._worker = old_worker

    dialog._on_failure(old_worker, "failed")

    assert dialog._worker is old_worker
    assert not dialog._download_btn.isEnabled()

    dialog._worker = replacement
    dialog._on_worker_stopped(old_worker)

    assert dialog._worker is replacement
    assert not dialog._download_btn.isEnabled()

    dialog._worker = old_worker
    dialog._on_worker_stopped(old_worker)

    assert dialog._worker is None
    assert dialog._download_btn.isEnabled()
    assert dialog._download_btn.text() == "Retry"


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
        lambda _feature, **_kwargs: next(checks),
    )
    monkeypatch.setattr("core.feature_registry.get_feature_size_estimate", lambda _feature: 450)
    monkeypatch.setattr(
        "core.feature_registry.install_for_feature",
        lambda _feature, _cb=None: True,
    )
    monkeypatch.setattr("ui.widgets.dependency_widgets.DependencyDownloadDialog", _FakeDialog)
    monkeypatch.setattr("ui.widgets.dependency_widgets.QMessageBox.question", lambda *args, **kwargs: QMessageBox.Yes)

    assert prompt_feature_download("shot_classify") is False


def test_readiness_cancel_keeps_gui_responsive_until_probe_stops(monkeypatch):
    import time

    from ui.widgets.dependency_widgets import (
        _ReadinessDialog,
        _check_feature_ready_responsive,
    )

    probe_stopped = []
    heartbeat = []

    def _blocked_probe(_feature, *, cancel_event=None):
        assert cancel_event is not None
        while not cancel_event.wait(0.01):
            pass
        time.sleep(0.05)
        probe_stopped.append(True)
        return True, []  # a late successful probe must not override cancellation

    monkeypatch.setattr("core.feature_registry.check_feature_ready", _blocked_probe)
    original_exec = _ReadinessDialog.exec

    def _exec_and_cancel(self):
        timer = QTimer(self)
        timer.timeout.connect(lambda: heartbeat.append(True))
        timer.start(5)
        QTimer.singleShot(10, self.reject)  # exercise the window-close path
        return original_exec(self)

    monkeypatch.setattr(_ReadinessDialog, "exec", _exec_and_cancel)

    assert _check_feature_ready_responsive("transcribe") == (
        False,
        ["runtime:readiness check cancelled"],
    )
    assert probe_stopped == [True]
    assert heartbeat  # nested Qt event processing continued during cancellation


def test_cancelled_readiness_does_not_offer_install(monkeypatch):
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets._check_feature_ready_responsive",
        lambda *_args: (False, ["runtime:readiness check cancelled"]),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.QMessageBox.question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled readiness must not offer installation")
        ),
    )

    assert prompt_feature_download("transcribe") is False


def test_analysis_gate_cancel_stops_fallback_and_install_prompt(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace()

    checks = []
    monkeypatch.setattr(
        "ui.main_window.get_operation_feature_candidates",
        lambda *_args, **_kwargs: ["describe_local", "describe_local_cpu"],
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets._check_feature_ready_responsive",
        lambda feature, *_args: checks.append(feature)
        or (False, ["runtime:readiness check cancelled"]),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled readiness must not start an install prompt")
        ),
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "describe") is False
    assert checks == ["describe_local"]


def test_prompt_feature_download_shows_full_repair_stack(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature, **_kwargs: (False, ["package:ultralytics"]),
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
        lambda _feature, **_kwargs: (False, ["runtime:broken"]),
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
        lambda _feature, **_kwargs: (False, ["package:paddleocr"]),
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

    def _check_feature_ready(feature_name, **_kwargs):
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


def test_analysis_gate_prompts_alternate_when_preferred_install_fails(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace(transcription_backend="auto")

    prompts = []

    monkeypatch.setattr(
        "ui.main_window.get_operation_feature_candidates",
        lambda *_args, **_kwargs: ["transcribe_mlx", "transcribe"],
    )
    monkeypatch.setattr(
        "core.feature_registry.check_feature_ready",
        lambda _feature, **_kwargs: (False, ["package:missing"]),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda feature_name, *_args, **_kwargs: prompts.append(feature_name) or feature_name == "transcribe",
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "transcribe") is True
    assert prompts == ["transcribe_mlx", "transcribe"]


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
        lambda feature_name, **_kwargs: ((feature_name == "describe_local"), []),
    )
    monkeypatch.setattr(
        "ui.widgets.dependency_widgets.prompt_feature_download",
        lambda feature_name, *_args, **_kwargs: prompts.append(feature_name) or True,
    )

    assert MainWindow._ensure_analysis_operation_available(Harness(), "describe") is True
    assert prompts == []


def test_groq_transcription_gate_requires_api_key(monkeypatch):
    class Harness:
        def __init__(self):
            self.settings = SimpleNamespace(transcription_backend="groq")
            self.status = []
            self.status_bar = SimpleNamespace(showMessage=lambda message: self.status.append(message))

    monkeypatch.setattr("core.settings.get_groq_api_key", lambda: "")

    harness = Harness()

    assert MainWindow._ensure_analysis_operation_available(harness, "transcribe") is False
    assert harness.status == [
        "Groq transcription selected but no Groq API key is configured. "
        "Add it in Settings > API Keys."
    ]
