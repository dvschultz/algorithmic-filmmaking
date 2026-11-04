"""Batch analysis failures are summarized with counts and can be retried (#184)."""

from types import SimpleNamespace
from unittest.mock import Mock

from core.operations.clip_analysis import clip_analysis_failure_report
from core.project import Project
from models.clip import Clip, Source
from ui.main_window import MainWindow


def _result(clip_ids, failed_ids, errors):
    return {
        "succeeded": [cid for cid in clip_ids if cid not in failed_ids],
        "failed": list(failed_ids),
        "cancelled": False,
        "operations": {
            "colors": {cid: "succeeded" for cid in clip_ids},
            "describe": {
                cid: ("failed" if cid in failed_ids else "succeeded")
                for cid in clip_ids
            },
        },
        "errors": errors,
        "analyzed_sources": [],
    }


def test_report_counts_and_targets_only_failed_work():
    clip_ids = [f"c{i}" for i in range(2000)]
    failed = clip_ids[:40]
    errors = [f"describe: {cid}: 503 Service Unavailable" for cid in failed]
    report = clip_analysis_failure_report(_result(clip_ids, failed, errors))

    assert report.succeeded == 1960
    assert report.failed == 40
    assert report.retry_clip_ids == tuple(failed)
    # Colors succeeded everywhere; only the failed operation is retried.
    assert report.retry_operations == ("describe",)
    assert "1960 clips succeeded" in report.summary
    assert "40 failed" in report.summary
    # The dialog body stays bounded however many clips fail.
    assert report.details.count("\n") < 20
    assert "... and" in report.details
    assert report.full_log.count("503") == 40


def _window(tmp_path, monkeypatch, n=3):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4")
    source.file_path.write_bytes(b"source")
    clips = [
        Clip(source_id=source.id, start_frame=i * 30, end_frame=i * 30 + 30)
        for i in range(n)
    ]
    project.add_source(source)
    project.add_clips(clips)
    window = SimpleNamespace(
        project=project,
        _gui_state=Mock(),
        analyze_tab=Mock(),
        progress_bar=Mock(),
        status_bar=Mock(),
        collect_tab=Mock(),
        _update_chat_project_state=Mock(),
        _run_analysis_pipeline=Mock(return_value=True),
    )
    controller = SimpleNamespace(
        owns_view=lambda: True,
        is_current=lambda: True,
        clips={c.id: c for c in clips},
        reply=None,
        force_rerun=False,
    )
    return window, controller, clips


def test_retry_failed_reruns_only_failed_clips(tmp_path, monkeypatch):
    window, controller, clips = _window(tmp_path, monkeypatch)
    prompt = Mock(return_value=True)
    monkeypatch.setattr("ui.main_window.ask_retry_failed_analysis", prompt)
    ids = [c.id for c in clips]
    MainWindow._on_clip_analysis_finished(
        window,
        controller,
        _result(ids, [ids[1]], [f"describe: {ids[1]}: 429 rate limited"]),
    )
    prompt.assert_called_once()
    report = prompt.call_args.args[1]
    assert report.failed == 1 and report.succeeded == 2
    assert "429 rate limited" in report.details
    window._run_analysis_pipeline.assert_called_once_with(
        [clips[1]], ["describe"], force_rerun=False
    )
    assert "1 failed" in window.status_bar.showMessage.call_args.args[0]


def test_declining_retry_does_not_rerun(tmp_path, monkeypatch):
    window, controller, clips = _window(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ui.main_window.ask_retry_failed_analysis", Mock(return_value=False)
    )
    ids = [c.id for c in clips]
    MainWindow._on_clip_analysis_finished(
        window, controller, _result(ids, [ids[0]], ["describe: boom"])
    )
    window._run_analysis_pipeline.assert_not_called()


def test_dialog_offers_retry_and_reports_choice(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from ui.dialogs.analysis_failure_dialog import ask_retry_failed_analysis

    report = clip_analysis_failure_report(
        _result(["a", "b"], ["b"], ["describe: b: 429"])
    )
    seen = {}

    def fake_exec(box):
        seen["text"] = box.text()
        seen["buttons"] = [b.text() for b in box.buttons()]
        retry = next(b for b in box.buttons() if b.text().startswith("Retry"))
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda _self: retry)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    assert ask_retry_failed_analysis(None, report) is True
    assert "1 failed" in seen["text"]
    assert "Retry Failed (1)" in seen["buttons"]
