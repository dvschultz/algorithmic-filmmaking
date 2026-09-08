"""Dependency gates preserve the original clip request across nested UI events."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.project import Project
from models.clip import Clip, Source
from ui.main_window import MainWindow


def dispatch_setup(tmp_path, monkeypatch):
    project = Project.new()
    source = Source(file_path=tmp_path / "source.mp4")
    source.file_path.write_bytes(b"source")
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    project.add_source(source)
    project.add_clips([clip])
    factory = Mock()
    monkeypatch.setattr("ui.workers.clip_analysis.ClipAnalysisController", factory)
    window = SimpleNamespace(
        project=project,
        _custom_query_text=None,
        _filter_available_analysis_operations=lambda ops: ops,
        _gui_state=Mock(),
        analyze_tab=Mock(),
        progress_bar=Mock(),
        _on_clip_analysis_progress=Mock(),
        _on_clip_analysis_status=Mock(),
        _on_clip_analysis_finished=Mock(),
    )
    return window, clip, factory


def test_only_available_enabled_work_is_dispatched(tmp_path, monkeypatch):
    window, clip, factory = dispatch_setup(tmp_path, monkeypatch)
    disabled = Clip(
        source_id=clip.source_id, start_frame=30, end_frame=60, disabled=True
    )
    window.project.add_clips([disabled])
    window._filter_available_analysis_operations = lambda ops: [
        op for op in ops if op != "shots"
    ]
    assert MainWindow._run_analysis_pipeline(
        window, [clip, disabled], ["colors", "shots"]
    )
    factory.assert_called_once_with(
        window, [clip], ["colors"], force_rerun=False, query=None
    )
    factory.return_value.start.assert_called_once()


@pytest.mark.parametrize(
    "change", ["project", "session", "clip", "source", "media", "path", "run"]
)
def test_gate_rejects_changed_request(tmp_path, monkeypatch, change):
    window, clip, factory = dispatch_setup(tmp_path, monkeypatch)

    def gate(operations):
        if change == "project":
            window.project = Project.new()
        elif change == "session":
            window.project.session.session_id = "replacement"
        elif change == "clip":
            window.project.clips_by_id[clip.id] = Clip(
                id=clip.id, source_id=clip.source_id, start_frame=0, end_frame=30
            )
        elif change == "source":
            window.project.sources_by_id[clip.source_id] = Source(
                id=clip.source_id, file_path=tmp_path / "source.mp4"
            )
        elif change == "media":
            (tmp_path / "source.mp4").write_bytes(b"replacement")
        elif change == "path":
            window.project.path = tmp_path / "other.json"
        elif change == "run":
            window._clip_analysis_controller = object()
        return operations

    window._filter_available_analysis_operations = gate
    assert not MainWindow._run_analysis_pipeline(window, [clip], ["colors"])
    factory.assert_not_called()


def test_force_dispatch_preserves_prior_results_until_replacement(
    tmp_path, monkeypatch
):
    window, clip, factory = dispatch_setup(tmp_path, monkeypatch)
    clip.dominant_colors = [(1, 2, 3)]
    assert MainWindow._run_analysis_pipeline(
        window, [clip], ["colors"], force_rerun=True
    )
    assert clip.dominant_colors == [(1, 2, 3)]
    assert factory.call_args.kwargs["force_rerun"]


def test_empty_disabled_and_blocked_requests_do_not_start(tmp_path, monkeypatch):
    window, clip, factory = dispatch_setup(tmp_path, monkeypatch)
    assert not MainWindow._run_analysis_pipeline(window, [], ["colors"])
    clip.disabled = True
    assert not MainWindow._run_analysis_pipeline(window, [clip], ["colors"])
    clip.disabled = False
    window._filter_available_analysis_operations = lambda ops: []
    assert not MainWindow._run_analysis_pipeline(window, [clip], ["colors"])
    factory.assert_not_called()


def test_ui_start_notification_cannot_replace_a_newer_run(tmp_path, monkeypatch):
    window, clip, factory = dispatch_setup(tmp_path, monkeypatch)
    replacement = object()
    window.analyze_tab.set_analyzing.side_effect = lambda *args: setattr(
        window, "_clip_analysis_controller", replacement
    )
    assert not MainWindow._run_analysis_pipeline(window, [clip], ["colors"])
    factory.return_value.start.assert_not_called()
    assert window._clip_analysis_controller is replacement


def test_manual_failure_shows_actionable_reason(tmp_path, monkeypatch):
    window, clip, _ = dispatch_setup(tmp_path, monkeypatch)
    window.status_bar = Mock()
    window.collect_tab = Mock()
    window._update_chat_project_state = Mock()
    warning = Mock()
    monkeypatch.setattr("ui.main_window.QMessageBox.warning", warning)
    controller = SimpleNamespace(
        owns_view=lambda: True,
        is_current=lambda: True,
        clips={clip.id: clip},
        reply=None,
    )
    MainWindow._on_clip_analysis_finished(
        window,
        controller,
        {
            "succeeded": [],
            "failed": [clip.id],
            "cancelled": False,
            "operations": {"describe": {clip.id: "failed"}},
            "errors": ["describe: provider unavailable"],
            "analyzed_sources": [],
        },
    )
    warning.assert_called_once()
    assert "provider unavailable" in warning.call_args.args[2]
