"""Exquisite Corpus proposals must not mutate or escape their project."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_description_operations import project_with_thumbnails

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def dialog(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from ui.dialogs.exquisite_corpus_dialog import ExquisiteCorpusDialog

    app = QApplication.instance() or QApplication([])
    project = project_with_thumbnails(tmp_path, 2)
    window = ExquisiteCorpusDialog(project.clips, project.sources_by_id, project)
    monkeypatch.setattr(window, "_generate_poem", Mock())
    yield window, project, app
    window.reject()


def test_proposal_clips_are_detached(dialog):
    window, project, _ = dialog
    assert window.clips[0] is not project.clips[0]
    assert window.sources_by_id[project.sources[0].id] is not project.sources[0]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("reopen", [False, True])
def test_saved_proposal_recovers_without_publishing(
    tmp_path, monkeypatch, empty, reopen
):
    from PySide6.QtWidgets import QApplication
    from core.project import Project
    from models.clip import ExtractedText
    from ui.dialogs.exquisite_corpus_dialog import ExquisiteCorpusDialog

    app = QApplication.instance() or QApplication([])
    project = project_with_thumbnails(tmp_path, 2)
    project.save(tmp_path / "project.json")
    original = project.path.read_bytes()
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(
            cache_dir=tmp_path,
            text_extraction_method="vlm",
            text_extraction_vlm_model="test",
        ),
    )
    provider = Mock(
        side_effect=lambda **kw: []
        if empty
        else [ExtractedText(kw["clip"].start_frame, "SIGN", 0.87654321, "vlm")]
    )
    monkeypatch.setattr("core.analysis.ocr.extract_text_from_clip", provider)
    windows = []

    def open_dialog(current):
        window = ExquisiteCorpusDialog(current.clips, current.sources_by_id, current)
        window._generate_poem = Mock()
        windows.append(window)
        return window

    window = open_dialog(project)
    try:
        for attempt in range(2):
            if attempt and reopen:
                window.reject()
                project = Project.load(project.path)
                window = open_dialog(project)
            window._start_extraction()
            assert window.worker.wait(10000)
            app.processEvents()
            assert window.worker.job_status == "completed"
            assert window.worker.cache is not None
            assert not project.metadata.job_results
            assert all(c.extracted_texts is None for c in project.clips)
            assert project.path.read_bytes() == original
            assert all(
                c.extracted_texts == []
                if empty
                else c.extracted_texts[0].confidence == 0.87654321
                for c in window.clips
            )
        assert provider.call_count == 2
    finally:
        for window in windows:
            window.reject()


def test_completed_ocr_enriches_only_proposal(dialog, monkeypatch):
    from core.operations.ocr import OcrOutcome, OcrText

    window, project, _ = dialog
    outcomes = tuple(
        OcrOutcome(
            c.id, "succeeded", texts=(OcrText(c.start_frame, "SIGN", 0.9, "vlm"),)
        )
        for c in project.clips
    )
    window.worker = SimpleNamespace(
        result=outcomes, is_cancelled=lambda: False, isRunning=lambda: False
    )
    window._on_extraction_finished({o.clip_id: o.to_models() for o in outcomes})
    assert all(c.extracted_texts is None for c in project.clips)
    assert all(c.extracted_texts[0].text == "SIGN" for c in window.clips)
    window._generate_poem.assert_called_once()


@pytest.mark.parametrize("mode", ["current", "edit", "closed", "worker", "context"])
def test_real_queued_worker_is_bound_to_proposal(dialog, monkeypatch, mode):
    from models.clip import ExtractedText

    window, project, app = dialog
    monkeypatch.setattr(
        "core.settings.load_settings",
        lambda: SimpleNamespace(
            text_extraction_method="vlm", text_extraction_vlm_model="test"
        ),
    )
    monkeypatch.setattr(
        "core.analysis.ocr.extract_text_from_clip",
        lambda **kwargs: [
            ExtractedText(kwargs["clip"].start_frame, "SIGN", 0.9, "vlm")
        ],
    )
    window._start_extraction()
    worker = window.worker
    assert worker.wait(5000)
    if mode == "edit":
        project.clips[0].notes = "unrelated edit"
        project.clips[0].end_frame += 1
    if mode == "closed":
        window.reject()
    if mode == "worker":
        window.worker = SimpleNamespace(
            is_cancelled=lambda: False, isRunning=lambda: False
        )
    if mode == "context":
        window._context_current = lambda: False
    app.processEvents()
    worker.extraction_completed.emit({})
    assert all(clip.extracted_texts is None for clip in project.clips)
    if mode == "current":
        window._generate_poem.assert_called_once()
        assert all(clip.extracted_texts[0].text == "SIGN" for clip in window.clips)
    else:
        window._generate_poem.assert_not_called()


def test_changed_input_cannot_generate_poem(dialog):
    window, project, _ = dialog
    project.clips[0].end_frame += 10
    window.worker = SimpleNamespace(
        result=(), is_cancelled=lambda: False, isRunning=lambda: False
    )
    window._on_extraction_finished({})
    assert "changed" in window.progress_label.text().lower()
    window._generate_poem.assert_not_called()


def test_empty_result_can_return_and_retry(dialog, monkeypatch):
    window, _, _ = dialog
    window.worker = SimpleNamespace(
        result=(), is_cancelled=lambda: False, isRunning=lambda: False
    )
    window.stack.setCurrentIndex(window.PAGE_PROGRESS)
    window._on_extraction_finished({})
    window.next_btn.click()
    assert window.stack.currentIndex() == window.PAGE_MOOD
    window.mood_input.setPlainText("quiet")
    start = Mock()
    monkeypatch.setattr(window, "_start_extraction", start)
    window.next_btn.click()
    start.assert_called_once()


@pytest.mark.parametrize("mode", ["text", "media", "session", "save_as", "closed"])
def test_finish_rejects_changed_or_closed_proposal(dialog, monkeypatch, tmp_path, mode):
    from models.clip import ExtractedText

    window, project, _ = dialog
    emitted = Mock()
    window.sequence_ready.connect(emitted)
    if mode == "text":
        project.clips[0].extracted_texts = [ExtractedText(0, "EDIT", 1, "vlm")]
    if mode == "media":
        project.sources[0].file_path.write_bytes(b"new media")
    if mode == "session":
        project.clear()
    if mode == "save_as":
        project.save(tmp_path / "copy.json")
    if mode == "closed":
        window.reject()
    window._finish()
    emitted.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    ["current", "project", "session", "workflow", "cancelled", "rejected_replacement"],
)
def test_intention_handoff_is_bound_to_original_workflow(tmp_path, monkeypatch, mode):
    from core.project import Project
    from core.intention_workflow import WorkflowState
    from ui.main_window import MainWindow
    from PySide6.QtWidgets import QDialog

    project = project_with_thumbnails(tmp_path, 1)
    workflow = SimpleNamespace(
        state=WorkflowState.BUILDING,
        get_all_sources=lambda: project.sources,
        cancel=Mock(),
    )
    replacement = SimpleNamespace(state=WorkflowState.BUILDING, cancel=Mock())
    window = SimpleNamespace(
        project=project,
        intention_workflow=workflow,
        _on_exquisite_corpus_sequence_ready=Mock(),
    )

    class Dialog:
        def __init__(self, **kwargs):
            self.recipe = None
            self.sequence_ready = SimpleNamespace(
                connect=lambda callback: setattr(self, "callback", callback)
            )

        def exec(self):
            if mode == "project":
                window.project = Project.new()
            if mode == "session":
                project.clear()
            if mode in ("workflow", "rejected_replacement"):
                window.intention_workflow = replacement
            if mode == "cancelled":
                workflow.state = WorkflowState.CANCELLED
            self.callback(["proposal"])
            return (
                QDialog.Rejected if mode == "rejected_replacement" else QDialog.Accepted
            )

    monkeypatch.setattr(
        "ui.dialogs.exquisite_corpus_dialog.ExquisiteCorpusDialog", Dialog
    )
    MainWindow._show_exquisite_corpus_dialog_for_intention(window, project.clips)
    if mode == "current":
        window._on_exquisite_corpus_sequence_ready.assert_called_once()
    else:
        window._on_exquisite_corpus_sequence_ready.assert_not_called()
    replacement.cancel.assert_not_called()


@pytest.mark.parametrize("mode", ["current", "project", "session"])
def test_sequence_tab_passes_project_and_rejects_stale_handoff(
    tmp_path, monkeypatch, mode
):
    from core.project import Project
    from ui.tabs.sequence_tab import SequenceTab

    project = project_with_thumbnails(tmp_path, 1)
    tab = SimpleNamespace(_project=project, _apply_exquisite_corpus_sequence=Mock())

    class Dialog:
        def __init__(self, *, clips, sources_by_id, project, parent, is_current):
            assert project is tab._project
            self.recipe = None
            self.sequence_ready = SimpleNamespace(
                connect=lambda callback: setattr(self, "callback", callback)
            )

        def exec(self):
            if mode == "project":
                tab._project = Project.new()
            if mode == "session":
                tab._project.clear()
            self.callback(["proposal"])
            self.callback(["proposal"])

    monkeypatch.setattr("ui.tabs.sequence_tab.ExquisiteCorpusDialog", Dialog)
    SequenceTab._show_exquisite_corpus_dialog(
        tab, [(project.clips[0], project.sources[0])]
    )
    if mode == "current":
        tab._apply_exquisite_corpus_sequence.assert_called_once_with(["proposal"], recipe=None)
    else:
        tab._apply_exquisite_corpus_sequence.assert_not_called()
