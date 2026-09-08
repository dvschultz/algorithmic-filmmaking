"""Tests for ``WordLLMComposerDialog``.

These exercise the dialog's pure-Python flow without invoking the real
Ollama HTTP path. The Ollama health probe is injected via the dialog's
``_ollama_health_fn`` keyword arg; the compose worker is monkeypatched to a
fake that produces deterministic ``SequenceClip[]`` payloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.transcription import TranscriptSegment, WordTimestamp
from models.clip import Clip, Source
from models.sequence import SequenceClip

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def verified_word_fixtures(tmp_path, monkeypatch):
    from tests.analysis_fixtures import verify_word_alignment
    original = _aligned_clip

    def verified(*args, **kwargs):
        clip, source = original(*args, **kwargs)
        verify_word_alignment(clip, source, tmp_path)
        return clip, source

    monkeypatch.setattr("core.operations.alignment_records.alignment_model_revision", lambda: "fixture-r1")
    monkeypatch.setitem(globals(), "_aligned_clip", verified)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _aligned_clip(clip_id: str = "clip-1", source_id: str = "src-1") -> tuple:
    source = Source(
        id=source_id,
        file_path=Path(f"/test/{source_id}.mp4"),
        duration_seconds=30.0,
        fps=24.0,
        width=640,
        height=360,
    )
    clip = Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=240,
    )
    clip.transcript = [
        TranscriptSegment(
            start_time=0.0,
            end_time=1.0,
            text="quiet light",
            confidence=0.9,
            words=[
                WordTimestamp(start=0.0, end=0.4, text="quiet"),
                WordTimestamp(start=0.4, end=1.0, text="light"),
            ],
            language="en",
        )
    ]
    return clip, source


def test_ollama_absent_shows_install_page(qapp):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (False, "Ollama is not running"),
    )

    # Page index 2 = ollama-absent page.
    assert dialog._stack.currentIndex() == 2
    assert "Local LLM not detected" in dialog._ollama_message_label.text()
    assert dialog._accept_btn.isEnabled() is False


def test_llm_composer_worker_default_timeout_is_dialog_safe(qapp):
    from ui.workers.llm_composer_worker import LLMComposerWorker

    worker = LLMComposerWorker(
        clips=[],
        prompt="test",
        target_length=1,
    )

    assert worker._timeout == 30.0


def test_ollama_healthy_shows_form(qapp):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    assert dialog._stack.currentIndex() == 0
    # Source picker populated.
    assert dialog._source_list.count() == 1


def test_seed_visible_only_for_random_policy(qapp):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    # Round-robin default → seed hidden.
    assert dialog._seed_container.isHidden() is True

    # Switch to random → seed shows.
    for i in range(dialog._repeat_combo.count()):
        if dialog._repeat_combo.itemData(i) == "random":
            dialog._repeat_combo.setCurrentIndex(i)
            break
    assert dialog._seed_container.isHidden() is False


def test_validation_requires_prompt(qapp):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    dialog._prompt_input.setPlainText("")
    dialog._refresh_validation()
    assert dialog._accept_btn.isEnabled() is False

    dialog._prompt_input.setPlainText("hello")
    dialog._refresh_validation()
    assert dialog._accept_btn.isEnabled() is True


def test_prompt_editor_has_room_for_long_prompts(qapp):
    from PySide6.QtWidgets import QSizePolicy

    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    assert dialog._prompt_input.minimumHeight() >= 160
    assert (
        dialog._prompt_input.sizePolicy().verticalPolicy()
        == QSizePolicy.Policy.MinimumExpanding
    )


def test_accept_dispatches_compose_worker(qapp, monkeypatch):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    starts: list = []

    class FakeComposeWorker:
        def __init__(self, *, clips, prompt, target_length, repeat_policy,
                     seed=None, handle_frames=0, parent=None, **kwargs):
            self._prompt = prompt
            self._target_length = target_length

            class _Sig:
                def connect(self, *args, **kwargs):
                    return None

            self.progress = _Sig()
            self.sequence_ready = _Sig()
            self.error = _Sig()

        def isRunning(self):
            return True

        def start(self):
            starts.append((self._prompt, self._target_length))

        def cancel(self):
            pass

        def wait(self, _ms):
            return True

    monkeypatch.setattr(
        "ui.workers.llm_composer_worker.LLMComposerWorker",
        FakeComposeWorker,
    )

    dialog._prompt_input.setPlainText("compose a quiet sentence")
    dialog._length_spin.setValue(7)
    dialog._refresh_validation()
    dialog._on_accept()

    assert starts == [("compose a quiet sentence", 7)]


def test_compose_ready_routes_to_review_then_apply_emits(qapp):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    captured: list = []
    dialog.sequence_ready.connect(captured.append)

    fake_seq = [
        SequenceClip(
            source_clip_id=clip.id,
            source_id=source.id,
            in_point=0,
            out_point=10,
            rationale="silence",
        )
    ]
    dialog._on_compose_ready(fake_seq)

    # Compose-ready now routes to the review page (index 3); nothing is
    # emitted until the user clicks Apply.
    assert captured == []
    assert dialog._stack.currentIndex() == 3
    assert "silence" in dialog._review_sentence.toPlainText()

    dialog._on_review_apply()
    assert captured == [fake_seq]


def test_compose_error_keeps_dialog_open(qapp):
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )

    dialog._stack.setCurrentIndex(1)  # progress page
    dialog._on_compose_error("LLM returned empty response")

    # Stack returned to form page, error label populated.
    assert dialog._stack.currentIndex() == 0
    assert "empty response" in dialog._error_label.text()


def test_close_during_alignment_cancels_worker(qapp):
    from PySide6.QtGui import QCloseEvent
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    clip, source = _aligned_clip()
    dialog = WordLLMComposerDialog(
        clips=[(clip, source)],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )
    calls: list[str] = []

    class FakeController:
        def is_running(self):
            return True

        def cancel(self):
            calls.append("cancel")

        def wait(self, ms):
            calls.append(f"wait:{ms}")

    dialog._alignment_ctrl = FakeController()
    dialog.closeEvent(QCloseEvent())

    assert calls == ["cancel", "wait:2000"]


def test_empty_corpus_disables_accept(qapp):
    """With no source clips at all, accept stays disabled."""
    from ui.dialogs.word_llm_composer_dialog import WordLLMComposerDialog

    dialog = WordLLMComposerDialog(
        clips=[],
        project=None,
        _ollama_health_fn=lambda: (True, ""),
    )
    dialog._prompt_input.setPlainText("hello")
    dialog._refresh_validation()
    assert dialog._accept_btn.isEnabled() is False
