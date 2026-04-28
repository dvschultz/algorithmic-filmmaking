"""Tests for the CassetteTapeDialog state transitions and emitted signals.

These tests focus on the dialog's UX contract — page navigation, button
enablement, signal emission shapes — rather than the matching algorithm
itself (covered in tests/test_cassette_tape.py).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.remix.cassette_tape import MatchResult
from core.transcription import TranscriptSegment
from models.clip import Clip, Source


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start_time=start, end_time=end, text=text)


def _make_clip(clip_id: str, source_id: str, segments) -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=900,
        transcript=segments,
    )


def _make_source(source_id: str = "s1", fps: float = 30.0) -> Source:
    return Source(
        id=source_id,
        file_path=Path(f"/test/{source_id}.mp4"),
        duration_seconds=60.0,
        fps=fps,
        width=1920,
        height=1080,
    )


@pytest.fixture
def transcribed_project(qapp):
    """A small project with two transcribed clips."""
    src = _make_source("s1")
    c1 = _make_clip("c1", "s1", [
        _seg(0.0, 1.5, "well thank you for coming today"),
        _seg(1.5, 3.0, "I really appreciate it"),
    ])
    c2 = _make_clip("c2", "s1", [_seg(0.0, 2.0, "thank you very much")])
    return [c1, c2], {"s1": src}, None  # (clips, sources_by_id, project)


@pytest.fixture
def empty_project(qapp):
    """A project with no transcribed clips."""
    src = _make_source("s1")
    c1 = Clip(id="c1", source_id="s1", start_frame=0, end_frame=900, transcript=None)
    return [c1], {"s1": src}, None


def _make_dialog(qapp, project_data):
    from ui.dialogs.cassette_tape_dialog import CassetteTapeDialog
    clips, sources, project = project_data
    return CassetteTapeDialog(clips=clips, sources_by_id=sources, project=project)


class TestSetupPage:
    def test_initializes_with_three_empty_phrase_rows(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        assert len(dialog.phrase_rows) == 3
        assert all(r.phrase() == "" for r in dialog.phrase_rows)

    def test_find_matches_disabled_until_phrase_entered(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        assert dialog.next_btn.isEnabled() is False
        # Type in the first row
        dialog.phrase_rows[0].line_edit.setText("thank you")
        assert dialog.next_btn.isEnabled() is True

    def test_no_transcripts_disables_find_matches_with_warning(self, qapp, empty_project):
        dialog = _make_dialog(qapp, empty_project)
        assert dialog._setup_no_transcripts is True
        # Even with phrase typed, button stays disabled
        dialog.phrase_rows[0].line_edit.setText("thank you")
        assert dialog.next_btn.isEnabled() is False

    def test_add_phrase_row_increases_count(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        before = len(dialog.phrase_rows)
        dialog._add_phrase_row()
        assert len(dialog.phrase_rows) == before + 1

    def test_remove_phrase_row_decreases_count(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        row = dialog.phrase_rows[0]
        before = len(dialog.phrase_rows)
        dialog._remove_phrase_row(row)
        assert len(dialog.phrase_rows) == before - 1

    def test_phrases_with_counts_skips_empty_rows(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        dialog.phrase_rows[0].line_edit.setText("thank you")
        dialog.phrase_rows[1].slider.setValue(2)  # but no phrase text
        dialog.phrase_rows[2].line_edit.setText("hello")
        dialog.phrase_rows[2].slider.setValue(4)
        result = dialog._phrases_with_counts()
        assert result == [("thank you", 3), ("hello", 4)]


class TestProgressPageAndWorkerFlow:
    def test_clicking_find_matches_advances_to_progress(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        dialog.phrase_rows[0].line_edit.setText("thank you")

        # Patch worker.start so it doesn't actually run a thread
        with patch("ui.dialogs.cassette_tape_dialog.CassetteTapeWorker.start"):
            dialog._on_next()

        assert dialog.stack.currentIndex() == dialog.PAGE_PROGRESS
        assert dialog.next_btn.text() == "Please wait…"
        assert dialog.next_btn.isEnabled() is False

    def test_matches_ready_advances_to_review(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        clips, _, _ = transcribed_project
        match = MatchResult(
            phrase="thank you", clip_id="c1", segment_index=0,
            segment=clips[0].transcript[0], score=92, match_start=5, match_end=14,
        )
        dialog._on_matches_ready({"thank you": [match]})
        assert dialog.stack.currentIndex() == dialog.PAGE_REVIEW
        assert dialog.next_btn.text() == "Generate Sequence"
        assert dialog.next_btn.isEnabled() is True
        assert len(dialog._match_rows) == 1

    def test_match_error_returns_to_setup(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        with patch("ui.dialogs.cassette_tape_dialog.QMessageBox.warning") as warn:
            dialog._on_match_error("rapidfuzz blew up")
        assert dialog.stack.currentIndex() == dialog.PAGE_SETUP
        # Error surfaces via QMessageBox, not the hidden progress page label.
        assert warn.called
        args = warn.call_args.args
        assert "rapidfuzz blew up" in args[2]


class TestCancellation:
    def test_match_phrases_aborts_when_is_cancelled_returns_true(self, qapp, transcribed_project):
        # match_phrases checks the cancel flag between phrases and returns {}.
        from core.remix.cassette_tape import match_phrases
        clips, _, _ = transcribed_project
        cancelled = {"flag": True}
        result = match_phrases(
            [("thank you", 1), ("hello", 1)],
            clips,
            is_cancelled=lambda: cancelled["flag"],
        )
        assert result == {}

    def test_cancel_during_progress_disconnects_signals_and_clears_worker(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        dialog.phrase_rows[0].line_edit.setText("thank you")
        with patch("ui.dialogs.cassette_tape_dialog.CassetteTapeWorker.start"):
            dialog._on_next()
        # Worker is alive but not running (start patched). Force isRunning to True.
        with patch.object(dialog.worker, "isRunning", return_value=True), \
             patch.object(dialog.worker, "wait") as wait_call, \
             patch.object(dialog.worker, "quit") as quit_call:
            dialog._stop_worker_if_running()
        wait_call.assert_called_once_with()  # unbounded
        quit_call.assert_called_once()
        assert dialog.worker is None

    def test_close_event_stops_worker(self, qapp, transcribed_project):
        from PySide6.QtGui import QCloseEvent
        dialog = _make_dialog(qapp, transcribed_project)
        dialog.phrase_rows[0].line_edit.setText("thank you")
        with patch("ui.dialogs.cassette_tape_dialog.CassetteTapeWorker.start"):
            dialog._on_next()
        called = {"stop": 0}
        with patch.object(dialog, "_stop_worker_if_running",
                          side_effect=lambda: called.update(stop=called["stop"] + 1)):
            dialog.closeEvent(QCloseEvent())
        assert called["stop"] == 1


class TestReviewPage:
    def _populate(self, dialog, clips, n=2):
        match1 = MatchResult(
            phrase="thank you", clip_id="c1", segment_index=0,
            segment=clips[0].transcript[0], score=92, match_start=5, match_end=14,
        )
        match2 = MatchResult(
            phrase="thank you", clip_id="c2", segment_index=0,
            segment=clips[1].transcript[0], score=88, match_start=0, match_end=9,
        )
        dialog._on_matches_ready({"thank you": [match1, match2][:n]})

    def test_all_matches_default_enabled(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        clips, _, _ = transcribed_project
        self._populate(dialog, clips, n=2)
        assert all(r.is_enabled() for r in dialog._match_rows)

    def test_toggling_all_off_disables_generate(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        clips, _, _ = transcribed_project
        self._populate(dialog, clips, n=2)
        for row in dialog._match_rows:
            row.checkbox.setChecked(False)
        assert dialog.next_btn.isEnabled() is False

    def test_back_returns_to_setup(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        clips, _, _ = transcribed_project
        self._populate(dialog, clips, n=1)
        dialog._on_back()
        assert dialog.stack.currentIndex() == dialog.PAGE_SETUP


class TestSequenceReadyEmission:
    def test_generate_emits_sequence_ready_with_4_tuples(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        clips, sources, _ = transcribed_project
        match = MatchResult(
            phrase="thank you", clip_id="c1", segment_index=0,
            segment=clips[0].transcript[0], score=92, match_start=5, match_end=14,
        )
        dialog._on_matches_ready({"thank you": [match]})

        captured = []
        dialog.sequence_ready.connect(lambda data: captured.append(data))
        dialog._finish_with_sequence()

        assert len(captured) == 1
        seq = captured[0]
        assert len(seq) == 1
        ret_clip, ret_source, in_frame, out_frame = seq[0]
        assert ret_clip.id == "c1"
        assert ret_source.id == "s1"
        # 0.0–1.5s @ 30fps → 0–45
        assert in_frame == 0
        assert out_frame == 45

    def test_generate_filters_disabled_matches(self, qapp, transcribed_project):
        dialog = _make_dialog(qapp, transcribed_project)
        clips, sources, _ = transcribed_project
        match1 = MatchResult(
            phrase="thank you", clip_id="c1", segment_index=0,
            segment=clips[0].transcript[0], score=92, match_start=5, match_end=14,
        )
        match2 = MatchResult(
            phrase="thank you", clip_id="c2", segment_index=0,
            segment=clips[1].transcript[0], score=88, match_start=0, match_end=9,
        )
        dialog._on_matches_ready({"thank you": [match1, match2]})

        # Disable the first match
        dialog._match_rows[0].checkbox.setChecked(False)

        captured = []
        dialog.sequence_ready.connect(lambda data: captured.append(data))
        dialog._finish_with_sequence()

        assert len(captured) == 1
        seq = captured[0]
        assert len(seq) == 1
        assert seq[0][0].id == "c2"


class TestHighlightRendering:
    def test_match_row_highlights_substring(self, qapp, transcribed_project):
        from ui.dialogs.cassette_tape_dialog import _MatchRow

        clips, _, _ = transcribed_project
        match = MatchResult(
            phrase="thank you", clip_id="c1", segment_index=0,
            segment=clips[0].transcript[0], score=92, match_start=5, match_end=14,
        )
        row = _MatchRow(match, "Test Clip")
        # Find the snippet QLabel by walking children
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel
        snippet_labels = [w for w in row.findChildren(QLabel)
                          if w.textFormat() == Qt.RichText]
        assert snippet_labels, "expected at least one rich-text snippet label"
        html = snippet_labels[0].text()
        assert "<b" in html  # the matched substring is wrapped in <b>
        # The exact matched text "thank you" should appear inside the bold
        assert "thank you" in html
