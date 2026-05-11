"""Tests for ``WordSequencerDialog``.

Run under offscreen Qt so these don't open windows in CI. Tests exercise the
dialog's pure-Python paths — picker population, validation, mode-conditional
visibility, accept-path delegation — without spinning up the Ollama or
alignment workers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.transcription import TranscriptSegment, WordTimestamp
from models.clip import Clip, Source

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _aligned_segment(words: list[tuple[str, float, float]], language: str = "en") -> TranscriptSegment:
    seg = TranscriptSegment(
        start_time=words[0][1] if words else 0.0,
        end_time=words[-1][2] if words else 0.0,
        text=" ".join(w[0] for w in words),
        confidence=0.9,
        words=[WordTimestamp(start=s, end=e, text=t) for t, s, e in words],
        language=language,
    )
    return seg


def _unaligned_segment(text: str, start: float, end: float, language: str = "en") -> TranscriptSegment:
    return TranscriptSegment(
        start_time=start,
        end_time=end,
        text=text,
        confidence=0.9,
        words=None,
        language=language,
    )


def _make_aligned_clip(clip_id: str = "clip-1", source_id: str = "src-1") -> tuple:
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
        _aligned_segment([
            ("the", 0.0, 0.2),
            ("apple", 0.2, 0.5),
            ("falls", 0.5, 0.9),
        ]),
    ]
    return clip, source


def _make_unaligned_clip(clip_id: str = "clip-2", source_id: str = "src-2") -> tuple:
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
    clip.transcript = [_unaligned_segment("the sky is blue", 0.0, 1.0)]
    return clip, source


def _make_unsupported_lang_clip(clip_id: str = "clip-3", source_id: str = "src-3") -> tuple:
    """Build a clip whose transcript language is not in the supported set."""
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
    # "xx" — guaranteed not to appear in any real ISO 639-1 list.
    clip.transcript = [_aligned_segment([("foo", 0.0, 0.2)], language="xx")]
    return clip, source


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dialog_opens_with_aligned_clips(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)

    # Source picker populated, row enabled and checked.
    assert dialog._source_list.count() == 1
    item = dialog._source_list.item(0)
    from PySide6.QtCore import Qt
    assert item.checkState() == Qt.Checked
    assert "✓ aligned" in item.text()

    # Accept enabled in alphabetical mode for a non-empty corpus.
    assert dialog._accept_btn.isEnabled() is True


def test_mode_visibility_toggles(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)

    # Initial mode = Alphabetical → no mode-specific container visible
    # (isHidden() reflects explicit setVisible state independent of parent).
    assert dialog._chosen_container.isHidden() is True
    assert dialog._property_container.isHidden() is True
    assert dialog._userlist_container.isHidden() is True
    assert dialog._frequency_container.isHidden() is True

    # Switch to Chosen Words.
    dialog._mode_combo.setCurrentIndex(1)
    assert dialog._chosen_container.isHidden() is False
    assert dialog._property_container.isHidden() is True

    # Switch to By Property.
    dialog._mode_combo.setCurrentIndex(3)
    assert dialog._property_container.isHidden() is False
    assert dialog._chosen_container.isHidden() is True

    # Switch to User-Curated.
    dialog._mode_combo.setCurrentIndex(4)
    assert dialog._userlist_container.isHidden() is False


def test_text_state_preserved_across_mode_switch(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)

    dialog._mode_combo.setCurrentIndex(1)  # chosen
    dialog._chosen_words_input.setPlainText("apple")
    dialog._mode_combo.setCurrentIndex(4)  # user-curated
    dialog._userlist_input.setPlainText("the the apple")

    # Switch back; both fields retain their values.
    dialog._mode_combo.setCurrentIndex(1)
    assert "apple" in dialog._chosen_words_input.toPlainText()
    dialog._mode_combo.setCurrentIndex(4)
    assert "the the apple" in dialog._userlist_input.toPlainText()


def test_unsupported_language_source_disabled(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog
    from PySide6.QtCore import Qt

    aligned_clip, aligned_source = _make_aligned_clip()
    bad_clip, bad_source = _make_unsupported_lang_clip()
    dialog = WordSequencerDialog(
        clips=[(aligned_clip, aligned_source), (bad_clip, bad_source)],
        project=None,
    )

    assert dialog._source_list.count() == 2

    bad_row = None
    aligned_row = None
    for i in range(dialog._source_list.count()):
        item = dialog._source_list.item(i)
        if "unsupported" in item.text():
            bad_row = item
        elif "aligned" in item.text():
            aligned_row = item
    assert bad_row is not None
    assert aligned_row is not None
    assert not (bad_row.flags() & Qt.ItemIsEnabled)
    assert bad_row.checkState() == Qt.Unchecked
    assert aligned_row.checkState() == Qt.Checked
    # Accept stays enabled because aligned row is still valid.
    assert dialog._accept_btn.isEnabled() is True


def test_chosen_words_validation(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)
    dialog._mode_combo.setCurrentIndex(1)  # chosen

    # Empty input → disabled.
    dialog._chosen_words_input.setPlainText("")
    assert dialog._accept_btn.isEnabled() is False

    # All entries miss the corpus → disabled, error label populated.
    dialog._chosen_words_input.setPlainText("zzz, qqq")
    assert dialog._accept_btn.isEnabled() is False
    assert "0 of 2" in dialog._error_label.text()

    # Mixed: one hit → enabled.
    dialog._chosen_words_input.setPlainText("apple, qqq")
    assert dialog._accept_btn.isEnabled() is True


def test_unchecking_all_disables_accept(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog
    from PySide6.QtCore import Qt

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)
    item = dialog._source_list.item(0)
    item.setCheckState(Qt.Unchecked)
    # Force re-validate (itemChanged should fire but be explicit).
    dialog._refresh_validation()
    assert dialog._accept_btn.isEnabled() is False


def test_accept_emits_sequence_for_alphabetical(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)

    captured = []
    dialog.sequence_ready.connect(captured.append)

    # Force-accept programmatically.
    dialog._mode_combo.setCurrentIndex(0)  # alphabetical
    dialog._on_accept()

    assert len(captured) == 1
    seq_clips = captured[0]
    # Three words → three SequenceClip entries, alphabetical: apple, falls, the.
    assert len(seq_clips) == 3
    # source_clip_id always set to our single clip.
    for sc in seq_clips:
        assert sc.source_clip_id == clip.id


def test_accept_with_unaligned_triggers_alignment_worker(qapp, monkeypatch):
    """Accept with missing word data must start ForcedAlignmentWorker, not
    call the sequencer directly."""
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    aligned_clip, aligned_source = _make_aligned_clip()
    bad_clip, bad_source = _make_unaligned_clip()
    dialog = WordSequencerDialog(
        clips=[(aligned_clip, aligned_source), (bad_clip, bad_source)],
        project=None,
    )

    starts = []

    class FakeAlignmentWorker:
        def __init__(self, *, clips, sources_by_id, skip_existing, parent=None):
            self._clips = clips
            from PySide6.QtCore import Signal

            # Use real Signal-shaped attribute holders so .connect works.
            class _Sig:
                def connect(self, *args, **kwargs):
                    return None

            self.progress = _Sig()
            self.clip_aligned = _Sig()
            self.alignment_completed = _Sig()
            self.error = _Sig()

        def isRunning(self):
            return True

        def start(self):
            starts.append(self._clips)

        def wait(self, _ms):
            return True

        def cancel(self):
            pass

    monkeypatch.setattr(
        "ui.workers.forced_alignment_worker.ForcedAlignmentWorker",
        FakeAlignmentWorker,
    )

    dialog._on_accept()
    # Exactly one alignment run started, only the unaligned clip queued.
    assert len(starts) == 1
    assert [c.id for c in starts[0]] == [bad_clip.id]


def test_collect_mode_params_for_by_property(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)
    dialog._mode_combo.setCurrentIndex(3)  # by_property
    dialog._property_combo.setCurrentIndex(2)  # log_frequency → defaults descending

    params = dialog._collect_mode_params()
    assert params["key"] == "log_frequency"
    assert params["order"] == "descending"


def test_user_curated_list_match_preview(qapp):
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_aligned_clip()
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)
    dialog._mode_combo.setCurrentIndex(4)  # from_word_list
    dialog._userlist_input.setPlainText("the the apple zzz")
    text = dialog._userlist_match_label.text()
    assert "4 slots" in text
    assert "1 unrecognized" in text
