"""Multi-step dialog for the Cassette Tape sequencer.

Guides the user through:
1. Phrase entry (text + per-phrase count slider 1-5)
2. Background matching across transcribed clips
3. Reviewing matches with confidence scores and a per-match toggle
4. Generating the final sub-clip sequence

Emits ``sequence_ready`` with a list of ``(Clip, Source, in_frame, out_frame)``
tuples; ``in_frame`` / ``out_frame`` are relative to ``clip.start_frame``,
matching the convention used by Signature Style.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.remix.cassette_tape import (
    MatchResult,
    build_sequence_data,
    flatten_matches_in_phrase_order,
    match_phrases,
)
from models.clip import Clip
from ui.theme import UISizes, theme
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


# Confidence color thresholds
CONFIDENCE_HIGH = 80
CONFIDENCE_MID = 50

# Slider range
SLIDER_MIN = 1
SLIDER_MAX = 5
SLIDER_DEFAULT = 3
DEFAULT_PHRASE_ROWS = 3


class CassetteTapeWorker(CancellableWorker):
    """Background worker that runs phrase matching off the UI thread.

    Inherits ``cancel()`` / ``is_cancelled()`` / ``error`` from
    ``CancellableWorker``; the cancel flag is plumbed into ``match_phrases``
    so a long-running scoring pass can actually abort instead of just
    suppressing the post-completion signal.
    """

    matches_ready = Signal(dict)  # phrase -> [MatchResult]

    def __init__(
        self,
        phrases_with_counts: list[tuple[str, int]],
        clips: list[Clip],
        parent=None,
    ):
        super().__init__(parent)
        self.phrases_with_counts = phrases_with_counts
        self.clips = clips

    def run(self):
        self._log_start()
        try:
            results = match_phrases(
                self.phrases_with_counts,
                self.clips,
                is_cancelled=self.is_cancelled,
            )
            if self.is_cancelled():
                self._log_cancelled()
                return
            self.matches_ready.emit(results)
            self._log_complete()
        except Exception as exc:
            if self.is_cancelled():
                self._log_cancelled()
                return
            logger.exception("Cassette Tape matching failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")


class _PhraseRow(QWidget):
    """One phrase entry row: text input + count slider + remove button."""

    text_changed = Signal()
    remove_requested = Signal(QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.line_edit = QLineEdit()
        self.line_edit.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        self.line_edit.setPlaceholderText("Enter a phrase to find…")
        self.line_edit.textChanged.connect(lambda _: self.text_changed.emit())
        layout.addWidget(self.line_edit, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.slider.setValue(SLIDER_DEFAULT)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setFixedWidth(140)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.count_label = QLabel(str(SLIDER_DEFAULT))
        self.count_label.setFixedWidth(20)
        self.count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_label)

        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(28, 28)
        self.remove_btn.setToolTip("Remove this phrase")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(self.remove_btn)

    def _on_slider_changed(self, value: int):
        self.count_label.setText(str(value))

    def phrase(self) -> str:
        return self.line_edit.text().strip()

    def count(self) -> int:
        return self.slider.value()


class _MatchRow(QWidget):
    """One match in the review screen: checkbox + snippet + score."""

    def __init__(self, match: MatchResult, clip_name: str, parent=None):
        super().__init__(parent)
        self.match = match

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.setToolTip("Include this match in the final sequence")
        layout.addWidget(self.checkbox)

        # Text body: clip name + segment snippet (matched substring highlighted)
        body = QVBoxLayout()
        body.setSpacing(2)

        name_label = QLabel(clip_name)
        name_font = QFont()
        name_font.setBold(True)
        name_label.setFont(name_font)
        body.addWidget(name_label)

        snippet_label = QLabel(self._build_snippet_html(match))
        snippet_label.setTextFormat(Qt.RichText)
        snippet_label.setWordWrap(True)
        body.addWidget(snippet_label)

        layout.addLayout(body, 1)

        # Confidence score badge + bar
        score_block = QVBoxLayout()
        score_block.setSpacing(2)
        score_label = QLabel(f"{match.score}")
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setStyleSheet(
            f"color: {self._score_color(match.score)}; font-weight: bold;"
        )
        score_label.setFixedWidth(40)
        score_block.addWidget(score_label)

        score_bar = QFrame()
        score_bar.setFixedSize(40, 4)
        score_bar.setStyleSheet(
            f"background-color: {self._score_color(match.score)}; border-radius: 2px;"
        )
        score_block.addWidget(score_bar)
        layout.addLayout(score_block)

    @staticmethod
    def _score_color(score: int) -> str:
        t = theme()
        if score >= CONFIDENCE_HIGH:
            return getattr(t, "accent_green", "#5cb85c")
        if score >= CONFIDENCE_MID:
            return getattr(t, "accent_yellow", "#f0ad4e")
        return getattr(t, "accent_red", "#d9534f")

    @staticmethod
    def _build_snippet_html(match: MatchResult) -> str:
        """Render the segment text with the matched substring highlighted."""
        text = match.segment.text or ""
        start = match.match_start
        end = match.match_end
        if 0 <= start < end <= len(text):
            before = _escape_html(text[:start])
            hit = _escape_html(text[start:end])
            after = _escape_html(text[end:])
            return f'{before}<b style="color: {theme().accent_blue};">{hit}</b>{after}'
        return _escape_html(text)

    def is_enabled(self) -> bool:
        return self.checkbox.isChecked()


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


class CassetteTapeDialog(QDialog):
    """Three-page modal dialog for the Cassette Tape sequencer.

    Pages:
        PAGE_SETUP    — phrase entry + count sliders
        PAGE_PROGRESS — matching in progress
        PAGE_REVIEW   — per-match toggle + confidence

    Signals:
        sequence_ready: emitted on Generate with
            ``list[tuple[Clip, Source, in_frame, out_frame]]``.
    """

    sequence_ready = Signal(list)

    PAGE_SETUP = 0
    PAGE_PROGRESS = 1
    PAGE_REVIEW = 2

    def __init__(self, clips, sources_by_id, project, parent=None):
        super().__init__(parent)
        self.all_clips = clips
        self.sources_by_id = sources_by_id
        self.project = project
        self.worker: Optional[CassetteTapeWorker] = None
        self.matches_by_phrase: dict[str, list[MatchResult]] = {}
        self._match_rows: list[_MatchRow] = []

        # Filter to clips that have a transcript and aren't disabled.
        # The matching logic also filters, but we need this for the setup-page banner.
        self.transcribed_clips = [
            c for c in clips
            if c.transcript and not c.disabled
        ]

        self.setWindowTitle("Cassette Tape")
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self._setup_ui()

    # ---------- UI construction ----------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.stack.addWidget(self._create_setup_page())
        self.stack.addWidget(self._create_progress_page())
        self.stack.addWidget(self._create_review_page())

        # Navigation
        nav = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._on_back)
        self.back_btn.setVisible(False)
        nav.addWidget(self.back_btn)

        nav.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        nav.addWidget(self.cancel_btn)

        self.next_btn = QPushButton("Find Matches")
        self.next_btn.clicked.connect(self._on_next)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

        self._update_nav_buttons()
        self._update_next_btn_enabled()

    def _create_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Cassette Tape")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        if not self.transcribed_clips:
            warn = QLabel(
                "This project has no transcribed clips. Run Analyze → Transcribe first."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {theme().accent_red};")
            layout.addWidget(warn)
            self._setup_no_transcripts = True
        else:
            info = QLabel(
                f"Find clips that say specific phrases. "
                f"{len(self.transcribed_clips)} transcribed clip(s) available. "
                f"Each phrase contributes 1–5 matches by similarity."
            )
            info.setWordWrap(True)
            layout.addWidget(info)
            self._setup_no_transcripts = False

        layout.addSpacing(12)

        # Header row above phrase list
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.addWidget(QLabel("Phrase"), 1)
        ct_label = QLabel("Matches per phrase")
        ct_label.setFixedWidth(160)
        head_row.addWidget(ct_label)
        head_row.addSpacing(34)  # Align with × buttons below
        layout.addLayout(head_row)

        # Scrollable phrase list
        self.phrase_container = QWidget()
        self.phrase_layout = QVBoxLayout(self.phrase_container)
        self.phrase_layout.setContentsMargins(0, 0, 0, 0)
        self.phrase_layout.setSpacing(6)
        self.phrase_layout.addStretch()  # Trailing stretch to keep rows top-aligned

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.phrase_container)
        layout.addWidget(scroll, 1)

        self.phrase_rows: list[_PhraseRow] = []
        for _ in range(DEFAULT_PHRASE_ROWS):
            self._add_phrase_row()

        # Add row button
        add_row = QHBoxLayout()
        add_btn = QPushButton("+ Add phrase")
        add_btn.setMinimumHeight(UISizes.BUTTON_MIN_HEIGHT)
        add_btn.clicked.connect(self._add_phrase_row)
        add_row.addWidget(add_btn)
        add_row.addStretch()
        layout.addLayout(add_row)

        return page

    def _create_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addStretch()

        header = QLabel("Finding matches…")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        self.progress_status = QLabel("")
        self.progress_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # indeterminate
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return page

    def _create_review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Review matches")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        self.review_summary = QLabel("")
        layout.addWidget(self.review_summary)

        layout.addSpacing(8)

        # Scrollable list of phrase groups
        self.review_container = QWidget()
        self.review_layout = QVBoxLayout(self.review_container)
        self.review_layout.setContentsMargins(0, 0, 0, 0)
        self.review_layout.setSpacing(10)
        self.review_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.review_container)
        layout.addWidget(scroll, 1)

        return page

    # ---------- Phrase row management ----------

    def _add_phrase_row(self):
        row = _PhraseRow()
        row.text_changed.connect(self._update_next_btn_enabled)
        row.remove_requested.connect(self._remove_phrase_row)
        # Insert before the trailing stretch
        self.phrase_layout.insertWidget(self.phrase_layout.count() - 1, row)
        self.phrase_rows.append(row)
        self._update_next_btn_enabled()

    def _remove_phrase_row(self, row: QWidget):
        if row in self.phrase_rows:
            self.phrase_rows.remove(row)
        self.phrase_layout.removeWidget(row)
        row.deleteLater()
        self._update_next_btn_enabled()

    def _phrases_with_counts(self) -> list[tuple[str, int]]:
        return [(r.phrase(), r.count()) for r in self.phrase_rows if r.phrase()]

    # ---------- Navigation / state ----------

    def _update_next_btn_enabled(self):
        page = self.stack.currentIndex() if hasattr(self, "stack") else self.PAGE_SETUP
        if page == self.PAGE_SETUP:
            if self._setup_no_transcripts:
                self.next_btn.setEnabled(False)
                return
            has_phrase = any(r.phrase() for r in getattr(self, "phrase_rows", []))
            self.next_btn.setEnabled(has_phrase)
        elif page == self.PAGE_REVIEW:
            self.next_btn.setEnabled(self._any_match_enabled())
        # Progress page button enable handled in _update_nav_buttons.

    def _update_nav_buttons(self):
        page = self.stack.currentIndex()
        if page == self.PAGE_SETUP:
            self.back_btn.setVisible(False)
            self.next_btn.setText("Find Matches")
            self._update_next_btn_enabled()
        elif page == self.PAGE_PROGRESS:
            self.back_btn.setVisible(False)
            self.next_btn.setText("Please wait…")
            self.next_btn.setEnabled(False)
        elif page == self.PAGE_REVIEW:
            self.back_btn.setVisible(True)
            self.next_btn.setText("Generate Sequence")
            self._update_next_btn_enabled()

    def _on_next(self):
        page = self.stack.currentIndex()
        if page == self.PAGE_SETUP:
            self._start_matching()
        elif page == self.PAGE_REVIEW:
            self._finish_with_sequence()

    def _on_back(self):
        if self.stack.currentIndex() == self.PAGE_REVIEW:
            self.stack.setCurrentIndex(self.PAGE_SETUP)
            self._update_nav_buttons()

    def _on_cancel(self):
        self._stop_worker_if_running()
        self.reject()

    def closeEvent(self, event):
        """Window-X / ESC must also stop the worker — never tear down a
        running QThread parented to this dialog."""
        self._stop_worker_if_running()
        super().closeEvent(event)

    def _stop_worker_if_running(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            # Disconnect signals so a late delivery doesn't try to populate
            # an already-closing dialog.
            try:
                self.worker.matches_ready.disconnect(self._on_matches_ready)
            except (RuntimeError, TypeError):
                pass
            try:
                self.worker.error.disconnect(self._on_match_error)
            except (RuntimeError, TypeError):
                pass
            self.worker.quit()
            # Unbounded wait — the cooperative cancel returns within one
            # phrase iteration, which is sub-second on typical projects.
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None

    # ---------- Matching ----------

    def _start_matching(self):
        phrases = self._phrases_with_counts()
        if not phrases:
            return

        self.stack.setCurrentIndex(self.PAGE_PROGRESS)
        self._update_nav_buttons()
        self.progress_status.setText(f"Matching {len(phrases)} phrase(s)…")

        # Always create a fresh worker; never reuse across runs.
        self.worker = CassetteTapeWorker(
            phrases_with_counts=phrases,
            clips=self.transcribed_clips,
            parent=self,
        )
        self.worker.matches_ready.connect(self._on_matches_ready)
        self.worker.error.connect(self._on_match_error)
        self.worker.start()

    def _on_matches_ready(self, results: dict):
        self.matches_by_phrase = results
        self._populate_review_page()
        self.stack.setCurrentIndex(self.PAGE_REVIEW)
        self._update_nav_buttons()

    def _on_match_error(self, message: str):
        logger.error("Cassette Tape error: %s", message)
        self.stack.setCurrentIndex(self.PAGE_SETUP)
        self._update_nav_buttons()
        QMessageBox.warning(
            self,
            "Cassette Tape — matching failed",
            f"Could not finish matching:\n\n{message}",
        )

    # ---------- Review page ----------

    def _populate_review_page(self):
        # Clear existing rows
        while self.review_layout.count() > 1:  # keep trailing stretch
            item = self.review_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._match_rows = []

        clips_by_id = {c.id: c for c in self.all_clips}
        total_matches = 0

        for phrase, matches in self.matches_by_phrase.items():
            group_label = QLabel(f"“{phrase}”  ({len(matches)} match{'es' if len(matches) != 1 else ''})")
            group_font = QFont()
            group_font.setBold(True)
            group_font.setPointSize(13)
            group_label.setFont(group_font)
            self.review_layout.insertWidget(self.review_layout.count() - 1, group_label)

            if not matches:
                placeholder = QLabel("  no matches")
                placeholder.setStyleSheet(f"color: {theme().text_muted};")
                self.review_layout.insertWidget(self.review_layout.count() - 1, placeholder)
                continue

            for match in matches:
                clip = clips_by_id.get(match.clip_id)
                source = self.sources_by_id.get(clip.source_id) if clip else None
                clip_name = clip.display_name(
                    source.file_path.name if source and source.file_path else "",
                    source.fps if source else 30.0,
                ) if clip else match.clip_id
                row = _MatchRow(match, clip_name)
                row.checkbox.stateChanged.connect(lambda _: self._update_next_btn_enabled())
                self._match_rows.append(row)
                self.review_layout.insertWidget(self.review_layout.count() - 1, row)
                total_matches += 1

        self.review_summary.setText(
            f"{total_matches} match{'es' if total_matches != 1 else ''} across "
            f"{sum(1 for v in self.matches_by_phrase.values() if v)} phrase(s). "
            f"Toggle off any you don't want before generating."
        )

    def _any_match_enabled(self) -> bool:
        return any(r.is_enabled() for r in self._match_rows)

    # ---------- Final emit ----------

    def _finish_with_sequence(self):
        enabled_keys = {
            (r.match.phrase, r.match.clip_id, r.match.segment_index)
            for r in self._match_rows
            if r.is_enabled()
        }
        flat = flatten_matches_in_phrase_order(self.matches_by_phrase, enabled_keys)
        clips_by_id = {c.id: c for c in self.all_clips}
        sequence_data = build_sequence_data(flat, clips_by_id, self.sources_by_id)

        if not sequence_data:
            logger.warning("Cassette Tape: no enabled matches to emit")
            self.reject()
            return

        logger.info("Cassette Tape: emitting %d sub-clips", len(sequence_data))
        self.sequence_ready.emit(sequence_data)
        self.accept()
