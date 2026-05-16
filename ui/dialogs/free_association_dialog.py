"""Interactive dialog for the Free Association sequencer.

The user selects a first clip, then the LLM proposes each next clip based
on clip metadata. Accept adds the clip and moves on; Reject asks for a
different proposal. A rationale log on the right side accumulates the
editorial record of the final sequence.

Design notes (see docs/plans/2026-04-12-001-feat-free-association-sequencer-plan.md):

- Single-step worker per proposal; dialog owns all state.
- Guard flag + Qt.UniqueConnection for worker signal safety.
- No worker.wait() on cancel — in-flight HTTP calls could block the UI
  thread up to 120s. Instead call worker.cancel() and worker.deleteLater().
- The emitted signal payload is list[tuple[Clip, Source, Optional[str]]] —
  the third element is the rationale for each transition (None for the
  user-selected first clip).
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.remix.free_association import (
    DEFAULT_RECENT_RATIONALES,
    DEFAULT_SHORTLIST_SIZE,
    build_id_mapping,
    format_clip_digest,
    format_clip_full_metadata,
    shortlist_candidates,
)
from ui.theme import theme
from ui.workers.free_association_worker import FreeAssociationWorker

logger = logging.getLogger(__name__)

# Page indices
PAGE_FIRST_CLIP_SELECT = 0
PAGE_LOADING = 1
PAGE_PROPOSAL = 2
PAGE_ERROR = 3
PAGE_POOL_EXHAUSTED = 4
PAGE_COMPLETE = 5

# Threshold for requiring confirmation on stop/discard
CONFIRMATION_THRESHOLD = 3


class FreeAssociationDialog(QDialog):
    """Interactive step-by-step sequencer dialog.

    Signals:
        sequence_ready(list): emitted with the built sequence as a list of
            (Clip, Source, Optional[str]) tuples, where the third element
            is the rationale for that transition (None for the first clip).
    """

    sequence_ready = Signal(list)

    def __init__(self, clips, sources_by_id, project, parent=None):
        super().__init__(parent)
        self.clips = list(clips)
        self.sources_by_id = sources_by_id
        self._project = project

        # Dialog state (source of truth — not kept on the worker)
        # sequence_built parallels rationales; rationales[0] is always None
        self.sequence_built: list[tuple] = []  # [(Clip, Source), ...]
        self.rationales: list[Optional[str]] = []
        self.available_pool: list[tuple] = self._initial_pool()
        # IDs rejected for the current position, cleared on accept
        self.rejected_for_position: set[str] = set()

        # Current proposal state (valid on PAGE_PROPOSAL)
        self._current_candidates: list[tuple] = []
        self._current_short_to_full: dict[str, str] = {}
        self._proposed_clip: Optional[tuple] = None  # (Clip, Source)
        self._proposed_rationale: str = ""
        self._last_rejected_proposal: Optional[tuple] = None  # for "Reconsider"

        # Worker lifecycle
        self._worker: Optional[FreeAssociationWorker] = None
        self._proposal_handled = False  # guard against duplicate signal delivery

        self.setWindowTitle("Free Association")
        self.setMinimumSize(900, 600)
        self.setModal(True)

        self._setup_ui()
        self._apply_theme()

        # Start on FIRST_CLIP_SELECT (or empty state)
        if not self.available_pool:
            self._show_empty_state()
        else:
            self.stack.setCurrentIndex(PAGE_FIRST_CLIP_SELECT)

    def _initial_pool(self) -> list[tuple]:
        """Build the initial pool from clips + sources."""
        pool = []
        for clip in self.clips:
            source = self.sources_by_id.get(clip.source_id)
            if source is not None:
                pool.append((clip, source))
        return pool

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: interaction stack
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_first_clip_page())
        self.stack.addWidget(self._build_loading_page())
        self.stack.addWidget(self._build_proposal_page())
        self.stack.addWidget(self._build_error_page())
        self.stack.addWidget(self._build_pool_exhausted_page())
        self.stack.addWidget(self._build_complete_page())
        splitter.addWidget(self.stack)

        # Right: rationale log panel
        splitter.addWidget(self._build_log_panel())

        # 70/30 split
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter, 1)

    def _build_first_clip_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Pick your opening clip")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        page_layout.addWidget(header)

        subhead = QLabel(
            "The Free Association sequencer will propose each subsequent clip "
            "based on the metadata of the one that came before. Select the clip "
            "that should open the sequence."
        )
        subhead.setWordWrap(True)
        page_layout.addWidget(subhead)

        # Scrollable list of available clips — simple list with thumbnails
        self.first_clip_list = QListWidget()
        self.first_clip_list.setIconSize(self._thumbnail_size())
        self.first_clip_list.itemDoubleClicked.connect(self._on_first_clip_double_clicked)
        for clip, source in self.available_pool:
            item = QListWidgetItem(self._clip_list_label(clip, source))
            item.setData(Qt.UserRole, clip.id)
            pixmap = self._thumbnail_for(clip)
            if pixmap is not None:
                item.setIcon(pixmap)
            self.first_clip_list.addItem(item)
        page_layout.addWidget(self.first_clip_list, 1)

        # Start button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel_from_first_page)
        btn_row.addWidget(cancel_btn)
        self.start_btn = QPushButton("Start")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._on_start_clicked)
        btn_row.addWidget(self.start_btn)
        page_layout.addLayout(btn_row)

        return page

    def _build_loading_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignCenter)

        header = QLabel("Thinking…")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        lay.addWidget(header)

        self.loading_status = QLabel("Finding the next clip…")
        self.loading_status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.loading_status)

        bar = QProgressBar()
        bar.setMinimum(0)
        bar.setMaximum(0)  # indeterminate
        bar.setFixedWidth(300)
        lay.addWidget(bar, alignment=Qt.AlignCenter)

        # Stop button
        stop_row = QHBoxLayout()
        stop_row.addStretch()
        loading_stop_btn = QPushButton("Stop")
        loading_stop_btn.clicked.connect(self._on_stop_clicked)
        stop_row.addWidget(loading_stop_btn)
        stop_row.addStretch()
        lay.addLayout(stop_row)

        return page

    def _build_proposal_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)

        header = QLabel("Next clip")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        lay.addWidget(header)

        # Proposed clip thumbnail + name
        self.proposal_thumbnail = QLabel()
        self.proposal_thumbnail.setAlignment(Qt.AlignCenter)
        self.proposal_thumbnail.setFixedHeight(200)
        lay.addWidget(self.proposal_thumbnail)

        self.proposal_name = QLabel("")
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        self.proposal_name.setFont(name_font)
        lay.addWidget(self.proposal_name)

        # Metadata summary
        self.proposal_metadata = QLabel("")
        self.proposal_metadata.setWordWrap(True)
        self.proposal_metadata.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(self.proposal_metadata)

        # Rationale
        rationale_heading = QLabel("Rationale")
        rationale_heading.setStyleSheet("font-weight: bold;")
        lay.addWidget(rationale_heading)

        self.proposal_rationale = QLabel("")
        self.proposal_rationale.setWordWrap(True)
        rationale_scroll = QScrollArea()
        rationale_scroll.setWidget(self.proposal_rationale)
        rationale_scroll.setWidgetResizable(True)
        rationale_scroll.setFrameShape(QScrollArea.NoFrame)
        rationale_scroll.setFixedHeight(80)
        lay.addWidget(rationale_scroll)

        lay.addStretch()

        # Action row: Accept / Reject / Stop
        action_row = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        action_row.addWidget(self.stop_btn)
        action_row.addStretch()
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self._on_reject_clicked)
        action_row.addWidget(self.reject_btn)
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setDefault(True)
        self.accept_btn.clicked.connect(self._on_accept_clicked)
        action_row.addWidget(self.accept_btn)
        lay.addLayout(action_row)

        return page

    def _build_error_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignCenter)

        header = QLabel("Something went wrong")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        lay.addWidget(header)

        self.error_message = QLabel("")
        self.error_message.setWordWrap(True)
        self.error_message.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.error_message)

        self.error_reassurance = QLabel("")
        self.error_reassurance.setAlignment(Qt.AlignCenter)
        self.error_reassurance.setStyleSheet("color: gray;")
        lay.addWidget(self.error_reassurance)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel_from_error)
        btn_row.addWidget(cancel_btn)
        retry_btn = QPushButton("Retry")
        retry_btn.setDefault(True)
        retry_btn.clicked.connect(self._on_retry_clicked)
        btn_row.addWidget(retry_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        return page

    def _build_pool_exhausted_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignCenter)

        header = QLabel("No more candidates")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        lay.addWidget(header)

        msg = QLabel(
            "Every clip that could follow here has been rejected. "
            "You can keep the sequence as it stands or reconsider the most "
            "recently rejected clip."
        )
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        lay.addWidget(msg)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        end_btn = QPushButton("End sequence")
        end_btn.clicked.connect(self._on_end_from_pool_exhausted)
        btn_row.addWidget(end_btn)
        self.reconsider_btn = QPushButton("Reconsider last rejected")
        self.reconsider_btn.setDefault(True)
        self.reconsider_btn.clicked.connect(self._on_reconsider_clicked)
        btn_row.addWidget(self.reconsider_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        return page

    def _build_complete_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)

        header = QLabel("Sequence ready")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        lay.addWidget(header)

        self.complete_summary = QLabel("")
        lay.addWidget(self.complete_summary)

        self.complete_list = QListWidget()
        lay.addWidget(self.complete_list, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close without applying")
        close_btn.clicked.connect(self._on_close_from_complete)
        btn_row.addWidget(close_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply_clicked)
        btn_row.addWidget(apply_btn)
        lay.addLayout(btn_row)

        return page

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        heading = QLabel("Rationale log")
        heading_font = QFont()
        heading_font.setPointSize(12)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        lay.addWidget(heading)

        self.log_list = QListWidget()
        self.log_list.setWordWrap(True)
        self.log_list.setAlternatingRowColors(True)
        lay.addWidget(self.log_list, 1)

        self.log_placeholder = QLabel(
            "Proposals and rationales will appear here as you build the sequence."
        )
        self.log_placeholder.setWordWrap(True)
        self.log_placeholder.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(self.log_placeholder)

        return panel

    def _apply_theme(self):
        try:
            self.setStyleSheet(
                f"""
                QDialog {{ background-color: {theme().background_primary}; }}
                QLabel {{ color: {theme().text_primary}; }}
                QListWidget {{
                    background-color: {theme().background_tertiary};
                    color: {theme().text_primary};
                    border: 1px solid {theme().border_primary};
                    border-radius: 4px;
                }}
                QListWidget::item {{ padding: 6px; }}
                QPushButton {{
                    background-color: {theme().background_tertiary};
                    color: {theme().text_primary};
                    border: 1px solid {theme().border_primary};
                    border-radius: 4px;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{ background-color: {theme().background_elevated}; }}
                QPushButton:disabled {{
                    background-color: {theme().background_secondary};
                    color: {theme().text_muted};
                }}
                QProgressBar {{
                    background-color: {theme().background_secondary};
                    border: 1px solid {theme().border_primary};
                    border-radius: 4px;
                }}
                QProgressBar::chunk {{ background-color: {theme().accent_blue}; }}
                """
            )
        except Exception:  # theme may not be available in some test contexts
            logger.debug("Skipping theme application (theme unavailable)")

    # ------------------------------------------------------------------
    # Empty state
    # ------------------------------------------------------------------

    def _show_empty_state(self):
        """Replace the first-clip page with an empty-state message."""
        # Disable the Start button and show a message inline
        self.first_clip_list.clear()
        empty_item = QListWidgetItem("No clips available — add clips in the Cut tab first")
        empty_item.setFlags(Qt.NoItemFlags)
        self.first_clip_list.addItem(empty_item)
        self.start_btn.setEnabled(False)
        self.stack.setCurrentIndex(PAGE_FIRST_CLIP_SELECT)

    # ------------------------------------------------------------------
    # First clip selection
    # ------------------------------------------------------------------

    def _on_first_clip_double_clicked(self, _item):
        # Double-click to confirm (convenience) — same path as Start button
        self._on_start_clicked()

    def _on_start_clicked(self):
        item = self.first_clip_list.currentItem()
        if item is None or not (item.flags() & Qt.ItemIsEnabled):
            QMessageBox.information(
                self, "Select a clip", "Choose a clip to open the sequence."
            )
            return

        clip_id = item.data(Qt.UserRole)
        # Find the (clip, source) in available_pool
        selected = next(
            ((c, s) for c, s in self.available_pool if c.id == clip_id), None
        )
        if selected is None:
            QMessageBox.warning(self, "Error", "Selected clip not found in pool.")
            return

        clip, source = selected
        self.sequence_built.append((clip, source))
        self.rationales.append(None)  # First clip has no rationale
        self.available_pool.remove((clip, source))
        self._log_placeholder_hide()
        self._append_log_entry(
            position=1, clip_name=self._clip_list_label(clip, source), rationale=None
        )
        self._request_next_proposal()

    # ------------------------------------------------------------------
    # Proposal request / response
    # ------------------------------------------------------------------

    def _request_next_proposal(self):
        """Shortlist candidates and spawn a worker to propose the next clip."""
        if not self.available_pool:
            # All clips placed — go directly to COMPLETE
            self._show_complete_page()
            return

        current_clip, _ = self.sequence_built[-1]
        candidates = shortlist_candidates(
            current_clip, self.available_pool, k=DEFAULT_SHORTLIST_SIZE
        )
        # Filter out candidates already rejected for this position
        unrejected = [(c, s) for c, s in candidates if c.id not in self.rejected_for_position]
        if not unrejected:
            # All candidates for this position have been rejected
            self._show_pool_exhausted()
            return

        self._current_candidates = unrejected
        short_to_full, full_to_short = build_id_mapping(unrejected)
        self._current_short_to_full = short_to_full

        candidate_digests = [
            (full_to_short[clip.id], format_clip_digest(clip)) for clip, _ in unrejected
        ]
        current_meta = format_clip_full_metadata(current_clip)
        recent_rationales = [
            r for r in self.rationales[-DEFAULT_RECENT_RATIONALES :] if r
        ]
        # Rejected IDs must be converted to the current short-ID space.
        # Since we already filtered unrejected, pass an empty rejected list
        # to propose_next_clip — all candidates in the prompt are valid.
        rejected_short_ids: list[str] = []

        self._spawn_worker(
            current_meta, candidate_digests, recent_rationales, rejected_short_ids
        )

    def _spawn_worker(
        self,
        current_meta: str,
        candidate_digests: list[tuple[str, str]],
        recent_rationales: list[str],
        rejected_short_ids: list[str],
    ):
        """Create and start a new single-step worker."""
        # Clean up any previous worker reference
        self._teardown_worker()
        self._proposal_handled = False

        self.stack.setCurrentIndex(PAGE_LOADING)
        position = len(self.sequence_built) + 1
        self.loading_status.setText(f"Finding clip for position {position}…")

        self._worker = FreeAssociationWorker(
            current_clip_metadata=current_meta,
            candidate_digests=candidate_digests,
            recent_rationales=recent_rationales,
            rejected_short_ids=rejected_short_ids,
            parent=self,
        )
        self._worker.proposal_ready.connect(
            self._on_proposal_ready, Qt.UniqueConnection
        )
        self._worker.error.connect(self._on_proposal_error, Qt.UniqueConnection)
        # Auto-cleanup when the worker finishes — avoids worker.wait() on cancel
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _teardown_worker(self):
        """Cancel and drop reference to the current worker, if any."""
        if self._worker is not None:
            try:
                self._worker.cancel()
            except Exception:
                logger.debug("Worker cancel raised; ignoring")
            self._worker = None

    def _on_proposal_ready(self, clip_short_id: str, rationale: str):
        # Guard against duplicate signal delivery (Qt finished can fire twice)
        if self._proposal_handled:
            logger.debug("Duplicate proposal_ready suppressed")
            return
        self._proposal_handled = True

        clip_id = self._current_short_to_full.get(clip_short_id)
        if clip_id is None:
            self._show_error(
                f"LLM returned an unknown clip ID ({clip_short_id}). Try again."
            )
            return

        # Find (clip, source) in current candidates
        selected = next(
            ((c, s) for c, s in self._current_candidates if c.id == clip_id), None
        )
        if selected is None:
            self._show_error(
                "The proposed clip is no longer in the candidate set. Try again."
            )
            return

        self._proposed_clip = selected
        self._proposed_rationale = rationale
        self._display_proposal(selected, rationale)
        self.stack.setCurrentIndex(PAGE_PROPOSAL)

    def _on_proposal_error(self, message: str):
        if self._proposal_handled:
            return
        self._proposal_handled = True
        self._show_error(message)

    def _show_error(self, message: str):
        self.error_message.setText(message)
        n_accepted = len(self.sequence_built)
        if n_accepted > 0:
            plural = "s" if n_accepted != 1 else ""
            self.error_reassurance.setText(
                f"Your {n_accepted} accepted clip{plural} will be preserved if you retry or cancel."
            )
        else:
            self.error_reassurance.setText("")
        self.stack.setCurrentIndex(PAGE_ERROR)

    def _display_proposal(self, proposed: tuple, rationale: str):
        clip, source = proposed
        pixmap = self._thumbnail_for(clip, scaled=True)
        if pixmap is not None:
            self.proposal_thumbnail.setPixmap(pixmap)
        else:
            self.proposal_thumbnail.setText("(no thumbnail)")
        self.proposal_name.setText(self._clip_list_label(clip, source))
        self.proposal_metadata.setText(format_clip_digest(clip))
        self.proposal_rationale.setText(rationale)

    # ------------------------------------------------------------------
    # Accept / Reject / Stop
    # ------------------------------------------------------------------

    def _on_accept_clicked(self):
        if self._proposed_clip is None:
            return
        clip, source = self._proposed_clip
        self.sequence_built.append((clip, source))
        self.rationales.append(self._proposed_rationale)
        self.available_pool.remove((clip, source))
        self.rejected_for_position.clear()
        self._last_rejected_proposal = None

        self._append_log_entry(
            position=len(self.sequence_built),
            clip_name=self._clip_list_label(clip, source),
            rationale=self._proposed_rationale,
        )

        self._proposed_clip = None
        self._proposed_rationale = ""
        self._request_next_proposal()

    def _on_reject_clicked(self):
        if self._proposed_clip is None:
            return
        clip, source = self._proposed_clip
        self.rejected_for_position.add(clip.id)
        self._last_rejected_proposal = (clip, source)
        self._proposed_clip = None
        self._proposed_rationale = ""
        self._request_next_proposal()

    def _on_reconsider_clicked(self):
        """User chose to reconsider the most recently rejected clip."""
        if self._last_rejected_proposal is None:
            # Shouldn't happen — fall through to complete
            self._show_complete_page()
            return
        clip, source = self._last_rejected_proposal
        self.rejected_for_position.discard(clip.id)
        # Synthesize a proposal from the resurrected clip — the LLM has
        # already provided a rationale for it earlier in this session.
        # We prompt the user to decide again with a fresh rationale stub;
        # this keeps the flow simple without another LLM round-trip.
        self._proposed_clip = (clip, source)
        self._proposed_rationale = (
            "(Resurrected from rejection — decide if this transition now fits.)"
        )
        self._display_proposal(self._proposed_clip, self._proposed_rationale)
        self.stack.setCurrentIndex(PAGE_PROPOSAL)

    def _on_stop_clicked(self):
        """User clicked Stop — confirm if >=3 clips accepted."""
        n = len(self.sequence_built)
        if n >= CONFIRMATION_THRESHOLD:
            reply = QMessageBox.question(
                self,
                "End sequence?",
                f"You have accepted {n} clips. End the sequence here?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._teardown_worker()
        self._show_complete_page()

    def _on_retry_clicked(self):
        """Retry the current position from ERROR state."""
        # Same position, same rejection set — just re-spawn the worker
        self._request_next_proposal()

    def _on_cancel_from_error(self):
        """Cancel from ERROR — preserve partial sequence, go to COMPLETE."""
        self._teardown_worker()
        self._show_complete_page()

    def _on_cancel_from_first_page(self):
        """Cancel before any clip is selected — close without applying."""
        self._teardown_worker()
        self.reject()

    def _on_end_from_pool_exhausted(self):
        self._teardown_worker()
        self._show_complete_page()

    # ------------------------------------------------------------------
    # Complete page
    # ------------------------------------------------------------------

    def _show_complete_page(self):
        self._populate_complete_summary()
        self.stack.setCurrentIndex(PAGE_COMPLETE)

    def _show_pool_exhausted(self):
        self.reconsider_btn.setEnabled(self._last_rejected_proposal is not None)
        self.stack.setCurrentIndex(PAGE_POOL_EXHAUSTED)

    def _populate_complete_summary(self):
        n = len(self.sequence_built)
        total_seconds = sum(
            clip.duration_seconds(source.fps) for clip, source in self.sequence_built
        )
        minutes, seconds = divmod(int(total_seconds), 60)
        self.complete_summary.setText(
            f"{n} clip{'s' if n != 1 else ''} · {minutes}:{seconds:02d} total"
        )
        self.complete_list.clear()
        for idx, (clip, source) in enumerate(self.sequence_built, start=1):
            self.complete_list.addItem(
                f"{idx}. {self._clip_list_label(clip, source)}"
            )

    def _on_apply_clicked(self):
        """Emit the final sequence as (Clip, Source, rationale) triples."""
        self._teardown_worker()
        payload = [
            (clip, source, self.rationales[idx])
            for idx, (clip, source) in enumerate(self.sequence_built)
        ]
        self.sequence_ready.emit(payload)
        self.accept()

    def _on_close_from_complete(self):
        """Close without applying — confirm if >=3 clips accepted."""
        n = len(self.sequence_built)
        if n >= CONFIRMATION_THRESHOLD:
            reply = QMessageBox.question(
                self,
                "Discard sequence?",
                f"Discard the sequence with {n} accepted clips?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._teardown_worker()
        self.reject()

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 — Qt override
        """Handle dialog close (X button) — cancel worker without blocking."""
        self._teardown_worker()
        # No worker.wait() — blocking HTTP calls could freeze up to 120s
        event.accept()

    # ------------------------------------------------------------------
    # Rationale log panel
    # ------------------------------------------------------------------

    def _append_log_entry(
        self, position: int, clip_name: str, rationale: Optional[str]
    ):
        """Append an accepted transition to the rationale log."""
        if rationale is None:
            text = f"{position}. {clip_name}\n    (opening clip — user-selected)"
        else:
            text = f"{position}. {clip_name}\n    {rationale}"
        item = QListWidgetItem(text)
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _log_placeholder_hide(self):
        self.log_placeholder.setVisible(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _thumbnail_size():
        from PySide6.QtCore import QSize

        return QSize(96, 54)

    def _thumbnail_for(self, clip, scaled: bool = False) -> Optional[QPixmap]:
        path = getattr(clip, "thumbnail_path", None)
        if not path:
            return None
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return None
        if scaled:
            return pixmap.scaledToHeight(180, Qt.SmoothTransformation)
        return pixmap

    def _clip_list_label(self, clip, source) -> str:
        source_name = getattr(source, "file_path", None)
        source_name = source_name.name if source_name is not None else ""
        return clip.display_name(source_filename=str(source_name), fps=source.fps)
