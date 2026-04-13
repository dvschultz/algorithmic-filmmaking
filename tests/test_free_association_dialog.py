"""Tests for FreeAssociationDialog.

Per the plan, tests cover state transitions and data flow with mocked
workers rather than full UI rendering. Workers are never started — we
drive state transitions by calling handler methods directly.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.clip import Clip, Source


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_source(source_id: str = "s1") -> Source:
    return Source(
        id=source_id,
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )


def _make_clip(clip_id: str, source_id: str = "s1", **extra) -> Clip:
    kwargs = dict(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=30,
        description=f"Description for {clip_id}",
        shot_type="close-up",
    )
    kwargs.update(extra)
    return Clip(**kwargs)


@pytest.fixture(autouse=True)
def _no_real_workers():
    """Replace _spawn_worker on the class for the test duration so that
    _request_next_proposal never actually starts a QThread."""
    from ui.dialogs.free_association_dialog import FreeAssociationDialog

    original = FreeAssociationDialog._spawn_worker

    def noop_spawn(self, *args, **kwargs):
        from ui.dialogs.free_association_dialog import PAGE_LOADING

        # Still switch to LOADING so any state-transition test sees the
        # stack update that would happen in production.
        self.stack.setCurrentIndex(PAGE_LOADING)
        self._proposal_handled = False

    FreeAssociationDialog._spawn_worker = noop_spawn
    yield
    FreeAssociationDialog._spawn_worker = original


def _build_dialog(clips, source=None):
    """Construct a dialog with the given clips. Workers never actually spawn
    (see the _no_real_workers fixture)."""
    from ui.dialogs.free_association_dialog import FreeAssociationDialog

    src = source or _make_source()
    sources_by_id = {src.id: src}
    project = MagicMock()
    return FreeAssociationDialog(clips, sources_by_id, project)


class TestInitialization:
    def test_opens_on_first_clip_select_page(self, qapp):
        from ui.dialogs.free_association_dialog import PAGE_FIRST_CLIP_SELECT

        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        assert dialog.stack.currentIndex() == PAGE_FIRST_CLIP_SELECT
        # Start button is enabled when clips exist
        assert dialog.start_btn.isEnabled() is True
        # List has the clips
        assert dialog.first_clip_list.count() == 2

    def test_empty_pool_disables_start_button(self, qapp):
        """With zero clips, the dialog opens on the empty-state."""
        dialog = _build_dialog([])
        assert dialog.start_btn.isEnabled() is False
        # The list shows a placeholder item (not enabled)
        assert dialog.first_clip_list.count() == 1
        item = dialog.first_clip_list.item(0)
        # Placeholder item has no enabled flag
        from PySide6.QtCore import Qt

        assert not (item.flags() & Qt.ItemIsEnabled)


class TestFirstClipSelection:
    def test_selecting_first_clip_adds_it_to_sequence(self, qapp):
        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)

        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()

        assert len(dialog.sequence_built) == 1
        assert dialog.sequence_built[0][0].id == "c1"
        # First clip has no rationale
        assert dialog.rationales == [None]
        # Pool was reduced
        assert len(dialog.available_pool) == 1
        # Log entry was added
        assert dialog.log_list.count() == 1

    def test_start_without_selection_shows_warning(self, qapp):
        clips = [_make_clip("c1")]
        dialog = _build_dialog(clips)
        # No selection made
        with patch(
            "ui.dialogs.free_association_dialog.QMessageBox.information"
        ) as mock_info:
            dialog._on_start_clicked()
            mock_info.assert_called_once()
        assert len(dialog.sequence_built) == 0


class TestProposalFlow:
    def test_proposal_ready_transitions_to_proposal_page(self, qapp):
        from ui.dialogs.free_association_dialog import (
            PAGE_FIRST_CLIP_SELECT,
            PAGE_PROPOSAL,
        )

        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        # Now simulate proposal_ready from a worker
        # Manually set up what _request_next_proposal would have prepared
        dialog._current_candidates = [
            (c, dialog.available_pool[i][1])
            for i, c in enumerate([p[0] for p in dialog.available_pool])
        ]
        dialog._current_short_to_full = {"c1": "c2", "c2": "c3"}
        dialog._proposal_handled = False

        dialog._on_proposal_ready("c1", "Matched on close-up framing")

        assert dialog.stack.currentIndex() == PAGE_PROPOSAL
        assert dialog._proposed_clip is not None
        assert dialog._proposed_clip[0].id == "c2"
        assert dialog._proposed_rationale == "Matched on close-up framing"
        assert dialog._proposal_handled is True

    def test_duplicate_proposal_ready_is_suppressed(self, qapp):
        """Guard flag prevents Qt's double-signal-fire from being processed twice."""
        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        dialog._current_candidates = [dialog.available_pool[0]]
        dialog._current_short_to_full = {"c1": "c2"}
        dialog._proposal_handled = False

        dialog._on_proposal_ready("c1", "First rationale")
        first_proposed = dialog._proposed_clip
        # Simulate duplicate signal fire
        dialog._on_proposal_ready("c1", "Second rationale")
        # Second call should not overwrite state
        assert dialog._proposed_clip is first_proposed

    def test_hallucinated_clip_id_routes_to_error(self, qapp):
        from ui.dialogs.free_association_dialog import PAGE_ERROR

        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        dialog._current_candidates = [dialog.available_pool[0]]
        dialog._current_short_to_full = {"c1": "c2"}
        dialog._proposal_handled = False

        # LLM returns a short ID not in the mapping
        dialog._on_proposal_ready("c99", "Won't work")

        assert dialog.stack.currentIndex() == PAGE_ERROR


class TestAcceptFlow:
    def test_accept_adds_clip_and_rationale(self, qapp):
        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()

        # Set up a pending proposal
        proposed_clip = dialog.available_pool[0]
        dialog._proposed_clip = proposed_clip
        dialog._proposed_rationale = "Nice warm transition"
        dialog.rejected_for_position = {"c99"}  # should be cleared

        dialog._on_accept_clicked()

        assert len(dialog.sequence_built) == 2
        assert dialog.sequence_built[1] == proposed_clip
        assert dialog.rationales == [None, "Nice warm transition"]
        # Rejection set cleared on accept
        assert dialog.rejected_for_position == set()
        # Clip removed from pool
        assert proposed_clip not in dialog.available_pool
        # Log was appended
        assert dialog.log_list.count() == 2


class TestRejectFlow:
    def test_reject_adds_to_rejection_set_without_logging(self, qapp):
        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        log_count_before = dialog.log_list.count()

        proposed_clip = dialog.available_pool[0]
        dialog._proposed_clip = proposed_clip
        dialog._proposed_rationale = "Reject me"

        dialog._on_reject_clicked()

        assert proposed_clip[0].id in dialog.rejected_for_position
        # Clip stays in pool (not removed)
        assert proposed_clip in dialog.available_pool
        # Log did NOT grow
        assert dialog.log_list.count() == log_count_before
        # Last rejected is stashed for Reconsider
        assert dialog._last_rejected_proposal == proposed_clip


class TestStopFlow:
    def test_stop_under_threshold_does_not_confirm(self, qapp):
        from ui.dialogs.free_association_dialog import PAGE_COMPLETE

        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        # Only 1 accepted — below threshold of 3

        with patch(
            "ui.dialogs.free_association_dialog.QMessageBox.question"
        ) as mock_confirm:
            dialog._on_stop_clicked()
            mock_confirm.assert_not_called()

        assert dialog.stack.currentIndex() == PAGE_COMPLETE

    def test_stop_at_or_above_threshold_asks_confirmation(self, qapp):
        from PySide6.QtWidgets import QMessageBox
        from ui.dialogs.free_association_dialog import PAGE_COMPLETE, PAGE_PROPOSAL

        clips = [_make_clip(f"c{i}") for i in range(5)]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        # Manually inflate sequence_built to reach threshold
        dialog.sequence_built.extend(
            [dialog.available_pool[i] for i in range(2)]
        )
        dialog.rationales.extend(["r1", "r2"])
        dialog.stack.setCurrentIndex(PAGE_PROPOSAL)

        # Case: user cancels the confirmation
        with patch(
            "ui.dialogs.free_association_dialog.QMessageBox.question",
            return_value=QMessageBox.No,
        ):
            dialog._on_stop_clicked()
        assert dialog.stack.currentIndex() == PAGE_PROPOSAL  # did not move

        # Case: user confirms
        with patch(
            "ui.dialogs.free_association_dialog.QMessageBox.question",
            return_value=QMessageBox.Yes,
        ):
            dialog._on_stop_clicked()
        assert dialog.stack.currentIndex() == PAGE_COMPLETE


class TestErrorRecovery:
    def test_error_routes_to_error_page_preserving_sequence(self, qapp):
        from ui.dialogs.free_association_dialog import PAGE_ERROR

        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        assert len(dialog.sequence_built) == 1  # first clip accepted

        dialog._proposal_handled = False
        dialog._on_proposal_error("Network timed out")

        assert dialog.stack.currentIndex() == PAGE_ERROR
        # Sequence is preserved
        assert len(dialog.sequence_built) == 1
        assert "preserved" in dialog.error_reassurance.text().lower()

    def test_cancel_from_error_routes_to_complete(self, qapp):
        from ui.dialogs.free_association_dialog import PAGE_COMPLETE, PAGE_ERROR

        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        dialog._proposal_handled = False
        dialog._on_proposal_error("An error")
        assert dialog.stack.currentIndex() == PAGE_ERROR

        dialog._on_cancel_from_error()
        assert dialog.stack.currentIndex() == PAGE_COMPLETE


class TestApply:
    def test_apply_emits_sequence_ready_with_three_tuples(self, qapp):
        clips = [_make_clip("c1"), _make_clip("c2"), _make_clip("c3")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        # Simulate accepting another clip
        dialog._proposed_clip = dialog.available_pool[0]
        dialog._proposed_rationale = "Second clip rationale"
        dialog._on_accept_clicked()

        emitted = []
        dialog.sequence_ready.connect(lambda payload: emitted.append(payload))
        dialog._on_apply_clicked()

        assert len(emitted) == 1
        payload = emitted[0]
        assert len(payload) == 2
        # First entry: clip, source, None
        assert payload[0][0].id == "c1"
        assert payload[0][2] is None
        # Second entry: has rationale
        assert payload[1][2] == "Second clip rationale"

    def test_close_without_applying_under_threshold(self, qapp):
        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        dialog._show_complete_page()

        emitted = []
        dialog.sequence_ready.connect(lambda p: emitted.append(p))

        with patch(
            "ui.dialogs.free_association_dialog.QMessageBox.question"
        ) as mock_confirm:
            dialog._on_close_from_complete()
            mock_confirm.assert_not_called()  # under threshold, no confirm

        # sequence_ready was NOT emitted
        assert emitted == []


class TestRationaleLog:
    def test_log_accumulates_only_accepted_transitions(self, qapp):
        clips = [_make_clip(f"c{i}") for i in range(4)]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        # First clip added to log
        assert dialog.log_list.count() == 1

        # Reject a proposal — should NOT add to log
        dialog._proposed_clip = dialog.available_pool[0]
        dialog._proposed_rationale = "rejected rationale"
        dialog._on_reject_clicked()
        assert dialog.log_list.count() == 1  # unchanged

        # Accept next — should add
        dialog._proposed_clip = dialog.available_pool[-1]
        dialog._proposed_rationale = "accepted rationale"
        dialog._on_accept_clicked()
        assert dialog.log_list.count() == 2

    def test_first_log_entry_marks_opening_clip(self, qapp):
        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()
        first_text = dialog.log_list.item(0).text()
        assert "opening clip" in first_text.lower()


class TestPoolExhausted:
    def test_shows_pool_exhausted_page_when_all_candidates_rejected(self, qapp):
        from ui.dialogs.free_association_dialog import PAGE_POOL_EXHAUSTED

        # Two clips total — after selecting one, only 1 candidate remains
        clips = [_make_clip("c1"), _make_clip("c2")]
        dialog = _build_dialog(clips)
        dialog.first_clip_list.setCurrentRow(0)
        dialog._on_start_clicked()

        # Reject the only remaining candidate
        dialog._proposed_clip = dialog.available_pool[0]
        dialog._proposed_rationale = "no good"
        dialog._on_reject_clicked()
        # _request_next_proposal should have transitioned to POOL_EXHAUSTED

        assert dialog.stack.currentIndex() == PAGE_POOL_EXHAUSTED
        # Reconsider button should be enabled (we have a last-rejected stash)
        assert dialog.reconsider_btn.isEnabled() is True
