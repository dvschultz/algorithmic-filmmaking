"""Tests for Free Association integration into SequenceTab.

Covers routing, apply-time rationale preservation, and end-to-end
save/load verification of R10 (rationale persistence).
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# SequenceTab contains video widgets — use offscreen for headless runs
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_source(source_id: str = "src-1", fps: float = 30.0) -> "Source":
    from models.clip import Source

    return Source(
        id=source_id,
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=fps,
        width=1920,
        height=1080,
    )


def _make_clip(clip_id: str, source_id: str = "src-1", **extra) -> "Clip":
    from models.clip import Clip

    kwargs = dict(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=30,
        description=f"Description for {clip_id}",
    )
    kwargs.update(extra)
    return Clip(**kwargs)


class TestRouting:
    def test_card_click_routes_free_association_to_dialog(self, qapp, monkeypatch):
        from ui.tabs.sequence_tab import SequenceTab

        tab = SequenceTab()
        source = _make_source()
        clip = _make_clip("c1")
        tab.set_available_clips(
            [(clip, source)], all_clips=[clip], sources_by_id={source.id: source}
        )
        tab._gui_state = SimpleNamespace(
            analyze_selected_ids=[clip.id], cut_selected_ids=[]
        )

        dialog_calls = []
        monkeypatch.setattr(
            tab,
            "_show_free_association_dialog",
            lambda clips: dialog_calls.append(clips),
        )

        tab._on_card_clicked("free_association")

        assert len(dialog_calls) == 1
        assert len(dialog_calls[0]) == 1

    def test_show_dialog_without_descriptions_warns_and_does_not_open(self, qapp):
        from ui.tabs.sequence_tab import SequenceTab

        tab = SequenceTab()
        source = _make_source()
        clip = _make_clip("c1", description=None)  # no description

        with patch(
            "ui.tabs.sequence_tab.QMessageBox.warning"
        ) as mock_warn, patch(
            "ui.tabs.sequence_tab.FreeAssociationDialog"
        ) as mock_dialog_class:
            tab._show_free_association_dialog([(clip, source)])

        mock_warn.assert_called_once()
        mock_dialog_class.assert_not_called()


class TestApplyRationalePreservation:
    def test_apply_attaches_rationales_to_sequence_clips(self, qapp):
        from ui.tabs.sequence_tab import SequenceTab

        tab = SequenceTab()
        source = _make_source(fps=30.0)
        clip1 = _make_clip("c1")
        clip1.end_frame = 30  # 30 frames
        clip2 = _make_clip("c2")
        clip2.end_frame = 60  # 60 frames
        clip3 = _make_clip("c3")
        clip3.end_frame = 45  # 45 frames

        # Seed the tab's source lookup so the timeline can resolve videos
        tab._sources = {source.id: source}

        # Stub the video player and timeline_preview to avoid file I/O
        tab.video_player = MagicMock()
        tab.timeline_preview = MagicMock()

        payload = [
            (clip1, source, None),  # First clip has no rationale
            (clip2, source, "Rationale for clip 2"),
            (clip3, source, "Rationale for clip 3"),
        ]

        tab._apply_free_association_sequence(payload)

        sequence = tab.timeline.get_sequence()
        seq_clips = sequence.get_all_clips()
        assert len(seq_clips) == 3

        # First clip: no rationale
        assert seq_clips[0].rationale is None
        # Second and third: rationales attached
        assert seq_clips[1].rationale == "Rationale for clip 2"
        assert seq_clips[2].rationale == "Rationale for clip 3"

        # Algorithm key set on the sequence
        assert sequence.algorithm == "free_association"

    def test_apply_with_empty_payload_is_no_op(self, qapp):
        from ui.tabs.sequence_tab import SequenceTab

        tab = SequenceTab()
        # Should not raise
        tab._apply_free_association_sequence([])


class TestEndToEndPersistence:
    def test_applied_sequence_survives_project_roundtrip(self, qapp):
        """Regression-guard for R10: the most important integration test.

        Build a real sequence via _apply_free_association_sequence, save a
        project containing it, load it back, and verify the rationales
        are still attached to the SequenceClips.
        """
        from core.project import Project
        from models.sequence import Sequence
        from ui.tabs.sequence_tab import SequenceTab

        tab = SequenceTab()
        source = _make_source(fps=30.0)
        clip1 = _make_clip("c1")
        clip1.end_frame = 30
        clip2 = _make_clip("c2")
        clip2.end_frame = 45

        tab._sources = {source.id: source}
        tab.video_player = MagicMock()
        tab.timeline_preview = MagicMock()

        payload = [
            (clip1, source, None),
            (clip2, source, "Both share close-up framing"),
        ]
        tab._apply_free_association_sequence(payload)
        built_sequence: Sequence = tab.timeline.get_sequence()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video = tmp / "video.mp4"
            video.touch()
            project_path = tmp / "project.json"

            project = Project.new(name="FA roundtrip")
            # Use a matching file path on the source we persist
            source.file_path = video
            project.add_source(source)
            project.add_clips([clip1, clip2])
            project.sequence = built_sequence

            assert project.save(project_path) is True

            loaded = Project.load(project_path)

        assert loaded.sequence is not None
        assert loaded.sequence.algorithm == "free_association"
        loaded_clips = loaded.sequence.get_all_clips()
        assert len(loaded_clips) == 2
        assert loaded_clips[0].rationale is None
        assert loaded_clips[1].rationale == "Both share close-up framing"


class TestCardAvailability:
    def test_free_association_appears_in_availability_mapping(self, qapp):
        """The card grid uses a hand-maintained dict — ensure free_association
        is registered so the card is visible. Stubs feature-registry probes
        so the test doesn't depend on transformers/other optional packages."""
        from ui.tabs.sequence_tab import SequenceTab

        tab = SequenceTab()
        source = _make_source()
        clip = _make_clip("c1")
        tab._clips = [clip]

        captured = {}
        tab.card_grid.set_algorithm_availability = (
            lambda d: captured.update(d) or None
        )

        with patch(
            "ui.tabs.sequence_tab.check_feature_ready", return_value=(True, [])
        ):
            tab._update_card_availability()

        assert "free_association" in captured
