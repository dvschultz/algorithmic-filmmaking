"""Preflight for Reference Guide sequencing with offline/mislinked footage (#179).

Projects keep offline sources after load, so Reference Guide used to match
on metadata and only fail later at playback/render without naming the
source. These tests pin that the dialog and the agent path refuse up front
and name the affected source.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.remix.reference_match import find_media_problems
from models.clip import Clip, Source


def _source(source_id: str, path: Path) -> Source:
    return Source(
        id=source_id,
        file_path=path,
        fps=30.0,
        duration_seconds=60.0,
        width=1920,
        height=1080,
    )


def _clip(clip_id: str, source_id: str) -> Clip:
    return Clip(id=clip_id, source_id=source_id, start_frame=0, end_frame=90)


@pytest.fixture
def pool(tmp_path):
    """Reference source is offline; user source exists on disk."""
    present = tmp_path / "user.mp4"
    present.write_bytes(b"")
    ref = _source("ref", tmp_path / "missing" / "GMT_Recording.m4a")
    user = _source("user", present)
    clips = [
        (_clip("r1", "ref"), ref),
        (_clip("r2", "ref"), ref),
        (_clip("u1", "user"), user),
    ]
    return ref, user, clips


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestFindMediaProblems:
    def test_all_present_returns_empty(self, tmp_path):
        path = tmp_path / "a.mp4"
        path.write_bytes(b"")
        src = _source("a", path)
        assert find_media_problems([(_clip("c1", "a"), src)]) == []

    def test_missing_file_named_once(self, pool):
        _, _, clips = pool
        problems = find_media_problems(clips)
        assert len(problems) == 1
        assert "GMT_Recording.m4a" in problems[0]
        assert "offline" in problems[0]

    def test_clip_source_id_mismatch_reported(self, tmp_path):
        path = tmp_path / "a.mp4"
        path.write_bytes(b"")
        src = _source("a", path)
        problems = find_media_problems([(_clip("c1", "other"), src)])
        assert len(problems) == 1
        assert "c1" in problems[0] and "a.mp4" in problems[0]


class TestSequenceTabAgentPath:
    def test_offline_reference_refuses_before_matching(self, qapp, pool):
        from ui.tabs.sequence_tab import SequenceTab

        ref, user, clips = pool
        tab = SequenceTab()
        tab._sources = {ref.id: ref, user.id: user}
        tab._available_clips = clips
        tab.video_player = MagicMock()
        tab.timeline_preview = MagicMock()

        with patch("core.remix.run_registry_algorithm") as run:
            result = tab.generate_reference_guided("ref", {"duration": 1.0})

        assert result["success"] is False
        assert "GMT_Recording.m4a" in result["error"]
        run.assert_not_called()


class TestReferenceGuideDialog:
    def test_offline_reference_warns_and_does_not_start_worker(self, qapp, pool):
        from ui.dialogs.reference_guide_dialog import ReferenceGuideDialog

        ref, user, clips = pool
        dialog = ReferenceGuideDialog(
            clips=clips, sources_by_id={ref.id: ref, user.id: user}
        )
        index = dialog.source_combo.findData("ref")
        dialog.source_combo.setCurrentIndex(index)
        # Force a dimension that is always available on.
        widgets = dialog._dimension_widgets["duration"]
        widgets["checkbox"].setChecked(True)
        widgets["slider"].setValue(100)

        with (
            patch("ui.dialogs.reference_guide_dialog.QMessageBox.warning") as warn,
            patch(
                "ui.dialogs.reference_guide_dialog.ReferenceMatchWorker"
            ) as worker_cls,
        ):
            dialog._on_generate()

        worker_cls.assert_not_called()
        warn.assert_called_once()
        assert "GMT_Recording.m4a" in warn.call_args.args[2]
