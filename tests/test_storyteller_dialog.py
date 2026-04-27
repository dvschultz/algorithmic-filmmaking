"""Tests for StorytellerDialog state transitions."""

from pathlib import Path
from unittest.mock import patch

import pytest

from models.clip import Clip, Source


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_source(source_id: str = "s1", fps: float = 24.0) -> Source:
    return Source(
        id=source_id,
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=fps,
        width=1920,
        height=1080,
    )


def _make_clip(clip_id: str, source_id: str = "s1") -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=48,
        description=f"Description for {clip_id}",
    )


def test_generation_error_retry_returns_to_config_and_can_restart(qapp):
    from ui.dialogs.storyteller_dialog import StorytellerDialog

    source = _make_source()
    clip = _make_clip("c1", source.id)
    dialog = StorytellerDialog(
        clips=[clip],
        sources_by_id={source.id: source},
        project=None,
    )

    dialog.stack.setCurrentIndex(dialog.PAGE_PROGRESS)
    dialog._update_nav_buttons()
    dialog._on_generation_error("LLM failed")

    assert dialog.next_btn.text() == "Try Again"
    assert dialog.stack.currentIndex() == dialog.PAGE_PROGRESS

    dialog.next_btn.click()
    assert dialog.stack.currentIndex() == dialog.PAGE_CONFIG

    with patch.object(dialog, "_start_generation") as start_mock:
        dialog.next_btn.click()

    assert dialog.stack.currentIndex() == dialog.PAGE_PROGRESS
    start_mock.assert_called_once()
