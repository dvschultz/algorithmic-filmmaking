"""Tests for StaccatoDialog generation wiring."""

import os
from pathlib import Path

import numpy as np
import pytest

from core.analysis.audio import AudioAnalysis
from models.clip import Clip, Source

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _DummySignal:
    def connect(self, *args, **kwargs):
        return None


def _make_clip_and_source():
    source = Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=60.0,
        fps=24.0,
        width=1920,
        height=1080,
    )
    clip = Clip(
        id="clip-1",
        source_id=source.id,
        start_frame=0,
        end_frame=48,
        description="clip",
    )
    clip.embedding = [1.0, 0.0]
    return clip, source


def test_generate_uses_preview_filtered_markers(qapp, monkeypatch):
    from ui.dialogs.staccato_dialog import StaccatoDialog

    clip, source = _make_clip_and_source()
    dialog = StaccatoDialog(clips=[(clip, source)])
    dialog._audio_analysis = AudioAnalysis(
        beat_times=[0.5, 1.0, 1.5],
        onset_times=[0.5, 1.0, 1.5],
        onset_strengths=[0.95, 0.4, 0.92],
        duration_seconds=2.0,
    )
    dialog._audio_samples = np.zeros(32, dtype=np.float32)
    dialog._strategy_combo.setCurrentText("Onsets")
    dialog._sensitivity_slider.setValue(1)  # threshold 0.9 -> keep 0.5 and 1.5

    captured = {}

    class FakeWorker:
        def __init__(self, clips, audio_analysis, strategy, cut_times=None, parent=None):
            captured["clips"] = clips
            captured["audio_analysis"] = audio_analysis
            captured["strategy"] = strategy
            captured["cut_times"] = cut_times
            self.progress_update = _DummySignal()
            self.progress_message = _DummySignal()
            self.finished_sequence = _DummySignal()
            self.error = _DummySignal()

        def start(self):
            captured["started"] = True

    monkeypatch.setattr("ui.dialogs.staccato_dialog.StaccatoGenerateWorker", FakeWorker)

    dialog._on_generate()

    assert captured["strategy"] == "onsets"
    assert captured["cut_times"] == [0.5, 1.5]
    assert captured["started"] is True
