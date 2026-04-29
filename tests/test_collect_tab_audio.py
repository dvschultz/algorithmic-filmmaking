"""Tests for the Collect tab audio import + library section."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def make_audio(tmp_path):
    """Build an AudioSource with file actually present on disk."""
    from models.audio_source import AudioSource

    def _make(name="song.wav", **kwargs):
        path = tmp_path / name
        path.write_bytes(b"")
        kwargs.setdefault("duration_seconds", 60.0)
        return AudioSource(file_path=path, **kwargs)

    return _make


def test_audio_library_list_renders_sources(qapp, make_audio):
    from ui.widgets.audio_library_list import AudioLibraryList

    a1 = make_audio("a.mp3", id="a1", duration_seconds=120.0)
    a2 = make_audio("b.wav", id="a2", duration_seconds=45.0)

    widget = AudioLibraryList()
    widget.set_sources([a1, a2])

    assert widget.count() == 2
    assert widget._table.rowCount() == 2
    assert widget._table.item(0, 0).text() == "a.mp3"
    assert widget._table.item(0, 1).text() == "2:00"
    assert widget._table.item(1, 0).text() == "b.wav"
    assert widget._table.item(1, 1).text() == "0:45"


def test_audio_library_list_emits_remove_request(qapp, make_audio):
    from ui.widgets.audio_library_list import AudioLibraryList

    a1 = make_audio("a.mp3", id="a1")
    widget = AudioLibraryList()
    widget.set_sources([a1])

    emissions: list[str] = []
    widget.remove_requested.connect(lambda aid: emissions.append(aid))

    # Simulate clicking the row's Remove button
    btn = widget._table.cellWidget(0, widget._COL_REMOVE)
    btn.click()

    assert emissions == ["a1"]


def test_audio_library_list_emits_transcribe_request(qapp, make_audio):
    from ui.widgets.audio_library_list import AudioLibraryList

    a1 = make_audio("a.mp3", id="a1")
    widget = AudioLibraryList()
    widget.set_sources([a1])

    emissions: list[str] = []
    widget.transcribe_requested.connect(lambda aid: emissions.append(aid))

    btn = widget._table.cellWidget(0, widget._COL_TRANSCRIBE)
    assert btn.text() == "Transcribe"
    assert btn.isEnabled() is True
    btn.click()

    assert emissions == ["a1"]


def test_audio_library_list_disables_transcribe_when_already_transcribed(qapp, make_audio):
    from core.transcription import TranscriptSegment
    from ui.widgets.audio_library_list import AudioLibraryList

    a1 = make_audio("a.mp3", id="a1")
    a1.transcript = [TranscriptSegment(start_time=0.0, end_time=1.0, text="hi")]

    widget = AudioLibraryList()
    widget.set_sources([a1])

    btn = widget._table.cellWidget(0, widget._COL_TRANSCRIBE)
    assert btn.text() == "Transcribed"
    assert btn.isEnabled() is False


def test_audio_library_list_emits_selection(qapp, make_audio):
    from ui.widgets.audio_library_list import AudioLibraryList

    a1 = make_audio("a.mp3", id="a1")
    widget = AudioLibraryList()
    widget.set_sources([a1])

    emissions: list[object] = []
    widget.audio_source_selected.connect(lambda audio: emissions.append(audio))

    widget._table.selectRow(0)
    qapp.processEvents()

    assert len(emissions) == 1
    assert emissions[0] is a1


def test_collect_tab_routes_audio_drops_to_audio_signal(qapp, tmp_path):
    from ui.tabs.collect_tab import CollectTab

    tab = CollectTab()

    audio_emissions: list[list[Path]] = []
    video_emissions: list[list[Path]] = []
    tab.audio_files_added.connect(lambda paths: audio_emissions.append(list(paths)))
    tab.videos_added.connect(lambda paths: video_emissions.append(list(paths)))

    audio_file = tmp_path / "a.wav"
    video_file = tmp_path / "v.mp4"
    other_file = tmp_path / "x.txt"

    tab._on_files_dropped([audio_file, video_file, other_file])

    assert audio_emissions == [[audio_file]]
    # Non-audio files (video and unknown) all flow through the videos signal —
    # main_window's video import path handles unknown extensions itself.
    assert video_emissions == [[video_file, other_file]]


def test_collect_tab_drops_with_only_videos_emits_only_videos_signal(qapp, tmp_path):
    from ui.tabs.collect_tab import CollectTab

    tab = CollectTab()
    audio_emissions, video_emissions = [], []
    tab.audio_files_added.connect(lambda paths: audio_emissions.append(list(paths)))
    tab.videos_added.connect(lambda paths: video_emissions.append(list(paths)))

    tab._on_files_dropped([tmp_path / "v.mp4"])

    assert audio_emissions == []
    assert len(video_emissions) == 1


def test_collect_tab_set_audio_sources_renders_in_library(qapp, make_audio):
    from ui.tabs.collect_tab import CollectTab

    tab = CollectTab()
    a1 = make_audio("a.mp3", id="a1")
    a2 = make_audio("b.wav", id="a2")

    tab.set_audio_sources([a1, a2])

    assert tab.audio_library.count() == 2
    assert tab.get_audio_sources() == [a1, a2]


def test_collect_tab_clear_resets_audio_library(qapp, make_audio):
    from ui.tabs.collect_tab import CollectTab

    tab = CollectTab()
    tab.set_audio_sources([make_audio()])
    assert tab.audio_library.count() == 1

    tab.clear()
    assert tab.audio_library.count() == 0


def test_collect_tab_audio_remove_signal_propagates(qapp, make_audio):
    from ui.tabs.collect_tab import CollectTab

    tab = CollectTab()
    a1 = make_audio("a.mp3", id="a1")
    tab.set_audio_sources([a1])

    emissions: list[str] = []
    tab.audio_remove_requested.connect(lambda aid: emissions.append(aid))

    # Simulate the inner library widget firing its remove signal
    tab.audio_library.remove_requested.emit("a1")
    assert emissions == ["a1"]


def test_audio_format_helpers():
    from core.audio_formats import AUDIO_EXTENSIONS, is_audio_file

    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS
    assert is_audio_file("song.MP3") is True  # case-insensitive
    assert is_audio_file(Path("/tmp/track.flac")) is True
    assert is_audio_file("video.mp4") is False
    assert is_audio_file("nope.txt") is False
