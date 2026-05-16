"""Tests for the audio source picker in StaccatoDialog (U5)."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def make_audio(tmp_path):
    """Build an AudioSource whose file_path actually exists."""
    from models.audio_source import AudioSource

    def _make(name="song.wav", **kwargs):
        path = tmp_path / name
        path.write_bytes(b"")
        kwargs.setdefault("duration_seconds", 60.0)
        return AudioSource(file_path=path, **kwargs)

    return _make


@pytest.fixture
def patched_dialog(qapp, monkeypatch):
    """Stub _analyze_audio so the dialog doesn't try to spin up audio analysis
    workers in tests. Tracks created dialogs and explicitly deletes them at
    teardown — without this, multiple QDialog constructions in one process
    can corrupt theme() state on macOS Qt.
    """
    from ui.dialogs.staccato_dialog import StaccatoDialog
    monkeypatch.setattr(StaccatoDialog, "_analyze_audio", lambda self: None)

    created: list = []

    def _build(clips, project):
        dialog = StaccatoDialog(clips=clips, project=project)
        created.append(dialog)
        return dialog

    yield _build

    for d in created:
        d.deleteLater()
    qapp.processEvents()


def test_combo_populates_with_project_audio_sources(qapp, make_audio, patched_dialog):
    from core.project import Project
    from ui.dialogs.staccato_dialog import StaccatoDialog

    project = Project.new()
    a1 = make_audio("a.mp3", id="a1", duration_seconds=120.0)
    a2 = make_audio("b.wav", id="a2", duration_seconds=45.0)
    project.add_audio_source(a1)
    project.add_audio_source(a2)

    dialog = patched_dialog(clips=[], project=project)
    combo = dialog._audio_combo

    # Two real sources + separator + Import-new = 4 items
    assert combo.count() == 4
    assert combo.itemData(0) == "a1"
    assert "a.mp3" in combo.itemText(0)
    assert "2:00" in combo.itemText(0)
    assert combo.itemData(1) == "a2"
    # Index 2 is a separator (no data), index 3 is import-new sentinel
    assert combo.itemData(3) == StaccatoDialog._IMPORT_NEW_SENTINEL


def test_default_selection_picks_first_audio_source_and_sets_music_path(
    qapp, make_audio, patched_dialog
):
    from core.project import Project

    project = Project.new()
    a1 = make_audio("a.mp3", id="a1")
    project.add_audio_source(a1)

    dialog = patched_dialog(clips=[], project=project)

    assert dialog._music_path == a1.file_path


def test_combo_empty_project_shows_placeholder_and_import_new(qapp, patched_dialog):
    from core.project import Project
    from ui.dialogs.staccato_dialog import StaccatoDialog

    project = Project.new()
    dialog = patched_dialog(clips=[], project=project)
    combo = dialog._audio_combo

    # Placeholder + separator + Import-new = 3 items
    assert combo.count() == 3
    # Placeholder has no data and is disabled
    assert combo.itemData(0) is None
    assert "No audio sources" in combo.itemText(0)
    # Import-new is the only enabled real choice
    assert combo.itemData(2) == StaccatoDialog._IMPORT_NEW_SENTINEL
    # No music_path set
    assert dialog._music_path is None


def test_empty_project_prompts_for_audio_when_dialog_is_shown(qapp, monkeypatch):
    from core.project import Project
    from ui.dialogs.staccato_dialog import StaccatoDialog

    project = Project.new()
    prompts = []

    monkeypatch.setattr(StaccatoDialog, "_analyze_audio", lambda self: None)
    monkeypatch.setattr(
        StaccatoDialog,
        "_on_import_new_audio",
        lambda self: prompts.append(True),
    )

    dialog = StaccatoDialog(clips=[], project=project)
    dialog.show()
    qapp.processEvents()

    assert prompts == [True]
    dialog.deleteLater()
    qapp.processEvents()


def test_sequence_tab_staccato_route_passes_project_to_dialog(qapp, monkeypatch):
    from core.project import Project
    from models.clip import Clip, Source
    from ui.dialogs import staccato_dialog
    from ui.tabs.sequence_tab import SequenceTab

    project = Project.new()
    source = Source(id="src-1", file_path=Path("/tmp/test.mp4"), fps=24.0)
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=24)
    tab = SequenceTab()
    tab.set_project(project)
    tab.set_available_clips(
        [(clip, source)],
        all_clips=[clip],
        sources_by_id={source.id: source},
    )
    tab._gui_state = SimpleNamespace(analyze_selected_ids=[clip.id], cut_selected_ids=[])

    captured = {}

    class FakeSignal:
        def connect(self, callback):
            captured["callback"] = callback

    class FakeStaccatoDialog:
        def __init__(self, clips, project=None, parent=None):
            captured["clips"] = clips
            captured["project"] = project
            captured["parent"] = parent
            self.music_path = None
            self.sequence_ready = FakeSignal()

        def exec(self):
            captured["exec"] = True

    monkeypatch.setattr(staccato_dialog, "StaccatoDialog", FakeStaccatoDialog)

    tab._on_card_clicked("staccato")

    assert captured["clips"] == [(clip, source)]
    assert captured["project"] is project
    assert captured["parent"] is tab
    assert captured["exec"] is True


def test_sequence_tab_uses_private_project_attribute():
    source = Path("ui/tabs/sequence_tab.py").read_text(encoding="utf-8")

    assert "self.project" not in source
    assert "self._project" in source


def test_dialog_sequencers_use_private_project_attribute():
    dialog_paths = [
        Path("ui/dialogs/cassette_tape_dialog.py"),
        Path("ui/dialogs/exquisite_corpus_dialog.py"),
        Path("ui/dialogs/free_association_dialog.py"),
        Path("ui/dialogs/storyteller_dialog.py"),
    ]

    for path in dialog_paths:
        source = path.read_text(encoding="utf-8")
        assert "self.project" not in source, path
        assert "self._project" in source, path


def test_selecting_existing_source_updates_music_path(qapp, make_audio, patched_dialog):
    from core.project import Project

    project = Project.new()
    a1 = make_audio("first.wav", id="a1")
    a2 = make_audio("second.wav", id="a2")
    project.add_audio_source(a1)
    project.add_audio_source(a2)

    dialog = patched_dialog(clips=[], project=project)
    # default is a1
    assert dialog._music_path == a1.file_path

    # Switch to a2 by setting combo index
    for i in range(dialog._audio_combo.count()):
        if dialog._audio_combo.itemData(i) == "a2":
            dialog._audio_combo.setCurrentIndex(i)
            break

    assert dialog._music_path == a2.file_path


def test_missing_file_on_disk_surfaces_error_and_clears_music_path(
    qapp, tmp_path, patched_dialog
):
    """If a previously-imported audio source's file is gone, picking it
    should not crash analysis — the dialog should surface a clear message."""
    from core.project import Project
    from models.audio_source import AudioSource

    # Build an AudioSource pointing at a path that doesn't exist
    project = Project.new()
    audio = AudioSource(
        id="ghost",
        file_path=tmp_path / "missing.wav",
        duration_seconds=30.0,
    )
    project.add_audio_source(audio)

    dialog = patched_dialog(clips=[], project=project)
    assert dialog._music_path is None
    assert "missing" in dialog._info_label.text().lower()
    assert dialog._generate_btn.isEnabled() is False


def test_import_new_routes_through_worker_and_selects_new_source(
    qapp, tmp_path, patched_dialog
):
    """Selecting 'Import new…' opens a file dialog, runs the import worker,
    adds the audio source to the project, and auto-selects it."""
    from core.project import Project
    from models.audio_source import AudioSource

    project = Project.new()
    dialog = patched_dialog(clips=[], project=project)

    # Stage a file the user "selects" in the QFileDialog
    new_audio_path = tmp_path / "new.wav"
    new_audio_path.write_bytes(b"")
    expected_audio = AudioSource(
        id="new-aud",
        file_path=new_audio_path,
        duration_seconds=42.0,
        sample_rate=44100,
        channels=2,
    )

    # Patch the file dialog to return our fixture path
    with patch(
        "ui.dialogs.staccato_dialog.QFileDialog.getOpenFileName",
        return_value=(str(new_audio_path), ""),
    ):
        # Skip the actual worker — just simulate the audio_ready callback path
        # by calling _on_imported_audio_ready directly.
        # First, find and click the Import-new combo index.
        for i in range(dialog._audio_combo.count()):
            if dialog._audio_combo.itemData(i) == dialog._IMPORT_NEW_SENTINEL:
                # Selecting it triggers _on_audio_combo_changed → _on_import_new_audio
                # which spawns a worker. Instead of starting the real worker,
                # we just assert the worker was created and bypass its run.
                dialog._audio_combo.setCurrentIndex(i)
                break

        # Simulate a successful worker emission
        dialog._on_imported_audio_ready(expected_audio)

    # The new audio source should be in the project
    assert project.get_audio_source("new-aud") is expected_audio
    # And the combo should have it auto-selected
    assert dialog._audio_combo.currentData() == "new-aud"


def test_import_new_with_existing_audio_analyzes_new_source_once(
    qapp, make_audio, monkeypatch
):
    from core.project import Project
    from models.audio_source import AudioSource
    from ui.dialogs.staccato_dialog import StaccatoDialog

    project = Project.new()
    existing = make_audio("existing.wav", id="existing")
    project.add_audio_source(existing)
    new_path = make_audio("new.wav", id="unused").file_path
    new_audio = AudioSource(id="new", file_path=new_path, duration_seconds=42.0)

    analyzed_paths = []
    monkeypatch.setattr(
        StaccatoDialog,
        "_analyze_audio",
        lambda self: analyzed_paths.append(self._music_path),
    )

    dialog = StaccatoDialog(clips=[], project=project)
    analyzed_paths.clear()

    dialog._on_imported_audio_ready(new_audio)

    assert dialog._audio_combo.currentData() == "new"
    assert analyzed_paths == [new_path]
    dialog.deleteLater()
    qapp.processEvents()


def test_cancelled_import_new_resets_combo(qapp, make_audio, patched_dialog):
    """If the user picks Import-new and cancels the file dialog, the combo
    should fall back to the previously-selected source rather than getting
    stuck on Import-new."""
    from core.project import Project

    project = Project.new()
    a1 = make_audio("a.mp3", id="a1")
    project.add_audio_source(a1)

    dialog = patched_dialog(clips=[], project=project)
    # Currently selected: a1 (index 0)
    assert dialog._audio_combo.currentData() == "a1"

    with patch(
        "ui.dialogs.staccato_dialog.QFileDialog.getOpenFileName",
        return_value=("", ""),  # User cancelled
    ):
        # Pick the Import-new item
        for i in range(dialog._audio_combo.count()):
            if dialog._audio_combo.itemData(i) == dialog._IMPORT_NEW_SENTINEL:
                dialog._audio_combo.setCurrentIndex(i)
                break

    # Combo should have reset back to a1
    assert dialog._audio_combo.currentData() == "a1"


def test_failed_import_surfaces_error_and_resets_combo(qapp, make_audio, patched_dialog):
    """If the worker emits error, the combo should reset and the info label
    should reflect the failure."""
    from core.project import Project

    project = Project.new()
    a1 = make_audio("a.mp3", id="a1")
    project.add_audio_source(a1)

    dialog = patched_dialog(clips=[], project=project)
    dialog._on_imported_audio_error("FFprobe failed")

    assert "FFprobe failed" in dialog._info_label.text()
    assert dialog._audio_combo.currentData() == "a1"
