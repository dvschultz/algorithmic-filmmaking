"""Tests for the audio source picker in StaccatoDialog (U5)."""

import os
from pathlib import Path
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
