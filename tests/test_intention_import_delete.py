"""Tests for deleting pending imports in the IntentionImportDialog."""

from pathlib import Path

import pytest
from PySide6.QtCore import Qt


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dialog(qapp):
    from ui.dialogs.intention_import_dialog import IntentionImportDialog

    dlg = IntentionImportDialog(algorithm="duration")
    return dlg


def test_delete_local_file_removes_from_pending(dialog):
    """Pressing Delete on a selected local file removes it."""
    path = Path("/test/video.mp4")
    dialog._local_paths.append(path)
    dialog._update_pending_list()

    assert dialog.pending_list.count() == 1

    # Select the item
    dialog.pending_list.setCurrentRow(0)

    # Remove it
    dialog._remove_selected_pending()

    assert dialog.pending_list.count() == 0
    assert path not in dialog._local_paths


def test_delete_local_file_removes_from_durations(dialog):
    """Deleting a local file also clears its cached duration."""
    path = Path("/test/video.mp4")
    dialog._local_paths.append(path)
    dialog._durations[str(path)] = 120.0
    dialog._update_pending_list()

    dialog.pending_list.setCurrentRow(0)
    dialog._remove_selected_pending()

    assert str(path) not in dialog._durations


def test_delete_url_removes_from_pending_and_text_input(dialog):
    """Pressing Delete on a selected URL removes it from the list and text input."""
    url = "https://youtube.com/watch?v=abc123"
    dialog.url_input.setPlainText(url)
    dialog._update_pending_list()

    assert dialog.pending_list.count() == 1

    # Select the URL item
    dialog.pending_list.setCurrentRow(0)
    dialog._remove_selected_pending()

    assert dialog.pending_list.count() == 0
    assert dialog.url_input.toPlainText().strip() == ""


def test_delete_url_preserves_other_urls(dialog):
    """Deleting one URL preserves other URLs in the text input."""
    urls = "https://youtube.com/watch?v=abc\nhttps://vimeo.com/123"
    dialog.url_input.setPlainText(urls)
    dialog._update_pending_list()

    assert dialog.pending_list.count() == 2

    # Select only the first URL item (files come first, but here there are none)
    dialog.pending_list.setCurrentRow(0)
    dialog._remove_selected_pending()

    assert dialog.pending_list.count() == 1
    assert "vimeo.com" in dialog.url_input.toPlainText()
    assert "youtube.com" not in dialog.url_input.toPlainText()


def test_delete_multiple_items(dialog):
    """Multi-select deletion removes all selected items."""
    path1 = Path("/test/video1.mp4")
    path2 = Path("/test/video2.mp4")
    dialog._local_paths.extend([path1, path2])
    dialog.url_input.setPlainText("https://youtube.com/watch?v=abc")
    dialog._update_pending_list()

    assert dialog.pending_list.count() == 3

    # Select all items
    for i in range(dialog.pending_list.count()):
        dialog.pending_list.item(i).setSelected(True)

    dialog._remove_selected_pending()

    assert dialog.pending_list.count() == 0
    assert len(dialog._local_paths) == 0
    assert dialog.url_input.toPlainText().strip() == ""


def test_delete_all_disables_start_button(dialog):
    """After deleting all items, the Start button is disabled."""
    path = Path("/test/video.mp4")
    dialog._local_paths.append(path)
    dialog._update_pending_list()

    assert dialog.start_btn.isEnabled()

    dialog.pending_list.setCurrentRow(0)
    dialog._remove_selected_pending()

    assert not dialog.start_btn.isEnabled()


def test_delete_with_no_selection_is_noop(dialog):
    """Calling remove with nothing selected does nothing."""
    path = Path("/test/video.mp4")
    dialog._local_paths.append(path)
    dialog._update_pending_list()

    # Clear selection
    dialog.pending_list.clearSelection()
    dialog._remove_selected_pending()

    assert dialog.pending_list.count() == 1
    assert path in dialog._local_paths
