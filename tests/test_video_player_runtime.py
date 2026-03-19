"""Tests for frozen-runtime video player setup."""

import ctypes.util
from pathlib import Path
from unittest.mock import patch


def test_find_bundled_mpv_library_windows(tmp_path):
    """Frozen Windows builds should detect a bundled mpv DLL."""
    runtime_dir = tmp_path / "mpv"
    runtime_dir.mkdir()
    dll_path = runtime_dir / "mpv-2.dll"
    dll_path.write_text("fake")

    import ui.video_player as video_player

    with patch.object(video_player.sys, "frozen", True, create=True), \
         patch.object(video_player.sys, "_MEIPASS", str(tmp_path), create=True), \
         patch.object(video_player.sys, "platform", "win32"):
        assert video_player._find_bundled_mpv_library() == dll_path


def test_prepare_frozen_mpv_import_patches_find_library(tmp_path):
    """Frozen builds should make bundled libmpv discoverable to python-mpv."""
    dylib_path = tmp_path / "libmpv.dylib"
    dylib_path.write_text("fake")

    import ui.video_player as video_player

    original_find_library = ctypes.util.find_library
    try:
        with patch.object(video_player.sys, "frozen", True, create=True), \
             patch.object(video_player.sys, "_MEIPASS", str(tmp_path), create=True), \
             patch.object(video_player.sys, "platform", "darwin"):
            prepared = video_player._prepare_frozen_mpv_import()
            assert prepared == dylib_path
            assert ctypes.util.find_library("mpv") == str(dylib_path)
    finally:
        ctypes.util.find_library = original_find_library
