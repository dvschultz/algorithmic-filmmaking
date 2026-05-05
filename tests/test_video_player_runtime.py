"""Tests for frozen-runtime video player setup."""

import ctypes.util
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch


def test_video_player_module_does_not_import_python_mpv_on_import():
    """Importing the UI module should not load libmpv into the process."""
    sys.modules.pop("ui.video_player", None)
    sys.modules.pop("mpv", None)

    video_player = importlib.import_module("ui.video_player")

    assert video_player.mpv is None
    assert "mpv" not in sys.modules


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


def test_prepare_system_mpv_import_uses_exact_env_library(tmp_path, monkeypatch):
    """Source builds should avoid broad DYLD_LIBRARY_PATH mutations for mpv."""
    dylib_path = tmp_path / "libmpv.dylib"
    dylib_path.write_text("fake")

    import ui.video_player as video_player

    original_find_library = ctypes.util.find_library
    try:
        monkeypatch.setenv("SCENE_RIPPER_LIBMPV_DYLIB", str(dylib_path))
        monkeypatch.setenv("DYLD_LIBRARY_PATH", "/existing")

        assert video_player._find_system_mpv_library() == dylib_path
        video_player._prepare_mpv_import(dylib_path)

        assert ctypes.util.find_library("mpv") == str(dylib_path)
        assert os.environ["DYLD_LIBRARY_PATH"] == "/existing"
    finally:
        ctypes.util.find_library = original_find_library
