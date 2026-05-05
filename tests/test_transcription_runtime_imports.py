"""Tests for transcription runtime import boundaries.

These tests defend the fix documented in
``docs/solutions/runtime-errors/macos-libmpv-pyav-ffmpeg-dylib-collision-20260504.md``.
Loading PyAV's bundled FFmpeg dylibs alongside Homebrew FFmpeg (pulled in
transitively by libmpv) triggers AVFoundation duplicate-class warnings on
macOS, so any startup or availability-check path that pulls in ``av`` is a
regression.
"""

import sys


def _purge_transcription_modules():
    """Drop cached transcription/PyAV modules so re-imports run cleanly."""
    for name in (
        "core.transcription",
        "faster_whisper",
        "av",
    ):
        sys.modules.pop(name, None)


def test_importing_core_transcription_does_not_load_pyav():
    """``import core.transcription`` must not transitively load PyAV.

    ``faster_whisper`` 1.x eagerly imports ``av`` from ``faster_whisper.audio``
    at module load. The transcription module therefore has to defer importing
    ``faster_whisper`` until transcription actually runs — otherwise the GUI
    process loads PyAV's bundled FFmpeg dylibs alongside libmpv's Homebrew
    FFmpeg dylibs and triggers the documented AVFoundation collision.
    """
    _purge_transcription_modules()

    import core.transcription  # noqa: F401

    assert "faster_whisper" not in sys.modules
    assert "av" not in sys.modules


def test_faster_whisper_availability_does_not_import_runtime_modules(monkeypatch):
    """Availability checks should not import PyAV into the GUI process."""
    import core.transcription as transcription

    monkeypatch.setattr(transcription, "_faster_whisper_available", None)
    sys.modules.pop("faster_whisper", None)
    sys.modules.pop("av", None)

    transcription.is_faster_whisper_available()

    assert "faster_whisper" not in sys.modules
    assert "av" not in sys.modules


def test_scene_detect_uses_opencv_backend_without_importing_pyav():
    """Scene detection imports should not load PyAV's FFmpeg dylibs."""
    sys.modules.pop("core.scene_detect", None)
    sys.modules.pop("scenedetect", None)
    sys.modules.pop("scenedetect.backends", None)
    sys.modules.pop("scenedetect.backends.pyav", None)
    sys.modules.pop("av", None)

    import core.scene_detect  # noqa: F401

    assert "av" not in sys.modules
    assert sys.modules.get("scenedetect.backends.pyav") is None
