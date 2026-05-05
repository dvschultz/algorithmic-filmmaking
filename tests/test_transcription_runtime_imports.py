"""Tests for transcription runtime import boundaries."""

import sys


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
