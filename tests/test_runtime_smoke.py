"""Tests for frozen runtime smoke helpers."""

import pytest

from core.runtime_smoke import (
    get_runtime_smoke_targets,
    run_runtime_smoke_target,
)


def test_runtime_smoke_targets_are_stable():
    """Smoke targets should enumerate the release validation surfaces."""
    assert get_runtime_smoke_targets() == ("imports", "project", "scene-detect", "updater")


def test_project_runtime_smoke_passes():
    """Project runtime smoke should exercise save/load successfully."""
    assert run_runtime_smoke_target("project") == "project"


def test_scene_detect_runtime_smoke_passes():
    """Scene detection runtime smoke should detect multiple synthetic clips."""
    assert run_runtime_smoke_target("scene-detect") == "scene-detect"


def test_runtime_smoke_rejects_unknown_target():
    """Unknown smoke targets should fail fast with a clear message."""
    with pytest.raises(ValueError, match="Unknown runtime smoke target"):
        run_runtime_smoke_target("nope")


def test_updater_runtime_smoke_surfaces_unavailable_status(monkeypatch):
    """Updater smoke should raise when the Windows updater is unavailable."""
    import core.runtime_smoke as runtime_smoke

    class _Status:
        available = False
        reason = "missing metadata"
        dll_path = None
        feed_url = ""
        public_key = ""

    monkeypatch.setattr(runtime_smoke.sys, "platform", "win32")

    def _fake_get_status(update_channel: str = "stable"):
        assert update_channel == "stable"
        return _Status()

    monkeypatch.setattr("core.windows_updater.get_status", _fake_get_status)

    with pytest.raises(RuntimeError, match="missing metadata"):
        run_runtime_smoke_target("updater")
