"""U14: native runtime families and the host <-> worker isolation seam.

Source-mode proof (the same standard U13 met before its packaged run):
allowlisted engine calls run in a real worker process, the host never
imports the runtime, provenance callbacks are replayed, dependency and
model failures keep their exception classes, and every family is off by
default until its packaged evidence lands.
"""

from __future__ import annotations

import math
import os
import sys
import wave
from pathlib import Path

import pytest

from core import runtime_families
from core.runtime_families import FAMILIES, enabled_families, isolated, run_isolated
from core.runtime_supervisor import RuntimeSupervisor, WorkerLaunch
from core.runtime_worker.calls import ISOLATED_CALLS

WORKER_ROOT = Path(__file__).resolve().parents[1] / "core"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker_supervisor(monkeypatch, tmp_path):
    import core.runtime_supervisor as supervisor_module

    sup = RuntimeSupervisor(staging_root=tmp_path / "staging")
    sup.launch_factory = lambda family: WorkerLaunch(
        interpreter=Path(sys.executable), worker_root=WORKER_ROOT, family=family, engine_root=REPO_ROOT,
    )
    monkeypatch.setattr(supervisor_module, "_default", sup)
    monkeypatch.setattr(supervisor_module, "default_supervisor", lambda: sup)
    yield sup
    sup.shutdown()


def test_allowlist_matches_decorated_engine_functions():
    import importlib

    for call, (module_name, function_name) in ISOLATED_CALLS.items():
        if call.startswith("selftest."):
            continue
        function = getattr(importlib.import_module(module_name), function_name)
        family, decorated_call = function.__isolated__
        assert decorated_call == call and family in FAMILIES, call
    with pytest.raises(ValueError, match="not an allowlisted"):
        isolated("vision", "os.system")(lambda: None)
    with pytest.raises(ValueError, match="Unknown runtime family"):
        isolated("shell", "selftest.identity")(lambda: None)


def test_families_are_off_by_default_except_transcription(monkeypatch):
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.delenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", raising=False)
    monkeypatch.setattr("core.settings.load_settings", lambda **kw: type("S", (), {"native_worker_families": ["transcription"]})())
    assert enabled_families() == frozenset({"transcription"})
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "audio, vision,bogus")
    assert enabled_families() == frozenset({"audio", "vision"})
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "0")
    assert enabled_families() == frozenset()
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_WORKER_PROCESS", "1")
    assert enabled_families() == frozenset()  # never re-forward inside a worker


def test_decorator_runs_in_process_when_the_family_is_off(monkeypatch):
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "0")
    calls = []

    @isolated("audio", "selftest.identity")
    def local(path: Path, *, on_execution=None):
        calls.append(path)
        return "local"

    assert local(Path("/x")) == "local" and calls == [Path("/x")]


def test_round_trip_runs_in_the_worker_and_replays_executions(monkeypatch, worker_supervisor):
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "audio")
    result = run_isolated("audio", "selftest.identity", {"path": "/tmp/x", "n": 2})
    assert result["kwargs"] == {"path": "/tmp/x", "n": 2} and result["pid"] != os.getpid()

    seen = []

    @isolated("audio", "selftest.execution", decode=lambda v: ("decoded", v))
    def engine_call(image_path: Path, *, on_execution=None, progress=None):
        raise AssertionError("must run in the worker, not here")

    value = engine_call(Path("/tmp/frame.jpg"), on_execution=seen.append, progress=lambda *_: None)
    assert value == ("decoded", {"ok": True})
    assert seen == [{"model": "selftest", "device": "cpu", "kwargs": {"image_path": "/tmp/frame.jpg"}}]


def test_worker_failures_keep_their_exception_classes(monkeypatch, worker_supervisor):
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision")
    with pytest.raises(ImportError, match="not installed in the worker"):
        run_isolated("vision", "selftest.missing", {})
    with pytest.raises(ValueError, match="not an allowlisted"):
        run_isolated("vision", "shutil.rmtree", {"path": "/"})
    with pytest.raises(runtime_families.IsolatedCallError):
        run_isolated("vision", "selftest.identity", {"unexpected": object()})  # not JSON-encodable


@pytest.mark.skipif(not __import__("importlib.util").util.find_spec("librosa"), reason="librosa not installed")
def test_audio_family_analyzes_in_the_worker_without_importing_librosa_here(monkeypatch, worker_supervisor, tmp_path):
    from core.analysis.audio import AudioAnalysis, analyze_audio

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "audio")
    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        for index in range(22050 * 2):
            beat = 1.0 if (index // 11025) % 2 == 0 else 0.2
            value = int(8000 * beat * math.sin(2 * math.pi * 220 * index / 22050))
            handle.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    for name in [m for m in sys.modules if m.split(".")[0] in ("librosa", "numba")]:
        sys.modules.pop(name)
    analysis = analyze_audio(wav, include_onsets=True)
    assert isinstance(analysis, AudioAnalysis)
    assert 1.9 < analysis.duration_seconds < 2.1 and analysis.sample_rate == 22050
    assert "librosa" not in sys.modules  # the worker imported it, this process did not
    worker = worker_supervisor.worker("audio")
    assert worker.pid != os.getpid()


def test_host_preimport_workarounds_lift_only_when_every_family_is_isolated(monkeypatch):
    from core.runtime_families import host_needs_runtime

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision,vlm")
    assert host_needs_runtime("torch")  # alignment and audio still import torch in-process
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision,vlm,alignment,audio")
    assert not host_needs_runtime("torch")
    monkeypatch.setattr("core.settings.load_settings", lambda **kw: type("S", (), {"transcription_backend": "faster-whisper", "native_worker_families": []})())
    assert not host_needs_runtime("mlx")
    monkeypatch.setattr("core.settings.load_settings", lambda **kw: type("S", (), {"transcription_backend": "auto", "native_worker_families": []})())
    assert host_needs_runtime("mlx")  # the MLX whisper backend is not isolated
    assert host_needs_runtime("unknown-runtime")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "0")
    assert host_needs_runtime("torch")


def test_settings_accept_only_known_families(monkeypatch, tmp_path):
    from core.spine.settings_io import update_settings

    monkeypatch.setenv("SCENE_RIPPER_CONFIG", str(tmp_path / "config.json"))
    assert update_settings("native_worker_families", "audio, vision")["success"]
    from core.settings import load_settings

    assert load_settings().native_worker_families == ["audio", "vision"]
    assert not update_settings("native_worker_families", ["audio", "shell"])["success"]
