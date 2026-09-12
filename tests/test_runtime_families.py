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

    sup = RuntimeSupervisor(staging_root=tmp_path / "staging", allow_test_tasks=True)
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


def test_analysis_payload_is_decoded_only_after_supervisor_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core.runtime_supervisor import WorkerProtocolViolation

    victim = tmp_path / "unrelated.json"
    victim.write_text('{"private": true}', encoding="utf-8")
    monkeypatch.setattr(
        "core.runtime_supervisor.default_supervisor",
        lambda: SimpleNamespace(
            run=lambda *args, **kwargs: {"value_path": str(victim)}
        ),
    )

    with pytest.raises(WorkerProtocolViolation):
        run_isolated("vision", "embeddings.thumbnails", {"thumbnail_paths": []})

    assert victim.exists()


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
    evicted = {m: sys.modules.pop(m) for m in list(sys.modules) if m.split(".")[0] in ("librosa", "numba")}
    try:
        analysis = analyze_audio(wav, include_onsets=True)
        assert isinstance(analysis, AudioAnalysis)
        assert 1.9 < analysis.duration_seconds < 2.1 and analysis.sample_rate == 22050
        assert "librosa" not in sys.modules  # the worker imported it, this process did not
        worker = worker_supervisor.worker("audio")
        assert worker.pid != os.getpid()
    finally:
        sys.modules.update(evicted)  # never leave native modules unloaded for later tests


def test_host_preimport_workarounds_lift_only_when_every_family_is_isolated(monkeypatch):
    from core.runtime_families import host_needs_runtime

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision,vlm")
    assert host_needs_runtime("torch")  # alignment and audio still import torch in-process
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision,vlm,alignment,audio")
    assert not host_needs_runtime("torch")
    monkeypatch.setattr("core.feature_registry.check_feature", lambda name: (False, ["package:lightning_whisper_mlx"]))
    assert not host_needs_runtime("mlx")
    monkeypatch.setattr("core.feature_registry.check_feature", lambda name: (True, []))
    assert host_needs_runtime("mlx")  # the MLX whisper backend runs in-process while it is installed
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


def test_selftest_calls_are_refused_without_test_tasks(monkeypatch, tmp_path):
    import core.runtime_supervisor as supervisor_module

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "audio")
    sup = RuntimeSupervisor(staging_root=tmp_path / "staging")  # production: no test tasks
    sup.launch_factory = lambda family: WorkerLaunch(
        interpreter=Path(sys.executable), worker_root=WORKER_ROOT, family=family, engine_root=REPO_ROOT,
    )
    monkeypatch.setattr(supervisor_module, "_default", sup)
    monkeypatch.setattr(supervisor_module, "default_supervisor", lambda: sup)
    try:
        with pytest.raises(runtime_families.IsolatedCallError, match="diagnostic"):
            run_isolated("audio", "selftest.identity", {})
    finally:
        sup.shutdown()


def test_dataclass_arguments_cross_the_boundary_and_bad_ones_fail_loudly(monkeypatch, worker_supervisor):
    from core.analysis.audio import OnsetDetectionConfig

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "audio")

    @isolated("audio", "selftest.identity")
    def configured(onset_config: OnsetDetectionConfig, path: Path, cancel_event=None):
        raise AssertionError("must run in the worker")

    echoed = configured(OnsetDetectionConfig(profile="tight", hop_length=256), Path("/tmp/a.wav"))
    assert echoed["kwargs"]["onset_config"]["hop_length"] == 256 and echoed["kwargs"]["path"] == "/tmp/a.wav"

    @isolated("audio", "selftest.identity")
    def unsendable(handle):
        raise AssertionError("must not run")

    with pytest.raises(runtime_families.IsolatedCallError, match="cannot be sent"):
        unsendable(object())


def test_worker_rebuilds_dataclass_and_path_arguments():
    from core.analysis.audio import OnsetDetectionConfig, analyze_audio
    from core.runtime_worker.tasks import _rebuild_arguments

    rebuilt = _rebuild_arguments(
        analyze_audio, {"audio_path": "/tmp/x.wav", "onset_config": {"profile": "tight", "hop_length": 128}, "sample_rate": 8000},
    )
    assert rebuilt["audio_path"] == Path("/tmp/x.wav")
    assert isinstance(rebuilt["onset_config"], OnsetDetectionConfig) and rebuilt["onset_config"].hop_length == 128
    assert rebuilt["sample_rate"] == 8000


def test_model_errors_progress_and_large_values_round_trip(monkeypatch, worker_supervisor):
    from core.errors import ModelDownloadError

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision")
    with pytest.raises(ModelDownloadError, match="weights"):
        run_isolated("vision", "selftest.model_error", {})
    messages = []
    big = run_isolated(
        "vision", "selftest.big_value", {"count": 40_000}, progress=lambda fraction, text: messages.append(text),
    )
    assert len(big) == 40_000 and big[-1] == 39_999  # > 256 KiB travelled through the staging file
    assert messages and messages[0] == "halfway"
    seen = []

    @isolated("vision", "selftest.execution_then_fail")
    def failing(image_path: Path, *, on_execution=None):
        raise AssertionError("must run in the worker")

    with pytest.raises(runtime_families.IsolatedCallError, match="after reporting"):
        failing(Path("/tmp/f.jpg"), on_execution=seen.append)
    assert seen == [{"model": "selftest", "device": "cpu"}]  # provenance of the failed attempt survives


def test_cancel_reaches_the_worker_for_functions_without_the_parameter(monkeypatch, worker_supervisor):
    import threading

    from core.runtime_supervisor import WorkerCancelled

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision")

    @isolated("vision", "selftest.wait_for_cancel")
    def blocking(image_path: Path):
        raise AssertionError("must run in the worker")

    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with pytest.raises(WorkerCancelled):
        blocking(Path("/tmp/x.jpg"), cancel_event=cancel)


def test_stems_decode_refuses_files_outside_the_output_dir(tmp_path):
    from core.analysis.stem_separation import _decode_stems

    good = _decode_stems({"vocals": str(tmp_path / "out" / "vocals.wav")}, {"output_dir": tmp_path / "out"})
    assert good == {"vocals": tmp_path / "out" / "vocals.wav"}
    with pytest.raises(RuntimeError, match="outside the output directory"):
        _decode_stems({"vocals": "/etc/passwd"}, {"output_dir": tmp_path / "out"})


def test_paddle_helpers_handle_both_engine_generations():
    from types import SimpleNamespace

    from core.analysis import ocr

    class Legacy:  # PaddleOCR 2.x: accepts show_log
        def __init__(self, use_angle_cls=True, lang="en", show_log=False):
            self.use_angle_cls = use_angle_cls

    class Modern:  # PaddleOCR 3.x: no show_log, predict() with rec_texts/rec_scores
        def __init__(self, lang="en", use_textline_orientation=True, use_doc_orientation_classify=False, use_doc_unwarping=False):
            pass

    legacy = ocr._construct_paddle_engine(Legacy)
    assert isinstance(legacy, Legacy) and legacy._scene_ripper_v3 is False
    modern = ocr._construct_paddle_engine(Modern)
    assert isinstance(modern, Modern) and modern._scene_ripper_v3 is True
    assert ocr._paddle_lines([[[None, ("SCENE", 0.9)], [None, ("RIPPER", 0.8)]]]) == [("SCENE", 0.9), ("RIPPER", 0.8)]
    page = SimpleNamespace(get=lambda key, default=None: {"rec_texts": ["SCENE", "RIPPER"], "rec_scores": [0.95, 0.85]}.get(key, default))
    assert ocr._paddle_lines([page]) == [("SCENE", 0.95), ("RIPPER", 0.85)]
    assert ocr._paddle_lines(None) == []


def test_runtime_isolate_cli_round_trips(monkeypatch, tmp_path):
    import json
    import subprocess

    env = {
        **os.environ, "SCENE_RIPPER_CONFIG": str(tmp_path / "config.json"),
        "SCENE_RIPPER_APP_SUPPORT_DIR": str(tmp_path / "support"), "SCENE_RIPPER_NATIVE_WORKERS": "1",
    }
    env.pop("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", None)
    run = lambda *args: subprocess.run([sys.executable, "-m", "cli.main", "--json", "runtime", *args], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)  # noqa: E731
    assert run("isolate", "audio", "vision").returncode == 0
    listed = json.loads(run("list").stdout)
    assert listed["isolated_families"] == ["audio", "transcription", "vision"]
    assert run("isolate", "--off", "vision").returncode == 0
    assert json.loads(run("list").stdout)["isolated_families"] == ["audio", "transcription"]
    bad = run("isolate", "bogus")
    assert bad.returncode == 7  # validation error, like an unknown profile
    assert run("isolate").returncode == 7


def test_chat_isolate_runtime_family_adds_and_removes_one_family(monkeypatch, tmp_path):
    from core.chat_tools import tools
    from core.settings import load_settings

    monkeypatch.setenv("SCENE_RIPPER_CONFIG", str(tmp_path / "config.json"))
    tool = tools.get("isolate_runtime_family").func
    assert tool("audio")["success"] and load_settings().native_worker_families == ["audio", "transcription"]
    assert tool("audio", enabled=False)["success"] and load_settings().native_worker_families == ["transcription"]
    assert tool("shell")["success"] is False
