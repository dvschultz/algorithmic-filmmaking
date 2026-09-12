"""Tests for frozen runtime smoke helpers."""

import os

import pytest

from core.runtime_smoke import (
    get_runtime_smoke_targets,
    run_runtime_smoke_target,
)


def test_runtime_smoke_targets_are_stable():
    """Smoke targets should enumerate the release validation surfaces."""
    assert get_runtime_smoke_targets() == (
        "imports",
        "project",
        "scene-detect",
        "transcription",
        "updater",
        "analyze-clip",
        "sequence-build",
        "render-short",
        "mcp-stdio",
        "native-worker",
        "native-analysis",
    )


def test_project_runtime_smoke_passes():
    """Project runtime smoke should exercise save/load successfully."""
    assert run_runtime_smoke_target("project") == "project"


def test_scene_detect_runtime_smoke_passes():
    """Scene detection runtime smoke should detect multiple synthetic clips."""
    assert run_runtime_smoke_target("scene-detect") == "scene-detect"


def test_transcription_runtime_smoke_passes():
    """Transcription smoke should exercise FFmpeg audio extraction."""
    assert run_runtime_smoke_target("transcription") == "transcription"


def test_analyze_clip_runtime_smoke_passes():
    """Analyze-clip smoke should run the non-ML color + brightness paths."""
    assert run_runtime_smoke_target("analyze-clip") == "analyze-clip"


def test_sequence_build_runtime_smoke_passes():
    """Sequence-build smoke should assemble a sequence from detected clips."""
    assert run_runtime_smoke_target("sequence-build") == "sequence-build"


def test_render_short_runtime_smoke_passes():
    """Render-short smoke should export a short sequence to a playable MP4."""
    assert run_runtime_smoke_target("render-short") == "render-short"


# Note: no live "mcp-stdio" pass-test here. That target spawns the MCP server as
# a subprocess, which is too heavy/slow for the unit suite; it is exercised via
# the frozen runtime smoke run in the release pipeline instead.


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


def test_native_worker_runtime_smoke_passes_in_source_mode(monkeypatch):
    """Handshake under an explicit interpreter plus real inference through the worker."""
    import importlib.util

    if importlib.util.find_spec("faster_whisper") is None:
        import pytest

        pytest.skip("faster-whisper not installed in this environment")
    monkeypatch.delenv("SCENE_RIPPER_WORKER_PYTHON", raising=False)
    # faster-whisper resolves model revisions through the Hub; offline runs need
    # network only when the tiny.en snapshot is not cached yet.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    before = dict(os.environ)
    assert run_runtime_smoke_target("native-worker") == "native-worker"
    assert os.environ.get("SCENE_RIPPER_NATIVE_WORKERS") == before.get("SCENE_RIPPER_NATIVE_WORKERS")


def test_native_analysis_runtime_smoke_runs_the_audio_family_in_source_mode(monkeypatch):
    """U14 packaged-proof target, exercised in source mode for the model-free audio family."""
    import importlib.util

    from core.runtime_profiles import profile_status

    if importlib.util.find_spec("librosa") is None:
        pytest.skip("librosa not installed")
    # The audio profile also covers Demucs stem separation (torch, torchaudio,
    # demucs_infer). Those live in requirements-optional.txt, so an environment
    # built from requirements.txt alone cannot run this family; the smoke would
    # report the profile "missing" rather than exercise the isolation seam.
    status = profile_status("audio-librosa")
    if not status["installed"]:
        pytest.skip("audio-librosa profile is not installed: " + ", ".join(status["missing"]))
    monkeypatch.setenv("SCENE_RIPPER_SMOKE_FAMILIES", "audio")
    monkeypatch.delenv("SCENE_RIPPER_SMOKE_INSTALL_PROFILES", raising=False)
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    assert run_runtime_smoke_target("native-analysis") == "native-analysis"


class _FakeWorker:
    """In-memory stand-in for ManagedWorker; records the launch it was started with."""

    _next_pid = 1000

    def __init__(self, launch):
        self.launch = launch
        self.python = launch.interpreter
        _FakeWorker._next_pid += 1
        self.pid = _FakeWorker._next_pid

    def run(self, kind, args, **options):
        assert kind == "echo"
        return {"echo": args["value"], "pid": self.pid}


class _FakeSupervisor:
    """RuntimeSupervisor stand-in: one worker per family, replaced by restart_family."""

    def __init__(self, **_kwargs):
        self.launch_factory = None
        self._workers = {}

    def worker(self, family="default"):
        worker = self._workers.get(family)
        if worker is None:
            worker = _FakeWorker(self.launch_factory(family))
            self._workers[family] = worker
        return worker

    def run(self, family, kind, args, **options):
        return self.worker(family).run(kind, args, **options)

    def restart_family(self, family):
        self._workers.pop(family, None)

    def shutdown(self):
        self._workers.clear()


def test_native_worker_smoke_transcribes_on_a_worker_that_sees_the_promoted_overlay(
    monkeypatch, tmp_path
):
    """Regression: the smoke must re-resolve the launch after installing the profile.

    A launch captured once, before the install, carries the package directories
    that existed at that moment. On a clean machine that set is empty, so the
    worker used for the transcription never had the promoted overlay on its
    path and faster-whisper was missing even though the install succeeded.
    """
    import core.runtime_profiles as runtime_profiles
    import core.runtime_supervisor as supervisor_module
    import core.transcription as transcription_module
    from core.runtime_supervisor import WorkerLaunch

    overlays = tmp_path / "packages-overlays"
    overlays.mkdir()
    promoted = overlays / "overlay-1-transcription-whisper"

    def fake_default_launch(family, *, ensure_interpreter=False, staged_paths=()):
        # Mirrors the real resolver: only directories that exist right now.
        return WorkerLaunch(
            interpreter=tmp_path / "python3",
            worker_root=tmp_path / "runtime_worker",
            package_paths=tuple(p for p in sorted(overlays.iterdir()) if p.is_dir()),
            family=family,
        )

    installed = {"done": False}

    def fake_profile_status(profile_id):
        return {"profile": profile_id, "installed": installed["done"], "missing": ["faster-whisper"]}

    def fake_install_profile(profile_id, *args, **kwargs):
        promoted.mkdir()
        installed["done"] = True
        # The real install retires the family worker through the default supervisor.
        supervisor_module.default_supervisor().restart_family("transcription")
        return {"profile": profile_id, "installed": True, "missing": [], "promoted_dir": str(promoted)}

    seen: dict = {}

    def fake_transcribe(media_path, model, language, progress, *, extract_audio, cancel_event=None):
        worker = supervisor_module.default_supervisor().worker("transcription")
        seen["package_paths"] = tuple(worker.launch.package_paths)
        return ([], "en")

    monkeypatch.setenv("SCENE_RIPPER_SMOKE_INSTALL_PROFILES", "1")
    monkeypatch.setattr("core.paths.is_frozen", lambda: False)
    previous_default = supervisor_module._default
    monkeypatch.setattr(supervisor_module, "RuntimeSupervisor", _FakeSupervisor)
    monkeypatch.setattr(supervisor_module, "default_launch", fake_default_launch)
    monkeypatch.setattr(runtime_profiles, "profile_status", fake_profile_status)
    monkeypatch.setattr(runtime_profiles, "install_profile", fake_install_profile)
    monkeypatch.setattr(transcription_module, "_transcribe_in_worker", fake_transcribe)

    assert run_runtime_smoke_target("native-worker") == "native-worker"
    assert seen["package_paths"] == (promoted,), (
        "transcription ran on a worker whose path lacks the freshly promoted overlay"
    )
    assert supervisor_module._default is previous_default


def test_native_analysis_records_every_family_when_one_install_raises(monkeypatch, tmp_path):
    """Regression: a raising install must not cost the evidence for later families.

    The packaged macOS run installed profiles in order and vlm-local's install
    raised a version conflict. That exception unwound the whole sweep, so no
    family after it was ever probed and the per-family evidence U14 needs was
    lost behind a single traceback.
    """
    import core.runtime_profiles as runtime_profiles
    import core.runtime_smoke as smoke
    from core.runtime_profiles import RuntimeProfile

    profiles = {
        "aaa-first": RuntimeProfile(
            id="aaa-first", family="ocr", features=("f",), task_kinds=("analysis",),
            description="", probe_module="aaa",
        ),
        "bbb-raises": RuntimeProfile(
            id="bbb-raises", family="vision", features=("f",), task_kinds=("analysis",),
            description="", probe_module="bbb",
        ),
        "ccc-last": RuntimeProfile(
            id="ccc-last", family="audio", features=("f",), task_kinds=("analysis",),
            description="", probe_module="ccc",
        ),
    }

    def fake_install_profile(profile_id, *args, **kwargs):
        if profile_id == "bbb-raises":
            raise RuntimeError("bbb runtime is incomplete in the vision worker")
        return {"profile": profile_id, "installed": True, "missing": []}

    monkeypatch.setenv("SCENE_RIPPER_SMOKE_INSTALL_PROFILES", "1")
    monkeypatch.delenv("SCENE_RIPPER_SMOKE_FAMILIES", raising=False)
    monkeypatch.setattr(runtime_profiles, "PROFILES", profiles)
    monkeypatch.setattr(runtime_profiles, "install_profile", fake_install_profile)
    monkeypatch.setattr(
        runtime_profiles, "profile_status",
        lambda pid: {"profile": pid, "installed": False, "missing": ["dep"]},
    )
    monkeypatch.setattr(
        runtime_profiles, "probe_profile_runtime", lambda pid, **kw: {"ok": True, "version": "1"}
    )
    monkeypatch.setattr(smoke, "_analysis_smoke_families", lambda: frozenset({"ocr", "vision", "audio"}))
    monkeypatch.setattr(smoke, "_FAMILY_SMOKE_CALLS", {})

    logged: list[str] = []
    monkeypatch.setattr(smoke.logger, "info", lambda msg, *a: logged.append(str(msg) % a if a else str(msg)))

    with pytest.raises(RuntimeError, match="bbb-raises"):
        run_runtime_smoke_target("native-analysis")

    reported = [line for line in logged if line.startswith("Native analysis family result")]
    assert any("aaa-first -> ok" in line for line in reported), reported
    assert any("ccc-last -> ok" in line for line in reported), reported
    assert any("bbb-raises -> failed" in line for line in reported), reported


def test_native_analysis_calls_a_failed_install_a_failure_not_a_missing_profile(monkeypatch, tmp_path):
    """An install that was attempted and did not land must fail the target."""
    import core.runtime_profiles as runtime_profiles
    import core.runtime_smoke as smoke
    from core.runtime_profiles import RuntimeProfile

    profile = RuntimeProfile(
        id="ocr-x", family="ocr", features=("f",), task_kinds=("analysis",),
        description="", probe_module="x",
    )
    monkeypatch.setenv("SCENE_RIPPER_SMOKE_INSTALL_PROFILES", "1")
    monkeypatch.delenv("SCENE_RIPPER_SMOKE_FAMILIES", raising=False)
    monkeypatch.setattr(runtime_profiles, "PROFILES", {"ocr-x": profile})
    monkeypatch.setattr(smoke, "_analysis_smoke_families", lambda: frozenset({"ocr"}))
    monkeypatch.setattr(
        runtime_profiles, "profile_status",
        lambda pid: {"profile": pid, "installed": False, "missing": ["paddlepaddle"]},
    )
    monkeypatch.setattr(
        runtime_profiles, "install_profile",
        lambda pid, *a, **k: {"profile": pid, "installed": False, "missing": ["paddlepaddle"],
                              "error": "f: conflict"},
    )

    with pytest.raises(RuntimeError, match="install did not complete"):
        run_runtime_smoke_target("native-analysis")
