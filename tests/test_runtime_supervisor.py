"""Supervisor: crash containment, cancellation with tree teardown, bounded protocol, staging."""

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from core.runtime_supervisor import (
    ManagedWorker, RuntimeSupervisor, WorkerCancelled, WorkerCrashed, WorkerError, WorkerLaunch,
    WorkerProtocolViolation, WorkerTaskError, WorkerUnavailable, default_launch, validate_result_paths,
)

WORKER_ROOT = Path(__file__).resolve().parents[1] / "core"


def _launch() -> WorkerLaunch:
    return WorkerLaunch(interpreter=Path(sys.executable), worker_root=WORKER_ROOT, family="test")


@pytest.fixture
def supervisor(tmp_path):
    sup = RuntimeSupervisor(staging_root=tmp_path / "staging", allow_test_tasks=True)
    sup.launch_factory = lambda family: _launch()
    yield sup
    sup.shutdown()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_handshake_reports_interpreter_and_capabilities(supervisor):
    worker = supervisor.worker("test")
    assert worker.alive and worker.python and worker.pid
    assert "transcribe" in worker.capabilities
    assert supervisor.run("test", "echo", {"value": {"n": 1}})["echo"] == {"n": 1}


# Scenario 1: a crash is contained and the host keeps editing/saving.

def test_worker_crash_is_contained_and_project_edits_continue(supervisor, tmp_path):
    from core.project import Project
    from models.clip import Clip, Source

    project = Project(sources=[Source(id="s", file_path=tmp_path / "v.mp4", fps=30, duration_seconds=5)],
                      clips=[Clip(id="c", source_id="s", start_frame=0, end_frame=30)])
    first = supervisor.worker("test")
    with pytest.raises(WorkerCrashed) as info:
        supervisor.run("test", "crash", {"code": 3})
    assert not first.alive
    assert info.value.returncode in (3, None)
    # The host is unaffected: edit and save the project.
    project.add_to_sequence(["c"])
    assert project.save(tmp_path / "after-crash.sceneripper")
    # A fresh worker is spawned transparently for the next task.
    second = supervisor.worker("test")
    assert second is not first and supervisor.run("test", "echo", {"value": 1})["echo"] == 1


def test_worker_exit_during_a_task_surfaces_as_a_crash_not_a_hang(supervisor):
    worker = supervisor.worker("test")
    started = threading.Event()

    def kill_soon():
        started.wait(5)
        time.sleep(0.3)
        os.kill(worker.pid, signal.SIGKILL)

    threading.Thread(target=kill_soon, daemon=True).start()
    started.set()
    with pytest.raises(WorkerCrashed):
        supervisor.run("test", "sleep", {"seconds": 30})


# Scenario 2: cancellation of a blocked task terminates the process tree after a grace period.

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group teardown")
def test_cancelling_a_blocked_task_kills_the_worker_tree_after_grace(supervisor, tmp_path):
    worker = supervisor.worker("test")
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(WorkerCancelled, match="grace"):
        supervisor.run(
            "test", "sleep", {"seconds": 60, "ignore_cancel": True, "spawn_child": True},
            cancel_event=cancel, cancel_grace=1.0,
        )
    elapsed = time.monotonic() - started
    assert 1.0 <= elapsed < 8.0
    child_files = list(worker.staging_dir.glob("*/child_pid"))
    assert child_files, "child process was not recorded"
    child_pid = int(child_files[0].read_text())
    deadline = time.monotonic() + 5
    while _pid_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _pid_alive(child_pid), "child FFmpeg-style process survived cancellation"
    assert not worker.alive


def test_cooperative_cancellation_returns_promptly(supervisor):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(WorkerCancelled):
        supervisor.run("test", "sleep", {"seconds": 60}, cancel_event=cancel, cancel_grace=5.0)
    assert time.monotonic() - started < 3.0
    assert supervisor.worker("test").alive  # cooperative cancel keeps the warm worker


# Scenario 3: protocol mismatch, truncated/excessive/malformed output fail the job without data.

def test_protocol_mismatch_is_refused(tmp_path, monkeypatch):
    import core.runtime_supervisor as module

    monkeypatch.setattr(module, "PROTOCOL_VERSION", 99)
    worker = ManagedWorker(_launch(), staging_root=tmp_path, allow_test_tasks=True)
    with pytest.raises(WorkerProtocolViolation, match="Protocol"):
        worker.start()
    worker.close()


def test_excessive_and_malformed_output_fail_the_task_without_a_result(supervisor):
    with pytest.raises(WorkerTaskError, match="exceeds"):
        supervisor.run("test", "big_output", {"size": 2_000_000})
    with pytest.raises(WorkerProtocolViolation, match="Malformed"):
        supervisor.run("test", "raw_stdout", {})
    # The violating worker was retired; the family recovers.
    assert supervisor.run("test", "echo", {"value": "again"})["echo"] == "again"


def test_truncated_output_from_a_dying_worker_is_a_crash(supervisor):
    worker = supervisor.worker("test")
    threading.Timer(0.2, lambda: os.kill(worker.pid, signal.SIGKILL)).start()
    with pytest.raises(WorkerCrashed):
        supervisor.run("test", "sleep", {"seconds": 10})


def test_task_timeout_terminates_the_worker(supervisor):
    with pytest.raises(WorkerError, match="timed out"):
        supervisor.run("test", "sleep", {"seconds": 60, "ignore_cancel": True}, timeout=0.5, cancel_grace=0.5)


# Scenario 5: results outside staging are rejected; installs only via allowlisted profiles.

def test_result_paths_outside_staging_are_rejected(supervisor, tmp_path):
    with pytest.raises(WorkerProtocolViolation, match="outside"):
        supervisor.run("test", "escape_staging", {"path": "/etc/passwd"})
    staging = tmp_path / "s"
    staging.mkdir()
    (staging / "ok.json").write_text("{}")
    validate_result_paths({"result_path": str(staging / "ok.json"), "nested": [{"path": str(staging / "ok.json")}]}, staging)
    with pytest.raises(WorkerProtocolViolation):
        validate_result_paths({"result_path": str(staging / ".." / "ok.json")}, staging)
    with pytest.raises(WorkerProtocolViolation):
        validate_result_paths({"result_path": "relative.json"}, staging)
    link = staging / "link.json"
    if not sys.platform.startswith("win"):
        link.symlink_to(tmp_path / "outside.json")
        (tmp_path / "outside.json").write_text("{}")
        with pytest.raises(WorkerProtocolViolation):
            validate_result_paths({"result_path": str(link)}, staging)


def test_install_requests_cannot_name_packages_or_executables(monkeypatch):
    from core import runtime_profiles

    calls = []
    monkeypatch.setattr("core.feature_registry.install_for_feature", lambda name, cb=None: calls.append(name) or True)
    monkeypatch.setattr("core.feature_registry.check_feature", lambda name: (True, []))
    for bad in ("torch", "/usr/bin/python", "http://evil/pkg.whl", "transcribe; rm -rf /", None, 3):
        with pytest.raises(ValueError, match="Unknown runtime profile"):
            runtime_profiles.install_profile(bad)
    status = runtime_profiles.install_profile("transcription-whisper")
    assert status["success"] and calls == ["transcribe"]
    assert runtime_profiles.profile_status("transcription-whisper")["task_kinds"] == ["transcribe"]


# Launch resolution: explicit interpreters only, credentials never inherited.

def test_launch_uses_explicit_interpreter_and_strips_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("GROQ_TOKEN", "secret")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    launch = _launch()
    env = launch.environment()
    assert "OPENAI_API_KEY" not in env and "GROQ_TOKEN" not in env
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(WORKER_ROOT)
    assert launch.command()[0] == sys.executable and launch.command()[-2:] == ["-m", "runtime_worker"]
    fake = tmp_path / "python"
    monkeypatch.setenv("SCENE_RIPPER_WORKER_PYTHON", str(fake))
    with pytest.raises(WorkerUnavailable):
        default_launch()
    fake.write_text("#!/bin/sh\n")
    assert default_launch().interpreter == fake


def test_managed_packages_only_accompany_the_managed_interpreter(monkeypatch, tmp_path):
    import core.runtime_supervisor as module

    managed = tmp_path / "python" / "bin" / "python3"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n")
    packages = tmp_path / "packages"
    packages.mkdir()
    monkeypatch.setattr(module, "managed_interpreter_path", lambda: managed)
    monkeypatch.setattr("core.paths.get_managed_package_search_paths", lambda: [packages])
    monkeypatch.setenv("SCENE_RIPPER_WORKER_PYTHON", sys.executable)
    assert default_launch().package_paths == ()  # developer interpreter: no 3.11 wheels shadowing it
    monkeypatch.setenv("SCENE_RIPPER_WORKER_PYTHON", str(managed))
    assert default_launch().package_paths == (packages,)


def test_frozen_host_never_uses_its_own_executable(monkeypatch):
    monkeypatch.delenv("SCENE_RIPPER_WORKER_PYTHON", raising=False)
    monkeypatch.setattr("core.paths.is_frozen", lambda: True)
    monkeypatch.setattr("core.paths.get_managed_python_dir", lambda: Path("/nonexistent/managed"))
    with pytest.raises(WorkerUnavailable, match="Managed Python"):
        default_launch()


# Transcription integration: the host maps worker results back to segments.

def test_transcription_reads_worker_output_and_maps_failures(monkeypatch, tmp_path):
    from core import transcription

    class FakeSupervisor:
        def run(self, family, kind, args, **options):
            assert family == "transcription" and kind == "transcribe" and args["ffmpeg"]
            out = tmp_path / "transcript.json"
            out.write_text('{"language": "en", "duration": 1.0, "model": "tiny.en", "segments": [{"start": 0.0, "end": 0.5, "text": "hi", "confidence": -0.1, "words": [{"start": 0.0, "end": 0.5, "text": "hi", "probability": 0.9}]}]}')
            options["progress"](1.0, "done")
            return {"result_path": str(out), "segment_count": 1, "language": "en"}

    monkeypatch.setattr("core.runtime_supervisor.default_supervisor", lambda: FakeSupervisor())
    monkeypatch.setattr(transcription, "_require_ffmpeg", lambda: "/usr/bin/ffmpeg")
    seen = []
    segments, language = transcription._transcribe_in_worker(tmp_path / "a.mp4", "tiny.en", "en", lambda f, m: seen.append(m), extract_audio=True)
    assert language == "en" and segments[0].text == "hi" and segments[0].words[0].text == "hi" and seen == ["done"]

    class Crashing:
        def run(self, *args, **kwargs):
            raise WorkerCrashed("boom", 9, "stderr tail")

    monkeypatch.setattr("core.runtime_supervisor.default_supervisor", lambda: Crashing())
    with pytest.raises(transcription.TranscriptionError, match="crashed"):
        transcription._transcribe_in_worker(tmp_path / "a.mp4", "tiny.en", "en", None, extract_audio=False)


def test_native_worker_toggle(monkeypatch):
    from core import transcription

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "0")
    assert transcription.native_worker_enabled() is False
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    assert transcription.native_worker_enabled() is True
