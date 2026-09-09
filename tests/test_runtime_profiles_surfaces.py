"""U14: runtime profile status and install/repair are consistent on every surface.

Scenario 1 (same requirement everywhere), scenario 2 (interrupted install
keeps the previous runtime; covered end to end in tests/test_runtime_supervisor.py
and here through the MCP job), scenario 3 (no install into a live
interpreter: the host never imports the runtime), plus agent-native parity
for chat, MCP and the CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_support(monkeypatch, tmp_path):
    root = tmp_path / "support"
    monkeypatch.setenv("SCENE_RIPPER_APP_SUPPORT_DIR", str(root))
    monkeypatch.setattr("core.paths.get_app_support_dir", lambda: root)
    return root


def _missing(monkeypatch, missing: list[str]):
    monkeypatch.setattr("core.feature_registry.check_feature", lambda name: (not missing, list(missing)))


def test_status_is_identical_across_spine_chat_and_cli(monkeypatch, isolated_support):
    from core.chat_tools import tools
    from core.spine.runtime import get_runtime_profile_status, list_runtime_profiles

    _missing(monkeypatch, ["package:faster-whisper"])
    spine = list_runtime_profiles()
    chat = tools.get("list_runtime_profiles").func()
    assert spine == chat and spine["profiles"][0]["missing"] == ["package:faster-whisper"]
    assert spine["profiles"][0]["installed"] is False
    single = get_runtime_profile_status("transcription-whisper")
    assert single["success"] and single["missing"] == ["package:faster-whisper"]
    assert tools.get("install_runtime_profile").conflicts_with_workers
    assert not tools.get("install_runtime_profile").requires_project
    assert get_runtime_profile_status("torch")["success"] is False  # package names are refused

    cli = subprocess.run(
        [sys.executable, "-m", "cli.main", "--json", "runtime", "status", "transcription-whisper"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "SCENE_RIPPER_APP_SUPPORT_DIR": str(isolated_support)},
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["profile"] == "transcription-whisper"
    bad = subprocess.run(
        [sys.executable, "-m", "cli.main", "--json", "runtime", "install", "torch"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert bad.returncode != 0 and "Unknown runtime profile" in bad.stdout + bad.stderr


def test_probe_status_reports_a_broken_runtime_without_importing_it(monkeypatch, isolated_support):
    from core import runtime_profiles
    from core.spine.runtime import get_runtime_profile_status

    _missing(monkeypatch, [])
    monkeypatch.setattr(
        runtime_profiles, "probe_profile_runtime",
        lambda profile, staged_paths=(): (_ for _ in ()).throw(RuntimeError("worker: no StorageView")),
    )
    modules_before = set(sys.modules)
    status = get_runtime_profile_status("transcription-whisper", probe=True)
    assert status["installed"] is False and "StorageView" in status["health_error"]
    assert not {m for m in set(sys.modules) - modules_before if m.split(".")[0] in ("faster_whisper", "ctranslate2")}


@pytest.mark.asyncio
async def test_mcp_install_job_is_durable_and_cancellable(monkeypatch, isolated_support, tmp_path):
    from scene_ripper_mcp.jobs import JobRuntime, JobStore
    from scene_ripper_mcp.jobs.store import STATUS_CANCELLED, STATUS_COMPLETED
    from scene_ripper_mcp.tools.jobs import cancel_job, get_job_result
    from scene_ripper_mcp.tools import runtime as runtime_tools
    from core import runtime_profiles

    store = JobStore(tmp_path / "jobs.db")
    job_runtime = JobRuntime(store, max_workers=2)
    ctx = AsyncMock()
    ctx.request_context = SimpleNamespace(lifespan_context={"job_store": store, "job_runtime": job_runtime})
    _missing(monkeypatch, [])
    monkeypatch.setattr(
        runtime_profiles, "probe_profile_runtime",
        lambda profile, staged_paths=(), **kw: {"ok": True, "file": str((staged_paths or (isolated_support / "packages",))[0] / "faster_whisper" / "__init__.py")},
    )

    def quick_stage(name, target_dir, progress_callback=None, cancel_event=None):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "ok").write_text(name)
        return True

    monkeypatch.setattr("core.feature_registry.stage_feature_packages", quick_stage)
    try:
        refused = json.loads(await runtime_tools.start_install_runtime_profile("torch", ctx=ctx))
        assert refused["success"] is False and refused["error"]["code"] == "unknown_profile"

        started = json.loads(await runtime_tools.start_install_runtime_profile("transcription-whisper", ctx=ctx))
        assert started["success"], started
        _wait(store, started["task_id"], STATUS_COMPLETED)
        output = json.loads(await get_job_result(started["task_id"], ctx=ctx))
        result = output["result"]["result"] if "result" in output["result"] else output["result"]
        assert result["success"] and Path(result["promoted_dir"]).is_dir()
        first_overlay = Path(result["promoted_dir"])

        gate = threading.Event()

        def slow_stage(name, target_dir, progress_callback=None, cancel_event=None):
            target_dir.mkdir(parents=True, exist_ok=True)
            gate.wait(5)
            return not (cancel_event is not None and cancel_event.is_set())

        monkeypatch.setattr("core.feature_registry.stage_feature_packages", slow_stage)
        entered = threading.Event()

        def slow_stage_signalling(name, target_dir, progress_callback=None, cancel_event=None):
            entered.set()
            return slow_stage(name, target_dir, progress_callback, cancel_event)

        monkeypatch.setattr("core.feature_registry.stage_feature_packages", slow_stage_signalling)
        second = json.loads(await runtime_tools.start_install_runtime_profile("transcription-whisper", ctx=ctx))
        assert second["success"]
        assert entered.wait(5)  # the job is inside the staged install now
        cancelled = json.loads(await cancel_job(second["task_id"], ctx=ctx))
        assert cancelled.get("ok") or cancelled.get("success")
        gate.set()
        final = _wait(store, second["task_id"], {STATUS_CANCELLED, STATUS_COMPLETED})
        assert final == STATUS_CANCELLED or json.loads(
            await get_job_result(second["task_id"], ctx=ctx)
        )["result"].get("cancelled")
        assert runtime_profiles.profile_overlays("transcription-whisper") == [first_overlay]
        assert not any((isolated_support / "packages-staging").iterdir())
        listed = json.loads(await runtime_tools.list_runtime_profiles(ctx=ctx))
        assert listed["profiles"][0]["overlays"] == [first_overlay.name]
        rolled = json.loads(await runtime_tools.rollback_runtime_profile("transcription-whisper", ctx=ctx))
        assert rolled["success"] and runtime_profiles.profile_overlays("transcription-whisper") == []
    finally:
        job_runtime.shutdown(wait=True)


def _wait(store, task_id, expected, timeout=10.0):
    targets = {expected} if isinstance(expected, str) else set(expected)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = store.get(task_id).status
        if status in targets:
            return status
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {targets}; last={status!r}")


def test_ui_install_prompt_uses_the_staged_profile_path_when_isolated(monkeypatch):
    from ui.widgets import dependency_widgets

    calls = []
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setattr("core.spine.runtime.install_runtime_profile", lambda profile, progress_callback=None, cancel_event=None: calls.append(("profile", profile)) or {"success": True})
    legacy = lambda name, cb: calls.append(("legacy", name)) or True  # noqa: E731
    assert dependency_widgets._install_feature("transcribe", None, legacy)
    assert dependency_widgets._install_feature("ocr", None, legacy)  # no profile yet: in-place path
    assert calls == [("profile", "transcription-whisper"), ("legacy", "ocr")]
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "0")
    assert dependency_widgets._install_feature("transcribe", None, legacy)
    assert calls[-1] == ("legacy", "transcribe")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setattr("core.spine.runtime.install_runtime_profile", lambda profile, **kw: {"success": False, "error": "Health check failed; previous runtime kept"})
    with pytest.raises(RuntimeError, match="previous runtime kept"):
        dependency_widgets._install_feature("transcribe", None, legacy)  # the dialog shows the real reason
