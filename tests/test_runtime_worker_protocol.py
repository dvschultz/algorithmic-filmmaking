"""Worker protocol: framing, bounds, handshake, and worker-side task loop."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime_worker.protocol import (
    MAX_MESSAGE_BYTES, PROTOCOL_VERSION, HOST_TYPES, WORKER_TYPES, ProtocolError, decode, encode,
)

WORKER_ROOT = Path(__file__).resolve().parents[1] / "core"


def _run_worker(lines: list[dict], *, timeout: float = 30.0, allow_test_tasks: bool = True, staging: Path | None = None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PYTHONPATH"] = str(WORKER_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "runtime_worker"], input=b"".join(encode(m) for m in lines),
        capture_output=True, timeout=timeout, env=env, cwd=str(staging or WORKER_ROOT),
    )
    messages = [decode(line, expected_types=WORKER_TYPES) for line in proc.stdout.splitlines() if line.strip()]
    return proc, messages


def _hello(staging: Path, **extra) -> dict:
    return {"type": "hello", "protocol": PROTOCOL_VERSION, "worker_id": "w", "staging_dir": str(staging), "allow_test_tasks": True, **extra}


class TestFraming:
    def test_encode_decode_round_trip_and_type_checks(self):
        data = encode({"type": "task", "id": "1", "kind": "echo", "args": {"value": [1, 2]}})
        assert data.endswith(b"\n") and b"\n" not in data[:-1]
        assert decode(data, expected_types=HOST_TYPES)["kind"] == "echo"
        with pytest.raises(ProtocolError):
            decode(data, expected_types=WORKER_TYPES)
        with pytest.raises(ProtocolError):
            encode({"no_type": 1})
        with pytest.raises(ProtocolError):
            encode({"type": "result", "value": float("nan")})

    def test_size_bound_is_enforced_both_ways(self):
        with pytest.raises(ProtocolError, match="exceeds"):
            encode({"type": "result", "blob": "x" * MAX_MESSAGE_BYTES})
        with pytest.raises(ProtocolError, match="exceeds"):
            decode(b"{" + b"x" * MAX_MESSAGE_BYTES, expected_types=WORKER_TYPES)
        with pytest.raises(ProtocolError, match="Malformed"):
            decode(b"not json\n", expected_types=WORKER_TYPES)
        with pytest.raises(ProtocolError, match="Unexpected"):
            decode(b'{"type": "wat"}\n', expected_types=WORKER_TYPES)


class TestWorkerLoop:
    def test_handshake_then_task_then_shutdown(self, tmp_path):
        proc, messages = _run_worker([
            _hello(tmp_path),
            {"type": "task", "id": "t1", "kind": "echo", "args": {"value": 5}},
            {"type": "shutdown"},
        ])
        assert proc.returncode == 0, proc.stderr
        assert messages[0]["type"] == "ready" and messages[0]["protocol"] == PROTOCOL_VERSION
        assert "transcribe" in messages[0]["capabilities"] and "echo" in messages[0]["capabilities"]
        assert messages[1] == {"type": "result", "id": "t1", "result": {"echo": 5, "pid": messages[0]["pid"]}}
        assert (tmp_path / "t1").is_dir()

    def test_protocol_version_mismatch_is_refused(self, tmp_path):
        proc, messages = _run_worker([{**_hello(tmp_path), "protocol": 99}])
        assert proc.returncode == 2
        assert messages[0]["type"] == "error" and "unsupported" in messages[0]["error"]

    def test_missing_staging_and_bad_first_message_are_refused(self, tmp_path):
        proc, messages = _run_worker([_hello(tmp_path / "missing")])
        assert proc.returncode == 2 and "Staging" in messages[0]["error"]
        proc, messages = _run_worker([{"type": "shutdown"}])
        assert proc.returncode == 2 and messages[0]["error"] == "Expected hello"

    def test_test_tasks_are_hidden_unless_the_host_allows_them(self, tmp_path):
        proc, messages = _run_worker([
            {**_hello(tmp_path), "allow_test_tasks": False},
            {"type": "task", "id": "t1", "kind": "echo", "args": {}},
            {"type": "shutdown"},
        ])
        assert "echo" not in messages[0]["capabilities"]
        assert messages[1]["type"] == "error" and messages[1]["kind"] == "unsupported"

    def test_task_failures_keep_the_worker_alive_and_logs_stay_off_stdout(self, tmp_path):
        proc, messages = _run_worker([
            _hello(tmp_path),
            {"type": "task", "id": "bad", "kind": "transcribe", "args": {"media_path": str(tmp_path / "none.mp4")}},
            {"type": "task", "id": "ok", "kind": "echo", "args": {"value": 1}},
            {"type": "shutdown"},
        ])
        assert proc.returncode == 0
        assert messages[1]["type"] == "error" and messages[1]["id"] == "bad" and "not found" in messages[1]["error"]
        assert messages[2]["type"] == "result" and messages[2]["id"] == "ok"
        assert b"Traceback" not in proc.stdout

    def test_oversized_result_is_reported_as_a_protocol_error_not_written(self, tmp_path):
        proc, messages = _run_worker([
            _hello(tmp_path),
            {"type": "task", "id": "big", "kind": "big_output", "args": {"size": 2_000_000}},
            {"type": "shutdown"},
        ])
        assert messages[1]["type"] == "error" and messages[1]["kind"] == "protocol"
        assert all(len(line) <= MAX_MESSAGE_BYTES for line in proc.stdout.splitlines())

    def test_malformed_host_line_shuts_the_worker_down(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(WORKER_ROOT)
        proc = subprocess.run(
            [sys.executable, "-m", "runtime_worker"], input=encode(_hello(tmp_path)) + b"garbage\n",
            capture_output=True, timeout=30, env=env,
        )
        assert proc.returncode == 0 and b"protocol error" in proc.stderr
        assert json.loads(proc.stdout.splitlines()[0])["type"] == "ready"

    def test_cancel_message_stops_a_cooperative_task(self, tmp_path):
        proc, messages = _run_worker([
            _hello(tmp_path),
            {"type": "task", "id": "s", "kind": "sleep", "args": {"seconds": 30}},
            {"type": "cancel", "id": "s"},
            {"type": "shutdown"},
        ], timeout=20)
        assert messages[1] == {"type": "cancelled", "id": "s"}
