"""Real-process writer exclusion and replacement-safe ownership."""

import asyncio
import os
import subprocess
import sys

import pytest

from core.project_lock import ProjectBusyError, ProjectWriter, project_writer


@pytest.fixture
def lock_root(tmp_path, monkeypatch):
    root = tmp_path / "locks"
    monkeypatch.setattr("core.project_lock._lock_directory", lambda: root)
    return root


def attempt(path, lock_root):
    code = """
import sys
from pathlib import Path
import core.project_lock as locks
locks._lock_directory = lambda: Path(sys.argv[2])
try:
    with locks.ProjectWriter(sys.argv[1]):
        print('acquired')
except locks.ProjectBusyError:
    print('busy')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(lock_root)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_excludes_other_process_and_releases_without_deleting_records(
    tmp_path, lock_root
):
    path = tmp_path / "project.json"
    with ProjectWriter(path):
        assert attempt(path, lock_root) == "busy"
    records = list(lock_root.glob("*.lock"))
    assert records
    assert attempt(path, lock_root) == "acquired"
    assert all(record.exists() for record in records)


def test_symlink_and_hardlink_aliases_share_ownership(tmp_path, lock_root):
    path = tmp_path / "project.json"
    path.write_text("old")
    symlink = tmp_path / "symlink.json"
    hardlink = tmp_path / "hardlink.json"
    symlink.symlink_to(path)
    os.link(path, hardlink)
    with ProjectWriter(path):
        assert attempt(symlink, lock_root) == "busy"
        assert attempt(hardlink, lock_root) == "busy"


def test_replacement_retains_path_and_new_file_identity(tmp_path, lock_root):
    path = tmp_path / "project.json"
    path.write_text("old")
    temporary = tmp_path / "replacement.tmp"
    temporary.write_text("new")
    with ProjectWriter(path) as writer:
        writer.replace(temporary)
        assert path.read_text() == "new"
        assert attempt(path, lock_root) == "busy"
        alias = tmp_path / "new-alias.json"
        os.link(path, alias)
        assert attempt(alias, lock_root) == "busy"
    assert attempt(path, lock_root) == "acquired"


def test_process_death_releases_ownership(tmp_path, lock_root):
    path = tmp_path / "project.json"
    code = """
import sys
from pathlib import Path
import core.project_lock as locks
locks._lock_directory = lambda: Path(sys.argv[2])
with locks.ProjectWriter(sys.argv[1]):
    print('ready', flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path), str(lock_root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        assert attempt(path, lock_root) == "busy"
        process.terminate()
        process.wait(timeout=10)
        assert attempt(path, lock_root) == "acquired"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_nested_scope_reuses_owner_but_not_inherited_async_task(tmp_path, lock_root):
    path = tmp_path / "project.json"

    async def contender():
        with pytest.raises(ProjectBusyError):
            with project_writer(path):
                pass

    async def run():
        with project_writer(path) as first:
            with project_writer(path) as second:
                assert first is second
            await asyncio.create_task(contender())
        with project_writer(path):
            pass

    asyncio.run(run())


def test_failed_replacement_keeps_existing_ownership(tmp_path, lock_root, monkeypatch):
    path = tmp_path / "project.json"
    path.write_text("old")
    temporary = tmp_path / "replacement.tmp"
    temporary.write_text("new")

    def fail(*args):
        raise OSError("replacement failed")

    with ProjectWriter(path) as writer:
        monkeypatch.setattr("core.project_lock.os.replace", fail)
        with pytest.raises(OSError):
            writer.replace(temporary)
        assert path.read_text() == "old"
        assert attempt(path, lock_root) == "busy"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork inheritance")
def test_forked_child_cannot_reuse_parent_writer_context(tmp_path, lock_root):
    code = """
import os, sys
from pathlib import Path
import core.project_lock as locks
locks._lock_directory = lambda: Path(sys.argv[2])
with locks.project_writer(sys.argv[1]):
    pid = os.fork()
    if pid == 0:
        try:
            with locks.project_writer(sys.argv[1]):
                os._exit(9)
        except locks.ProjectBusyError:
            os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "project.json"), str(lock_root)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_shared_save_refuses_competing_owner(tmp_path, lock_root):
    from core.project import Project

    path = tmp_path / "project.json"
    project = Project.new("Original")
    assert project.save(path)
    original = path.read_bytes()
    with ProjectWriter(path):
        project.rename("Unsaved")
        assert not project.save(path)
        assert path.read_bytes() == original
        assert project.is_dirty


def test_mcp_mutation_owns_file_before_loading_and_releases_after_return(tmp_path, lock_root, monkeypatch):
    import core.spine.project_io as project_io
    from core.project import Project
    from scene_ripper_mcp.tools.clips import add_clip_note
    import json

    path = tmp_path / "project.sceneripper"
    assert Project.new().save(path)
    load = project_io.load_with_mtime
    observed = []

    def inspect_load(target):
        observed.append(attempt(target, lock_root))
        return load(target)

    monkeypatch.setattr(project_io, "load_with_mtime", inspect_load)
    result = json.loads(asyncio.run(add_clip_note(str(path), "missing", "note")))
    assert not result["success"]
    assert observed == ["busy"]
    assert attempt(path, lock_root) == "acquired"


def test_mcp_writer_conflict_is_structured_and_does_not_load(tmp_path, lock_root, monkeypatch):
    import core.spine.project_io as project_io
    from core.project import Project
    from scene_ripper_mcp.tools.clips import add_clip_note
    import json

    path = tmp_path / "project.sceneripper"
    assert Project.new().save(path)
    original = path.read_bytes()
    monkeypatch.setattr(project_io, "load_with_mtime", lambda path: pytest.fail("loaded without ownership"))
    with ProjectWriter(path):
        result = json.loads(asyncio.run(add_clip_note(str(path), "missing", "note")))
    assert result["error"]["code"] == "project_busy"
    assert path.read_bytes() == original
