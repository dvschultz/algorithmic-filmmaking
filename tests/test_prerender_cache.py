"""Prerender reuse follows actual media, range, and rendering inputs."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.remix.prerender import prerender_clip


@pytest.fixture
def render(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(f"render-{len(calls)}".encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("core.remix.prerender.find_binary", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("core.remix.prerender.subprocess.run", run)
    args = dict(source_path=source, start_frame=0, end_frame=30, fps=30.0,
                hflip=True, vflip=False, reverse=False, output_dir=tmp_path / "cache", clip_id="clip")
    return args, calls


@pytest.mark.parametrize("change", ["source", "start", "end", "fps", "transform"])
def test_prerender_invalidates_changed_inputs(render, change):
    args, calls = render
    first = prerender_clip(**args)
    assert first is not None
    assert prerender_clip(**args) == first and len(calls) == 1
    if change == "source":
        before = args["source_path"].stat()
        args["source_path"].write_bytes(b"change")
        os.utime(args["source_path"], ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "start":
        args["start_frame"] = 1
    elif change == "end":
        args["end_frame"] = 31
    elif change == "fps":
        args["fps"] = 24.0
    else:
        args["vflip"] = True
    second = prerender_clip(**args)
    assert second is not None and second != first and len(calls) == 2


def test_prerender_does_not_trust_old_filename_or_damaged_payload(render):
    args, calls = render
    legacy = args["output_dir"] / "clip_1_0_0.mp4"
    legacy.parent.mkdir()
    legacy.write_bytes(b"unverified")
    first = prerender_clip(**args)
    assert first != legacy and len(calls) == 1
    first.write_bytes(b"corrupt")
    assert prerender_clip(**args) is not None and len(calls) == 2
    assert legacy.read_bytes() == b"unverified"


def test_failed_prerender_does_not_publish_partial_output(render, monkeypatch):
    args, calls = render

    def fail(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="failed")

    monkeypatch.setattr("core.remix.prerender.subprocess.run", fail)
    assert prerender_clip(**args) is None
    assert prerender_clip(**args) is None
    assert len(calls) == 2
    assert not list(args["output_dir"].glob("render-*"))


def test_prerender_identity_is_independent_of_editor_clip_id(render):
    args, calls = render
    first = prerender_clip(**args)
    args["clip_id"] = "another-editor-occurrence"
    assert prerender_clip(**args) == first
    assert len(calls) == 1


def test_prerender_runtime_change_invalidates_cached_output(render, monkeypatch):
    args, calls = render
    binary = args["source_path"].parent / "ffmpeg"
    binary.write_bytes(b"runtime v1")
    monkeypatch.setattr("core.remix.prerender.find_binary", lambda _: str(binary))
    first = prerender_clip(**args)
    binary.write_bytes(b"runtime v2")
    assert prerender_clip(**args) != first and len(calls) == 2


def test_batch_hashes_a_shared_source_once(render, monkeypatch):
    from core.remix.prerender import prerender_batch

    args, calls = render
    source = SimpleNamespace(file_path=args["source_path"], fps=30.0)
    entries = [(SimpleNamespace(id=str(i), start_frame=i, end_frame=i + 30), source,
                {"hflip": True}) for i in range(4)]
    opens = []
    original = Path.open

    def track_open(path, *positional, **kwargs):
        if path == source.file_path and positional == ("rb",):
            opens.append(path)
        return original(path, *positional, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    results = prerender_batch(entries, args["output_dir"])
    assert len(results) == 4 and all(result[2] is not None for result in results)
    assert len(opens) == 1


def test_source_change_during_prerender_is_not_published(render, monkeypatch):
    args, calls = render

    def mutate(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"stale")
        args["source_path"].write_bytes(b"changed")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("core.remix.prerender.subprocess.run", mutate)
    assert prerender_clip(**args) is None
    assert not list(args["output_dir"].glob("render-*"))


def test_project_local_prerender_copy_cannot_mutate_shared_cache(render):
    from core.project import Project
    from models.sequence import SequenceClip

    args, calls = render
    output = prerender_clip(**args)
    assert output is not None
    original = output.read_bytes()
    project = Project.new()
    project.sequence.tracks[0].clips.append(SequenceClip(out_point=30, prerendered_path=str(output)))
    saved_path = args["source_path"].parent / "saved" / "project.json"
    assert project.save(saved_path)
    restored = Project.load(saved_path)
    local = Path(restored.sequence.get_all_clips()[0].prerendered_path)
    assert local != output and not local.samefile(output)
    local.write_bytes(b"edited project file")
    assert output.read_bytes() == original
    assert prerender_clip(**args) == output and len(calls) == 1


def test_runtime_change_during_prerender_is_not_published(render, monkeypatch):
    args, calls = render
    binary = args["source_path"].parent / "ffmpeg"
    binary.write_bytes(b"runtime v1")
    monkeypatch.setattr("core.remix.prerender.find_binary", lambda _: str(binary))

    def mutate(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"unverified runtime")
        binary.write_bytes(b"runtime v2")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("core.remix.prerender.subprocess.run", mutate)
    assert prerender_clip(**args) is None
